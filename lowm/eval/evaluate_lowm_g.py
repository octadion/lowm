"""Evaluate LOWM-G candidate rollouts with OMC critic reranking."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from lowm.data.dataset import LOWMSynthRankingDataset, ranking_config_from_mapping
from lowm.data.simulate import config_from_mapping, step_state
from lowm.eval.evaluate_all import _ensure_split, load_run_model
from lowm.models.lowm_g import OperatorConditionedProposalModel, lowm_g_config_from_mapping


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return loaded if isinstance(loaded, dict) else {}


def _load_proposal(run_dir: Path, checkpoint: str, device: torch.device) -> tuple[OperatorConditionedProposalModel, dict[str, Any]]:
    ckpt = run_dir / "checkpoints" / checkpoint
    if not ckpt.exists():
        ckpt = run_dir / "checkpoints" / "last.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing LOWM-G checkpoint under {run_dir / 'checkpoints'}")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    config = payload.get("config")
    if not isinstance(config, dict):
        config = _load_yaml(run_dir / "config.yaml")
    model = OperatorConditionedProposalModel(lowm_g_config_from_mapping(config))
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, config


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    active = mask[:, 1:, :, None].to(pred.dtype)
    diff = (pred[:, 1:, :, 0:4] - target[:, 1:, :, 0:4]) * active
    denom = active.sum(dim=(1, 2, 3)).clamp_min(1.0) * 4.0
    return diff.square().sum(dim=(1, 2, 3)) / denom


def _candidate_diversity(candidates: torch.Tensor, mask: torch.Tensor) -> float:
    if candidates.shape[0] < 2:
        return 0.0
    vals = []
    for i in range(candidates.shape[0]):
        for j in range(i + 1, candidates.shape[0]):
            vals.append(float(_masked_mse(candidates[i : i + 1], candidates[j : j + 1], mask[:1]).item()))
    return float(np.mean(vals)) if vals else 0.0


def _score_candidates(
    critic: torch.nn.Module,
    context_states: torch.Tensor,
    context_actions: torch.Tensor,
    context_mask: torch.Tensor,
    cand_states: torch.Tensor,
    actions: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    k = cand_states.shape[0]
    cs = context_states.unsqueeze(0).to(device)
    ca = context_actions.unsqueeze(0).to(device)
    cm = context_mask.unsqueeze(0).to(device)
    states = cand_states.unsqueeze(0).to(device)
    acts = actions.unsqueeze(0).expand(k, -1, -1, -1).unsqueeze(0).to(device)
    masks = mask.unsqueeze(0).expand(k, -1, -1).unsqueeze(0).to(device)
    with torch.no_grad():
        if hasattr(critic, "encode_lambda") and hasattr(critic, "energy"):
            _, _, lam = critic.encode_lambda(cs, ca, cm)
            return critic.energy(states, acts, masks, lam).squeeze(0).detach().cpu()
        batch = {"context_states": cs, "context_actions": ca, "context_mask": cm, "cand_states": states, "cand_actions": acts, "cand_mask": masks, "labels": torch.zeros(1, dtype=torch.long, device=device)}
        out = critic(batch)
        energies = out["energies"] if isinstance(out, Mapping) else out
        return energies.squeeze(0).detach().cpu()


def _context_from_episode(dataset: LOWMSynthRankingDataset, episode: int, context_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    context_len = max(1, min(context_len, dataset.T))
    states = np.zeros((context_len, 2, dataset.nmax, dataset.d_object), dtype=np.float32)
    actions = np.zeros((context_len, dataset.nmax, 2), dtype=np.float32)
    mask = np.zeros((context_len, 2, dataset.nmax), dtype=np.float32)
    for t in range(context_len):
        states[t, 0] = dataset.states[episode, t]
        states[t, 1] = dataset.states[episode, t + 1]
        actions[t] = dataset.actions[episode, t]
        mask[t, 0] = dataset.mask[episode, t]
        mask[t, 1] = dataset.mask[episode, t + 1]
    return torch.from_numpy(states), torch.from_numpy(actions), torch.from_numpy(mask)


def _distractors(dataset: LOWMSynthRankingDataset, op_id: int, count: int, rng: np.random.Generator) -> list[int]:
    candidates = [int(i) for i in range(dataset.num_episodes) if int(dataset.op_id[i]) != int(op_id)]
    rng.shuffle(candidates)
    return candidates[:count]


def _operator_coherence(
    critic: torch.nn.Module,
    dataset: LOWMSynthRankingDataset,
    item: Mapping[str, Any],
    selected: torch.Tensor,
    actions: torch.Tensor,
    mask: torch.Tensor,
    context_len: int,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    own = (item["context_states"][:context_len], item["context_actions"][:context_len], item["context_mask"][:context_len])
    eps = _distractors(dataset, int(item["query_op_id"].item()), 3, rng)
    contexts = [own, *[_context_from_episode(dataset, ep, context_len) for ep in eps]]
    energies = []
    for ctx in contexts:
        energies.append(float(_score_candidates(critic, ctx[0], ctx[1], ctx[2], selected.unsqueeze(0), actions, mask, device)[0].item()))
    wrong = np.asarray(energies[1:], dtype=np.float32)
    own_e = float(energies[0])
    return {
        "own_operator_energy": own_e,
        "mean_wrong_operator_energy": float(wrong.mean()) if wrong.size else 0.0,
        "min_wrong_operator_energy": float(wrong.min()) if wrong.size else 0.0,
        "own_vs_wrong_gap_mean": float(wrong.mean() - own_e) if wrong.size else 0.0,
        "own_vs_wrong_gap_min": float(wrong.min() - own_e) if wrong.size else 0.0,
        "own_vs_wrong_pair_acc": float(np.mean(own_e < wrong)) if wrong.size else 0.0,
    }


def _generate_candidates(
    proposal: OperatorConditionedProposalModel,
    item: Mapping[str, Any],
    num_candidates: int,
    context_length: int,
    noise_scale: float,
    device: torch.device,
) -> torch.Tensor:
    cs = item["context_states"][:context_length].unsqueeze(0).expand(num_candidates, -1, -1, -1, -1).to(device)
    ca = item["context_actions"][:context_length].unsqueeze(0).expand(num_candidates, -1, -1, -1).to(device)
    cm = item["context_mask"][:context_length].unsqueeze(0).expand(num_candidates, -1, -1, -1).to(device)
    init = item["pos_states"][0].unsqueeze(0).expand(num_candidates, -1, -1).to(device)
    actions = item["pos_actions"].unsqueeze(0).expand(num_candidates, -1, -1, -1).to(device)
    mask = item["pos_mask"].unsqueeze(0).expand(num_candidates, -1, -1).to(device)
    with torch.no_grad():
        out = proposal(cs, ca, cm, init, actions, mask, noise_scale=noise_scale)
    return out["pred_states"].detach().cpu()


def _simulate_counterfactual_gt(proposal_config: Mapping[str, Any], dataset: LOWMSynthRankingDataset, item: Mapping[str, Any], target_ep: int) -> torch.Tensor:
    data_cfg = dict(proposal_config.get("data", {}))
    sim_cfg = config_from_mapping(_load_yaml(Path(data_cfg.get("dataset_config", "configs/lowm_synth_ood_param.yaml"))))
    actions = item["pos_actions"].numpy()
    s0 = item["pos_states"][0].numpy()
    mask0 = item["pos_mask"][0].numpy()
    n = int(np.sum(mask0 > 0.5))
    states = np.zeros_like(item["pos_states"].numpy())
    states[0] = s0
    for t in range(actions.shape[0]):
        states[t + 1] = step_state(states[t], actions[t], n, int(dataset.op_id[target_ep]), dataset.op_params[target_ep], sim_cfg)
    return torch.from_numpy(states.astype(np.float32))


def _normal_p_from_z(z: float) -> float:
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def _wilcoxon_like_p(values: np.ndarray) -> float:
    nonzero = values[np.abs(values) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return 1.0
    order = np.argsort(np.abs(nonzero))
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    w_plus = ranks[nonzero > 0].sum()
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    return _normal_p_from_z((w_plus - mean) / math.sqrt(max(var, 1e-12)))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0].keys()) if rows else ["sample_id"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_outputs(selected_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], out_dir: Path) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    if selected_rows:
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        ax.scatter([r["mse_first_candidate"] for r in selected_rows], [r["mse_omc_reranked"] for r in selected_rows], alpha=0.7)
        lim = max([r["mse_first_candidate"] for r in selected_rows] + [r["mse_omc_reranked"] for r in selected_rows] + [1e-6])
        ax.plot([0, lim], [0, lim], color="black", linewidth=0.8)
        ax.set_xlabel("first candidate MSE")
        ax.set_ylabel("OMC reranked MSE")
        fig.tight_layout()
        fig.savefig(plots / "mse_first_vs_reranked.png", dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(4.2, 3.5))
        ax.hist([r["candidate_diversity"] for r in selected_rows], bins=20)
        ax.set_xlabel("candidate diversity")
        fig.tight_layout()
        fig.savefig(plots / "candidate_diversity.png", dpi=160)
        plt.close(fig)
    if candidate_rows:
        fig, ax = plt.subplots(figsize=(4.4, 3.8))
        ax.scatter([r["energy"] for r in candidate_rows], [r["mse"] for r in candidate_rows], alpha=0.5)
        ax.set_xlabel("critic energy")
        ax.set_ylabel("candidate MSE")
        fig.tight_layout()
        fig.savefig(plots / "candidate_energy_vs_mse.png", dpi=160)
        plt.close(fig)
        # Compact placeholder example plot using candidate MSE traces.
        fig, ax = plt.subplots(figsize=(5.0, 3.5))
        for sample_id in sorted({int(r["sample_id"]) for r in candidate_rows})[:3]:
            rows = [r for r in candidate_rows if int(r["sample_id"]) == sample_id]
            ax.plot([r["candidate_id"] for r in rows], [r["mse"] for r in rows], marker="o", label=f"sample {sample_id}")
        ax.set_xlabel("candidate")
        ax.set_ylabel("MSE")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plots / "example_rollouts.png", dpi=160)
        plt.close(fig)


def _summarize(selected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in selected_rows])) if selected_rows else 0.0

    metrics = {
        "num_samples": len(selected_rows),
        "mse_random_candidate": mean("mse_random_candidate"),
        "mse_first_candidate": mean("mse_first_candidate"),
        "mse_omc_reranked": mean("mse_omc_reranked"),
        "mse_oracle_best_candidate": mean("mse_oracle_best_candidate"),
        "law_energy_random": mean("law_energy_random"),
        "law_energy_omc_reranked": mean("law_energy_omc_reranked"),
        "candidate_diversity": mean("candidate_diversity"),
        "rerank_gain_mse": mean("rerank_gain_mse"),
        "rerank_gain_energy": mean("rerank_gain_energy"),
        "fraction_omc_improves_mse": float(np.mean([r["mse_omc_reranked"] < r["mse_first_candidate"] for r in selected_rows])) if selected_rows else 0.0,
        "fraction_omc_improves_energy": float(np.mean([r["law_energy_omc_reranked"] < r["law_energy_first_candidate"] for r in selected_rows])) if selected_rows else 0.0,
        "oracle_gap": mean("oracle_gap"),
        "own_vs_wrong_pair_acc": mean("own_vs_wrong_pair_acc"),
        "own_vs_wrong_gap_mean": mean("own_vs_wrong_gap_mean"),
        "own_vs_wrong_gap_min": mean("own_vs_wrong_gap_min"),
    }
    c1 = metrics["fraction_omc_improves_energy"] > 0.60
    c3 = metrics["fraction_omc_improves_mse"] >= 0.45
    c4 = metrics["mse_omc_reranked"] <= metrics["mse_first_candidate"] * 1.10 if metrics["mse_first_candidate"] > 0 else True
    c5 = metrics["mse_oracle_best_candidate"] < metrics["mse_first_candidate"]
    passed = sum([c1, c3, c4, c5])
    metrics["go_no_go"] = {"criteria": {"C1_energy_improves": c1, "C3_mse_not_catastrophic": c3, "C4_mse_within_10pct": c4, "C5_oracle_diversity": c5}, "num_passed_without_cross_critic": passed, "verdict": "GO" if passed == 4 else "WEAK GO" if passed >= 3 else "NO-GO"}
    return metrics


def evaluate_lowm_g(
    proposal_run: Path,
    critic_run: Path | None,
    split: str = "val",
    proposal_checkpoint: str = "best_pred.pt",
    critic_checkpoint: str = "best_law_pair.pt",
    num_samples: int = 500,
    num_candidates: int = 16,
    context_length: int = 2,
    candidate_noise_scale: float = 0.5,
    horizon: int | None = None,
    mode: str = "normal",
    out_dir: Path | None = None,
    compare_critics: list[Path] | None = None,
    device_name: str = "auto",
    seed: int = 0,
) -> dict[str, Any]:
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    proposal, proposal_config = _load_proposal(proposal_run, proposal_checkpoint, device)
    split_path = _ensure_split(proposal_config, split)
    ranking_cfg = ranking_config_from_mapping(proposal_config)
    if horizon is not None:
        ranking_cfg = replace(ranking_cfg, H=int(horizon))
    dataset = LOWMSynthRankingDataset(split_path, ranking_cfg, num_samples=None)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    if len(indices) > num_samples:
        indices = rng.choice(indices, size=num_samples, replace=False)

    critics: list[tuple[str, torch.nn.Module, Path]] = []
    if compare_critics:
        for run in compare_critics:
            critic, _, detected = load_run_model(run, checkpoint_name=critic_checkpoint, device=device)
            critics.append((detected if detected else run.name, critic, run))
    elif critic_run is not None:
        critic, _, detected = load_run_model(critic_run, checkpoint_name=critic_checkpoint, device=device)
        critics.append((detected, critic, critic_run))
    else:
        raise ValueError("critic_run or compare_critics is required")
    primary_name, primary_critic, _ = critics[0]

    candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    cross_compare_rows: list[dict[str, Any]] = []

    for sample_out_idx, idx in enumerate(indices):
        item = dataset[int(idx)]
        target = item["pos_states"]
        if mode == "counterfactual":
            distractor = _distractors(dataset, int(item["query_op_id"].item()), 1, rng)
            if not distractor:
                continue
            target_ep = distractor[0]
            cs, ca, cm = _context_from_episode(dataset, target_ep, context_length)
            item = dict(item)
            item["context_states"], item["context_actions"], item["context_mask"] = cs, ca, cm
            target = _simulate_counterfactual_gt(proposal_config, dataset, item, target_ep)
        candidates = _generate_candidates(proposal, item, num_candidates, context_length, candidate_noise_scale, device)
        target_batch = target.unsqueeze(0).expand(num_candidates, -1, -1, -1)
        mask_batch = item["pos_mask"].unsqueeze(0).expand(num_candidates, -1, -1)
        mses = _masked_mse(candidates, target_batch, mask_batch).numpy()
        energies = _score_candidates(primary_critic, item["context_states"][:context_length], item["context_actions"][:context_length], item["context_mask"][:context_length], candidates, item["pos_actions"], item["pos_mask"], device).numpy()
        random_idx = int(rng.integers(0, num_candidates))
        first_idx = 0
        omc_idx = int(np.argmin(energies))
        oracle_idx = int(np.argmin(mses))
        diversity = _candidate_diversity(candidates, item["pos_mask"].unsqueeze(0))
        for cand_id in range(num_candidates):
            candidate_rows.append({"sample_id": sample_out_idx, "candidate_id": cand_id, "mse": float(mses[cand_id]), "energy": float(energies[cand_id])})
        coherence = _operator_coherence(primary_critic, dataset, item, candidates[omc_idx], item["pos_actions"], item["pos_mask"], context_length, device, rng)
        selected = {
            "sample_id": sample_out_idx,
            "mse_random_candidate": float(mses[random_idx]),
            "mse_first_candidate": float(mses[first_idx]),
            "mse_omc_reranked": float(mses[omc_idx]),
            "mse_oracle_best_candidate": float(mses[oracle_idx]),
            "law_energy_random": float(energies[random_idx]),
            "law_energy_first_candidate": float(energies[first_idx]),
            "law_energy_omc_reranked": float(energies[omc_idx]),
            "candidate_diversity": diversity,
            "rerank_gain_mse": float(mses[first_idx] - mses[omc_idx]),
            "rerank_gain_energy": float(energies[first_idx] - energies[omc_idx]),
            "oracle_gap": float(mses[omc_idx] - mses[oracle_idx]),
            "selected_omc_idx": omc_idx,
            "selected_oracle_idx": oracle_idx,
            **coherence,
        }
        selected_rows.append(selected)
        cross_rows.append({"sample_id": sample_out_idx, **coherence})
        if len(critics) > 1:
            picks = []
            for name, critic, run in critics:
                e = _score_candidates(critic, item["context_states"][:context_length], item["context_actions"][:context_length], item["context_mask"][:context_length], candidates, item["pos_actions"], item["pos_mask"], device).numpy()
                pick = int(np.argmin(e))
                coh = _operator_coherence(critic, dataset, item, candidates[pick], item["pos_actions"], item["pos_mask"], context_length, device, rng)
                row = {"sample_id": sample_out_idx, "critic": run.name, "selected_idx": pick, "selected_mse": float(mses[pick]), "selected_energy": float(e[pick]), **coh}
                cross_compare_rows.append(row)
                picks.append(row)

    metrics = _summarize(selected_rows)
    metrics.update({"critic_model": primary_name, "split": split, "num_candidates": num_candidates, "context_length": context_length, "candidate_noise_scale": candidate_noise_scale, "mode": mode})
    if compare_critics and len(critics) >= 2:
        by_sample: dict[int, list[dict[str, Any]]] = {}
        for row in cross_compare_rows:
            by_sample.setdefault(int(row["sample_id"]), []).append(row)
        deltas = []
        wins = []
        for rows in by_sample.values():
            if len(rows) >= 2:
                delta = float(rows[0]["selected_mse"]) - float(rows[1]["selected_mse"])
                deltas.append(delta)
                wins.append(delta < 0)
        energy_wins = []
        for rows in by_sample.values():
            if len(rows) >= 2:
                energy_wins.append(float(rows[0]["selected_energy"]) < float(rows[1]["selected_energy"]))
        arr = np.asarray(deltas, dtype=np.float64)
        metrics["cross_critic"] = {
            "fraction_omc_rerank_beats_no_law_mse": float(np.mean(wins)) if wins else 0.0,
            "fraction_omc_rerank_beats_no_law_energy": float(np.mean(energy_wins)) if energy_wins else 0.0,
            "mean_mse_selected_by_omc": float(np.mean([r["selected_mse"] for r in cross_compare_rows[0::2]])) if cross_compare_rows else 0.0,
            "mean_mse_selected_by_no_law": float(np.mean([r["selected_mse"] for r in cross_compare_rows[1::2]])) if len(cross_compare_rows) > 1 else 0.0,
            "own_vs_wrong_pair_acc_selected_by_omc": float(np.mean([r["own_vs_wrong_pair_acc"] for r in cross_compare_rows[0::2]])) if cross_compare_rows else 0.0,
            "own_vs_wrong_pair_acc_selected_by_no_law": float(np.mean([r["own_vs_wrong_pair_acc"] for r in cross_compare_rows[1::2]])) if len(cross_compare_rows) > 1 else 0.0,
            "wilcoxon_signed_rank_p_value": _wilcoxon_like_p(arr) if arr.size else 1.0,
        }

    out = out_dir or (proposal_run / "eval" / split / "lowm_g_rerank")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    _write_csv(candidate_rows, out / "per_sample_candidates.csv")
    _write_csv(selected_rows, out / "selected_candidates.csv")
    _write_csv(cross_rows, out / "cross_operator_energy.csv")
    if cross_compare_rows:
        _write_csv(cross_compare_rows, out / "cross_critic_rerank.csv")
        with (out / "cross_critic_summary.json").open("w", encoding="utf-8") as f:
            json.dump(metrics.get("cross_critic", {}), f, indent=2, sort_keys=True)
        cc = metrics.get("cross_critic", {})
        lines = ["|metric|value|", "|---|---|"]
        for key, value in cc.items():
            lines.append(f"|{key}|{value:.6f}|" if isinstance(value, float) else f"|{key}|{value}|")
        (out / "cross_critic_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "rerank_summary.md").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    _plot_outputs(selected_rows, candidate_rows, out)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-run", type=Path, required=True)
    parser.add_argument("--critic-run", type=Path, default=None)
    parser.add_argument("--compare-critics", type=Path, nargs="*", default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--proposal-checkpoint", type=str, default="best_pred.pt")
    parser.add_argument("--critic-checkpoint", type=str, default="best_law_pair.pt")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=2)
    parser.add_argument("--candidate-noise-scale", type=float, default=0.5)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--mode", type=str, default="normal", choices=["normal", "counterfactual"])
    parser.add_argument("--ood", action="store_true")
    parser.add_argument("--capacity-tag", type=str, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    metrics = evaluate_lowm_g(
        args.proposal_run,
        args.critic_run,
        split="test_ood_param" if args.ood else args.split,
        proposal_checkpoint=args.proposal_checkpoint,
        critic_checkpoint=args.critic_checkpoint,
        num_samples=args.num_samples,
        num_candidates=args.num_candidates,
        context_length=args.context_length,
        candidate_noise_scale=args.candidate_noise_scale,
        horizon=args.horizon,
        mode=args.mode,
        out_dir=args.out,
        compare_critics=args.compare_critics,
        device_name=args.device,
        seed=args.seed,
    )
    print(json.dumps({"mse_first": metrics["mse_first_candidate"], "mse_omc": metrics["mse_omc_reranked"], "fraction_omc_improves_energy": metrics["fraction_omc_improves_energy"]}, indent=2))


if __name__ == "__main__":
    main()
