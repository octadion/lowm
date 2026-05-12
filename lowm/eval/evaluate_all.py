"""Unified evaluation for trained LOWM and baseline runs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from lowm.data.dataset import LOWMSynthRankingDataset, make_ranking_dataloader, ranking_config_from_mapping
from lowm.data.generate_dataset import generate_dataset
from lowm.data.negatives import REQUIRED_NEGATIVE_TYPES, make_state_corrupted
from lowm.eval.metrics import METRIC_VERSION, RankingMetricAccumulator
from lowm.models.baselines import baseline_config_from_mapping, build_baseline
from lowm.models.lowm import LOWM, lowm_config_from_mapping
from lowm.training.losses import nce_ranking_loss


GROUPS = ("valid", "state_mismatch", "law_mismatch", "both_mismatch")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    return loaded if isinstance(loaded, dict) else {}


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _score_model(model: torch.nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
    output = model(batch)
    if isinstance(output, dict):
        return output["energies"]
    return output


def _detect_model_type(run_dir: Path, checkpoint: Mapping[str, Any], metadata: Mapping[str, Any], override: str | None) -> str:
    if override:
        return override
    if "baseline" in checkpoint:
        return str(checkpoint["baseline"])
    if "baseline" in metadata:
        return str(metadata["baseline"])
    if str(metadata.get("model", "")).lower() == "lowm":
        return "lowm"
    config = checkpoint.get("config", {})
    if isinstance(config, Mapping):
        training = config.get("training", {})
        if isinstance(training, Mapping) and "baseline" in training:
            return str(training["baseline"])
        model = config.get("model", {})
        if isinstance(model, Mapping) and "lambda_dim" in model:
            return "lowm"
    if "lowm" in run_dir.name.lower():
        return "lowm"
    raise ValueError("could not detect model type; pass --model_type")


def _resolve_checkpoint_path(run_dir: Path, checkpoint_name: str) -> Path:
    ckpt_path = run_dir / "checkpoints" / checkpoint_name
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "last.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint under {run_dir / 'checkpoints'}")
    return ckpt_path


def load_run_model(run_dir: Path, model_type: str | None = None, checkpoint_name: str = "best.pt", device: torch.device | None = None) -> tuple[torch.nn.Module, dict[str, Any], str]:
    ckpt_path = _resolve_checkpoint_path(run_dir, checkpoint_name)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        config = _load_yaml(run_dir / "config.yaml")
    metadata = _load_json(run_dir / "metadata.json")
    detected = _detect_model_type(run_dir, checkpoint, metadata, model_type)
    if detected == "lowm":
        model = LOWM(lowm_config_from_mapping(config))
    else:
        model = build_baseline(detected, baseline_config_from_mapping(config))
    model.load_state_dict(checkpoint["model_state"])
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, config, detected


def _ensure_split(config: Mapping[str, Any], split: str) -> Path:
    data_cfg = dict(config.get("data", {}))
    root = Path(data_cfg.get("root", "data/lowm_synth_v0"))
    split_key = f"{split}_split"
    split_file = str(data_cfg.get(split_key, f"{split}.npz"))
    split_path = root / split_file
    if split_path.exists():
        return split_path
    if not bool(data_cfg.get("generate_if_missing", False)):
        raise FileNotFoundError(f"missing split file {split_path}")
    dataset_config = Path(data_cfg.get("dataset_config", "configs/lowm_synth_v0.yaml"))
    generate_dataset(dataset_config, root, [split_file.replace(".npz", "")])
    return split_path


def _write_debug_energies(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "candidate_id", "label", "negative_type", "energy", "is_positive"])
        writer.writeheader()
        writer.writerows(rows)


def evaluate_ranking(
    model: torch.nn.Module,
    dataset: LOWMSynthRankingDataset,
    batch_size: int,
    device: torch.device,
    debug_rows: int = 32,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    loader = make_ranking_dataloader(dataset, batch_size=batch_size, shuffle=False)
    acc = RankingMetricAccumulator()
    debug: list[dict[str, Any]] = []
    sample_offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            energies = _score_model(model, batch)
            loss = nce_ranking_loss(energies, batch["labels"])
            acc.update(energies, batch["labels"], batch["negative_types"], float(loss.item()))
            energies_cpu = energies.detach().cpu()
            labels_cpu = batch["labels"].detach().cpu()
            if sample_offset < debug_rows:
                for b in range(energies_cpu.shape[0]):
                    sample_id = sample_offset + b
                    if sample_id >= debug_rows:
                        break
                    label = int(labels_cpu[b].item())
                    for cand_id in range(energies_cpu.shape[1]):
                        neg_type = batch["negative_types"][b][cand_id]
                        debug.append(
                            {
                                "sample_id": sample_id,
                                "candidate_id": cand_id,
                                "label": label,
                                "negative_type": neg_type,
                                "energy": float(energies_cpu[b, cand_id].item()),
                                "is_positive": cand_id == label,
                            }
                        )
            sample_offset += energies_cpu.shape[0]
    metrics = acc.compute()
    return metrics, debug


def _write_negative_breakdown(metrics: Mapping[str, Any], path: Path) -> None:
    rows = metrics.get("by_negative_type", {})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["negative_type", "pairwise_acc", "mean_energy_gap", "count", "top1_acc_on_samples_with_type"])
        writer.writeheader()
        for name in REQUIRED_NEGATIVE_TYPES:
            values = rows.get(name, {"pairwise_acc": 0.0, "mean_energy_gap": 0.0, "count": 0, "top1_acc_on_samples_with_type": 0.0})
            writer.writerow({"negative_type": name, **values})


def _candidate_arrays(sample: Mapping[str, Any], candidate_type: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if candidate_type == "positive":
        idx = int(sample["labels"].item())
    else:
        matches = [i for i, name in enumerate(sample["negative_types"]) if name == candidate_type]
        if not matches:
            return None
        idx = matches[0]
    return sample["cand_states"][idx], sample["cand_actions"][idx], sample["cand_mask"][idx]


def _disentanglement_batch(sample: Mapping[str, Any], rng: np.random.Generator) -> dict[str, Any] | None:
    valid = _candidate_arrays(sample, "positive")
    state = _candidate_arrays(sample, "state_corrupted")
    law = _candidate_arrays(sample, "law_mismatch")
    if valid is None or state is None or law is None:
        return None
    law_states, law_actions, law_mask = (x.numpy() for x in law)
    both_states, both_actions, both_mask = make_state_corrupted(law_states, law_actions, law_mask, rng)
    groups = {
        "valid": valid,
        "state_mismatch": state,
        "law_mismatch": law,
        "both_mismatch": (
            torch.from_numpy(both_states),
            torch.from_numpy(both_actions),
            torch.from_numpy(both_mask),
        ),
    }
    batch = {
        "context_states": sample["context_states"].unsqueeze(0),
        "context_actions": sample["context_actions"].unsqueeze(0),
        "context_mask": sample["context_mask"].unsqueeze(0),
        "cand_states": torch.stack([groups[name][0] for name in GROUPS], dim=0).unsqueeze(0),
        "cand_actions": torch.stack([groups[name][1] for name in GROUPS], dim=0).unsqueeze(0),
        "cand_mask": torch.stack([groups[name][2] for name in GROUPS], dim=0).unsqueeze(0),
        "labels": torch.tensor([0], dtype=torch.long),
        "negative_types": [["positive", "state_mismatch", "law_mismatch", "both_mismatch"]],
    }
    return batch


def evaluate_disentanglement(model: torch.nn.Module, dataset: LOWMSynthRankingDataset, device: torch.device, max_samples: int = 128) -> dict[str, float]:
    rng = np.random.default_rng(2027)
    sums = {name: 0.0 for name in GROUPS}
    counts = {name: 0 for name in GROUPS}
    limit = min(len(dataset), max_samples)
    with torch.no_grad():
        for idx in range(limit):
            batch = _disentanglement_batch(dataset[idx], rng)
            if batch is None:
                continue
            batch = _move_batch_to_device(batch, device)
            energies = _score_model(model, batch).detach().cpu().squeeze(0)
            for group_idx, name in enumerate(GROUPS):
                sums[name] += float(energies[group_idx].item())
                counts[name] += 1
    return {name: sums[name] / max(1, counts[name]) for name in GROUPS}


def _write_disentanglement(matrix: Mapping[str, float], csv_path: Path, plot_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", *GROUPS])
        writer.writeheader()
        writer.writerow({"metric": "mean_energy", **{name: matrix.get(name, 0.0) for name in GROUPS}})
    values = np.array([[matrix.get(name, 0.0) for name in GROUPS]], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6.5, 2.3))
    im = ax.imshow(values, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(GROUPS)), GROUPS, rotation=20, ha="right")
    ax.set_yticks([0], ["energy"])
    for i, value in enumerate(values[0]):
        ax.text(i, 0, f"{value:.2f}", ha="center", va="center", color="white" if value < values.max() * 0.7 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def _make_retrieval_batch(query: Mapping[str, Any], contexts: list[Mapping[str, Any]]) -> dict[str, Any]:
    pos_idx = int(query["labels"].item())
    tau_states = query["cand_states"][pos_idx]
    tau_actions = query["cand_actions"][pos_idx]
    tau_mask = query["cand_mask"][pos_idx]
    num_contexts = len(contexts)
    return {
        "context_states": torch.stack([ctx["context_states"] for ctx in contexts], dim=0),
        "context_actions": torch.stack([ctx["context_actions"] for ctx in contexts], dim=0),
        "context_mask": torch.stack([ctx["context_mask"] for ctx in contexts], dim=0),
        "cand_states": tau_states.unsqueeze(0).unsqueeze(0).expand(num_contexts, 1, -1, -1, -1).clone(),
        "cand_actions": tau_actions.unsqueeze(0).unsqueeze(0).expand(num_contexts, 1, -1, -1, -1).clone(),
        "cand_mask": tau_mask.unsqueeze(0).unsqueeze(0).expand(num_contexts, 1, -1, -1).clone(),
        "labels": torch.zeros(num_contexts, dtype=torch.long),
        "negative_types": [["positive"] for _ in contexts],
    }


def _context_pool(dataset: LOWMSynthRankingDataset, query_idx: int, pool_size: int) -> tuple[list[dict[str, Any]], int] | None:
    query = dataset[query_idx]
    query_op = int(query["query_op_id"].item())
    contexts = [query]
    for offset in range(1, len(dataset) + 1):
        if len(contexts) >= pool_size:
            break
        candidate = dataset[(query_idx + offset) % len(dataset)]
        if int(candidate["query_op_id"].item()) != query_op:
            contexts.append(candidate)
    if len(contexts) < 2:
        return None
    return contexts, 0


def evaluate_retrieval(model: torch.nn.Module, dataset: LOWMSynthRankingDataset, device: torch.device, pool_size: int = 8, max_queries: int = 64) -> dict[str, Any]:
    ranks: list[int] = []
    evaluated = 0
    with torch.no_grad():
        for idx in range(min(len(dataset), max_queries)):
            pool = _context_pool(dataset, idx, pool_size)
            if pool is None:
                continue
            contexts, positive_idx = pool
            batch = _move_batch_to_device(_make_retrieval_batch(dataset[idx], contexts), device)
            energies = _score_model(model, batch).detach().cpu().squeeze(1)
            positive_energy = energies[positive_idx]
            rank = int((energies < positive_energy).sum().item()) + 1
            ranks.append(rank)
            evaluated += 1
    if not ranks:
        return {"num_queries": 0, "retrieval_acc": 0.0, "mrr": 0.0, "recall_at_1": 0.0, "recall_at_3": 0.0, "recall_at_5": 0.0, "random": {}}
    ranks_arr = np.asarray(ranks, dtype=np.float32)
    actual_pool = min(pool_size, len(dataset))
    random_metrics = {
        "recall_at_1": 1.0 / max(1, actual_pool),
        "recall_at_3": min(3, actual_pool) / max(1, actual_pool),
        "recall_at_5": min(5, actual_pool) / max(1, actual_pool),
    }
    return {
        "num_queries": evaluated,
        "retrieval_acc": float(np.mean(ranks_arr == 1)),
        "mrr": float(np.mean(1.0 / ranks_arr)),
        "recall_at_1": float(np.mean(ranks_arr <= 1)),
        "recall_at_3": float(np.mean(ranks_arr <= 3)),
        "recall_at_5": float(np.mean(ranks_arr <= 5)),
        "random": random_metrics,
    }


def _mrr_from_energy_matrix(energies: torch.Tensor) -> tuple[float, float, float, float]:
    labels = torch.arange(energies.shape[0], device=energies.device)
    logits = -energies
    tau_pred = torch.argmax(logits, dim=1)
    lambda_pred = torch.argmax(logits.T, dim=1)
    diag = energies.diag()
    tau_ranks = (energies < diag[:, None]).sum(dim=1) + 1
    lambda_ranks = (energies < diag[None, :]).sum(dim=0) + 1
    return (
        float((tau_pred == labels).float().mean().item()),
        float((lambda_pred == labels).float().mean().item()),
        float((1.0 / tau_ranks.float()).mean().item()),
        float((1.0 / lambda_ranks.float()).mean().item()),
    )


def evaluate_occl_alignment_summary(
    model: torch.nn.Module,
    dataset: LOWMSynthRankingDataset,
    batch_size: int,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    if not hasattr(model, "energy_matrix"):
        return {}
    loader = make_ranking_dataloader(dataset, batch_size=batch_size, shuffle=False)
    tau_acc_sum = lambda_acc_sum = tau_mrr_sum = lambda_mrr_sum = 0.0
    diag_sum = off_sum = 0.0
    diag_count = off_count = batch_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = _move_batch_to_device(batch, device)
            output = model(batch)
            if not isinstance(output, Mapping) or "lambda" not in output:
                continue
            e_matrix = model.energy_matrix(batch["pos_states"], batch["pos_actions"], batch["pos_mask"], output["lambda"])
            tau_acc, lambda_acc, tau_mrr, lambda_mrr = _mrr_from_energy_matrix(e_matrix)
            diag = e_matrix.diag()
            off_mask = ~torch.eye(e_matrix.shape[0], dtype=torch.bool, device=e_matrix.device)
            off = e_matrix[off_mask]
            tau_acc_sum += tau_acc
            lambda_acc_sum += lambda_acc
            tau_mrr_sum += tau_mrr
            lambda_mrr_sum += lambda_mrr
            diag_sum += float(diag.sum().item())
            off_sum += float(off.sum().item())
            diag_count += int(diag.numel())
            off_count += int(off.numel())
            batch_count += 1
    if batch_count == 0:
        return {}
    diag_mean = diag_sum / max(1, diag_count)
    off_mean = off_sum / max(1, off_count)
    return {
        "tau_to_lambda_acc": tau_acc_sum / batch_count,
        "lambda_to_tau_acc": lambda_acc_sum / batch_count,
        "mrr_tau_to_lambda": tau_mrr_sum / batch_count,
        "mrr_lambda_to_tau": lambda_mrr_sum / batch_count,
        "diagonal_energy_mean": diag_mean,
        "off_diagonal_energy_mean": off_mean,
        "diagonal_vs_offdiag_gap": off_mean - diag_mean,
    }
def _plot_ranking(metrics: Mapping[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_type = metrics.get("by_negative_type", {})
    names = list(REQUIRED_NEGATIVE_TYPES)
    pairwise = [by_type.get(name, {}).get("pairwise_acc", 0.0) for name in names]
    gaps = [by_type.get(name, {}).get("mean_energy_gap", 0.0) for name in names]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(names, pairwise)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Pairwise accuracy")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "pairwise_accuracy_by_negative_type.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(names, gaps)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("E(negative) - E(positive)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "energy_gap_by_negative_type.png", dpi=160)
    plt.close(fig)


def evaluate_run(
    run_dir: Path,
    split: str = "val",
    model_type: str | None = None,
    checkpoint_name: str = "best.pt",
    batch_size: int | None = None,
    num_samples: int | None = None,
    seed: int | None = None,
    device_name: str = "auto",
) -> dict[str, Any]:
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    checkpoint_path = _resolve_checkpoint_path(run_dir, checkpoint_name)
    checkpoint_stem = checkpoint_path.stem
    model, config, detected = load_run_model(run_dir, model_type=model_type, checkpoint_name=checkpoint_name, device=device)
    split_path = _ensure_split(config, split)
    ranking_cfg = ranking_config_from_mapping(config)
    if seed is not None:
        ranking_cfg = replace(ranking_cfg, seed=int(seed))
    eval_cfg = dict(config.get("evaluation", {}))
    sample_count = num_samples if num_samples is not None else eval_cfg.get("num_samples", config.get("training", {}).get("val_samples"))
    dataset = LOWMSynthRankingDataset(split_path, ranking_cfg, num_samples=int(sample_count) if sample_count else None)
    bs = int(batch_size or config.get("training", {}).get("batch_size", 64))

    legacy_out_dir = run_dir / "eval" / split
    out_dir = legacy_out_dir / checkpoint_stem
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    ranking, debug = evaluate_ranking(model, dataset, bs, device, debug_rows=int(eval_cfg.get("debug_energy_samples", 32)))
    with (out_dir / "ranking_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(ranking, f, indent=2, sort_keys=True)
    _write_negative_breakdown(ranking, out_dir / "negative_type_breakdown.csv")
    _write_debug_energies(debug, out_dir / "debug_energies.csv")
    _plot_ranking(ranking, plots_dir)

    disentanglement = evaluate_disentanglement(model, dataset, device, max_samples=int(eval_cfg.get("disentanglement_samples", min(128, len(dataset)))))
    _write_disentanglement(disentanglement, out_dir / "disentanglement_matrix.csv", plots_dir / "disentanglement_heatmap.png")

    retrieval = evaluate_retrieval(
        model,
        dataset,
        device,
        pool_size=int(eval_cfg.get("retrieval_pool_size", 8)),
        max_queries=int(eval_cfg.get("retrieval_queries", min(64, len(dataset)))),
    )
    with (out_dir / "retrieval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(retrieval, f, indent=2, sort_keys=True)
    occl_alignment = evaluate_occl_alignment_summary(
        model,
        dataset,
        bs,
        device,
        max_batches=eval_cfg.get("occl_alignment_batches"),
    )

    summary = {
        "model_type": detected,
        "split": split,
        "checkpoint_requested": checkpoint_name,
        "checkpoint_used": checkpoint_path.name,
        "checkpoint_stem": checkpoint_stem,
        "ranking_seed": ranking_cfg.seed,
        "num_samples": len(dataset),
        "metric_version": METRIC_VERSION,
        "ranking": ranking,
        "disentanglement": disentanglement,
        "retrieval": retrieval,
        "occl_alignment": occl_alignment,
    }
    with (out_dir / "eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    legacy_out_dir.mkdir(parents=True, exist_ok=True)
    for filename in [
        "ranking_metrics.json",
        "negative_type_breakdown.csv",
        "debug_energies.csv",
        "disentanglement_matrix.csv",
        "retrieval_metrics.json",
        "eval_summary.json",
    ]:
        source = out_dir / filename
        if source.exists():
            target = legacy_out_dir / filename
            target.write_bytes(source.read_bytes())
    legacy_plots = legacy_out_dir / "plots"
    legacy_plots.mkdir(parents=True, exist_ok=True)
    for source in plots_dir.glob("*.png"):
        (legacy_plots / source.name).write_bytes(source.read_bytes())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--model_type", type=str, default=None, choices=["fixed_energy", "direct_context_energy", "lowm"])
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    summary = evaluate_run(args.run, args.split, args.model_type, args.checkpoint, args.batch_size, args.num_samples, args.seed, args.device)
    print(json.dumps({"model_type": summary["model_type"], "split": summary["split"], "top1_acc": summary["ranking"]["top1_acc"], "law_pair": summary["ranking"]["law_pair"], "law_gap": summary["ranking"]["law_gap"]}, indent=2))


if __name__ == "__main__":
    main()
