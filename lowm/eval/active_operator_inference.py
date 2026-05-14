"""Active Operator Inference evaluation on LOWM-Synth.

AOI uses a trained world-relative energy critic to choose an action that should
make candidate latent operators easier to distinguish after one observation.
This is an evaluation-only diagnostic: simulator operator metadata is used to
construct controlled outcomes, never as model input.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from lowm.data.dataset import LOWMSynthRankingDataset, RankingConfig, ranking_config_from_mapping
from lowm.data.negatives import law_distance
from lowm.data.simulate import SimulationConfig, config_from_mapping, step_state
from lowm.eval.evaluate_all import _ensure_split, _move_batch_to_device, _resolve_checkpoint_path, load_run_model


AOI_METHODS = (
    "random_action",
    "max_motion_action",
    "aoi_energy_separation",
    "aoi_entropy_reduction",
    "oracle_aoi",
)


def posterior_from_energies(energies: np.ndarray | torch.Tensor, temperature: float = 1.0) -> np.ndarray:
    """Return p(lambda | tau) proportional to exp(-E / T)."""

    values = np.asarray(energies.detach().cpu().numpy() if torch.is_tensor(energies) else energies, dtype=np.float64)
    temp = max(float(temperature), 1e-6)
    logits = -values / temp
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    denom = float(np.sum(probs))
    if denom <= 0.0 or not np.isfinite(denom):
        return np.full_like(probs, 1.0 / max(1, probs.size), dtype=np.float64)
    return probs / denom


def entropy(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def _rank_of_true(energies: np.ndarray, true_idx: int) -> int:
    true_energy = float(energies[true_idx])
    return int(np.sum(energies < true_energy)) + 1


def _action_set(num_actions: int, nmax: int, active_count: int, action_scale: float, rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    actions = np.zeros((max(1, num_actions), nmax, 2), dtype=np.float32)
    names: list[str] = []
    active_idx = 0 if active_count > 0 else 0
    base = [
        ("zero", (0.0, 0.0)),
        ("right", (action_scale, 0.0)),
        ("left", (-action_scale, 0.0)),
        ("up", (0.0, action_scale)),
        ("down", (0.0, -action_scale)),
    ]
    for idx in range(min(len(base), num_actions)):
        names.append(base[idx][0])
        actions[idx, active_idx] = np.asarray(base[idx][1], dtype=np.float32)
    for idx in range(len(names), num_actions):
        names.append(f"random_{idx - len(base) + 1}")
        actions[idx, active_idx] = rng.uniform(-action_scale, action_scale, size=2).astype(np.float32)
    return actions[:num_actions], names[:num_actions]


def _simulate_short_trajectory(
    current_state: np.ndarray,
    current_mask: np.ndarray,
    action: np.ndarray,
    op_id: int,
    op_params: np.ndarray,
    cfg: SimulationConfig,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(np.sum(current_mask > 0.5))
    states = np.zeros((horizon + 1, cfg.nmax, cfg.d_object), dtype=np.float32)
    actions = np.zeros((horizon, cfg.nmax, 2), dtype=np.float32)
    mask = np.zeros((horizon + 1, cfg.nmax), dtype=np.float32)
    states[0] = current_state.astype(np.float32)
    mask[:, :n] = 1.0
    for t in range(horizon):
        if t == 0:
            actions[t] = action.astype(np.float32)
        states[t + 1] = step_state(states[t], actions[t], n, int(op_id), op_params.astype(np.float32), cfg)
    return states, actions, mask


def _context_from_episode(dataset: LOWMSynthRankingDataset, episode: int, context_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    context_len = min(context_len, dataset.T)
    states = np.zeros((context_len, 2, dataset.nmax, dataset.d_object), dtype=np.float32)
    actions = np.zeros((context_len, dataset.nmax, 2), dtype=np.float32)
    mask = np.zeros((context_len, 2, dataset.nmax), dtype=np.float32)
    for t in range(context_len):
        states[t, 0] = dataset.states[episode, t]
        states[t, 1] = dataset.states[episode, t + 1]
        actions[t] = dataset.actions[episode, t]
        mask[t, 0] = dataset.mask[episode, t]
        mask[t, 1] = dataset.mask[episode, t + 1]
    return states, actions, mask


def _select_hypothesis_episodes(
    dataset: LOWMSynthRankingDataset,
    true_episode: int,
    num_hypotheses: int,
    min_param_distance: float,
    rng: np.random.Generator,
) -> list[int]:
    true_op = int(dataset.op_id[true_episode])
    true_params = dataset.op_params[true_episode]
    same_family = [
        int(idx)
        for idx in range(dataset.num_episodes)
        if idx != true_episode
        and int(dataset.op_id[idx]) == true_op
        and law_distance(true_op, true_params, int(dataset.op_id[idx]), dataset.op_params[idx]) >= min_param_distance
    ]
    rng.shuffle(same_family)
    selected = [int(true_episode), *same_family[: max(0, num_hypotheses - 1)]]
    if len(selected) < num_hypotheses:
        fallback = [int(idx) for idx in range(dataset.num_episodes) if idx not in selected]
        rng.shuffle(fallback)
        selected.extend(fallback[: num_hypotheses - len(selected)])
    return selected[:num_hypotheses]


def _make_context_batch(dataset: LOWMSynthRankingDataset, episodes: list[int], context_len: int) -> dict[str, torch.Tensor]:
    contexts = [_context_from_episode(dataset, ep, context_len) for ep in episodes]
    return {
        "context_states": torch.from_numpy(np.stack([item[0] for item in contexts]).astype(np.float32)),
        "context_actions": torch.from_numpy(np.stack([item[1] for item in contexts]).astype(np.float32)),
        "context_mask": torch.from_numpy(np.stack([item[2] for item in contexts]).astype(np.float32)),
        "op_id": torch.tensor([int(dataset.op_id[ep]) for ep in episodes], dtype=torch.long),
        "op_params": torch.from_numpy(np.stack([dataset.op_params[ep] for ep in episodes]).astype(np.float32)),
    }


def _score_energy_matrix(
    model: torch.nn.Module,
    context_batch: Mapping[str, torch.Tensor],
    traj_states: torch.Tensor,
    traj_actions: torch.Tensor,
    traj_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return M[i,j] = E(tau_i, context/lambda_j)."""

    k = int(traj_states.shape[0])
    context_device = {
        "context_states": context_batch["context_states"].to(device),
        "context_actions": context_batch["context_actions"].to(device),
        "context_mask": context_batch["context_mask"].to(device),
    }
    traj_states = traj_states.to(device)
    traj_actions = traj_actions.to(device)
    traj_mask = traj_mask.to(device)
    with torch.no_grad():
        if hasattr(model, "energy_matrix") and hasattr(model, "encode_lambda"):
            output = model(
                {
                    **context_device,
                    "cand_states": traj_states[:, None],
                    "cand_actions": traj_actions[:, None],
                    "cand_mask": traj_mask[:, None],
                    "labels": torch.zeros(k, dtype=torch.long, device=device),
                }
            )
            return model.energy_matrix(traj_states, traj_actions, traj_mask, output["lambda"]).detach().cpu()

        batch = {
            **context_device,
            "cand_states": traj_states[None].expand(k, k, *traj_states.shape[1:]).clone(),
            "cand_actions": traj_actions[None].expand(k, k, *traj_actions.shape[1:]).clone(),
            "cand_mask": traj_mask[None].expand(k, k, *traj_mask.shape[1:]).clone(),
            "labels": torch.zeros(k, dtype=torch.long, device=device),
        }
        output = model(batch)
        energies = output["energies"] if isinstance(output, Mapping) else output
        return energies.detach().cpu().T


def _score_observed(
    model: torch.nn.Module,
    context_batch: Mapping[str, torch.Tensor],
    obs_states: torch.Tensor,
    obs_actions: torch.Tensor,
    obs_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    k = int(context_batch["context_states"].shape[0])
    context_device = {
        "context_states": context_batch["context_states"].to(device),
        "context_actions": context_batch["context_actions"].to(device),
        "context_mask": context_batch["context_mask"].to(device),
    }
    obs_states = obs_states.to(device)
    obs_actions = obs_actions.to(device)
    obs_mask = obs_mask.to(device)
    with torch.no_grad():
        if hasattr(model, "encode_lambda") and hasattr(model, "energy"):
            _, _, lambdas = model.encode_lambda(
                context_device["context_states"],
                context_device["context_actions"],
                context_device["context_mask"],
            )
            return model.energy(
                obs_states[None, None].expand(k, 1, *obs_states.shape).clone(),
                obs_actions[None, None].expand(k, 1, *obs_actions.shape).clone(),
                obs_mask[None, None].expand(k, 1, *obs_mask.shape).clone(),
                lambdas,
            ).squeeze(1).detach().cpu()
        batch = {
            **context_device,
            "cand_states": obs_states[None, None].expand(k, 1, *obs_states.shape).clone(),
            "cand_actions": obs_actions[None, None].expand(k, 1, *obs_actions.shape).clone(),
            "cand_mask": obs_mask[None, None].expand(k, 1, *obs_mask.shape).clone(),
            "labels": torch.zeros(k, dtype=torch.long, device=device),
        }
        output = model(batch)
        energies = output["energies"] if isinstance(output, Mapping) else output
        return energies.squeeze(1).detach().cpu()


def _matrix_scores(matrix: np.ndarray, temperature: float) -> dict[str, float]:
    k = int(matrix.shape[0])
    off_mask = ~np.eye(k, dtype=bool)
    diag = np.diag(matrix)
    off_means = matrix[off_mask].reshape(k, max(1, k - 1)).mean(axis=1)
    posteriors = np.stack([posterior_from_energies(row, temperature) for row in matrix], axis=0)
    entropies = np.asarray([entropy(row) for row in posteriors], dtype=np.float64)
    mean_posterior = posteriors.mean(axis=0)
    return {
        "energy_separation": float(np.mean(off_means - diag)),
        "posterior_entropy": float(np.mean(entropies)),
        "expected_entropy_reduction": float(math.log(max(1, k)) - np.mean(entropies)),
        "mutual_information": float(entropy(mean_posterior) - np.mean(entropies)),
        "diag_energy_mean": float(np.mean(diag)),
        "offdiag_energy_mean": float(np.mean(matrix[off_mask])) if off_mask.any() else 0.0,
    }


def _oracle_matrix(trajectories: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    k = len(trajectories)
    matrix = np.zeros((k, k), dtype=np.float32)
    for i in range(k):
        for j in range(k):
            diff = trajectories[i][0] - trajectories[j][0]
            active = trajectories[i][2][..., None] * trajectories[j][2][..., None]
            denom = max(1.0, float(active.sum()))
            matrix[i, j] = float(((diff[..., 0:4] ** 2) * active).sum() / denom)
    return matrix


def evaluate_action_disambiguation(
    model: torch.nn.Module,
    candidate_contexts: Mapping[str, torch.Tensor],
    current_state: np.ndarray,
    candidate_actions: np.ndarray,
    simulator_or_dataset_sampler: Callable[[np.ndarray, int, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    num_operator_hypotheses: int = 4,
    horizon: int = 3,
    temperature: float = 1.0,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Score candidate actions by the separability of operator hypotheses."""

    del current_state, num_operator_hypotheses, horizon
    device = device or torch.device("cpu")
    op_ids = candidate_contexts["op_id"].detach().cpu().numpy().astype(np.int64)
    op_params = candidate_contexts["op_params"].detach().cpu().numpy().astype(np.float32)
    action_scores: list[dict[str, float]] = []
    matrices: list[np.ndarray] = []
    oracle_matrices: list[np.ndarray] = []
    for action in candidate_actions:
        outcomes = [simulator_or_dataset_sampler(action, int(op_ids[i]), op_params[i]) for i in range(len(op_ids))]
        traj_states = torch.from_numpy(np.stack([item[0] for item in outcomes]).astype(np.float32))
        traj_actions = torch.from_numpy(np.stack([item[1] for item in outcomes]).astype(np.float32))
        traj_mask = torch.from_numpy(np.stack([item[2] for item in outcomes]).astype(np.float32))
        matrix = _score_energy_matrix(model, candidate_contexts, traj_states, traj_actions, traj_mask, device).numpy()
        matrices.append(matrix)
        oracle_matrices.append(_oracle_matrix(outcomes))
        action_scores.append(_matrix_scores(matrix, temperature))
    energy_separation = np.asarray([score["energy_separation"] for score in action_scores], dtype=np.float64)
    entropy_reduction = np.asarray([score["expected_entropy_reduction"] for score in action_scores], dtype=np.float64)
    posterior_entropies = np.asarray([score["posterior_entropy"] for score in action_scores], dtype=np.float64)
    return {
        "action_scores": action_scores,
        "best_action": int(np.argmax(energy_separation)),
        "best_action_entropy": int(np.argmax(entropy_reduction)),
        "energy_matrices": matrices,
        "oracle_energy_matrices": oracle_matrices,
        "posterior_entropies": posterior_entropies,
        "expected_entropy_reduction": entropy_reduction,
        "energy_separation": energy_separation,
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return loaded if isinstance(loaded, dict) else {}


def _simulation_config_from_run(config: Mapping[str, Any], horizon: int) -> SimulationConfig:
    data_cfg = dict(config.get("data", {}))
    dataset_config = Path(data_cfg.get("dataset_config", "configs/lowm_synth_v0.yaml"))
    base = config_from_mapping(_load_yaml(dataset_config) if dataset_config.exists() else {})
    return replace(base, T=int(horizon))


def _fit_temperature(
    model: torch.nn.Module,
    dataset: LOWMSynthRankingDataset,
    sim_cfg: SimulationConfig,
    device: torch.device,
    num_samples: int = 32,
    num_hypotheses: int = 4,
    horizon: int = 3,
    seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    losses: dict[float, list[float]] = {float(t): [] for t in np.logspace(-2, 2, 41)}
    diag_gaps: list[float] = []
    count = min(num_samples, dataset.num_episodes)
    for offset in range(count):
        ep = int(offset % dataset.num_episodes)
        episodes = _select_hypothesis_episodes(dataset, ep, num_hypotheses, dataset.cfg.min_law_param_distance, rng)
        context_batch = _make_context_batch(dataset, episodes, context_len=2)
        t0 = max(0, min(2, dataset.T - horizon))
        current = dataset.states[ep, t0].copy()
        mask = dataset.mask[ep, t0].copy()
        action = np.zeros((dataset.nmax, 2), dtype=np.float32)

        def sampler(a: np.ndarray, op_id: int, params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return _simulate_short_trajectory(current, mask, a, op_id, params, sim_cfg, horizon)

        result = evaluate_action_disambiguation(
            model,
            context_batch,
            current,
            action[None],
            sampler,
            num_operator_hypotheses=len(episodes),
            horizon=horizon,
            temperature=1.0,
            device=device,
        )
        matrix = result["energy_matrices"][0]
        diag = np.diag(matrix)
        off = matrix[~np.eye(matrix.shape[0], dtype=bool)]
        diag_gaps.append(float(np.mean(off) - np.mean(diag)))
        for temp, values in losses.items():
            for i in range(matrix.shape[0]):
                probs = posterior_from_energies(matrix[i], temp)
                values.append(-math.log(max(float(probs[i]), 1e-12)))
    best_temp = min(losses, key=lambda temp: float(np.mean(losses[temp])) if losses[temp] else float("inf"))
    return {
        "temperature": float(best_temp),
        "validation_nll": float(np.mean(losses[best_temp])) if losses[best_temp] else 0.0,
        "discriminability_diag_offdiag_gap": float(np.mean(diag_gaps)) if diag_gaps else 0.0,
        "num_temperature_samples": count,
    }


def _evaluate_method(
    method: str,
    action_result: Mapping[str, Any],
    action_norms: np.ndarray,
    rng: np.random.Generator,
) -> int:
    if method == "random_action":
        return int(rng.integers(0, len(action_norms)))
    if method == "max_motion_action":
        return int(np.argmax(action_norms))
    if method == "aoi_energy_separation":
        return int(np.argmax(action_result["energy_separation"]))
    if method == "aoi_entropy_reduction":
        return int(np.argmax(action_result["expected_entropy_reduction"]))
    if method == "oracle_aoi":
        oracle_scores = [_matrix_scores(matrix, temperature=1.0)["expected_entropy_reduction"] for matrix in action_result["oracle_energy_matrices"]]
        return int(np.argmax(np.asarray(oracle_scores, dtype=np.float64)))
    raise ValueError(f"unknown AOI method '{method}'")


def _write_plots(per_method: Mapping[str, Mapping[str, float]], out_dir: Path) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    methods = list(per_method)

    def bar(metric: str, ylabel: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(methods)), 3.6))
        ax.bar(methods, [float(per_method[m].get(metric, 0.0)) for m in methods])
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(plots / filename, dpi=160)
        plt.close(fig)

    bar("entropy_reduction_mean", "Entropy reduction", "entropy_reduction_by_method.png")
    bar("identification_accuracy", "Identification accuracy", "operator_identification_accuracy.png")

    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(methods)), 3.6))
    ax.bar(x - width / 2, [float(per_method[m].get("posterior_entropy_before_mean", 0.0)) for m in methods], width, label="before")
    ax.bar(x + width / 2, [float(per_method[m].get("posterior_entropy_after_mean", 0.0)) for m in methods], width, label="after")
    ax.set_xticks(x, methods, rotation=25, ha="right")
    ax.set_ylabel("Posterior entropy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "posterior_entropy_before_after.png", dpi=160)
    plt.close(fig)


def evaluate_active_operator_inference(
    run_dir: Path,
    split: str = "test_iid",
    checkpoint_name: str = "best_law_pair.pt",
    model_type: str | None = None,
    num_episodes: int = 200,
    num_operator_hypotheses: int = 4,
    num_actions: int = 8,
    horizon: int = 3,
    action_scale: float = 0.5,
    seed: int = 0,
    device_name: str = "auto",
    temperature: float | None = None,
) -> dict[str, Any]:
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    checkpoint_path = _resolve_checkpoint_path(run_dir, checkpoint_name)
    model, config, detected = load_run_model(run_dir, model_type=model_type, checkpoint_name=checkpoint_name, device=device)
    split_path = _ensure_split(config, split)
    ranking_cfg = ranking_config_from_mapping(config)
    ranking_cfg = replace(ranking_cfg, K=2, H=max(1, int(horizon)), M=max(2, int(num_operator_hypotheses)))
    dataset = LOWMSynthRankingDataset(split_path, ranking_cfg, num_samples=None)
    sim_cfg = _simulation_config_from_run(config, horizon)

    val_split = "val"
    calibration: dict[str, float]
    if temperature is None:
        try:
            val_dataset = LOWMSynthRankingDataset(_ensure_split(config, val_split), ranking_cfg, num_samples=None)
            calibration = _fit_temperature(model, val_dataset, sim_cfg, device, num_samples=min(32, num_episodes), num_hypotheses=num_operator_hypotheses, horizon=horizon, seed=seed + 99)
        except FileNotFoundError as exc:
            calibration = {"temperature": 1.0, "validation_nll": 0.0, "discriminability_diag_offdiag_gap": 0.0, "num_temperature_samples": 0, "warning": str(exc)}
    else:
        calibration = {"temperature": float(temperature), "validation_nll": 0.0, "discriminability_diag_offdiag_gap": 0.0, "num_temperature_samples": 0}
    temp = float(calibration["temperature"])

    rng = np.random.default_rng(seed)
    episode_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    method_values: dict[str, list[dict[str, float]]] = {method: [] for method in AOI_METHODS}
    count = min(int(num_episodes), dataset.num_episodes)

    for episode_idx in range(count):
        ep = int(episode_idx % dataset.num_episodes)
        episodes = _select_hypothesis_episodes(dataset, ep, num_operator_hypotheses, ranking_cfg.min_law_param_distance, rng)
        if len(episodes) < 2:
            continue
        context_batch = _make_context_batch(dataset, episodes, context_len=2)
        true_idx = 0
        t0 = max(0, min(2, dataset.T - horizon))
        current = dataset.states[ep, t0].copy()
        mask = dataset.mask[ep, t0].copy()
        actions, action_names = _action_set(num_actions, dataset.nmax, int(mask.sum()), action_scale, rng)

        def sampler(a: np.ndarray, op_id: int, params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return _simulate_short_trajectory(current, mask, a, op_id, params, sim_cfg, horizon)

        action_result = evaluate_action_disambiguation(
            model,
            context_batch,
            current,
            actions,
            sampler,
            num_operator_hypotheses=len(episodes),
            horizon=horizon,
            temperature=temp,
            device=device,
        )
        action_norms = np.linalg.norm(actions.reshape(actions.shape[0], -1), axis=1)
        for action_id, name in enumerate(action_names):
            score_rows.append(
                {
                    "episode": episode_idx,
                    "action_id": action_id,
                    "action_name": name,
                    "energy_separation": float(action_result["energy_separation"][action_id]),
                    "expected_entropy_reduction": float(action_result["expected_entropy_reduction"][action_id]),
                    "posterior_entropy": float(action_result["posterior_entropies"][action_id]),
                }
            )

        for method in AOI_METHODS:
            action_idx = _evaluate_method(method, action_result, action_norms, rng)
            obs_states, obs_actions, obs_mask = sampler(actions[action_idx], int(dataset.op_id[ep]), dataset.op_params[ep])
            if method == "oracle_aoi":
                oracle_outcomes = [sampler(actions[action_idx], int(dataset.op_id[h_ep]), dataset.op_params[h_ep]) for h_ep in episodes]
                oracle_energies = np.asarray([
                    float(((obs_states - outcome[0])[:, :, 0:4] ** 2 * obs_mask[:, :, None]).sum() / max(1.0, float(obs_mask.sum())))
                    for outcome in oracle_outcomes
                ])
                posterior = posterior_from_energies(oracle_energies, temperature=1e-3)
                energies = oracle_energies
            else:
                energies = _score_observed(
                    model,
                    context_batch,
                    torch.from_numpy(obs_states.astype(np.float32)),
                    torch.from_numpy(obs_actions.astype(np.float32)),
                    torch.from_numpy(obs_mask.astype(np.float32)),
                    device,
                ).numpy()
                posterior = posterior_from_energies(energies, temp)
            pred = int(np.argmax(posterior))
            rank = _rank_of_true(energies, true_idx)
            before = math.log(len(episodes))
            after = entropy(posterior)
            record = {
                "episode": episode_idx,
                "method": method,
                "chosen_action": action_idx,
                "chosen_action_name": action_names[action_idx],
                "identification_correct": float(pred == true_idx),
                "posterior_entropy_before": before,
                "posterior_entropy_after": after,
                "entropy_reduction": before - after,
                "true_operator_rank": float(rank),
                "mrr": 1.0 / float(rank),
                "chosen_action_score": float(
                    action_result["energy_separation"][action_idx]
                    if method in {"aoi_energy_separation", "max_motion_action", "random_action"}
                    else action_result["expected_entropy_reduction"][action_idx]
                ),
                "predicted_operator_index": pred,
                "true_operator_index": true_idx,
            }
            episode_rows.append(record)
            method_values[method].append(record)

    per_method: dict[str, dict[str, float]] = {}
    for method, rows in method_values.items():
        if not rows:
            per_method[method] = {
                "identification_accuracy": 0.0,
                "entropy_reduction_mean": 0.0,
                "mrr_mean": 0.0,
                "posterior_entropy_before_mean": math.log(max(1, num_operator_hypotheses)),
                "posterior_entropy_after_mean": math.log(max(1, num_operator_hypotheses)),
            }
            continue
        per_method[method] = {
            "identification_accuracy": float(np.mean([row["identification_correct"] for row in rows])),
            "entropy_reduction_mean": float(np.mean([row["entropy_reduction"] for row in rows])),
            "entropy_reduction_std": float(np.std([row["entropy_reduction"] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
            "mrr_mean": float(np.mean([row["mrr"] for row in rows])),
            "mrr_std": float(np.std([row["mrr"] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
            "posterior_entropy_before_mean": float(np.mean([row["posterior_entropy_before"] for row in rows])),
            "posterior_entropy_after_mean": float(np.mean([row["posterior_entropy_after"] for row in rows])),
            "true_operator_rank_mean": float(np.mean([row["true_operator_rank"] for row in rows])),
        }

    out_dir = run_dir / "eval" / split / "aoi"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model_type": detected,
        "split": split,
        "checkpoint_requested": checkpoint_name,
        "checkpoint_used": checkpoint_path.name,
        "temperature": temp,
        "calibration": calibration,
        "num_episodes": count,
        "num_operator_hypotheses": num_operator_hypotheses,
        "num_actions": num_actions,
        "horizon": horizon,
        "action_scale": action_scale,
        "methods": per_method,
    }
    with (out_dir / "aoi_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    with (out_dir / "aoi_per_episode.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()) if episode_rows else ["episode", "method"])
        writer.writeheader()
        writer.writerows(episode_rows)
    with (out_dir / "action_score_examples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(score_rows[0].keys()) if score_rows else ["episode", "action_id"])
        writer.writeheader()
        writer.writerows(score_rows)
    _write_plots(per_method, out_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test_iid")
    parser.add_argument("--checkpoint", type=str, default="best_law_pair.pt")
    parser.add_argument("--model_type", type=str, default=None, choices=["fixed_energy", "direct_context_energy", "lowm"])
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--num-operator-hypotheses", type=int, default=4)
    parser.add_argument("--num-actions", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--action-scale", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()
    result = evaluate_active_operator_inference(
        args.run,
        split=args.split,
        checkpoint_name=args.checkpoint,
        model_type=args.model_type,
        num_episodes=args.num_episodes,
        num_operator_hypotheses=args.num_operator_hypotheses,
        num_actions=args.num_actions,
        horizon=args.horizon,
        action_scale=args.action_scale,
        seed=args.seed,
        device_name=args.device,
        temperature=args.temperature,
    )
    compact = {
        "model_type": result["model_type"],
        "split": result["split"],
        "temperature": result["temperature"],
        "aoi_energy_separation_acc": result["methods"]["aoi_energy_separation"]["identification_accuracy"],
        "aoi_entropy_reduction_acc": result["methods"]["aoi_entropy_reduction"]["identification_accuracy"],
        "random_action_acc": result["methods"]["random_action"]["identification_accuracy"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
