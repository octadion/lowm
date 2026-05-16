"""Energy-Based Trajectory World Model inference pilot.

This module probes whether an already trained LOWM/OMC energy critic can be
used as an inference-time trajectory optimizer. It does not add a training
objective and it does not update model parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
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
from lowm.data.negatives import is_law_mismatch
from lowm.data.simulate import SimulationConfig, config_from_mapping, step_state
from lowm.eval.evaluate_all import _ensure_split, _resolve_checkpoint_path, load_run_model


STATE_DYN = slice(0, 4)
POSITION = slice(0, 2)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return loaded if isinstance(loaded, dict) else {}


def _simulation_config_from_run(config: Mapping[str, Any], horizon: int) -> SimulationConfig:
    data_cfg = dict(config.get("data", {}))
    dataset_config = Path(data_cfg.get("dataset_config", "configs/lowm_synth_v0.yaml"))
    base = config_from_mapping(_load_yaml(dataset_config))
    return replace(base, T=int(horizon))


def _context_from_episode(dataset: LOWMSynthRankingDataset, episode: int, context_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    context_len = max(1, min(int(context_len), dataset.T))
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


def _to_batch(x: torch.Tensor, ndim: int) -> torch.Tensor:
    return x.unsqueeze(0) if x.ndim == ndim - 1 else x


def _model_lambda(
    model: torch.nn.Module,
    context_states: torch.Tensor,
    context_actions: torch.Tensor,
    context_mask: torch.Tensor,
    use_mu: bool,
) -> torch.Tensor | None:
    if not hasattr(model, "context_encoder"):
        return None
    mu, logvar = model.context_encoder(context_states, context_actions, context_mask)
    if use_mu:
        return mu.detach()
    std = torch.exp(0.5 * logvar)
    return (mu + torch.randn_like(std) * std).detach()


def _score_trajectory(
    model: torch.nn.Module,
    context_states: torch.Tensor,
    context_actions: torch.Tensor,
    context_mask: torch.Tensor,
    states: torch.Tensor,
    actions: torch.Tensor,
    mask: torch.Tensor,
    lambdas: torch.Tensor | None,
) -> torch.Tensor:
    """Score one candidate trajectory per context.

    Shapes:
    context: [B,K,2,N,D], states: [B,H+1,N,D], actions: [B,H,N,2].
    Returns [B].
    """

    if lambdas is not None and hasattr(model, "energy"):
        return model.energy(states[:, None], actions[:, None], mask[:, None], lambdas).squeeze(1)
    batch = {
        "context_states": context_states,
        "context_actions": context_actions,
        "context_mask": context_mask,
        "cand_states": states[:, None],
        "cand_actions": actions[:, None],
        "cand_mask": mask[:, None],
        "labels": torch.zeros(states.shape[0], dtype=torch.long, device=states.device),
    }
    output = model(batch)
    energies = output["energies"] if isinstance(output, Mapping) else output
    return energies.squeeze(1)


def _regularizers(
    states: torch.Tensor,
    init_states: torch.Tensor,
    mask: torch.Tensor,
    eta_init: float,
    eta_smooth: float,
    eta_anchor: float,
    eta_bounds: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    active = mask[..., None].to(states.dtype)
    init_reg = ((states[:, 0] - init_states[:, 0]) ** 2 * active[:, 0]).sum(dim=(1, 2))
    diffs = states[:, 1:, :, STATE_DYN] - states[:, :-1, :, STATE_DYN]
    smooth_mask = (mask[:, 1:] * mask[:, :-1])[..., None].to(states.dtype)
    smooth = (diffs.square() * smooth_mask).sum(dim=(1, 2, 3))
    anchor = (((states[:, 1:, :, STATE_DYN] - init_states[:, 1:, :, STATE_DYN]) ** 2) * active[:, 1:]).sum(dim=(1, 2, 3))
    pos = states[..., POSITION]
    bounds = (torch.relu(-pos).square() + torch.relu(pos - 1.0).square())
    bounds = (bounds * active).sum(dim=(1, 2, 3))
    total = eta_init * init_reg + eta_smooth * smooth + eta_anchor * anchor + eta_bounds * bounds
    return total, {"init": init_reg, "smooth": smooth, "anchor": anchor, "bounds": bounds}


def _masked_dynamic_mse(states: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    active = mask[..., None].to(states.dtype)
    diff = (states[..., STATE_DYN] - target[..., STATE_DYN]) * active
    denom = active.sum(dim=(1, 2, 3)).clamp_min(1.0) * 4.0
    return diff.square().sum(dim=(1, 2, 3)) / denom


def _trajectory_delta(states: torch.Tensor, init_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(_masked_dynamic_mse(states, init_states, mask).clamp_min(0.0))


def _build_states(s0: torch.Tensor, future_dyn: torch.Tensor, init_future_static: torch.Tensor) -> torch.Tensor:
    future = torch.cat([future_dyn, init_future_static], dim=-1)
    return torch.cat([s0[:, None], future], dim=1)


def optimize_trajectory_energy(
    model: torch.nn.Module,
    context_states: torch.Tensor,
    context_actions: torch.Tensor,
    context_mask: torch.Tensor,
    init_states: torch.Tensor,
    actions: torch.Tensor,
    init_mask: torch.Tensor,
    num_steps: int = 100,
    step_size: float = 1e-2,
    eta_init: float = 10.0,
    eta_smooth: float = 0.01,
    eta_anchor: float = 0.01,
    eta_bounds: float = 0.0,
    clamp_bounds: tuple[float, float] | None = None,
    use_mu: bool = True,
    optimizer: str = "adam",
    langevin_noise: float = 0.0,
    gt_states: torch.Tensor | None = None,
    data_range: float | None = None,
) -> dict[str, Any]:
    """Optimize future states s_1:H to reduce E_theta(tau, lambda).

    The function freezes model parameters and only differentiates through the
    trajectory variables. Static object attributes radius/mass/type are kept
    fixed; only position and velocity dimensions are optimized.
    """

    model.eval()
    device = next(model.parameters()).device
    original_flags = [p.requires_grad for p in model.parameters()]
    for param in model.parameters():
        param.requires_grad_(False)

    context_states = _to_batch(context_states.to(device=device, dtype=torch.float32), 5)
    context_actions = _to_batch(context_actions.to(device=device, dtype=torch.float32), 4)
    context_mask = _to_batch(context_mask.to(device=device, dtype=torch.float32), 4)
    init_states = _to_batch(init_states.to(device=device, dtype=torch.float32), 4)
    actions = _to_batch(actions.to(device=device, dtype=torch.float32), 4)
    init_mask = _to_batch(init_mask.to(device=device, dtype=torch.float32), 3)
    gt = _to_batch(gt_states.to(device=device, dtype=torch.float32), 4) if gt_states is not None else None

    if init_states.shape[0] != context_states.shape[0]:
        if context_states.shape[0] == 1:
            context_states = context_states.expand(init_states.shape[0], -1, -1, -1, -1)
            context_actions = context_actions.expand(init_states.shape[0], -1, -1, -1)
            context_mask = context_mask.expand(init_states.shape[0], -1, -1, -1)
        else:
            raise ValueError("context batch size must match trajectory batch size")

    s0 = init_states[:, 0].detach()
    static = init_states[:, 1:, :, 4:].detach()
    future_dyn = init_states[:, 1:, :, STATE_DYN].detach().clone().requires_grad_(True)
    data_limit = float(data_range if data_range is not None else init_states[..., STATE_DYN].detach().std().clamp_min(1e-3).item())
    divergence_limit = max(1e-3, 10.0 * data_limit)
    lambdas = _model_lambda(model, context_states, context_actions, context_mask, use_mu)

    def current_states() -> torch.Tensor:
        return _build_states(s0, future_dyn, static)

    def objective(states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        energy = _score_trajectory(model, context_states, context_actions, context_mask, states, actions, init_mask, lambdas)
        reg, regs = _regularizers(states, init_states, init_mask, eta_init, eta_smooth, eta_anchor, eta_bounds)
        return energy + reg, energy, regs

    with torch.enable_grad():
        before_states = current_states()
        obj_before, energy_before, regs_before = objective(before_states)
        initial_energy_abs = energy_before.detach().abs().clamp_min(1e-6)

        opt_name = optimizer.lower()
        if opt_name == "adam":
            opt = torch.optim.Adam([future_dyn], lr=float(step_size))
        elif opt_name == "sgd":
            opt = torch.optim.SGD([future_dyn], lr=float(step_size))
        else:
            raise ValueError("optimizer must be 'adam' or 'sgd'")

        curves: list[dict[str, float]] = [
            {
                "step": 0,
                "energy_mean": float(energy_before.detach().mean().cpu().item()),
                "objective_mean": float(obj_before.detach().mean().cpu().item()),
            }
        ]
        diverged = torch.zeros(init_states.shape[0], dtype=torch.bool, device=device)
        successful_steps = torch.zeros(init_states.shape[0], dtype=torch.long, device=device)

        for step in range(1, int(num_steps) + 1):
            opt.zero_grad(set_to_none=True)
            states = current_states()
            obj, energy, _ = objective(states)
            loss = obj[~diverged].mean() if (~diverged).any() else obj.mean() * 0.0
            loss.backward()
            opt.step()
            if float(langevin_noise) > 0:
                with torch.no_grad():
                    future_dyn.add_(torch.randn_like(future_dyn) * float(langevin_noise))
            with torch.no_grad():
                if clamp_bounds is not None:
                    lo, hi = float(clamp_bounds[0]), float(clamp_bounds[1])
                    future_dyn[..., POSITION].clamp_(lo, hi)
                states_now = current_states()
                obj_now, energy_now, _ = objective(states_now)
                sample_bad = (
                    ~torch.isfinite(states_now).flatten(1).all(dim=1)
                    | (states_now[..., STATE_DYN].detach().abs().flatten(1).amax(dim=1) > divergence_limit)
                    | (energy_now.detach() > 100.0 * initial_energy_abs)
                    | ~torch.isfinite(energy_now.detach())
                )
                diverged |= sample_bad
                successful_steps += (~diverged).long()
                curves.append(
                    {
                        "step": step,
                        "energy_mean": float(energy_now.detach()[~diverged].mean().cpu().item()) if (~diverged).any() else float("nan"),
                        "objective_mean": float(obj_now.detach()[~diverged].mean().cpu().item()) if (~diverged).any() else float("nan"),
                    }
                )

        final_states = current_states().detach()
        obj_after, energy_after, regs_after = objective(final_states)

    for param, flag in zip(model.parameters(), original_flags):
        param.requires_grad_(flag)

    result: dict[str, Any] = {
        "optimized_states": final_states.detach().cpu(),
        "energy_before": energy_before.detach().cpu(),
        "energy_after": energy_after.detach().cpu(),
        "objective_before": obj_before.detach().cpu(),
        "objective_after": obj_after.detach().cpu(),
        "optimization_curve": curves,
        "trajectory_delta": _trajectory_delta(final_states, init_states, init_mask).detach().cpu(),
        "regularizer_values": {f"before_{k}": v.detach().cpu() for k, v in regs_before.items()}
        | {f"after_{k}": v.detach().cpu() for k, v in regs_after.items()},
        "diverged": diverged.detach().cpu(),
        "optimization_steps_successful": successful_steps.detach().cpu(),
    }
    if gt is not None:
        result["mse_to_gt_before"] = _masked_dynamic_mse(init_states, gt, init_mask).detach().cpu()
        result["mse_to_gt_after"] = _masked_dynamic_mse(final_states, gt, init_mask).detach().cpu()
    return result


def _sample_items(dataset: LOWMSynthRankingDataset, num_samples: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    if len(indices) > num_samples:
        indices = rng.choice(indices, size=num_samples, replace=False)
    return [dataset[int(i)] for i in indices]


def _data_std(dataset: LOWMSynthRankingDataset) -> tuple[np.ndarray, float]:
    active = dataset.mask > 0.5
    dyn = dataset.states[..., STATE_DYN]
    vals = dyn[active]
    if vals.size == 0:
        return np.ones((4,), dtype=np.float32), 1.0
    per_dim = np.maximum(vals.reshape(-1, 4).std(axis=0), 1e-3).astype(np.float32)
    return per_dim, float(np.mean(per_dim))


def _make_corruption(
    item: Mapping[str, Any],
    dataset: LOWMSynthRankingDataset,
    sim_cfg: SimulationConfig,
    corruption_type: str,
    noise_std: float,
    data_std_per_dim: np.ndarray,
    rng: np.random.Generator,
) -> torch.Tensor:
    gt = item["pos_states"].detach().cpu().numpy().copy()
    actions = item["pos_actions"].detach().cpu().numpy()
    mask = item["pos_mask"].detach().cpu().numpy()
    init = gt.copy()
    if corruption_type == "gaussian":
        noise = rng.normal(0.0, noise_std, size=init[1:, :, STATE_DYN].shape).astype(np.float32)
        init[1:, :, STATE_DYN] += noise * data_std_per_dim[None, None, :]
        init[1:, :, POSITION] = np.clip(init[1:, :, POSITION], 0.0, 1.0)
    elif corruption_type == "temporal_shuffle":
        if init.shape[0] > 2:
            perm = rng.permutation(np.arange(1, init.shape[0]))
            init[1:] = init[perm]
    elif corruption_type == "wrong_operator":
        true_op = int(item["query_op_id"].item())
        candidates = np.where(dataset.op_id != true_op)[0]
        wrong_ep = int(rng.choice(candidates)) if len(candidates) else int((int(item["query_episode"].item()) + 1) % dataset.num_episodes)
        init, _, _ = _simulate_from_s0(gt[0], mask[0], actions, int(dataset.op_id[wrong_ep]), dataset.op_params[wrong_ep], sim_cfg)
    else:
        raise ValueError("corruption_type must be gaussian, temporal_shuffle, or wrong_operator")
    init[0] = gt[0]
    return torch.from_numpy(init.astype(np.float32))


def _simulate_from_s0(
    s0: np.ndarray,
    mask0: np.ndarray,
    actions: np.ndarray,
    op_id: int,
    op_params: np.ndarray,
    cfg: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizon = actions.shape[0]
    n = int(np.sum(mask0 > 0.5))
    states = np.zeros((horizon + 1, cfg.nmax, cfg.d_object), dtype=np.float32)
    traj_mask = np.zeros((horizon + 1, cfg.nmax), dtype=np.float32)
    states[0] = s0.astype(np.float32)
    traj_mask[:, :n] = 1.0
    for t in range(horizon):
        states[t + 1] = step_state(states[t], actions[t], n, int(op_id), op_params.astype(np.float32), cfg)
    return states, actions.astype(np.float32), traj_mask


def _distractor_episodes(dataset: LOWMSynthRankingDataset, item: Mapping[str, Any], count: int, rng: np.random.Generator) -> list[int]:
    true_op = int(item["query_op_id"].item())
    true_params = item["query_op_params"].detach().cpu().numpy()
    candidates = [
        int(i)
        for i in range(dataset.num_episodes)
        if is_law_mismatch(true_op, true_params, int(dataset.op_id[i]), dataset.op_params[i], dataset.cfg.min_law_param_distance)
    ]
    rng.shuffle(candidates)
    return candidates[:count]


def _context_tensor_from_episode(dataset: LOWMSynthRankingDataset, episode: int, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states, actions, mask = _context_from_episode(dataset, episode, k)
    return torch.from_numpy(states), torch.from_numpy(actions), torch.from_numpy(mask)


def _score_against_contexts(
    model: torch.nn.Module,
    traj_states: torch.Tensor,
    traj_actions: torch.Tensor,
    traj_mask: torch.Tensor,
    contexts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    use_mu: bool,
) -> torch.Tensor:
    cs = torch.stack([c[0] for c in contexts], dim=0).to(device)
    ca = torch.stack([c[1] for c in contexts], dim=0).to(device)
    cm = torch.stack([c[2] for c in contexts], dim=0).to(device)
    states = traj_states.to(device).unsqueeze(0).expand(len(contexts), -1, -1, -1).clone()
    actions = traj_actions.to(device).unsqueeze(0).expand(len(contexts), -1, -1, -1).clone()
    mask = traj_mask.to(device).unsqueeze(0).expand(len(contexts), -1, -1).clone()
    with torch.no_grad():
        lambdas = _model_lambda(model, cs, ca, cm, use_mu)
        return _score_trajectory(model, cs, ca, cm, states, actions, mask, lambdas).detach().cpu()


def _cross_operator_metrics(
    model: torch.nn.Module,
    dataset: LOWMSynthRankingDataset,
    item: Mapping[str, Any],
    traj_states: torch.Tensor,
    traj_actions: torch.Tensor,
    traj_mask: torch.Tensor,
    device: torch.device,
    rng: np.random.Generator,
    use_mu: bool,
    distractors: int = 3,
) -> dict[str, float]:
    own_context = (item["context_states"], item["context_actions"], item["context_mask"])
    wrong_eps = _distractor_episodes(dataset, item, distractors, rng)
    contexts = [own_context, *[_context_tensor_from_episode(dataset, ep, item["context_states"].shape[0]) for ep in wrong_eps]]
    energies = _score_against_contexts(model, traj_states, traj_actions, traj_mask, contexts, device, use_mu)
    own = float(energies[0].item())
    wrong = energies[1:].numpy() if len(energies) > 1 else np.asarray([], dtype=np.float32)
    return {
        "own_operator_energy": own,
        "mean_wrong_operator_energy": float(np.mean(wrong)) if wrong.size else 0.0,
        "min_wrong_operator_energy": float(np.min(wrong)) if wrong.size else 0.0,
        "own_vs_wrong_gap_mean": float(np.mean(wrong) - own) if wrong.size else 0.0,
        "own_vs_wrong_gap_min": float(np.min(wrong) - own) if wrong.size else 0.0,
        "own_vs_wrong_pair_acc": float(np.mean(own < wrong)) if wrong.size else 0.0,
    }


def _gradient_sanity_check(
    model: torch.nn.Module,
    items: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, float]:
    batch = items[: min(4, len(items))]
    if not batch:
        raise RuntimeError("gradient sanity failed: no samples")
    cs = torch.stack([x["context_states"] for x in batch], dim=0).to(device)
    ca = torch.stack([x["context_actions"] for x in batch], dim=0).to(device)
    cm = torch.stack([x["context_mask"] for x in batch], dim=0).to(device)
    states = torch.stack([x["pos_states"] for x in batch], dim=0).to(device)
    actions = torch.stack([x["pos_actions"] for x in batch], dim=0).to(device)
    mask = torch.stack([x["pos_mask"] for x in batch], dim=0).to(device)
    future_dyn = states[:, 1:, :, STATE_DYN].detach().clone().requires_grad_(True)
    full_states = _build_states(states[:, 0].detach(), future_dyn, states[:, 1:, :, 4:].detach())
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    lambdas = _model_lambda(model, cs, ca, cm, use_mu=True)
    energy = _score_trajectory(model, cs, ca, cm, full_states, actions, mask, lambdas)
    energy.mean().backward()
    grad = future_dyn.grad
    if grad is None:
        raise RuntimeError("gradient sanity failed: trajectory gradient is None")
    norm = float(grad.norm().detach().cpu().item())
    if not torch.isfinite(grad).all():
        raise RuntimeError("gradient sanity failed: non-finite trajectory gradient")
    if norm <= 1e-8:
        raise RuntimeError(f"gradient sanity failed: gradient norm too small ({norm:.3e})")
    if norm >= 1e6:
        raise RuntimeError(f"gradient sanity failed: gradient norm too large ({norm:.3e})")
    return {"grad_norm": norm, "num_samples": len(batch)}


def _run_preflight(
    model: torch.nn.Module,
    dataset: LOWMSynthRankingDataset,
    items: list[dict[str, Any]],
    device: torch.device,
    num_steps: int,
    noise_std: float,
    data_std_per_dim: np.ndarray,
    data_std_scalar: float,
    rng: np.random.Generator,
    eta_smooth: float,
    eta_anchor: float,
    eta_bounds: float,
    sim_cfg: SimulationConfig,
) -> dict[str, Any]:
    started = time.perf_counter()
    checks: dict[str, Any] = {"gradient_sanity": _gradient_sanity_check(model, items, device)}
    step_results: list[dict[str, float]] = []
    candidate_steps = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    sweep_items = items[: min(10, len(items))]
    for step_size in candidate_steps:
        improvements = []
        for item in sweep_items:
            init = _make_corruption(item, dataset, sim_cfg, "gaussian", noise_std, data_std_per_dim, rng)
            result = optimize_trajectory_energy(
                model,
                item["context_states"],
                item["context_actions"],
                item["context_mask"],
                init,
                item["pos_actions"],
                item["pos_mask"],
                num_steps=min(30, max(5, num_steps)),
                step_size=step_size,
                eta_smooth=eta_smooth,
                eta_anchor=eta_anchor,
                eta_bounds=eta_bounds,
                clamp_bounds=(0.0, 1.0),
                gt_states=item["pos_states"],
                data_range=data_std_scalar,
            )
            before = float(result["mse_to_gt_before"].mean().item())
            after = float(result["mse_to_gt_after"].mean().item())
            if not bool(result["diverged"].any().item()):
                improvements.append(before - after)
        step_results.append({"step_size": float(step_size), "mean_mse_improvement": float(np.mean(improvements)) if improvements else float("-inf")})
    chosen = max(step_results, key=lambda row: row["mean_mse_improvement"])["step_size"]
    checks["step_size_sweep"] = step_results
    checks["chosen_step_size"] = chosen

    smoke = []
    for item in items[: min(5, len(items))]:
        init = _make_corruption(item, dataset, sim_cfg, "gaussian", noise_std, data_std_per_dim, rng)
        result = optimize_trajectory_energy(
            model,
            item["context_states"],
            item["context_actions"],
            item["context_mask"],
            init,
            item["pos_actions"],
            item["pos_mask"],
            num_steps=10,
            step_size=chosen,
            eta_smooth=eta_smooth,
            eta_anchor=eta_anchor,
            eta_bounds=eta_bounds,
            clamp_bounds=(0.0, 1.0),
            gt_states=item["pos_states"],
            data_range=data_std_scalar,
        )
        decreased = bool((result["energy_after"] < result["energy_before"]).item())
        diverged = bool(result["diverged"].item())
        smoke.append({"energy_decreased": decreased, "diverged": diverged})
    good = sum(row["energy_decreased"] and not row["diverged"] for row in smoke)
    if good < min(4, len(smoke)):
        raise RuntimeError(f"smoke test failed: {good}/{len(smoke)} samples decreased energy without divergence")
    checks["smoke_test"] = {"passed_samples": good, "num_samples": len(smoke), "details": smoke}
    checks["pre_flight_runtime_seconds"] = time.perf_counter() - started
    return checks


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    valid = [row for row in rows if not row["diverged"]]

    def mean(key: str, pool: list[dict[str, Any]] = valid) -> float:
        return float(np.mean([row[key] for row in pool])) if pool else 0.0

    metrics = {
        "num_samples": len(rows),
        "num_valid_samples": len(valid),
        "fraction_nan_or_diverged": 1.0 - (len(valid) / max(1, len(rows))),
        "mean_energy_before": mean("energy_before"),
        "mean_energy_after": mean("energy_after"),
        "mean_energy_drop": mean("energy_before") - mean("energy_after"),
        "fraction_energy_decreased": float(np.mean([row["energy_after"] < row["energy_before"] for row in valid])) if valid else 0.0,
        "mean_objective_before": mean("objective_before"),
        "mean_objective_after": mean("objective_after"),
        "mean_mse_to_gt_before": mean("mse_to_gt_before"),
        "mean_mse_to_gt_after": mean("mse_to_gt_after"),
        "fraction_mse_improved": float(np.mean([row["mse_to_gt_after"] < row["mse_to_gt_before"] for row in valid])) if valid else 0.0,
        "mean_trajectory_delta": mean("trajectory_delta"),
        "mean_final_smoothness": mean("final_smoothness"),
        "mean_optimization_steps_successful": mean("optimization_steps_successful", rows),
        "own_vs_wrong_pair_acc_before": mean("own_vs_wrong_pair_acc_before"),
        "own_vs_wrong_pair_acc_after": mean("own_vs_wrong_pair_acc_after"),
        "own_vs_wrong_gap_mean_before": mean("own_vs_wrong_gap_mean_before"),
        "own_vs_wrong_gap_mean_after": mean("own_vs_wrong_gap_mean_after"),
        "own_vs_wrong_gap_min_before": mean("own_vs_wrong_gap_min_before"),
        "own_vs_wrong_gap_min_after": mean("own_vs_wrong_gap_min_after"),
    }
    return metrics


def _go_no_go(metrics: Mapping[str, float], noise_std: float, data_std: float) -> dict[str, Any]:
    criteria = {
        "C1_optimization_works": float(metrics.get("fraction_energy_decreased", 0.0)) > 0.90,
        "C2_recovery_happens": float(metrics.get("fraction_mse_improved", 0.0)) > 0.55,
        "C3_operator_coherence_preserved": float(metrics.get("own_vs_wrong_pair_acc_after", 0.0)) > 0.65,
        "C4_no_coherence_collapse": float(metrics.get("own_vs_wrong_pair_acc_after", 0.0))
        >= float(metrics.get("own_vs_wrong_pair_acc_before", 0.0)) - 0.05,
        "C5_no_excessive_drift": float(metrics.get("mean_trajectory_delta", float("inf"))) < 2.0 * float(noise_std) * float(data_std),
    }
    passed = sum(bool(v) for v in criteria.values())
    verdict = "STRONG GO" if passed == 5 else "WEAK GO" if passed == 4 else "AMBIGUOUS" if passed == 3 else "NO-GO"
    return {"criteria": criteria, "num_passed": passed, "verdict": verdict}


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0].keys()) if rows else ["sample_id"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_curve_csv(curves: list[dict[str, Any]], path: Path) -> None:
    fieldnames = ["sample_id", "step", "energy_mean", "objective_mean"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(curves)


def _plot_curves(curves: list[dict[str, Any]], path: Path) -> None:
    if not curves:
        return
    by_step: dict[int, list[float]] = {}
    for row in curves:
        value = float(row.get("objective_mean", 0.0))
        if np.isfinite(value):
            by_step.setdefault(int(row["step"]), []).append(value)
    xs = sorted(by_step)
    ys = [float(np.mean(by_step[x])) for x in xs]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(xs, ys, marker="o", markersize=2)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Objective")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_examples(examples: list[dict[str, np.ndarray]], path: Path) -> None:
    if not examples:
        return
    cols = min(3, len(examples))
    fig, axes = plt.subplots(1, cols, figsize=(4.0 * cols, 3.8), squeeze=False)
    for ax, ex in zip(axes[0], examples[:cols]):
        mask = ex["mask"] > 0.5
        for name, color in [("gt", "black"), ("init", "tab:orange"), ("opt", "tab:blue")]:
            states = ex[name]
            for obj in range(states.shape[1]):
                active = mask[:, obj]
                if active.any():
                    ax.plot(states[active, obj, 0], states[active, obj, 1], color=color, alpha=0.75, linewidth=1.2, label=name if obj == 0 else None)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate_ebtwm_inference(
    run_dir: Path,
    split: str,
    checkpoint: str,
    num_samples: int = 100,
    num_steps: int = 100,
    step_size: float = 1e-2,
    noise_std: float = 0.05,
    corruption_type: str = "gaussian",
    horizon: int | None = None,
    device: str = "auto",
    mode: str = "denoise",
    eta_smooth: float = 0.01,
    eta_anchor: float = 0.01,
    eta_bounds: float = 0.0,
    langevin_noise: float = 0.0,
    seed: int = 0,
    skip_preflight: bool = False,
    use_mu: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    device_obj = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    checkpoint_path = _resolve_checkpoint_path(run_dir, checkpoint)
    model, config, detected = load_run_model(run_dir, checkpoint_name=checkpoint, device=device_obj)
    split_path = _ensure_split(config, split)
    ranking_cfg = ranking_config_from_mapping(config)
    eval_horizon = int(horizon or ranking_cfg.H)
    ranking_cfg = replace(ranking_cfg, H=eval_horizon)
    dataset = LOWMSynthRankingDataset(split_path, ranking_cfg, num_samples=None)
    sim_cfg = _simulation_config_from_run(config, eval_horizon)
    rng = np.random.default_rng(seed)
    data_std_per_dim, data_std_scalar = _data_std(dataset)
    items = _sample_items(dataset, num_samples, seed)
    if not items:
        raise RuntimeError("EBTWM inference has no samples to evaluate")

    preflight: dict[str, Any] = {"skipped": bool(skip_preflight), "chosen_step_size": float(step_size), "pre_flight_runtime_seconds": 0.0}
    if not skip_preflight:
        preflight = _run_preflight(
            model,
            dataset,
            items,
            device_obj,
            num_steps=num_steps,
            noise_std=noise_std,
            data_std_per_dim=data_std_per_dim,
            data_std_scalar=data_std_scalar,
            rng=rng,
            eta_smooth=eta_smooth,
            eta_anchor=eta_anchor,
            eta_bounds=eta_bounds,
            sim_cfg=sim_cfg,
        )
        step_size = float(preflight["chosen_step_size"])

    main_started = time.perf_counter()
    per_sample: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    examples: list[dict[str, np.ndarray]] = []

    for sample_id, item in enumerate(items):
        if mode == "counterfactual":
            source_ep = int(item["query_episode"].item())
            target_candidates = np.where(dataset.op_id != dataset.op_id[source_ep])[0]
            target_ep = int(target_candidates[sample_id % len(target_candidates)]) if len(target_candidates) else int((source_ep + 1) % dataset.num_episodes)
            target_context = _context_tensor_from_episode(dataset, target_ep, item["context_states"].shape[0])
            context_states, context_actions, context_mask = target_context
            gt_np, _, _ = _simulate_from_s0(
                item["pos_states"][0].numpy(),
                item["pos_mask"][0].numpy(),
                item["pos_actions"].numpy(),
                int(dataset.op_id[target_ep]),
                dataset.op_params[target_ep],
                sim_cfg,
            )
            gt_states = torch.from_numpy(gt_np.astype(np.float32))
            init_states = item["pos_states"].clone()
        else:
            context_states, context_actions, context_mask = item["context_states"], item["context_actions"], item["context_mask"]
            gt_states = item["pos_states"]
            init_states = _make_corruption(item, dataset, sim_cfg, corruption_type, noise_std, data_std_per_dim, rng)

        result = optimize_trajectory_energy(
            model,
            context_states,
            context_actions,
            context_mask,
            init_states,
            item["pos_actions"],
            item["pos_mask"],
            num_steps=num_steps,
            step_size=step_size,
            eta_smooth=eta_smooth,
            eta_anchor=eta_anchor,
            eta_bounds=eta_bounds,
            clamp_bounds=(0.0, 1.0),
            use_mu=use_mu,
            langevin_noise=langevin_noise,
            gt_states=gt_states,
            data_range=data_std_scalar,
        )
        opt_states = result["optimized_states"].squeeze(0)
        init_b = init_states.unsqueeze(0) if init_states.ndim == 3 else init_states
        mask_b = item["pos_mask"].unsqueeze(0)
        smooth = float(_regularizers(opt_states.unsqueeze(0), init_b, mask_b, 0.0, 1.0, 0.0, 0.0)[1]["smooth"].item())
        before_cross = _cross_operator_metrics(model, dataset, item, init_states, item["pos_actions"], item["pos_mask"], device_obj, rng, use_mu)
        after_cross = _cross_operator_metrics(model, dataset, item, opt_states, item["pos_actions"], item["pos_mask"], device_obj, rng, use_mu)
        diverged = bool(result["diverged"].item())
        row = {
            "sample_id": sample_id,
            "mode": mode,
            "corruption_type": corruption_type,
            "diverged": diverged,
            "energy_before": float(result["energy_before"].item()),
            "energy_after": float(result["energy_after"].item()),
            "objective_before": float(result["objective_before"].item()),
            "objective_after": float(result["objective_after"].item()),
            "mse_to_gt_before": float(result["mse_to_gt_before"].item()),
            "mse_to_gt_after": float(result["mse_to_gt_after"].item()),
            "trajectory_delta": float(result["trajectory_delta"].item()),
            "final_smoothness": smooth,
            "optimization_steps_successful": int(result["optimization_steps_successful"].item()),
            "own_vs_wrong_pair_acc_before": before_cross["own_vs_wrong_pair_acc"],
            "own_vs_wrong_pair_acc_after": after_cross["own_vs_wrong_pair_acc"],
            "own_vs_wrong_gap_mean_before": before_cross["own_vs_wrong_gap_mean"],
            "own_vs_wrong_gap_mean_after": after_cross["own_vs_wrong_gap_mean"],
            "own_vs_wrong_gap_min_before": before_cross["own_vs_wrong_gap_min"],
            "own_vs_wrong_gap_min_after": after_cross["own_vs_wrong_gap_min"],
        }
        if mode == "counterfactual":
            source_context = (item["context_states"], item["context_actions"], item["context_mask"])
            target_context = (context_states, context_actions, context_mask)
            before_source_target = _score_against_contexts(model, init_states, item["pos_actions"], item["pos_mask"], [source_context, target_context], device_obj, use_mu)
            after_source_target = _score_against_contexts(model, opt_states, item["pos_actions"], item["pos_mask"], [source_context, target_context], device_obj, use_mu)
            row.update(
                {
                    "energy_under_A_before": float(before_source_target[0].item()),
                    "energy_under_B_before": float(before_source_target[1].item()),
                    "energy_under_A_after": float(after_source_target[0].item()),
                    "energy_under_B_after": float(after_source_target[1].item()),
                    "cross_operator_gap_before": float(before_source_target[0].item() - before_source_target[1].item()),
                    "cross_operator_gap_after": float(after_source_target[0].item() - after_source_target[1].item()),
                }
            )
        per_sample.append(row)
        cross_rows.append({"sample_id": sample_id, "phase": "before", **before_cross})
        cross_rows.append({"sample_id": sample_id, "phase": "after", **after_cross})
        for curve in result["optimization_curve"]:
            curve_rows.append({"sample_id": sample_id, **curve})
        if len(examples) < 3:
            examples.append({"gt": gt_states.numpy(), "init": init_states.numpy(), "opt": opt_states.numpy(), "mask": item["pos_mask"].numpy()})

    metrics = _aggregate(per_sample)
    metrics["go_no_go_decision"] = _go_no_go(metrics, noise_std, data_std_scalar)
    main_runtime = time.perf_counter() - main_started
    total_runtime = time.perf_counter() - started
    metrics.update(
        {
            "model_type": detected,
            "split": split,
            "checkpoint_requested": checkpoint,
            "checkpoint_used": checkpoint_path.name,
            "mode": mode,
            "corruption_type": corruption_type,
            "num_steps": int(num_steps),
            "step_size": float(step_size),
            "noise_std": float(noise_std),
            "data_std": float(data_std_scalar),
            "pre_flight": preflight,
            "wall_clock_seconds": total_runtime,
            "pre_flight_runtime_seconds": float(preflight.get("pre_flight_runtime_seconds", 0.0)),
            "main_pilot_runtime_seconds": main_runtime,
            "samples_per_second": len(per_sample) / max(total_runtime, 1e-9),
            "gpu_memory_peak_mb": float(torch.cuda.max_memory_allocated(device_obj) / (1024**2)) if device_obj.type == "cuda" else 0.0,
        }
    )

    out_dir = run_dir / "eval" / split / "ebtwm_inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    _write_csv(per_sample, out_dir / "per_sample_metrics.csv")
    _write_curve_csv(curve_rows, out_dir / "optimization_curves.csv")
    _write_csv(cross_rows, out_dir / "cross_operator_energy.csv")
    _plot_curves(curve_rows, out_dir / "optimization_curves.png")
    _plot_examples(examples, out_dir / "before_after_examples.png")
    (out_dir / "README.txt").write_text(
        "\n".join(
            [
                "EBTWM inference pilot settings",
                f"run_dir: {run_dir}",
                f"split: {split}",
                f"checkpoint: {checkpoint_path.name}",
                f"mode: {mode}",
                f"corruption_type: {corruption_type}",
                f"num_samples: {len(per_sample)}",
                f"num_steps: {num_steps}",
                f"step_size: {step_size}",
                f"go_no_go: {metrics['go_no_go_decision']['verdict']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return metrics


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


def compare_cross_critic(
    omc_run: Path,
    baseline_run: Path,
    out_dir: Path,
    split: str = "test_iid",
) -> dict[str, Any]:
    omc_csv = omc_run / "eval" / split / "ebtwm_inference" / "per_sample_metrics.csv"
    base_csv = baseline_run / "eval" / split / "ebtwm_inference" / "per_sample_metrics.csv"
    if not omc_csv.exists() or not base_csv.exists():
        raise FileNotFoundError("both runs must already contain EBTWM per_sample_metrics.csv")
    def read(path: Path) -> dict[int, dict[str, str]]:
        with path.open("r", newline="", encoding="utf-8") as f:
            return {int(row["sample_id"]): row for row in csv.DictReader(f)}
    omc = read(omc_csv)
    base = read(base_csv)
    rows = []
    deltas = []
    for sample_id in sorted(set(omc) & set(base)):
        omc_mse = float(omc[sample_id]["mse_to_gt_after"])
        base_mse = float(base[sample_id]["mse_to_gt_after"])
        delta = omc_mse - base_mse
        deltas.append(delta)
        rows.append({"sample_id": sample_id, "mse_after_omc": omc_mse, "mse_after_baseline": base_mse, "delta_mse": delta, "omc_beats_baseline": delta < 0.0})
    arr = np.asarray(deltas, dtype=np.float64)
    summary = {
        "num_pairs": int(len(rows)),
        "mean_delta_mse": float(arr.mean()) if arr.size else 0.0,
        "delta_mse_ci95_low": float(arr.mean() - 1.96 * arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0,
        "delta_mse_ci95_high": float(arr.mean() + 1.96 * arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0,
        "fraction_omc_beats_baseline": float(np.mean(arr < 0.0)) if arr.size else 0.0,
        "wilcoxon_signed_rank_p_value": _wilcoxon_like_p(arr) if arr.size else 1.0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, out_dir / "cross_critic_paired.csv")
    with (out_dir / "cross_critic_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--checkpoint", type=str, default="best_law_pair.pt")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--step-size", type=float, default=1e-2)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--corruption-type", type=str, default="gaussian", choices=["gaussian", "temporal_shuffle", "wrong_operator"])
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--mode", type=str, default="denoise", choices=["denoise", "counterfactual"])
    parser.add_argument("--eta-smooth", type=float, default=0.01)
    parser.add_argument("--eta-anchor", type=float, default=0.01)
    parser.add_argument("--eta-bounds", type=float, default=0.0)
    parser.add_argument("--langevin-noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--compare-runs", type=Path, nargs=2, default=None, metavar=("OMC_RUN", "BASELINE_RUN"))
    parser.add_argument("--compare-out", type=Path, default=None)
    args = parser.parse_args()
    if args.compare_runs:
        out = args.compare_out or (Path("runs") / "ebtwm_cross_critic")
        summary = compare_cross_critic(args.compare_runs[0], args.compare_runs[1], out, split=args.split)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.run is None:
        raise SystemExit("--run is required unless --compare-runs is used")
    metrics = evaluate_ebtwm_inference(
        args.run,
        split=args.split,
        checkpoint=args.checkpoint,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        step_size=args.step_size,
        noise_std=args.noise_std,
        corruption_type=args.corruption_type,
        horizon=args.horizon,
        device=args.device,
        mode=args.mode,
        eta_smooth=args.eta_smooth,
        eta_anchor=args.eta_anchor,
        eta_bounds=args.eta_bounds,
        langevin_noise=args.langevin_noise,
        seed=args.seed,
        skip_preflight=args.skip_preflight,
    )
    print(
        json.dumps(
            {
                "verdict": metrics["go_no_go_decision"]["verdict"],
                "fraction_energy_decreased": metrics["fraction_energy_decreased"],
                "fraction_mse_improved": metrics["fraction_mse_improved"],
                "own_vs_wrong_pair_acc_after": metrics["own_vs_wrong_pair_acc_after"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
