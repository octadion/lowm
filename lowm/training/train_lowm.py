"""Train LOWM v0 on LOWM-Synth ranking data."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from lowm.data.dataset import LOWMSynthRankingDataset, make_ranking_dataloader, ranking_config_from_mapping
from lowm.data.generate_dataset import generate_dataset
from lowm.eval.metrics import RankingMetricAccumulator, format_metrics
from lowm.models.lowm import LOWM, lowm_config_from_mapping
from lowm.training.checkpointing import MultiMetricCheckpointer, checkpoint_payload
from lowm.training.losses import law_stability_loss, lowm_total_loss, operator_coherence_contrastive_loss


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    return result.stdout.strip()


def _ensure_data(config: Mapping[str, Any]) -> tuple[Path, Path]:
    data_cfg = dict(config.get("data", {}))
    root = Path(data_cfg.get("root", "data/lowm_synth_v0"))
    train_path = root / str(data_cfg.get("train_split", "train.npz"))
    val_path = root / str(data_cfg.get("val_split", "val.npz"))
    if train_path.exists() and val_path.exists():
        return train_path, val_path
    if not bool(data_cfg.get("generate_if_missing", False)):
        missing = [str(path) for path in [train_path, val_path] if not path.exists()]
        raise FileNotFoundError(f"missing dataset files: {missing}")
    dataset_config = Path(data_cfg.get("dataset_config", "configs/lowm_synth_v0.yaml"))
    splits = [data_cfg.get("train_split", "train.npz").replace(".npz", ""), data_cfg.get("val_split", "val.npz").replace(".npz", "")]
    generate_dataset(dataset_config, root, splits)
    return train_path, val_path


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


class LossAverages:
    def __init__(self) -> None:
        self.sums: defaultdict[str, float] = defaultdict(float)
        self.count = 0

    def update(self, losses: Mapping[str, torch.Tensor], batch_size: int) -> None:
        self.count += batch_size
        for key, value in losses.items():
            self.sums[key] += float(value.detach().cpu().item()) * batch_size

    def compute(self) -> dict[str, float]:
        denom = max(1, self.count)
        return {key: value / denom for key, value in self.sums.items()}


def _stability_from_batch(model: LOWM, batch: Mapping[str, torch.Tensor], enabled: bool) -> torch.Tensor | None:
    if not enabled:
        return None
    k = batch["context_states"].shape[1]
    if k < 2:
        return None
    mid = k // 2
    if mid == 0 or mid == k:
        return None
    mu_a, _ = model.context_encoder(
        batch["context_states"][:, :mid],
        batch["context_actions"][:, :mid],
        batch["context_mask"][:, :mid],
    )
    mu_b, _ = model.context_encoder(
        batch["context_states"][:, mid:],
        batch["context_actions"][:, mid:],
        batch["context_mask"][:, mid:],
    )
    return law_stability_loss(mu_a, mu_b)


def _occl_from_batch(
    model: LOWM,
    batch: Mapping[str, torch.Tensor],
    lambdas: torch.Tensor,
    enabled: bool,
    temperature: float,
) -> dict[str, torch.Tensor] | None:
    if not enabled or batch["pos_states"].shape[0] < 2:
        return None
    e_matrix = model.energy_matrix(batch["pos_states"], batch["pos_actions"], batch["pos_mask"], lambdas)
    return operator_coherence_contrastive_loss(e_matrix, temperature=temperature)


def _operator_batch_stats(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    op_id = batch["query_op_id"].detach()
    params = batch["query_op_params"].detach()
    batch_size = int(op_id.shape[0])
    zero = torch.zeros((), device=op_id.device, dtype=params.dtype)
    if batch_size < 2:
        return {
            "occl_batch_mean_pairwise_op_param_distance": zero,
            "occl_batch_fraction_same_op_id": zero,
            "occl_batch_effective_unique_op_id": torch.ones((), device=op_id.device, dtype=params.dtype),
        }
    idx_i, idx_j = torch.triu_indices(batch_size, batch_size, offset=1, device=op_id.device)
    distances = torch.linalg.norm(params[idx_i] - params[idx_j], dim=-1)
    same = (op_id[idx_i] == op_id[idx_j]).to(params.dtype)
    unique_ids, counts = torch.unique(op_id, return_counts=True)
    probs = counts.to(params.dtype) / counts.sum().to(params.dtype)
    entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum()
    return {
        "occl_batch_mean_pairwise_op_param_distance": distances.mean(),
        "occl_batch_fraction_same_op_id": same.mean(),
        "occl_batch_effective_unique_op_id": torch.exp(entropy),
    }


def _add_occl_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    loss_terms = metrics.get("loss_terms", {})
    for key in [
        "occl_loss",
        "tau_to_lambda_loss",
        "lambda_to_tau_loss",
        "occl_acc_tau_to_lambda",
        "occl_acc_lambda_to_tau",
        "occl_batch_mean_pairwise_op_param_distance",
        "occl_batch_fraction_same_op_id",
        "occl_batch_effective_unique_op_id",
    ]:
        if key in loss_terms:
            metrics[key] = loss_terms[key]
    if "occl_acc_tau_to_lambda" in metrics and "occl_acc_lambda_to_tau" in metrics:
        metrics["occl_acc"] = 0.5 * (metrics["occl_acc_tau_to_lambda"] + metrics["occl_acc_lambda_to_tau"])
    return metrics


def evaluate_lowm(
    model: LOWM,
    loader,
    device: torch.device,
    beta_kl: float,
    alpha_stable: float,
    use_stability: bool,
    alpha_occl: float = 0.0,
    use_occl: bool = False,
    occl_temperature: float = 1.0,
) -> dict[str, Any]:
    model.eval()
    metrics_acc = RankingMetricAccumulator()
    loss_acc = LossAverages()
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            output = model(batch)
            stability = _stability_from_batch(model, batch, use_stability)
            occl = _occl_from_batch(model, batch, output["lambda"], use_occl, occl_temperature)
            losses = lowm_total_loss(
                output["energies"],
                batch["labels"],
                output["mu"],
                output["logvar"],
                beta_kl=beta_kl,
                stability=stability,
                alpha_stable=alpha_stable,
                occl_loss=occl["occl_loss"] if occl else None,
                alpha_occl=alpha_occl,
            )
            if occl:
                losses.update(occl)
                losses.update(_operator_batch_stats(batch))
            batch_size = int(batch["labels"].shape[0])
            loss_acc.update(losses, batch_size)
            metrics_acc.update(output["energies"], batch["labels"], batch["negative_types"], float(losses["total"].item()))
    metrics = metrics_acc.compute()
    metrics["loss_terms"] = loss_acc.compute()
    return _add_occl_metrics(metrics)


def train_one_epoch(
    model: LOWM,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta_kl: float,
    alpha_stable: float,
    use_stability: bool,
    alpha_occl: float = 0.0,
    use_occl: bool = False,
    occl_temperature: float = 1.0,
    max_steps: int | None = None,
) -> dict[str, Any]:
    model.train()
    metrics_acc = RankingMetricAccumulator()
    loss_acc = LossAverages()
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        batch = _move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        stability = _stability_from_batch(model, batch, use_stability)
        occl = _occl_from_batch(model, batch, output["lambda"], use_occl, occl_temperature)
        losses = lowm_total_loss(
            output["energies"],
            batch["labels"],
            output["mu"],
            output["logvar"],
            beta_kl=beta_kl,
            stability=stability,
            alpha_stable=alpha_stable,
            occl_loss=occl["occl_loss"] if occl else None,
            alpha_occl=alpha_occl,
        )
        if occl:
            losses.update(occl)
            losses.update(_operator_batch_stats(batch))
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        batch_size = int(batch["labels"].shape[0])
        loss_acc.update(losses, batch_size)
        metrics_acc.update(output["energies"].detach(), batch["labels"], batch["negative_types"], float(losses["total"].item()))
    metrics = metrics_acc.compute()
    metrics["loss_terms"] = loss_acc.compute()
    return _add_occl_metrics(metrics)


def _format_lowm_metrics(metrics: Mapping[str, Any], prefix: str) -> str:
    loss_terms = metrics.get("loss_terms", {})
    return (
        f"{format_metrics(metrics, prefix)} "
        f"{prefix}total={loss_terms.get('total', float('nan')):.4f} "
        f"{prefix}nce={loss_terms.get('nce', float('nan')):.4f} "
        f"{prefix}kl={loss_terms.get('kl', float('nan')):.4f} "
        f"{prefix}stable={loss_terms.get('stability', float('nan')):.4f} "
        f"{prefix}occl_loss={loss_terms.get('occl_loss', float('nan')):.4f} "
        f"{prefix}tau_to_lambda_loss={loss_terms.get('tau_to_lambda_loss', float('nan')):.4f} "
        f"{prefix}lambda_to_tau_loss={loss_terms.get('lambda_to_tau_loss', float('nan')):.4f} "
        f"{prefix}occl_acc_tau_to_lambda={loss_terms.get('occl_acc_tau_to_lambda', float('nan')):.4f} "
        f"{prefix}occl_acc_lambda_to_tau={loss_terms.get('occl_acc_lambda_to_tau', float('nan')):.4f} "
        f"{prefix}occl_op_param_dist={loss_terms.get('occl_batch_mean_pairwise_op_param_distance', float('nan')):.4f} "
        f"{prefix}occl_same_op_frac={loss_terms.get('occl_batch_fraction_same_op_id', float('nan')):.4f} "
        f"{prefix}occl_effective_unique_op={loss_terms.get('occl_batch_effective_unique_op_id', float('nan')):.4f}"
    )


def _print_validation_metrics(metrics: Mapping[str, Any]) -> str:
    return (
        f"val_top1={metrics.get('top1_acc', 0.0):.4f} "
        f"val_loss={metrics.get('loss', float('nan')):.4f} "
        f"val_law_pair={metrics.get('law_pair', 0.0):.4f} "
        f"val_law_gap={metrics.get('law_gap', 0.0):.4f} "
        f"val_state_corrupted_pair={metrics.get('state_corrupted_pair_acc', 0.0):.4f} "
        f"val_temporal_shuffled_pair={metrics.get('temporal_shuffled_pair_acc', 0.0):.4f} "
        f"val_law_mismatch_pair={metrics.get('law_mismatch_pair_acc', 0.0):.4f} "
        f"val_random_impossible_pair={metrics.get('random_impossible_pair_acc', 0.0):.4f}"
    )


def train_lowm(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    train_cfg = dict(config.get("training", {}))
    seed = int(train_cfg.get("seed", config.get("seed", 0)))
    _set_seed(seed)

    train_path, val_path = _ensure_data(config)
    ranking_cfg = ranking_config_from_mapping(config)
    train_samples = train_cfg.get("train_samples_per_epoch")
    val_samples = train_cfg.get("val_samples")
    train_dataset = LOWMSynthRankingDataset(train_path, ranking_cfg, num_samples=int(train_samples) if train_samples else None)
    val_dataset = LOWMSynthRankingDataset(val_path, ranking_cfg, num_samples=int(val_samples) if val_samples else None)

    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 0))
    ensure_distinct = bool(train_cfg.get("ensure_distinct_operators_in_batch", ranking_cfg.ensure_distinct_operators_in_batch))
    train_loader = make_ranking_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        ensure_distinct_operators_in_batch=ensure_distinct,
    )
    val_loader = make_ranking_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        ensure_distinct_operators_in_batch=ensure_distinct,
    )

    device_name = str(train_cfg.get("device", "auto"))
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    model = LOWM(lowm_config_from_mapping(config)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    beta_kl = float(train_cfg.get("beta_kl", 1e-4))
    alpha_stable = float(train_cfg.get("alpha_stable", 0.1))
    use_stability = bool(train_cfg.get("use_stability", True))
    use_occl = bool(train_cfg.get("use_occl", False))
    alpha_occl = float(train_cfg.get("alpha_occl", 1.0))
    occl_temperature = float(train_cfg.get("occl_temperature", 1.0))
    selection_metric = str(train_cfg.get("selection_metric", "law_pair"))

    output_root = Path(train_cfg.get("output_dir", "runs/lowm_synth_v0"))
    run_name = str(train_cfg.get("run_name", f"lowm_seed{seed}"))
    run_dir = output_root / run_name
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, run_dir / "config.yaml")
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "LOWM",
                "seed": seed,
                "device": str(device),
                "train_path": str(train_path),
                "val_path": str(val_path),
                "git_commit": _git_commit(),
            },
            f,
            indent=2,
            sort_keys=True,
        )

    history: list[dict[str, Any]] = []
    checkpointer = MultiMetricCheckpointer(checkpoints, selection_metric=selection_metric)
    epochs = int(train_cfg.get("epochs", 10))
    max_steps = train_cfg.get("max_train_steps_per_epoch")
    max_steps = int(max_steps) if max_steps is not None else None
    print(f"training LOWM on {device} -> {run_dir}")
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            beta_kl,
            alpha_stable,
            use_stability,
            alpha_occl,
            use_occl,
            occl_temperature,
            max_steps,
        )
        val_metrics = evaluate_lowm(
            model,
            val_loader,
            device,
            beta_kl,
            alpha_stable,
            use_stability,
            alpha_occl,
            use_occl,
            occl_temperature,
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch {epoch:03d} {_format_lowm_metrics(train_metrics, 'train_')} | "
            f"{_format_lowm_metrics(val_metrics, 'val_')} | {_print_validation_metrics(val_metrics)}"
        )
        payload = checkpoint_payload(model, config, epoch, val_metrics, "lowm")
        checkpointer.update(payload, val_metrics)

    metrics = {
        "history": history,
        "best_scores": checkpointer.best_scores,
        "selection_metric": selection_metric,
        "final_val": history[-1]["val"] if history else {},
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    train_lowm(args.config)


if __name__ == "__main__":
    main()
