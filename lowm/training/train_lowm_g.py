"""Train LOWM-G supervised operator-conditioned proposal dynamics."""

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
from lowm.models.lowm_g import (
    OperatorConditionedProposalModel,
    lowm_g_config_from_mapping,
    masked_delta_mse,
    masked_rollout_mse,
    rollout_mse_by_horizon,
)


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
    root = Path(data_cfg.get("root", "data/lowm_synth_ood_param"))
    train_path = root / str(data_cfg.get("train_split", "train.npz"))
    val_path = root / str(data_cfg.get("val_split", "val.npz"))
    if train_path.exists() and val_path.exists():
        return train_path, val_path
    if not bool(data_cfg.get("generate_if_missing", False)):
        raise FileNotFoundError(f"missing dataset files: {[str(p) for p in [train_path, val_path] if not p.exists()]}")
    dataset_config = Path(data_cfg.get("dataset_config", "configs/lowm_synth_ood_param.yaml"))
    splits = [data_cfg.get("train_split", "train.npz").replace(".npz", ""), data_cfg.get("val_split", "val.npz").replace(".npz", "")]
    generate_dataset(dataset_config, root, splits)
    return train_path, val_path


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


class Averages:
    def __init__(self) -> None:
        self.sums: defaultdict[str, float] = defaultdict(float)
        self.count = 0

    def update(self, values: Mapping[str, float], n: int) -> None:
        self.count += n
        for key, value in values.items():
            self.sums[key] += float(value) * n

    def compute(self) -> dict[str, float]:
        denom = max(1, self.count)
        return {key: value / denom for key, value in self.sums.items()}


def _run_model(model: OperatorConditionedProposalModel, batch: Mapping[str, torch.Tensor], noise_scale: float) -> dict[str, torch.Tensor]:
    return model(
        batch["context_states"],
        batch["context_actions"],
        batch["context_mask"],
        batch["pos_states"][:, 0],
        batch["pos_actions"],
        batch["pos_mask"],
        noise_scale=noise_scale,
    )


def _metrics_from_output(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    alpha_delta: float,
    alpha_smooth: float,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    pred = output["pred_states"]
    target = batch["pos_states"]
    mask = batch["pos_mask"]
    pred_mse = masked_rollout_mse(pred, target, mask)
    delta = masked_delta_mse(pred, target, mask)
    dyn_delta = pred[:, 1:, :, 0:4] - pred[:, :-1, :, 0:4]
    active = (mask[:, 1:] * mask[:, :-1])[:, :, :, None].to(pred.dtype)
    smooth = (dyn_delta.square() * active).sum() / (active.sum().clamp_min(1.0) * 4.0)
    loss = pred_mse + float(alpha_delta) * delta + float(alpha_smooth) * smooth
    by_h = rollout_mse_by_horizon(pred, target, mask).detach()
    values = {
        "loss": float(loss.detach().cpu().item()),
        "pred_mse": float(pred_mse.detach().cpu().item()),
        "delta_mse": float(delta.detach().cpu().item()),
        "smoothness": float(smooth.detach().cpu().item()),
        "rollout_mse_final_step": float(by_h[-1].detach().cpu().item()) if by_h.numel() else 0.0,
        "lambda_norm": float(output["lambda"].detach().norm(dim=-1).mean().cpu().item()),
    }
    for idx, value in enumerate(by_h.detach().cpu().tolist(), start=1):
        values[f"rollout_mse_t{idx}"] = float(value)
    return loss, values, by_h


def train_one_epoch(
    model: OperatorConditionedProposalModel,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha_delta: float,
    alpha_smooth: float,
    noise_scale: float,
    max_steps: int | None = None,
) -> dict[str, float]:
    model.train()
    avg = Averages()
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = _run_model(model, batch, noise_scale)
        loss, values, _ = _metrics_from_output(output, batch, alpha_delta, alpha_smooth)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        avg.update(values, int(batch["pos_states"].shape[0]))
    return avg.compute()


def evaluate(
    model: OperatorConditionedProposalModel,
    loader,
    device: torch.device,
    alpha_delta: float,
    alpha_smooth: float,
    noise_scale: float = 0.0,
) -> dict[str, float]:
    model.eval()
    avg = Averages()
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            output = _run_model(model, batch, noise_scale)
            _, values, _ = _metrics_from_output(output, batch, alpha_delta, alpha_smooth)
            avg.update(values, int(batch["pos_states"].shape[0]))
    return avg.compute()


def train_lowm_g(config_path: Path) -> dict[str, Any]:
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
    train_loader = make_ranking_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = make_ranking_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device_name = str(train_cfg.get("device", "auto"))
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    model = OperatorConditionedProposalModel(lowm_g_config_from_mapping(config)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 1e-3)), weight_decay=float(train_cfg.get("weight_decay", 1e-4)))
    alpha_delta = float(train_cfg.get("alpha_delta", 0.0))
    alpha_smooth = float(train_cfg.get("alpha_smooth", 0.0))
    noise_scale = float(train_cfg.get("candidate_noise_scale_train", train_cfg.get("noise_scale_train", 1.0)))

    output_root = Path(train_cfg.get("output_dir", "runs/lowm_synth_ood_param/lowm_g"))
    run_name = str(train_cfg.get("run_name", f"lowm_g_seed{seed}"))
    run_dir = output_root / run_name
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, run_dir / "config.yaml")
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump({"model": "LOWM-G", "seed": seed, "device": str(device), "git_commit": _git_commit()}, f, indent=2, sort_keys=True)

    best_pred = float("inf")
    history: list[dict[str, Any]] = []
    epochs = int(train_cfg.get("epochs", 30))
    max_steps = train_cfg.get("max_train_steps_per_epoch")
    max_steps = int(max_steps) if max_steps is not None else None
    print(f"training LOWM-G on {device} -> {run_dir}")
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, alpha_delta, alpha_smooth, noise_scale, max_steps=max_steps)
        val_metrics = evaluate(model, val_loader, device, alpha_delta, alpha_smooth, noise_scale=0.0)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch {epoch:03d} train_pred_mse={train_metrics.get('pred_mse', 0.0):.6f} "
            f"val_pred_mse={val_metrics.get('pred_mse', 0.0):.6f} "
            f"train_delta_mse={train_metrics.get('delta_mse', 0.0):.6f} "
            f"val_delta_mse={val_metrics.get('delta_mse', 0.0):.6f}"
        )
        payload = {"model_state": model.state_dict(), "config": dict(config), "epoch": epoch, "val_metrics": val_metrics, "model_type": "lowm_g"}
        torch.save(payload, checkpoints / "last.pt")
        if val_metrics.get("pred_mse", float("inf")) <= best_pred:
            best_pred = float(val_metrics.get("pred_mse", float("inf")))
            torch.save(payload, checkpoints / "best_pred.pt")

    metrics = {"history": history, "best_pred_mse": best_pred, "final_val": history[-1]["val"] if history else {}}
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    train_lowm_g(args.config)


if __name__ == "__main__":
    main()
