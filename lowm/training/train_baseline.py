"""Train Milestone 3 energy baselines on LOWM-Synth ranking data."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from lowm.data.dataset import LOWMSynthRankingDataset, make_ranking_dataloader, ranking_config_from_mapping
from lowm.data.generate_dataset import generate_dataset
from lowm.eval.metrics import RankingMetricAccumulator, format_metrics
from lowm.models.baselines import baseline_config_from_mapping, build_baseline
from lowm.training.losses import nce_ranking_loss


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
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
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
        missing = [str(p) for p in [train_path, val_path] if not p.exists()]
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


def evaluate(model: torch.nn.Module, loader, device: torch.device) -> dict[str, Any]:
    model.eval()
    accumulator = RankingMetricAccumulator()
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            energies = model(batch)
            loss = nce_ranking_loss(energies, batch["labels"])
            accumulator.update(energies, batch["labels"], batch["negative_types"], float(loss.item()))
    return accumulator.compute()


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_steps: int | None = None,
) -> dict[str, Any]:
    model.train()
    accumulator = RankingMetricAccumulator()
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        batch = _move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        energies = model(batch)
        loss = nce_ranking_loss(energies, batch["labels"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        accumulator.update(energies.detach(), batch["labels"], batch["negative_types"], float(loss.item()))
    return accumulator.compute()


def train_baseline(config_path: Path, baseline: str | None = None) -> dict[str, Any]:
    config = _load_yaml(config_path)
    train_cfg = dict(config.get("training", {}))
    baseline_overridden = baseline is not None
    baseline_name = baseline or str(train_cfg.get("baseline", "fixed_energy"))
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
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    model = build_baseline(baseline_name, baseline_config_from_mapping(config)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    output_root = Path(train_cfg.get("output_dir", "runs/lowm_synth_v0"))
    run_name = f"{baseline_name}_seed{seed}" if baseline_overridden else str(train_cfg.get("run_name", f"{baseline_name}_seed{seed}"))
    run_dir = output_root / run_name
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, run_dir / "config.yaml")

    metadata = {
        "baseline": baseline_name,
        "seed": seed,
        "device": str(device),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "git_commit": _git_commit(),
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    history: list[dict[str, Any]] = []
    best_val = -1.0
    epochs = int(train_cfg.get("epochs", 10))
    max_steps = train_cfg.get("max_train_steps_per_epoch")
    max_steps = int(max_steps) if max_steps is not None else None

    print(f"training {baseline_name} on {device} -> {run_dir}")
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, max_steps=max_steps)
        val_metrics = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(f"epoch {epoch:03d} {format_metrics(train_metrics, 'train_')} | {format_metrics(val_metrics, 'val_')}")

        if float(val_metrics["top1_acc"]) >= best_val:
            best_val = float(val_metrics["top1_acc"])
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "baseline": baseline_name,
                    "config": config,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                checkpoints / "best.pt",
            )
        torch.save(
            {
                "model_state": model.state_dict(),
                "baseline": baseline_name,
                "config": config,
                "epoch": epoch,
                "val_metrics": val_metrics,
            },
            checkpoints / "last.pt",
        )

    metrics = {"history": history, "best_val_top1_acc": best_val, "final_val": history[-1]["val"] if history else {}}
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline", type=str, default=None, choices=["fixed_energy", "direct_context_energy"])
    args = parser.parse_args()
    train_baseline(args.config, args.baseline)


if __name__ == "__main__":
    main()
