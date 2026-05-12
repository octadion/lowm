"""Checkpoint selection helpers shared by training scripts."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping

import torch


SELECTION_METRICS = ("top1", "loss", "law_pair", "law_gap", "occl_acc", "composite")


def metric_score(metrics: Mapping[str, Any], metric: str) -> float:
    if metric == "top1":
        return float(metrics.get("top1_acc", 0.0))
    if metric == "loss":
        return -float(metrics.get("loss", float("inf")))
    if metric == "law_pair":
        return float(metrics.get("law_pair", metrics.get("law_mismatch", {}).get("pairwise_acc", 0.0)))
    if metric == "law_gap":
        return float(metrics.get("law_gap", metrics.get("law_mismatch", {}).get("mean_energy_gap", 0.0)))
    if metric == "occl_acc":
        return float(
            metrics.get(
                "occl_acc",
                0.5
                * (
                    float(metrics.get("occl_acc_tau_to_lambda", 0.0))
                    + float(metrics.get("occl_acc_lambda_to_tau", 0.0))
                ),
            )
        )
    if metric == "composite":
        return metric_score(metrics, "law_pair") + 0.1 * metric_score(metrics, "top1") + 0.1 * metric_score(metrics, "occl_acc")
    raise ValueError(f"unknown selection metric '{metric}'")


def checkpoint_payload(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    epoch: int,
    val_metrics: Mapping[str, Any],
    model_type: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "config": dict(config),
        "epoch": epoch,
        "val_metrics": dict(val_metrics),
        "model_type": model_type,
    }
    if extra:
        payload.update(dict(extra))
    return payload


class MultiMetricCheckpointer:
    def __init__(self, checkpoints_dir: Path, selection_metric: str = "top1") -> None:
        if selection_metric not in SELECTION_METRICS:
            raise ValueError(f"selection_metric must be one of {SELECTION_METRICS}")
        self.checkpoints_dir = checkpoints_dir
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.selection_metric = selection_metric
        self.best_scores = {
            "top1": float("-inf"),
            "loss": float("-inf"),
            "law_pair": float("-inf"),
            "law_gap": float("-inf"),
            "occl_acc": float("-inf"),
            "composite": float("-inf"),
        }

    def update(self, payload: Mapping[str, Any], val_metrics: Mapping[str, Any]) -> dict[str, float]:
        torch.save(dict(payload), self.checkpoints_dir / "last.pt")
        saved: dict[str, float] = {}
        for metric in SELECTION_METRICS:
            score = metric_score(val_metrics, metric)
            if score >= self.best_scores[metric]:
                self.best_scores[metric] = score
                filename = f"best_{metric}.pt"
                torch.save(dict(payload), self.checkpoints_dir / filename)
                saved[metric] = score
        top1_path = self.checkpoints_dir / "best_top1.pt"
        if top1_path.exists():
            shutil.copyfile(top1_path, self.checkpoints_dir / "best.pt")
        return saved
