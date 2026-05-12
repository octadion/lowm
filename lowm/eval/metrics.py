"""Ranking metrics for energy-based LOWM experiments."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping

import torch

NEGATIVE_TYPE_ORDER = (
    "state_corrupted",
    "temporal_shuffled",
    "law_mismatch",
    "random_impossible",
)
METRIC_VERSION = "ranking_v2_pairwise_gap"


def ranking_metrics_from_energies(energies: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    if energies.ndim != 2:
        raise ValueError("energies must be [B,M]")
    preds = torch.argmin(energies, dim=1)
    top1 = (preds == labels).float().mean().item()
    label_energy = energies.gather(1, labels[:, None])
    ranks = (energies < label_energy).sum(dim=1) + 1
    return {
        "top1_acc": top1,
        "mean_rank": ranks.float().mean().item(),
        "mrr": (1.0 / ranks.float()).mean().item(),
    }


class RankingMetricAccumulator:
    def __init__(self) -> None:
        self.num_samples = 0
        self.num_correct = 0
        self.rank_sum = 0.0
        self.reciprocal_rank_sum = 0.0
        self.loss_sum = 0.0
        self.loss_count = 0
        self.type_wins: Counter[str] = Counter()
        self.type_counts: Counter[str] = Counter()
        self.type_energy_gap_sum: defaultdict[str, float] = defaultdict(float)
        self.top1_subset_correct: Counter[str] = Counter()
        self.top1_subset_counts: Counter[str] = Counter()

    def update(
        self,
        energies: torch.Tensor,
        labels: torch.Tensor,
        negative_types: list[list[str]],
        loss: float | None = None,
    ) -> None:
        energies_cpu = energies.detach().cpu()
        labels_cpu = labels.detach().cpu()
        preds = torch.argmin(energies_cpu, dim=1)
        label_energy = energies_cpu.gather(1, labels_cpu[:, None]).squeeze(1)
        ranks = (energies_cpu < label_energy[:, None]).sum(dim=1) + 1

        batch_size = energies_cpu.shape[0]
        self.num_samples += batch_size
        self.num_correct += int((preds == labels_cpu).sum().item())
        self.rank_sum += float(ranks.float().sum().item())
        self.reciprocal_rank_sum += float((1.0 / ranks.float()).sum().item())
        if loss is not None:
            self.loss_sum += float(loss) * batch_size
            self.loss_count += batch_size

        for b, types in enumerate(negative_types):
            pos_energy = float(label_energy[b].item())
            sample_types = set(t for t in types if t != "positive")
            for type_name in sample_types:
                self.top1_subset_counts[type_name] += 1
                if int(preds[b].item()) == int(labels_cpu[b].item()):
                    self.top1_subset_correct[type_name] += 1
            for m, type_name in enumerate(types):
                if type_name == "positive":
                    continue
                gap = float(energies_cpu[b, m].item()) - pos_energy
                self.type_counts[type_name] += 1
                self.type_energy_gap_sum[type_name] += gap
                if gap > 0:
                    self.type_wins[type_name] += 1

    def compute(self) -> dict[str, Any]:
        denom = max(1, self.num_samples)
        metrics: dict[str, Any] = {
            "num_samples": self.num_samples,
            "top1_acc": self.num_correct / denom,
            "mean_rank": self.rank_sum / denom,
            "mrr": self.reciprocal_rank_sum / denom,
        }
        if self.loss_count:
            metrics["loss"] = self.loss_sum / self.loss_count

        by_type: dict[str, dict[str, float | int]] = {}
        for type_name in sorted(self.type_counts):
            count = self.type_counts[type_name]
            by_type[type_name] = {
                "pairwise_acc": self.type_wins[type_name] / max(1, count),
                "mean_energy_gap": self.type_energy_gap_sum[type_name] / max(1, count),
                "count": count,
                "top1_acc_on_samples_with_type": self.top1_subset_correct[type_name]
                / max(1, self.top1_subset_counts[type_name]),
            }
        metrics["by_negative_type"] = by_type
        metrics["law_mismatch"] = by_type.get(
            "law_mismatch",
            {"pairwise_acc": 0.0, "mean_energy_gap": 0.0, "count": 0, "top1_acc_on_samples_with_type": 0.0},
        )
        metrics["law_pair"] = metrics["law_mismatch"]["pairwise_acc"]
        metrics["law_gap"] = metrics["law_mismatch"]["mean_energy_gap"]
        metrics["metric_version"] = METRIC_VERSION
        for type_name in NEGATIVE_TYPE_ORDER:
            values = by_type.get(
                type_name,
                {"pairwise_acc": 0.0, "mean_energy_gap": 0.0, "count": 0, "top1_acc_on_samples_with_type": 0.0},
            )
            metrics[f"{type_name}_pair_acc"] = values["pairwise_acc"]
            metrics[f"{type_name}_gap"] = values["mean_energy_gap"]
            metrics[f"{type_name}_count"] = values["count"]
        return metrics


def format_metrics(metrics: Mapping[str, Any], prefix: str = "") -> str:
    base = (
        f"{prefix}loss={metrics.get('loss', float('nan')):.4f} "
        f"top1={metrics['top1_acc']:.3f} "
        f"rank={metrics['mean_rank']:.2f} "
        f"mrr={metrics['mrr']:.3f}"
    )
    base += (
        f" law_pair={metrics.get('law_pair', metrics.get('law_mismatch', {}).get('pairwise_acc', 0.0)):.3f}"
        f" law_gap={metrics.get('law_gap', metrics.get('law_mismatch', {}).get('mean_energy_gap', 0.0)):.3f}"
    )
    for type_name in NEGATIVE_TYPE_ORDER:
        base += f" {type_name}_pair_acc={metrics.get(f'{type_name}_pair_acc', 0.0):.3f}"
    return base
