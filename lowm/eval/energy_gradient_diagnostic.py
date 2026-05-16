"""Local energy-gradient geometry diagnostic for trajectory EBMs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lowm.data.dataset import LOWMSynthRankingDataset, ranking_config_from_mapping
from lowm.eval.ebtwm_inference import _data_std, _masked_dynamic_mse, _model_lambda, _score_trajectory
from lowm.eval.evaluate_all import _ensure_split, _resolve_checkpoint_path, load_run_model


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0].keys()) if rows else ["sample_id"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_gradient_diagnostic(
    run_dir: Path,
    split: str = "test_iid",
    checkpoint: str = "best_law_pair.pt",
    num_samples: int = 100,
    noise_std: float = 0.05,
    seed: int = 0,
    device_name: str = "auto",
) -> dict[str, Any]:
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    checkpoint_path = _resolve_checkpoint_path(run_dir, checkpoint)
    model, config, detected = load_run_model(run_dir, checkpoint_name=checkpoint, device=device)
    ranking_cfg = ranking_config_from_mapping(config)
    split_path = _ensure_split(config, split)
    dataset = LOWMSynthRankingDataset(split_path, ranking_cfg, num_samples=None)
    data_std_per_dim, _ = _data_std(dataset)
    data_std = torch.from_numpy(data_std_per_dim).to(device=device, dtype=torch.float32).view(1, 1, 1, 4)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    if len(indices) > num_samples:
        indices = rng.choice(indices, size=num_samples, replace=False)

    rows: list[dict[str, Any]] = []
    for out_idx, idx in enumerate(indices):
        item = dataset[int(idx)]
        clean = item["pos_states"].to(device).unsqueeze(0)
        actions = item["pos_actions"].to(device).unsqueeze(0)
        mask = item["pos_mask"].to(device).unsqueeze(0)
        cs = item["context_states"].to(device).unsqueeze(0)
        ca = item["context_actions"].to(device).unsqueeze(0)
        cm = item["context_mask"].to(device).unsqueeze(0)
        noisy = clean.detach().clone()
        sigma = float(noise_std) * data_std
        noisy[:, 1:, :, 0:4] += torch.randn_like(noisy[:, 1:, :, 0:4]) * sigma
        noisy[:, 1:, :, 0:2].clamp_(0.0, 1.0)
        noisy.requires_grad_(True)

        model.eval()
        lambdas = _model_lambda(model, cs, ca, cm, use_mu=True)
        clean_energy = _score_trajectory(model, cs, ca, cm, clean, actions, mask, lambdas)
        noisy_energy = _score_trajectory(model, cs, ca, cm, noisy, actions, mask, lambdas)
        grad = torch.autograd.grad(noisy_energy.sum(), noisy, create_graph=False, retain_graph=False)[0]

        active = mask[:, 1:, :, None]
        neg_grad = -grad[:, 1:, :, 0:4] * active
        clean_dir = (clean[:, 1:, :, 0:4] - noisy[:, 1:, :, 0:4]) * active
        target = (noisy[:, 1:, :, 0:4] - clean[:, 1:, :, 0:4]) / sigma.square()
        grad_flat = neg_grad.reshape(1, -1)
        dir_flat = clean_dir.reshape(1, -1)
        dot = (grad_flat * dir_flat).sum(dim=1)
        grad_norm = torch.sqrt(grad_flat.square().sum(dim=1).clamp_min(1e-12))
        dir_norm = torch.sqrt(dir_flat.square().sum(dim=1).clamp_min(1e-12))
        target_norm = torch.sqrt(((target * active).reshape(1, -1).square()).sum(dim=1).clamp_min(1e-12))
        cosine = dot / (grad_norm * dir_norm).clamp_min(1e-12)
        projection = dot / dir_norm.clamp_min(1e-12)
        rows.append(
            {
                "sample_id": out_idx,
                "grad_norm": float(grad_norm.item()),
                "target_norm": float(target_norm.item()),
                "cosine_to_clean_direction": float(cosine.item()),
                "projection_to_clean_direction": float(projection.item()),
                "gradient_nan": float(not torch.isfinite(grad).all().item()),
                "gradient_exploding": float(grad_norm.item() > 1e6),
                "energy_clean": float(clean_energy.item()),
                "energy_noisy": float(noisy_energy.item()),
                "clean_vs_noisy_pair_acc": float(clean_energy.item() < noisy_energy.item()),
                "mse_noisy_to_clean": float(_masked_dynamic_mse(noisy.detach(), clean, mask).item()),
            }
        )

    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in rows])) if rows else 0.0

    metrics = {
        "model_type": detected,
        "split": split,
        "checkpoint_requested": checkpoint,
        "checkpoint_used": checkpoint_path.name,
        "num_samples": len(rows),
        "noise_std": float(noise_std),
        "grad_norm_mean": mean("grad_norm"),
        "grad_norm_std": float(np.std([row["grad_norm"] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
        "target_norm_mean": mean("target_norm"),
        "cosine_similarity_between_negative_grad_and_clean_direction": mean("cosine_to_clean_direction"),
        "fraction_cosine_positive": float(np.mean([row["cosine_to_clean_direction"] > 0 for row in rows])) if rows else 0.0,
        "mean_projection_to_clean_direction": mean("projection_to_clean_direction"),
        "gradient_nan_fraction": mean("gradient_nan"),
        "gradient_exploding_fraction": mean("gradient_exploding"),
        "energy_clean": mean("energy_clean"),
        "energy_noisy": mean("energy_noisy"),
        "clean_vs_noisy_pair_acc": mean("clean_vs_noisy_pair_acc"),
    }

    out_dir = run_dir / "eval" / split / "gradient_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    _write_csv(rows, out_dir / "per_sample.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    ax.hist([row["cosine_to_clean_direction"] for row in rows], bins=20, range=(-1, 1))
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("cos(-grad, clean - noisy)")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_dir / "cosine_histogram.png", dpi=160)
    plt.close(fig)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test_iid")
    parser.add_argument("--checkpoint", type=str, default="best_law_pair.pt")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    metrics = evaluate_gradient_diagnostic(args.run, args.split, args.checkpoint, args.num_samples, args.noise_std, args.seed, args.device)
    print(
        json.dumps(
            {
                "cosine": metrics["cosine_similarity_between_negative_grad_and_clean_direction"],
                "fraction_cosine_positive": metrics["fraction_cosine_positive"],
                "clean_vs_noisy_pair_acc": metrics["clean_vs_noisy_pair_acc"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
