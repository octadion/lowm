"""Evaluate OCCL trajectory-operator alignment for LOWM runs."""

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

from lowm.data.dataset import LOWMSynthRankingDataset, make_ranking_dataloader, ranking_config_from_mapping
from lowm.eval.evaluate_all import (
    _ensure_split,
    _move_batch_to_device,
    _resolve_checkpoint_path,
    evaluate_occl_alignment_summary,
    load_run_model,
)
from lowm.eval.metrics import METRIC_VERSION


def _write_matrix_csv(matrix: torch.Tensor, path: Path) -> None:
    matrix_np = matrix.detach().cpu().numpy()
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trajectory_index", *[f"lambda_{j}" for j in range(matrix_np.shape[1])]])
        for i, row in enumerate(matrix_np):
            writer.writerow([i, *[float(x) for x in row]])


def _write_heatmap(matrix: torch.Tensor, path: Path) -> None:
    values = matrix.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    im = ax.imshow(values, cmap="viridis", aspect="auto")
    ax.set_xlabel("lambda index")
    ax.set_ylabel("trajectory index")
    ax.set_title("OCCL energy matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate_occl_alignment(
    run_dir: Path,
    split: str = "val",
    checkpoint_name: str = "best_occl_acc.pt",
    model_type: str | None = None,
    batch_size: int | None = None,
    num_samples: int | None = None,
    seed: int | None = None,
    device_name: str = "auto",
) -> dict[str, Any]:
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    checkpoint_path = _resolve_checkpoint_path(run_dir, checkpoint_name)
    model, config, detected = load_run_model(run_dir, model_type=model_type, checkpoint_name=checkpoint_name, device=device)
    if not hasattr(model, "energy_matrix"):
        raise ValueError("OCCL alignment evaluation requires LOWM.energy_matrix")
    split_path = _ensure_split(config, split)
    ranking_cfg = ranking_config_from_mapping(config)
    if seed is not None:
        ranking_cfg = replace(ranking_cfg, seed=int(seed))
    eval_cfg = dict(config.get("evaluation", {}))
    sample_count = num_samples if num_samples is not None else eval_cfg.get("num_samples", config.get("training", {}).get("val_samples"))
    dataset = LOWMSynthRankingDataset(split_path, ranking_cfg, num_samples=int(sample_count) if sample_count else None)
    bs = int(batch_size or config.get("training", {}).get("batch_size", 64))

    metrics = evaluate_occl_alignment_summary(
        model,
        dataset,
        bs,
        device,
        same_operator_threshold=ranking_cfg.min_law_param_distance,
    )
    metrics.update(
        {
            "model_type": detected,
            "split": split,
            "checkpoint_requested": checkpoint_name,
            "checkpoint_used": checkpoint_path.name,
            "checkpoint_stem": checkpoint_path.stem,
            "ranking_seed": ranking_cfg.seed,
            "num_samples": len(dataset),
            "metric_version": METRIC_VERSION,
        }
    )

    out_dir = run_dir / "eval" / split / checkpoint_path.stem
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = make_ranking_dataloader(dataset, batch_size=min(bs, len(dataset)), shuffle=False)
    first_batch = next(iter(loader))
    with torch.no_grad():
        batch = _move_batch_to_device(first_batch, device)
        output = model(batch)
        e_matrix = model.energy_matrix(batch["pos_states"], batch["pos_actions"], batch["pos_mask"], output["lambda"])
    _write_matrix_csv(e_matrix, out_dir / "occl_energy_matrix.csv")
    _write_heatmap(e_matrix, plots_dir / "occl_energy_matrix_heatmap.png")
    with (out_dir / "occl_alignment_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    legacy = run_dir / "eval" / split
    legacy.mkdir(parents=True, exist_ok=True)
    for filename in ["occl_alignment_metrics.json", "occl_energy_matrix.csv"]:
        (legacy / filename).write_bytes((out_dir / filename).read_bytes())
    legacy_plots = legacy / "plots"
    legacy_plots.mkdir(parents=True, exist_ok=True)
    (legacy_plots / "occl_energy_matrix_heatmap.png").write_bytes((plots_dir / "occl_energy_matrix_heatmap.png").read_bytes())
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--checkpoint", type=str, default="best_occl_acc.pt")
    parser.add_argument("--model_type", type=str, default=None, choices=["lowm"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    metrics = evaluate_occl_alignment(
        args.run,
        split=args.split,
        checkpoint_name=args.checkpoint,
        model_type=args.model_type,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        seed=args.seed,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "tau_to_lambda_acc": metrics.get("tau_to_lambda_acc", 0.0),
                "lambda_to_tau_acc": metrics.get("lambda_to_tau_acc", 0.0),
                "same_operator_retrieval_accuracy_tau_to_lambda": metrics.get("same_operator_retrieval_accuracy_tau_to_lambda", 0.0),
                "same_operator_retrieval_accuracy_lambda_to_tau": metrics.get("same_operator_retrieval_accuracy_lambda_to_tau", 0.0),
                "recall_at_3_tau_to_lambda": metrics.get("recall_at_3_tau_to_lambda", 0.0),
                "recall_at_3_lambda_to_tau": metrics.get("recall_at_3_lambda_to_tau", 0.0),
                "diagonal_vs_offdiag_gap": metrics.get("diagonal_vs_offdiag_gap", 0.0),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
