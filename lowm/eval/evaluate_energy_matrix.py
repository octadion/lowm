"""Evaluate diagonal structure of trajectory-context energy matrices."""

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
from lowm.eval.evaluate_all import _ensure_split, _move_batch_to_device, _resolve_checkpoint_path, _score_model, load_run_model


METRIC_VERSION = "paper1_energy_matrix_v1"


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _energy_matrix_for_batch(model: torch.nn.Module, batch: Mapping[str, torch.Tensor], model_type: str) -> torch.Tensor:
    """Return M[i, j] = E(tau_i, context/lambda_j), lower is better."""

    batch_size = int(batch["pos_states"].shape[0])
    if model_type == "lowm" and hasattr(model, "energy_matrix"):
        output = model(batch)
        lambdas = output["lambda"] if isinstance(output, Mapping) and "lambda" in output else output["mu"]
        return model.energy_matrix(batch["pos_states"], batch["pos_actions"], batch["pos_mask"], lambdas)

    # Generic fallback for direct context baselines: materialize all N x N
    # trajectory/context pairs as one-candidate ranking examples.
    pos_states = batch["pos_states"][:, None].expand(batch_size, batch_size, *batch["pos_states"].shape[1:])
    pos_actions = batch["pos_actions"][:, None].expand(batch_size, batch_size, *batch["pos_actions"].shape[1:])
    pos_mask = batch["pos_mask"][:, None].expand(batch_size, batch_size, *batch["pos_mask"].shape[1:])
    context_states = batch["context_states"][None, :, ...].expand(batch_size, batch_size, *batch["context_states"].shape[1:])
    context_actions = batch["context_actions"][None, :, ...].expand(batch_size, batch_size, *batch["context_actions"].shape[1:])
    context_mask = batch["context_mask"][None, :, ...].expand(batch_size, batch_size, *batch["context_mask"].shape[1:])
    pair_batch: dict[str, Any] = {
        "context_states": context_states.reshape(batch_size * batch_size, *batch["context_states"].shape[1:]),
        "context_actions": context_actions.reshape(batch_size * batch_size, *batch["context_actions"].shape[1:]),
        "context_mask": context_mask.reshape(batch_size * batch_size, *batch["context_mask"].shape[1:]),
        "cand_states": pos_states.reshape(batch_size * batch_size, 1, *batch["pos_states"].shape[1:]),
        "cand_actions": pos_actions.reshape(batch_size * batch_size, 1, *batch["pos_actions"].shape[1:]),
        "cand_mask": pos_mask.reshape(batch_size * batch_size, 1, *batch["pos_mask"].shape[1:]),
        "labels": torch.zeros(batch_size * batch_size, dtype=torch.long, device=batch["pos_states"].device),
        "negative_types": [["positive"] for _ in range(batch_size * batch_size)],
    }
    energies = _score_model(model, pair_batch).reshape(batch_size, batch_size)
    return energies


def _batch_metrics(matrix: torch.Tensor) -> dict[str, Any]:
    values = matrix.detach().cpu().numpy().astype(np.float64)
    diag = np.diag(values)
    off_mask = ~np.eye(values.shape[0], dtype=bool)
    off = values[off_mask]
    row_ranks = (values < diag[:, None]).sum(axis=1) + 1
    col_ranks = (values < diag[None, :]).sum(axis=0) + 1
    return {
        "diag": diag,
        "off": off,
        "row_ranks": row_ranks.astype(np.int64),
        "col_ranks": col_ranks.astype(np.int64),
        "row_top1": row_ranks == 1,
        "col_top1": col_ranks == 1,
    }


def _operator_batch_stats(op_id: torch.Tensor, op_params: torch.Tensor) -> dict[str, float]:
    op_id_cpu = op_id.detach().cpu()
    params_cpu = op_params.detach().cpu()
    n = int(op_id_cpu.shape[0])
    if n < 2:
        return {"mean_pairwise_op_param_distance": 0.0, "fraction_same_op_id": 0.0, "effective_unique_op_id": 1.0}
    idx_i, idx_j = torch.triu_indices(n, n, offset=1)
    distances = torch.linalg.norm(params_cpu[idx_i] - params_cpu[idx_j], dim=-1)
    same = (op_id_cpu[idx_i] == op_id_cpu[idx_j]).float()
    _, counts = torch.unique(op_id_cpu, return_counts=True)
    probs = counts.float() / counts.sum().float()
    entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum()
    return {
        "mean_pairwise_op_param_distance": float(distances.mean().item()),
        "fraction_same_op_id": float(same.mean().item()),
        "effective_unique_op_id": float(torch.exp(entropy).item()),
    }


def _aggregate(
    matrices: list[np.ndarray],
    row_ranks: list[np.ndarray],
    col_ranks: list[np.ndarray],
    diag_values: list[np.ndarray],
    off_values: list[np.ndarray],
    batch_stats: list[dict[str, float]],
) -> dict[str, Any]:
    if not matrices:
        return {
            "metric_version": METRIC_VERSION,
            "num_batches": 0,
            "num_matrices": 0,
            "diagonal_energy_mean": 0.0,
            "off_diagonal_energy_mean": 0.0,
            "diagonal_vs_offdiag_gap": 0.0,
            "diagonal_top1_accuracy": 0.0,
            "row_wise_correct_lambda_rank": 0.0,
            "column_wise_correct_trajectory_rank": 0.0,
            "mrr": 0.0,
        }
    row = np.concatenate(row_ranks)
    col = np.concatenate(col_ranks)
    diag = np.concatenate(diag_values)
    off = np.concatenate(off_values) if off_values else np.asarray([], dtype=np.float64)
    row_top1 = np.mean(row == 1)
    col_top1 = np.mean(col == 1)
    stat_mean = {
        key: float(np.mean([stats[key] for stats in batch_stats])) for key in batch_stats[0]
    }
    diag_mean = float(np.mean(diag))
    off_mean = float(np.mean(off)) if off.size else 0.0
    return {
        "metric_version": METRIC_VERSION,
        "num_batches": int(len(matrices)),
        "num_matrices": int(len(matrices)),
        "matrix_size_mean": float(np.mean([matrix.shape[0] for matrix in matrices])),
        "diagonal_energy_mean": diag_mean,
        "off_diagonal_energy_mean": off_mean,
        "diagonal_vs_offdiag_gap": float(off_mean - diag_mean),
        "diagonal_energy_std": float(np.std(diag)),
        "off_diagonal_energy_std": float(np.std(off)) if off.size else 0.0,
        "diagonal_top1_accuracy": float(0.5 * (row_top1 + col_top1)),
        "row_top1_accuracy": float(row_top1),
        "column_top1_accuracy": float(col_top1),
        "row_wise_correct_lambda_rank": float(np.mean(row)),
        "column_wise_correct_trajectory_rank": float(np.mean(col)),
        "row_mrr": float(np.mean(1.0 / row)),
        "column_mrr": float(np.mean(1.0 / col)),
        "mrr": float(0.5 * (np.mean(1.0 / row) + np.mean(1.0 / col))),
        **stat_mean,
    }


def _write_matrix_csv(matrix: np.ndarray, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trajectory_index", *[f"lambda_{j}" for j in range(matrix.shape[1])]])
        for i, row in enumerate(matrix):
            writer.writerow([i, *[float(value) for value in row]])


def _write_pair_csv(matrices: list[np.ndarray], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["batch", "trajectory_index", "lambda_index", "energy", "is_diagonal"])
        writer.writeheader()
        for batch_idx, matrix in enumerate(matrices):
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    writer.writerow(
                        {
                            "batch": batch_idx,
                            "trajectory_index": i,
                            "lambda_index": j,
                            "energy": float(matrix[i, j]),
                            "is_diagonal": int(i == j),
                        }
                    )


def _plot_heatmap(matrix: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xlabel("context / lambda index")
    ax.set_ylabel("trajectory index")
    ax.set_title("Energy matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_summary(metrics: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Energy Matrix Diagonal Summary",
        "",
        f"- Batches: {metrics.get('num_batches', 0)}",
        f"- Mean matrix size: {metrics.get('matrix_size_mean', 0.0):.2f}",
        f"- Diagonal energy mean: {metrics.get('diagonal_energy_mean', 0.0):.4f}",
        f"- Off-diagonal energy mean: {metrics.get('off_diagonal_energy_mean', 0.0):.4f}",
        f"- Diagonal vs off-diagonal gap: {metrics.get('diagonal_vs_offdiag_gap', 0.0):.4f}",
        f"- Diagonal top1 accuracy: {metrics.get('diagonal_top1_accuracy', 0.0):.4f}",
        f"- Row-wise correct lambda rank: {metrics.get('row_wise_correct_lambda_rank', 0.0):.4f}",
        f"- Column-wise correct trajectory rank: {metrics.get('column_wise_correct_trajectory_rank', 0.0):.4f}",
        f"- MRR: {metrics.get('mrr', 0.0):.4f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_energy_matrix(
    run_dir: Path,
    split: str = "val",
    checkpoint_name: str = "best_law_pair.pt",
    model_type: str | None = None,
    matrix_size: int = 16,
    max_batches: int = 8,
    num_samples: int | None = None,
    seed: int | None = None,
    device_name: str = "auto",
) -> dict[str, Any]:
    device = _resolve_device(device_name)
    checkpoint_path = _resolve_checkpoint_path(run_dir, checkpoint_name)
    model, config, detected = load_run_model(run_dir, model_type=model_type, checkpoint_name=checkpoint_name, device=device)
    split_path = _ensure_split(config, split)
    base_ranking_cfg = ranking_config_from_mapping(config)
    ranking_cfg = replace(
        base_ranking_cfg,
        seed=int(seed if seed is not None else base_ranking_cfg.seed),
        ensure_distinct_operators_in_batch=True,
    )
    eval_cfg = dict(config.get("evaluation", {}))
    requested_samples = num_samples if num_samples is not None else eval_cfg.get("energy_matrix_samples")
    if requested_samples is None:
        requested_samples = int(matrix_size) * int(max_batches)
    dataset = LOWMSynthRankingDataset(split_path, ranking_cfg, num_samples=int(requested_samples))
    loader = make_ranking_dataloader(
        dataset,
        batch_size=int(matrix_size),
        shuffle=False,
        ensure_distinct_operators_in_batch=True,
    )

    matrices: list[np.ndarray] = []
    row_ranks: list[np.ndarray] = []
    col_ranks: list[np.ndarray] = []
    diag_values: list[np.ndarray] = []
    off_values: list[np.ndarray] = []
    batch_stats: list[dict[str, float]] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= int(max_batches):
                break
            if int(batch["pos_states"].shape[0]) < 2:
                continue
            batch = _move_batch_to_device(batch, device)
            matrix = _energy_matrix_for_batch(model, batch, detected)
            metrics = _batch_metrics(matrix)
            matrix_np = matrix.detach().cpu().numpy().astype(np.float32)
            matrices.append(matrix_np)
            row_ranks.append(metrics["row_ranks"])
            col_ranks.append(metrics["col_ranks"])
            diag_values.append(metrics["diag"])
            off_values.append(metrics["off"])
            batch_stats.append(_operator_batch_stats(batch["query_op_id"], batch["query_op_params"]))

    metrics = _aggregate(matrices, row_ranks, col_ranks, diag_values, off_values, batch_stats)
    metrics.update(
        {
            "model_type": detected,
            "split": split,
            "checkpoint_requested": checkpoint_name,
            "checkpoint_used": checkpoint_path.name,
            "checkpoint_stem": checkpoint_path.stem,
            "ranking_seed": ranking_cfg.seed,
            "requested_matrix_size": int(matrix_size),
            "requested_max_batches": int(max_batches),
            "num_samples": int(len(dataset)),
        }
    )

    out_dir = run_dir / "eval" / split / "energy_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    if matrices:
        _write_matrix_csv(matrices[0], out_dir / "energy_matrix.csv")
        _write_pair_csv(matrices, out_dir / "energy_matrix_pairs.csv")
        _plot_heatmap(matrices[0], out_dir / "energy_matrix_heatmap.png")
    else:
        _write_matrix_csv(np.zeros((0, 0), dtype=np.float32), out_dir / "energy_matrix.csv")
    _write_summary(metrics, out_dir / "diagonal_summary.md")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--checkpoint", type=str, default="best_law_pair.pt")
    parser.add_argument("--model_type", type=str, default=None, choices=["fixed_energy", "direct_context_energy", "lowm"])
    parser.add_argument("--matrix-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    metrics = evaluate_energy_matrix(
        args.run,
        split=args.split,
        checkpoint_name=args.checkpoint,
        model_type=args.model_type,
        matrix_size=args.matrix_size,
        max_batches=args.max_batches,
        num_samples=args.num_samples,
        seed=args.seed,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "diagonal_energy_mean": metrics["diagonal_energy_mean"],
                "off_diagonal_energy_mean": metrics["off_diagonal_energy_mean"],
                "diagonal_vs_offdiag_gap": metrics["diagonal_vs_offdiag_gap"],
                "diagonal_top1_accuracy": metrics["diagonal_top1_accuracy"],
                "row_wise_correct_lambda_rank": metrics["row_wise_correct_lambda_rank"],
                "column_wise_correct_trajectory_rank": metrics["column_wise_correct_trajectory_rank"],
                "mrr": metrics["mrr"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
