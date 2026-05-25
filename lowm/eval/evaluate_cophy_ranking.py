"""Evaluate CoPhy-style wrong-confounder ranking and stratification."""

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
from lowm.eval.evaluate_energy_matrix import _batch_metrics, _energy_matrix_for_batch


METRIC_VERSION = "cophy_ranking_v1"


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _candidate_index(types: list[str], label: int, candidate_type: str) -> int:
    if candidate_type == "positive":
        return label
    matches = [idx for idx, name in enumerate(types) if name == candidate_type]
    if not matches:
        raise ValueError(f"missing candidate type {candidate_type}")
    return int(matches[0])


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "metric_version": METRIC_VERSION,
            "num_samples": 0,
            "same_lt_wrong": 0.0,
            "wrong_lt_noise": 0.0,
            "full_stratification": 0.0,
            "wrong_confounder_pair_acc": 0.0,
            "confounder_only_top1": 0.0,
            "generic_top1": 0.0,
        }
    same = np.asarray([row["E_same"] for row in rows], dtype=np.float64)
    wrong = np.asarray([row["E_wrong"] for row in rows], dtype=np.float64)
    noise = np.asarray([row["E_noise"] for row in rows], dtype=np.float64)
    energies = np.stack([same, wrong, noise], axis=1)
    same_lt_wrong = same < wrong
    wrong_lt_noise = wrong < noise
    full = same_lt_wrong & wrong_lt_noise
    return {
        "metric_version": METRIC_VERSION,
        "num_samples": int(len(rows)),
        "same_lt_wrong": float(np.mean(same_lt_wrong)),
        "wrong_lt_noise": float(np.mean(wrong_lt_noise)),
        "full_stratification": float(np.mean(full)),
        "wrong_confounder_pair_acc": float(np.mean(same_lt_wrong)),
        "confounder_only_top1": float(np.mean(same_lt_wrong)),
        "generic_top1": float(np.mean(np.argmin(energies, axis=1) == 0)),
        "gap_same_wrong": float(np.mean(wrong - same)),
        "gap_wrong_noise": float(np.mean(noise - wrong)),
        "mean_energies": {
            "same": float(np.mean(same)),
            "wrong_confounder": float(np.mean(wrong)),
            "noise": float(np.mean(noise)),
        },
    }


def _matrix_summary(matrices: list[np.ndarray], row_ranks: list[np.ndarray], col_ranks: list[np.ndarray], diag_values: list[np.ndarray], off_values: list[np.ndarray]) -> dict[str, Any]:
    if not matrices:
        return {
            "energy_matrix_mrr": 0.0,
            "energy_matrix_top1": 0.0,
            "diagonal_vs_offdiag_gap": 0.0,
        }
    row = np.concatenate(row_ranks)
    col = np.concatenate(col_ranks)
    diag = np.concatenate(diag_values)
    off = np.concatenate(off_values)
    row_mrr = float(np.mean(1.0 / row))
    col_mrr = float(np.mean(1.0 / col))
    row_top1 = float(np.mean(row == 1))
    col_top1 = float(np.mean(col == 1))
    return {
        "energy_matrix_mrr": 0.5 * (row_mrr + col_mrr),
        "energy_matrix_top1": 0.5 * (row_top1 + col_top1),
        "diagonal_vs_offdiag_gap": float(np.mean(off) - np.mean(diag)),
        "energy_matrix_batches": int(len(matrices)),
    }


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "sample_id",
        "query_episode",
        "query_confounder_id",
        "wrong_confounder_id",
        "E_same",
        "E_wrong",
        "E_noise",
        "gap_same_wrong",
        "gap_wrong_noise",
        "same_lt_wrong",
        "wrong_lt_noise",
        "full_stratification",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _plot_energies(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    if rows:
        values = [[row["E_same"] for row in rows], [row["E_wrong"] for row in rows], [row["E_noise"] for row in rows]]
        ax.boxplot(values, showmeans=True)
        ax.set_xticks([1, 2, 3], ["same", "wrong", "noise"])
    else:
        ax.text(0.5, 0.5, "No samples", transform=ax.transAxes, ha="center", va="center")
    ax.set_ylabel("Energy")
    ax.set_title("CoPhy coherence energies")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_matrix(matrix: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xlabel("context index")
    ax.set_ylabel("trajectory index")
    ax.set_title("CoPhy energy matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_summary(metrics: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# CoPhy Ranking Summary",
        "",
        f"- Samples: {metrics.get('num_samples', 0)}",
        f"- Generic top1: {metrics.get('generic_top1', 0.0):.4f}",
        f"- Same < wrong confounder: {metrics.get('same_lt_wrong', 0.0):.4f}",
        f"- Wrong < noise: {metrics.get('wrong_lt_noise', 0.0):.4f}",
        f"- Full stratification: {metrics.get('full_stratification', 0.0):.4f}",
        f"- Gap same-wrong: {metrics.get('gap_same_wrong', 0.0):.4f}",
        f"- Gap wrong-noise: {metrics.get('gap_wrong_noise', 0.0):.4f}",
        f"- Energy matrix MRR: {metrics.get('energy_matrix_mrr', 0.0):.4f}",
        f"- Energy matrix top1: {metrics.get('energy_matrix_top1', 0.0):.4f}",
        f"- Diagonal/off-diagonal gap: {metrics.get('diagonal_vs_offdiag_gap', 0.0):.4f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_cophy_ranking(
    run_dir: Path,
    split: str = "val",
    scenario: str | None = None,
    checkpoint_name: str = "best_law_pair.pt",
    model_type: str | None = None,
    batch_size: int | None = None,
    num_samples: int | None = None,
    seed: int | None = None,
    matrix_size: int = 16,
    max_batches: int = 8,
    device_name: str = "auto",
) -> dict[str, Any]:
    device = _resolve_device(device_name)
    checkpoint_path = _resolve_checkpoint_path(run_dir, checkpoint_name)
    model, config, detected = load_run_model(run_dir, model_type=model_type, checkpoint_name=checkpoint_name, device=device)
    split_path = _ensure_split(config, split)
    base_ranking_cfg = ranking_config_from_mapping(config)
    ranking_cfg = replace(
        base_ranking_cfg,
        M=3,
        negative_types=("law_mismatch", "random_impossible"),
        seed=int(seed if seed is not None else base_ranking_cfg.seed),
        ensure_distinct_operators_in_batch=False,
    )
    eval_cfg = dict(config.get("evaluation", {}))
    sample_count = num_samples if num_samples is not None else eval_cfg.get("num_samples", config.get("training", {}).get("val_samples"))
    dataset = LOWMSynthRankingDataset(split_path, ranking_cfg, num_samples=int(sample_count) if sample_count else None)
    bs = int(batch_size or config.get("training", {}).get("batch_size", 64))
    loader = make_ranking_dataloader(dataset, batch_size=bs, shuffle=False)

    rows: list[dict[str, Any]] = []
    matrices: list[np.ndarray] = []
    row_ranks: list[np.ndarray] = []
    col_ranks: list[np.ndarray] = []
    diag_values: list[np.ndarray] = []
    off_values: list[np.ndarray] = []
    sample_offset = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = _move_batch_to_device(batch, device)
            energies = _score_model(model, batch).detach().cpu()
            labels = batch["labels"].detach().cpu()
            candidate_op_id = batch["candidate_op_id"].detach().cpu()
            query_op_id = batch["query_op_id"].detach().cpu()
            query_episode = batch["query_episode"].detach().cpu()
            for b, types in enumerate(batch["negative_types"]):
                label = int(labels[b].item())
                same_idx = _candidate_index(types, label, "positive")
                wrong_idx = _candidate_index(types, label, "law_mismatch")
                noise_idx = _candidate_index(types, label, "random_impossible")
                e_same = float(energies[b, same_idx].item())
                e_wrong = float(energies[b, wrong_idx].item())
                e_noise = float(energies[b, noise_idx].item())
                rows.append(
                    {
                        "sample_id": sample_offset + b,
                        "query_episode": int(query_episode[b].item()),
                        "query_confounder_id": int(query_op_id[b].item()),
                        "wrong_confounder_id": int(candidate_op_id[b, wrong_idx].item()),
                        "E_same": e_same,
                        "E_wrong": e_wrong,
                        "E_noise": e_noise,
                        "gap_same_wrong": e_wrong - e_same,
                        "gap_wrong_noise": e_noise - e_wrong,
                        "same_lt_wrong": int(e_same < e_wrong),
                        "wrong_lt_noise": int(e_wrong < e_noise),
                        "full_stratification": int(e_same < e_wrong < e_noise),
                    }
                )
            sample_offset += int(energies.shape[0])

            if len(matrices) < max_batches and int(batch["pos_states"].shape[0]) >= 2:
                matrix_batch = {
                    key: value[: min(matrix_size, value.shape[0])] if torch.is_tensor(value) and value.shape[:1] == batch["pos_states"].shape[:1] else value
                    for key, value in batch.items()
                }
                matrix = _energy_matrix_for_batch(model, matrix_batch, detected)
                m = _batch_metrics(matrix)
                matrices.append(matrix.detach().cpu().numpy().astype(np.float32))
                row_ranks.append(m["row_ranks"])
                col_ranks.append(m["col_ranks"])
                diag_values.append(m["diag"])
                off_values.append(m["off"])

    metrics = _summarize(rows)
    metrics.update(_matrix_summary(matrices, row_ranks, col_ranks, diag_values, off_values))
    metrics.update(
        {
            "model_type": detected,
            "split": split,
            "scenario": scenario or Path(split_path).parent.name,
            "checkpoint_requested": checkpoint_name,
            "checkpoint_used": checkpoint_path.name,
            "checkpoint_stem": checkpoint_path.stem,
            "ranking_seed": ranking_cfg.seed,
            "num_dataset_samples": len(dataset),
        }
    )
    out_dir = run_dir / "eval" / split / "cophy_ranking"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    _write_rows(rows, out_dir / "per_sample.csv")
    _plot_energies(rows, out_dir / "energy_by_type.png")
    if matrices:
        _plot_matrix(matrices[0], out_dir / "energy_matrix_heatmap.png")
    _write_summary(metrics, out_dir / "summary.md")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="best_law_pair.pt")
    parser.add_argument("--model_type", type=str, default=None, choices=["fixed_energy", "direct_context_energy", "lowm"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--matrix-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    metrics = evaluate_cophy_ranking(
        args.run,
        split=args.split,
        scenario=args.scenario,
        checkpoint_name=args.checkpoint,
        model_type=args.model_type,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        seed=args.seed,
        matrix_size=args.matrix_size,
        max_batches=args.max_batches,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "generic_top1": metrics["generic_top1"],
                "same_lt_wrong": metrics["same_lt_wrong"],
                "wrong_lt_noise": metrics["wrong_lt_noise"],
                "full_stratification": metrics["full_stratification"],
                "energy_matrix_mrr": metrics["energy_matrix_mrr"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
