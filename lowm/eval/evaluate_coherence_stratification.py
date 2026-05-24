"""Evaluate coherent-here vs plausible-elsewhere vs incoherent energies."""

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


METRIC_VERSION = "paper1_coherence_stratification_v1"
STRATA = ("same", "wrong", "noise")


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _candidate_index(types: list[str], label: int, candidate_type: str) -> int:
    if candidate_type == "positive":
        return label
    matches = [idx for idx, name in enumerate(types) if name == candidate_type]
    if not matches:
        raise ValueError(f"sample is missing candidate type '{candidate_type}'")
    return int(matches[0])


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "metric_version": METRIC_VERSION,
            "num_samples": 0,
            "fraction_same_lt_wrong": 0.0,
            "fraction_wrong_lt_noise": 0.0,
            "fraction_same_lt_wrong_lt_noise": 0.0,
            "stratification_accuracy": 0.0,
            "pairwise_stratification_accuracy": 0.0,
        }

    same = np.asarray([row["E_same"] for row in rows], dtype=np.float64)
    wrong = np.asarray([row["E_wrong"] for row in rows], dtype=np.float64)
    noise = np.asarray([row["E_noise"] for row in rows], dtype=np.float64)
    same_lt_wrong = same < wrong
    wrong_lt_noise = wrong < noise
    same_lt_noise = same < noise
    chain = same_lt_wrong & wrong_lt_noise
    gap_same_wrong = wrong - same
    gap_wrong_noise = noise - wrong
    metrics: dict[str, Any] = {
        "metric_version": METRIC_VERSION,
        "num_samples": int(len(rows)),
        "fraction_same_lt_wrong": float(np.mean(same_lt_wrong)),
        "fraction_wrong_lt_noise": float(np.mean(wrong_lt_noise)),
        "fraction_same_lt_noise": float(np.mean(same_lt_noise)),
        "fraction_same_lt_wrong_lt_noise": float(np.mean(chain)),
        "stratification_accuracy": float(np.mean(chain)),
        "pairwise_stratification_accuracy": float(0.5 * (np.mean(same_lt_wrong) + np.mean(wrong_lt_noise))),
        "gap_same_wrong": float(np.mean(gap_same_wrong)),
        "gap_wrong_noise": float(np.mean(gap_wrong_noise)),
        "gap_same_noise": float(np.mean(noise - same)),
        "mean_energies": {
            "same": float(np.mean(same)),
            "wrong": float(np.mean(wrong)),
            "noise": float(np.mean(noise)),
        },
        "std_energies": {
            "same": float(np.std(same)),
            "wrong": float(np.std(wrong)),
            "noise": float(np.std(noise)),
        },
        "median_energies": {
            "same": float(np.median(same)),
            "wrong": float(np.median(wrong)),
            "noise": float(np.median(noise)),
        },
    }
    return metrics


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "sample_id",
        "query_episode",
        "query_t0",
        "query_op_id",
        "wrong_op_id",
        "noise_op_id",
        "E_same",
        "E_wrong",
        "E_noise",
        "gap_same_wrong",
        "gap_wrong_noise",
        "same_lt_wrong",
        "wrong_lt_noise",
        "same_lt_wrong_lt_noise",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _plot_energy_by_type(rows: list[dict[str, Any]], path: Path) -> None:
    values = [[float(row[f"E_{name}"]) for row in rows] for name in STRATA]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    if rows:
        ax.boxplot(values, showmeans=True)
        ax.set_xticks([1, 2, 3], ["coherent here", "plausible elsewhere", "noise"])
        means = [np.mean(v) if v else 0.0 for v in values]
        ax.plot(np.arange(1, len(means) + 1), means, color="black", marker="o", linewidth=1.0)
    else:
        ax.text(0.5, 0.5, "No samples", transform=ax.transAxes, ha="center", va="center")
    ax.set_ylabel("Energy (lower is more coherent)")
    ax.set_title("Coherence stratification")
    ax.tick_params(axis="x", rotation=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_summary(metrics: Mapping[str, Any], path: Path) -> None:
    mean = metrics.get("mean_energies", {})
    lines = [
        "# Coherence Stratification Summary",
        "",
        f"- Samples: {metrics.get('num_samples', 0)}",
        f"- Same < wrong: {metrics.get('fraction_same_lt_wrong', 0.0):.4f}",
        f"- Wrong < noise: {metrics.get('fraction_wrong_lt_noise', 0.0):.4f}",
        f"- Same < wrong < noise: {metrics.get('fraction_same_lt_wrong_lt_noise', 0.0):.4f}",
        f"- Stratification accuracy: {metrics.get('stratification_accuracy', 0.0):.4f}",
        f"- Gap E_wrong - E_same: {metrics.get('gap_same_wrong', 0.0):.4f}",
        f"- Gap E_noise - E_wrong: {metrics.get('gap_wrong_noise', 0.0):.4f}",
        "",
        "Mean energies:",
        f"- coherent-here: {float(mean.get('same', 0.0)):.4f}",
        f"- plausible-elsewhere: {float(mean.get('wrong', 0.0)):.4f}",
        f"- incoherent/noise: {float(mean.get('noise', 0.0)):.4f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_coherence_stratification(
    run_dir: Path,
    split: str = "val",
    checkpoint_name: str = "best_law_pair.pt",
    model_type: str | None = None,
    batch_size: int | None = None,
    num_samples: int | None = None,
    seed: int | None = None,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Run the Paper 1 coherence-stratification diagnostic for a trained run."""

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
    sample_offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            energies = _score_model(model, batch).detach().cpu()
            labels = batch["labels"].detach().cpu()
            query_episode = batch["query_episode"].detach().cpu()
            query_t0 = batch["query_t0"].detach().cpu()
            query_op_id = batch["query_op_id"].detach().cpu()
            candidate_op_id = batch["candidate_op_id"].detach().cpu()
            for b, types in enumerate(batch["negative_types"]):
                label = int(labels[b].item())
                same_idx = _candidate_index(types, label, "positive")
                wrong_idx = _candidate_index(types, label, "law_mismatch")
                noise_idx = _candidate_index(types, label, "random_impossible")
                e_same = float(energies[b, same_idx].item())
                e_wrong = float(energies[b, wrong_idx].item())
                e_noise = float(energies[b, noise_idx].item())
                same_lt_wrong = e_same < e_wrong
                wrong_lt_noise = e_wrong < e_noise
                rows.append(
                    {
                        "sample_id": sample_offset + b,
                        "query_episode": int(query_episode[b].item()),
                        "query_t0": int(query_t0[b].item()),
                        "query_op_id": int(query_op_id[b].item()),
                        "wrong_op_id": int(candidate_op_id[b, wrong_idx].item()),
                        "noise_op_id": int(candidate_op_id[b, noise_idx].item()),
                        "E_same": e_same,
                        "E_wrong": e_wrong,
                        "E_noise": e_noise,
                        "gap_same_wrong": e_wrong - e_same,
                        "gap_wrong_noise": e_noise - e_wrong,
                        "same_lt_wrong": int(same_lt_wrong),
                        "wrong_lt_noise": int(wrong_lt_noise),
                        "same_lt_wrong_lt_noise": int(same_lt_wrong and wrong_lt_noise),
                    }
                )
            sample_offset += int(energies.shape[0])

    metrics = _summarize(rows)
    metrics.update(
        {
            "model_type": detected,
            "split": split,
            "checkpoint_requested": checkpoint_name,
            "checkpoint_used": checkpoint_path.name,
            "checkpoint_stem": checkpoint_path.stem,
            "ranking_seed": ranking_cfg.seed,
            "negative_types": list(ranking_cfg.negative_types),
        }
    )

    out_dir = run_dir / "eval" / split / "coherence_stratification"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    _write_rows(rows, out_dir / "per_sample.csv")
    _plot_energy_by_type(rows, out_dir / "energy_by_type.png")
    _write_summary(metrics, out_dir / "stratification_summary.md")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--checkpoint", type=str, default="best_law_pair.pt")
    parser.add_argument("--model_type", type=str, default=None, choices=["fixed_energy", "direct_context_energy", "lowm"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    metrics = evaluate_coherence_stratification(
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
                "fraction_same_lt_wrong": metrics["fraction_same_lt_wrong"],
                "fraction_wrong_lt_noise": metrics["fraction_wrong_lt_noise"],
                "fraction_same_lt_wrong_lt_noise": metrics["fraction_same_lt_wrong_lt_noise"],
                "gap_same_wrong": metrics.get("gap_same_wrong", 0.0),
                "gap_wrong_noise": metrics.get("gap_wrong_noise", 0.0),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
