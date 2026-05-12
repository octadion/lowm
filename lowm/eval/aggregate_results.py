"""Aggregate evaluation outputs across runs."""

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

from lowm.data.negatives import REQUIRED_NEGATIVE_TYPES


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return loaded


def _find_eval_dir(run_dir: Path, split: str | None, checkpoint: str | None = None) -> Path:
    eval_root = run_dir / "eval"
    if split:
        path = eval_root / split
        if checkpoint:
            path = path / Path(checkpoint).stem
        if not path.exists():
            raise FileNotFoundError(f"missing eval directory {path}")
        return path
    candidates = [p for p in eval_root.iterdir() if p.is_dir()] if eval_root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"no eval outputs found under {eval_root}")
    return sorted(candidates)[0]


def _read_breakdown(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["negative_type"]
            out[name] = {
                "pairwise_acc": float(row.get("pairwise_acc", 0.0)),
                "mean_energy_gap": float(row.get("mean_energy_gap", 0.0)),
                "count": float(row.get("count", 0.0)),
            }
    return out


def _run_label(run_dir: Path, summary: dict[str, Any]) -> str:
    model = str(summary.get("model_type", run_dir.name))
    checkpoint = str(summary.get("checkpoint_stem", summary.get("checkpoint_used", ""))).replace(".pt", "")
    suffix = f":{checkpoint}" if checkpoint else ""
    return f"{model}:{run_dir.name}{suffix}"


def collect_run(run_dir: Path, split: str | None = None, checkpoint: str | None = None) -> dict[str, Any]:
    eval_dir = _find_eval_dir(run_dir, split, checkpoint)
    summary_path = eval_dir / "eval_summary.json"
    if summary_path.exists():
        summary = _load_json(summary_path)
        ranking = summary["ranking"]
        retrieval = summary.get("retrieval", {})
        model_type = str(summary.get("model_type", run_dir.name))
    else:
        ranking = _load_json(eval_dir / "ranking_metrics.json")
        retrieval = _load_json(eval_dir / "retrieval_metrics.json") if (eval_dir / "retrieval_metrics.json").exists() else {}
        model_type = run_dir.name
        summary = {"model_type": model_type, "checkpoint_used": checkpoint or ""}
    breakdown = _read_breakdown(eval_dir / "negative_type_breakdown.csv")
    row = {
        "run": str(run_dir),
        "model": model_type,
        "label": _run_label(run_dir, summary),
        "split": str(summary.get("split", split or eval_dir.name)),
        "checkpoint": str(summary.get("checkpoint_used", checkpoint or "")).replace(".pt", ""),
        "top1_acc": float(ranking.get("top1_acc", 0.0)),
        "mean_rank": float(ranking.get("mean_rank", 0.0)),
        "mrr": float(ranking.get("mrr", 0.0)),
        "loss": float(ranking.get("loss", 0.0)),
        "law_pair": float(ranking.get("law_pair", ranking.get("law_mismatch", {}).get("pairwise_acc", 0.0))),
        "law_gap": float(ranking.get("law_gap", ranking.get("law_mismatch", {}).get("mean_energy_gap", 0.0))),
        "retrieval_acc": float(retrieval.get("retrieval_acc", 0.0)),
        "retrieval_mrr": float(retrieval.get("mrr", 0.0)),
        "breakdown": breakdown,
    }
    for name in REQUIRED_NEGATIVE_TYPES:
        row[f"{name}_pairwise_acc"] = breakdown.get(name, {}).get("pairwise_acc", 0.0)
        row[f"{name}_gap"] = breakdown.get(name, {}).get("mean_energy_gap", 0.0)
    return row


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "model",
        "checkpoint",
        "label",
        "split",
        "top1_acc",
        "mean_rank",
        "mrr",
        "loss",
        "law_pair",
        "law_gap",
        "retrieval_acc",
        "retrieval_mrr",
        *[f"{name}_pairwise_acc" for name in REQUIRED_NEGATIVE_TYPES],
        *[f"{name}_gap" for name in REQUIRED_NEGATIVE_TYPES],
        "run",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    columns = ["model", "checkpoint", "top1_acc", "law_pair", "law_gap", "retrieval_acc", "mrr"]
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("|" + "|".join(values) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bar_plot(rows: list[dict[str, Any]], key: str, ylabel: str, path: Path) -> None:
    labels = [row["label"] for row in rows]
    values = [float(row.get(key, 0.0)) for row in rows]
    fig, ax = plt.subplots(figsize=(max(5, 2.3 * len(rows)), 3.6))
    ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _grouped_negative_plot(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [row["label"] for row in rows]
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(rows)), 4.0))
    for i, name in enumerate(REQUIRED_NEGATIVE_TYPES):
        values = [float(row.get(f"{name}_pairwise_acc", 0.0)) for row in rows]
        ax.bar(x + (i - 1.5) * width, values, width, label=name)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Pairwise accuracy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def aggregate_results(run_dirs: list[Path], out_dir: Path, split: str | None = None, checkpoints: list[str] | None = None) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if checkpoints:
        for run_dir in run_dirs:
            for checkpoint in checkpoints:
                try:
                    rows.append(collect_run(run_dir, split, checkpoint))
                except FileNotFoundError:
                    continue
    else:
        rows = [collect_run(run_dir, split) for run_dir in run_dirs]
    _write_csv(rows, out_dir / "summary_table.csv")
    _write_markdown(rows, out_dir / "summary_table.md")
    _bar_plot(rows, "top1_acc", "Overall top-1", out_dir / "ranking_bar_by_model.png")
    _bar_plot(rows, "law_pair", "Law-mismatch pairwise accuracy", out_dir / "law_pair_by_model.png")
    _bar_plot(rows, "law_gap", "Law-mismatch energy gap", out_dir / "law_gap_by_model.png")

    fig, axes = plt.subplots(1, 2, figsize=(max(7, 2.2 * len(rows)), 3.6))
    labels = [row["label"] for row in rows]
    axes[0].bar(labels, [float(row.get("law_pair", 0.0)) for row in rows])
    axes[0].set_ylabel("Law pair")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(labels, [float(row.get("law_gap", 0.0)) for row in rows])
    axes[1].set_ylabel("Law gap")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_dir / "law_pair_gap_by_model.png", dpi=160)
    plt.close(fig)

    _grouped_negative_plot(rows, out_dir / "pairwise_accuracy_by_negative_type.png")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--checkpoints", type=str, nargs="*", default=None)
    args = parser.parse_args()
    rows = aggregate_results(args.runs, args.out, args.split, args.checkpoints)
    print(f"aggregated {len(rows)} runs into {args.out}")


if __name__ == "__main__":
    main()
