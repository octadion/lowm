"""Aggregate Active Operator Inference outputs across runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    return loaded if isinstance(loaded, dict) else {}


def _find_aoi_metrics(run: Path, split: str | None = None) -> list[Path]:
    if split:
        path = run / "eval" / split / "aoi" / "aoi_metrics.json"
        return [path] if path.exists() else []
    eval_root = run / "eval"
    if not eval_root.exists():
        return []
    return sorted(eval_root.glob("*/aoi/aoi_metrics.json"))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def collect_aoi_rows(runs: list[Path], split: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for metrics_path in _find_aoi_metrics(run, split):
            payload = _load_json(metrics_path)
            methods = payload.get("methods", {})
            if not isinstance(methods, dict):
                continue
            for method, values in methods.items():
                if not isinstance(values, dict):
                    continue
                rows.append(
                    {
                        "run": str(run),
                        "run_name": run.name,
                        "split": payload.get("split", metrics_path.parents[1].name),
                        "model_type": payload.get("model_type", ""),
                        "checkpoint": str(payload.get("checkpoint_used", "")).replace(".pt", ""),
                        "method": method,
                        "temperature": payload.get("temperature", 1.0),
                        "num_episodes": payload.get("num_episodes", 0),
                        "identification_accuracy": values.get("identification_accuracy", 0.0),
                        "entropy_reduction_mean": values.get("entropy_reduction_mean", 0.0),
                        "entropy_reduction_std": values.get("entropy_reduction_std", 0.0),
                        "mrr_mean": values.get("mrr_mean", 0.0),
                        "mrr_std": values.get("mrr_std", 0.0),
                        "posterior_entropy_after_mean": values.get("posterior_entropy_after_mean", 0.0),
                    }
                )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("split", "")), str(row.get("run_name", "")), str(row.get("method", "")))
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (split, run_name, method), group in sorted(groups.items()):
        acc = [float(row.get("identification_accuracy", 0.0)) for row in group]
        er = [float(row.get("entropy_reduction_mean", 0.0)) for row in group]
        mrr = [float(row.get("mrr_mean", 0.0)) for row in group]
        out.append(
            {
                "split": split,
                "run_name": run_name,
                "model_type": group[0].get("model_type", ""),
                "checkpoint": group[0].get("checkpoint", ""),
                "method": method,
                "n": len(group),
                "identification_accuracy_mean": sum(acc) / max(1, len(acc)),
                "identification_accuracy_std": _std(acc),
                "entropy_reduction_mean": sum(er) / max(1, len(er)),
                "entropy_reduction_std": _std(er),
                "mrr_mean": sum(mrr) / max(1, len(mrr)),
                "mrr_std": _std(mrr),
            }
        )
    return out


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "split",
        "run_name",
        "model_type",
        "checkpoint",
        "method",
        "n",
        "identification_accuracy_mean",
        "identification_accuracy_std",
        "entropy_reduction_mean",
        "entropy_reduction_std",
        "mrr_mean",
        "mrr_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_md(rows: list[dict[str, Any]], path: Path) -> None:
    columns = ["split", "run_name", "method", "identification_accuracy", "entropy_reduction", "mrr"]
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append(
            "|"
            + "|".join(
                [
                    str(row.get("split", "")),
                    str(row.get("run_name", "")),
                    str(row.get("method", "")),
                    f"{row.get('identification_accuracy_mean', 0.0):.4f} +/- {row.get('identification_accuracy_std', 0.0):.4f}",
                    f"{row.get('entropy_reduction_mean', 0.0):.4f} +/- {row.get('entropy_reduction_std', 0.0):.4f}",
                    f"{row.get('mrr_mean', 0.0):.4f} +/- {row.get('mrr_std', 0.0):.4f}",
                ]
            )
            + "|"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bar_plot(rows: list[dict[str, Any]], metric: str, ylabel: str, path: Path) -> None:
    if not rows:
        return
    labels = [f"{row['run_name']}:{row['method']}" for row in rows]
    values = [float(row.get(metric, 0.0)) for row in rows]
    fig, ax = plt.subplots(figsize=(max(7, 0.75 * len(labels)), 4.0))
    ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def aggregate_aoi(runs: list[Path], out_dir: Path, split: str | None = None) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_aoi_rows(runs, split)
    summary = summarize_rows(rows)
    _write_csv(summary, out_dir / "aoi_summary.csv")
    _write_md(summary, out_dir / "aoi_summary.md")
    _bar_plot(summary, "identification_accuracy_mean", "Identification accuracy", out_dir / "aoi_identification_accuracy.png")
    _bar_plot(summary, "entropy_reduction_mean", "Entropy reduction", out_dir / "aoi_entropy_reduction.png")
    _bar_plot(summary, "mrr_mean", "MRR", out_dir / "aoi_mrr.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", type=str, default=None)
    args = parser.parse_args()
    rows = aggregate_aoi(args.runs, args.out, args.split)
    print(f"aggregated {len(rows)} AOI rows into {args.out}")


if __name__ == "__main__":
    main()
