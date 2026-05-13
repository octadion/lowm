"""Aggregate LOWM-OCCL ablation sweep outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from lowm.eval.aggregate_results import collect_run


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    return loaded if isinstance(loaded, dict) else {}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return loaded if isinstance(loaded, dict) else {}


def _run_dirs(sweep_dir: Path) -> list[Path]:
    manifest = _load_json(sweep_dir / "manifest.json")
    if manifest.get("runs"):
        return [Path(path) for path in manifest["runs"]]
    root = sweep_dir / "runs"
    if root.exists():
        return sorted(path for path in root.iterdir() if path.is_dir())
    return sorted(path for path in sweep_dir.iterdir() if path.is_dir() and (path / "config.yaml").exists())


def _law_only_metrics(run_dir: Path, split: str, checkpoint: str) -> dict[str, Any]:
    stem = Path(checkpoint).stem
    candidates = [
        run_dir / "eval" / split / f"law_mismatch_only_{stem}" / "law_mismatch_only_metrics.json",
        run_dir / "eval" / split / "law_mismatch_only_metrics.json",
    ]
    for path in candidates:
        if path.exists():
            return _load_json(path)
    return {}


def _occl_metrics(run_dir: Path, split: str, checkpoint: str) -> dict[str, Any]:
    stem = Path(checkpoint).stem
    candidates = [
        run_dir / "eval" / split / stem / "occl_alignment_metrics.json",
        run_dir / "eval" / split / "occl_alignment_metrics.json",
    ]
    for path in candidates:
        if path.exists():
            return _load_json(path)
    return {}


def _sweep_params(run_dir: Path) -> dict[str, Any]:
    config = _load_yaml(run_dir / "config.yaml")
    params = dict(config.get("sweep_params", {}))
    model = dict(config.get("model", {}))
    training = dict(config.get("training", {}))
    params.setdefault("alpha_occl", training.get("alpha_occl"))
    params.setdefault("lambda_dim", model.get("lambda_dim"))
    params.setdefault("use_pairwise_energy", model.get("use_pairwise_energy"))
    params.setdefault("use_stability", training.get("use_stability"))
    params.setdefault("beta_kl", training.get("beta_kl"))
    params.setdefault("seed", training.get("seed"))
    params.setdefault("negative_set", params.get("name"))
    params.setdefault("component", params.get("name"))
    params.setdefault("negative_types", config.get("ranking", {}).get("negative_types"))
    return params


def collect_sweep_rows(
    sweep_dir: Path,
    split: str = "val",
    ranking_checkpoint: str = "best_law_pair.pt",
    law_checkpoint: str = "best_law_pair.pt",
    occl_checkpoint: str = "best_occl_acc.pt",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in _run_dirs(sweep_dir):
        try:
            ranking = collect_run(run_dir, split=split, checkpoint=ranking_checkpoint)
        except FileNotFoundError:
            continue
        law = _law_only_metrics(run_dir, split, law_checkpoint)
        occl = _occl_metrics(run_dir, split, occl_checkpoint)
        params = _sweep_params(run_dir)
        row = {
            "run": str(run_dir),
            "run_name": run_dir.name,
            "checkpoint": ranking.get("checkpoint", Path(ranking_checkpoint).stem),
            **params,
            "top1_acc": ranking.get("top1_acc", 0.0),
            "law_pair": ranking.get("law_pair", 0.0),
            "law_gap": ranking.get("law_gap", 0.0),
            "law_only_top1": law.get("law_only_top1", law.get("top1_law_only", 0.0)),
            "law_only_pairwise_acc": law.get("pairwise_acc_law_only", law.get("law_pair", 0.0)),
            "mean_law_gap": law.get("mean_law_gap", law.get("law_gap", 0.0)),
            "occl_tau_to_lambda_acc": occl.get("tau_to_lambda_acc", 0.0),
            "occl_lambda_to_tau_acc": occl.get("lambda_to_tau_acc", 0.0),
            "occl_same_operator_tau_to_lambda": occl.get("same_operator_retrieval_accuracy_tau_to_lambda", 0.0),
            "occl_same_operator_lambda_to_tau": occl.get("same_operator_retrieval_accuracy_lambda_to_tau", 0.0),
            "occl_gap": occl.get("diagonal_vs_offdiag_gap", 0.0),
        }
        rows.append(row)
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "run_name",
        "checkpoint",
        "alpha_occl",
        "lambda_dim",
        "use_pairwise_energy",
        "use_stability",
        "beta_kl",
        "seed",
        "component",
        "negative_set",
        "negative_types",
        "top1_acc",
        "law_pair",
        "law_gap",
        "law_only_top1",
        "law_only_pairwise_acc",
        "mean_law_gap",
        "occl_tau_to_lambda_acc",
        "occl_lambda_to_tau_acc",
        "occl_same_operator_tau_to_lambda",
        "occl_same_operator_lambda_to_tau",
        "occl_gap",
        "run",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_md(rows: list[dict[str, Any]], path: Path) -> None:
    columns = ["run_name", "component", "alpha_occl", "lambda_dim", "seed", "law_only_top1", "law_pair", "law_gap", "top1_acc", "occl_tau_to_lambda_acc"]
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            vals.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("|" + "|".join(vals) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean_by(rows: list[dict[str, Any]], x_key: str, y_key: str) -> tuple[list[Any], list[float]]:
    groups: dict[Any, list[float]] = {}
    for row in rows:
        x = row.get(x_key)
        if x is None:
            continue
        groups.setdefault(x, []).append(float(row.get(y_key, 0.0)))
    xs = sorted(groups)
    ys = [sum(groups[x]) / max(1, len(groups[x])) for x in xs]
    return xs, ys


def _line_plot(rows: list[dict[str, Any]], x_key: str, y_key: str, ylabel: str, path: Path) -> None:
    xs, ys = _mean_by(rows, x_key, y_key)
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel(x_key)
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _bar_plot(rows: list[dict[str, Any]], x_key: str, y_key: str, ylabel: str, path: Path) -> None:
    xs, ys = _mean_by(rows, x_key, y_key)
    labels = [str(x) for x in xs]
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    ax.bar(labels, ys)
    ax.set_xlabel(x_key)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_sweep_plots(rows: list[dict[str, Any]], out_dir: Path) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    _line_plot(rows, "alpha_occl", "law_only_top1", "Law-only top1", plots / "law_only_top1_vs_alpha.png")
    _line_plot(rows, "alpha_occl", "law_pair", "Law pair", plots / "law_pair_vs_alpha.png")
    _line_plot(rows, "alpha_occl", "law_gap", "Law gap", plots / "law_gap_vs_alpha.png")
    _line_plot(rows, "alpha_occl", "top1_acc", "Overall top1", plots / "top1_vs_alpha.png")
    _bar_plot(rows, "lambda_dim", "law_pair", "Law pair", plots / "lambda_dim_ablation.png")
    _bar_plot(rows, "use_pairwise_energy", "law_pair", "Law pair", plots / "pairwise_energy_ablation.png")
    if any(row.get("negative_set") for row in rows):
        _bar_plot(rows, "negative_set", "top1_acc", "Generic top1", plots / "negative_set_top1.png")
        _bar_plot(rows, "negative_set", "law_only_top1", "Law-only top1", plots / "negative_set_law_only_top1.png")
        _bar_plot(rows, "negative_set", "law_pair", "Law pair", plots / "negative_set_law_pair.png")
        _bar_plot(rows, "negative_set", "law_gap", "Law gap", plots / "negative_set_law_gap.png")
    if any(row.get("component") for row in rows):
        _bar_plot(rows, "component", "law_pair", "Law pair", plots / "component_ablation_law_pair.png")
        _bar_plot(rows, "component", "law_gap", "Law gap", plots / "component_ablation_law_gap.png")
        _bar_plot(rows, "component", "law_only_top1", "Law-only top1", plots / "component_ablation_law_only_top1.png")


def aggregate_sweep(
    sweep_dir: Path,
    out_dir: Path,
    split: str = "val",
    ranking_checkpoint: str = "best_law_pair.pt",
    law_checkpoint: str = "best_law_pair.pt",
    occl_checkpoint: str = "best_occl_acc.pt",
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_sweep_rows(sweep_dir, split, ranking_checkpoint, law_checkpoint, occl_checkpoint)
    _write_csv(rows, out_dir / "ablation_summary.csv")
    _write_md(rows, out_dir / "ablation_summary.md")
    write_sweep_plots(rows, out_dir)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--ranking-checkpoint", type=str, default="best_law_pair.pt")
    parser.add_argument("--law-checkpoint", type=str, default="best_law_pair.pt")
    parser.add_argument("--occl-checkpoint", type=str, default="best_occl_acc.pt")
    args = parser.parse_args()
    rows = aggregate_sweep(args.sweep_dir, args.out, args.split, args.ranking_checkpoint, args.law_checkpoint, args.occl_checkpoint)
    print(f"aggregated {len(rows)} sweep runs into {args.out}")


if __name__ == "__main__":
    main()
