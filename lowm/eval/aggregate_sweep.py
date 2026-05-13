"""Aggregate LOWM-OCCL ablation sweep outputs."""

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
    params.setdefault("model_type", params.get("model_type", "lowm" if "lambda_dim" in model else training.get("baseline")))
    params.setdefault("variant", params.get("variant", params.get("component", params.get("negative_set", run_dir.name))))
    params.setdefault("negative_set", params.get("name"))
    params.setdefault("component", params.get("name"))
    params.setdefault("negative_types", config.get("ranking", {}).get("negative_types"))
    return params


def collect_sweep_rows(
    sweep_dir: Path,
    split: str | list[str] = "val",
    ranking_checkpoint: str = "best_law_pair.pt",
    law_checkpoint: str = "best_law_pair.pt",
    occl_checkpoint: str = "best_occl_acc.pt",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    splits = split if isinstance(split, list) else [split]
    for run_dir in _run_dirs(sweep_dir):
        params = _sweep_params(run_dir)
        for split_name in splits:
            try:
                ranking = collect_run(run_dir, split=split_name, checkpoint=ranking_checkpoint)
            except FileNotFoundError:
                continue
            law = _law_only_metrics(run_dir, split_name, law_checkpoint)
            occl = _occl_metrics(run_dir, split_name, occl_checkpoint)
            row = {
                "run": str(run_dir),
                "run_name": run_dir.name,
                "split": split_name,
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
        "split",
        "checkpoint",
        "model_type",
        "variant",
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
    columns = ["run_name", "split", "variant", "component", "alpha_occl", "lambda_dim", "seed", "law_only_top1", "law_pair", "law_gap", "top1_acc", "occl_tau_to_lambda_acc"]
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


def _group_label(row: dict[str, Any]) -> str:
    return str(row.get("variant") or row.get("component") or row.get("negative_set") or row.get("model_type") or row.get("run_name"))


def _summary_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ["top1_acc", "law_only_top1", "law_pair", "law_gap"]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("split", "val")), str(row.get("model_type", "")), _group_label(row))
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (split, model_type, variant), group in sorted(groups.items()):
        record: dict[str, Any] = {
            "split": split,
            "model_type": model_type,
            "variant": variant,
            "n": len(group),
        }
        for metric in metrics:
            values = [float(row.get(metric, 0.0)) for row in group]
            mean = sum(values) / max(1, len(values))
            var = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1) if len(values) > 1 else 0.0
            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = math.sqrt(var)
        out.append(record)
    return out


def _write_stats_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "split",
        "model_type",
        "variant",
        "n",
        "top1_acc_mean",
        "top1_acc_std",
        "law_only_top1_mean",
        "law_only_top1_std",
        "law_pair_mean",
        "law_pair_std",
        "law_gap_mean",
        "law_gap_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_stats_md(rows: list[dict[str, Any]], path: Path) -> None:
    columns = ["split", "variant", "n", "top1_acc", "law_only_top1", "law_pair", "law_gap"]
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append(
            "|"
            + "|".join(
                [
                    str(row.get("split", "")),
                    str(row.get("variant", "")),
                    str(row.get("n", "")),
                    f"{row.get('top1_acc_mean', 0.0):.4f} +/- {row.get('top1_acc_std', 0.0):.4f}",
                    f"{row.get('law_only_top1_mean', 0.0):.4f} +/- {row.get('law_only_top1_std', 0.0):.4f}",
                    f"{row.get('law_pair_mean', 0.0):.4f} +/- {row.get('law_pair_std', 0.0):.4f}",
                    f"{row.get('law_gap_mean', 0.0):.4f} +/- {row.get('law_gap_std', 0.0):.4f}",
                ]
            )
            + "|"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ood_metric_plot(stats: list[dict[str, Any]], metric: str, ylabel: str, path: Path) -> None:
    filtered = [row for row in stats if row.get("split") in {"val", "test_iid", "test_ood_param"}]
    variants = sorted({_group_label(row) for row in filtered})
    splits = ["val", "test_iid", "test_ood_param"]
    if not variants:
        return
    x = range(len(splits))
    width = min(0.75 / max(1, len(variants)), 0.25)
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(variants)), 3.8))
    for idx, variant in enumerate(variants):
        values = []
        for split in splits:
            match = next((row for row in filtered if row.get("split") == split and _group_label(row) == variant), None)
            values.append(float(match.get(f"{metric}_mean", 0.0)) if match else 0.0)
        offsets = [pos + (idx - (len(variants) - 1) / 2) * width for pos in x]
        ax.bar(offsets, values, width=width, label=variant)
    ax.set_xticks(list(x), ["val IID", "test IID", "test OOD"])
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _ood_degradation_plot(stats: list[dict[str, Any]], path: Path) -> None:
    variants = sorted({_group_label(row) for row in stats})
    labels: list[str] = []
    values: list[float] = []
    for variant in variants:
        iid = next((row for row in stats if row.get("split") == "test_iid" and _group_label(row) == variant), None)
        ood = next((row for row in stats if row.get("split") == "test_ood_param" and _group_label(row) == variant), None)
        if iid and ood:
            labels.append(variant)
            values.append(float(iid.get("law_pair_mean", 0.0)) - float(ood.get("law_pair_mean", 0.0)))
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(max(5.5, 1.5 * len(labels)), 3.6))
    ax.bar(labels, values)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("test IID law_pair - test OOD law_pair")
    ax.tick_params(axis="x", rotation=20)
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
    stats = _summary_stats(rows)
    if any(row.get("split") == "test_ood_param" for row in rows):
        _ood_metric_plot(stats, "law_pair", "Law pair", plots / "iid_vs_ood_law_pair.png")
        _ood_metric_plot(stats, "law_only_top1", "Law-only top1", plots / "iid_vs_ood_law_only_top1.png")
        _ood_degradation_plot(stats, plots / "ood_degradation.png")


def aggregate_sweep(
    sweep_dir: Path,
    out_dir: Path,
    split: str | list[str] = "val",
    ranking_checkpoint: str = "best_law_pair.pt",
    law_checkpoint: str = "best_law_pair.pt",
    occl_checkpoint: str = "best_occl_acc.pt",
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_sweep_rows(sweep_dir, split, ranking_checkpoint, law_checkpoint, occl_checkpoint)
    _write_csv(rows, out_dir / "ablation_summary.csv")
    _write_md(rows, out_dir / "ablation_summary.md")
    stats = _summary_stats(rows)
    _write_stats_csv(stats, out_dir / "ablation_summary_stats.csv")
    _write_stats_md(stats, out_dir / "ablation_summary_stats.md")
    write_sweep_plots(rows, out_dir)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--splits", type=str, nargs="*", default=None)
    parser.add_argument("--ranking-checkpoint", type=str, default="best_law_pair.pt")
    parser.add_argument("--law-checkpoint", type=str, default="best_law_pair.pt")
    parser.add_argument("--occl-checkpoint", type=str, default="best_occl_acc.pt")
    args = parser.parse_args()
    split_arg: str | list[str] = args.splits if args.splits else args.split
    rows = aggregate_sweep(args.sweep_dir, args.out, split_arg, args.ranking_checkpoint, args.law_checkpoint, args.occl_checkpoint)
    print(f"aggregated {len(rows)} sweep runs into {args.out}")


if __name__ == "__main__":
    main()
