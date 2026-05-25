"""Aggregate Paper 1 direct-context OMC backbone comparison results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml

from lowm.eval.aggregate_results import collect_run


METRICS = (
    "top1_acc",
    "law_pair",
    "law_gap",
    "law_only_top1",
    "fraction_same_lt_wrong",
    "fraction_same_lt_wrong_lt_noise",
    "gap_same_wrong",
    "energy_matrix_mrr",
    "energy_matrix_diagonal_top1_accuracy",
    "energy_matrix_diagonal_vs_offdiag_gap",
)


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
    if isinstance(manifest.get("runs"), list):
        return [Path(path) for path in manifest["runs"]]
    root = sweep_dir / "runs"
    if root.exists():
        return sorted(path for path in root.iterdir() if path.is_dir())
    return sorted(path for path in sweep_dir.iterdir() if path.is_dir() and (path / "config.yaml").exists())


def _sweep_params(run_dir: Path) -> dict[str, Any]:
    config = _load_yaml(run_dir / "config.yaml")
    params = dict(config.get("sweep_params", {}))
    training = dict(config.get("training", {}))
    params.setdefault("variant", params.get("negative_set", run_dir.name))
    params.setdefault("model_type", params.get("model_type", training.get("baseline", "")))
    params.setdefault("seed", training.get("seed", params.get("seed", "")))
    params.setdefault("negative_set", params.get("variant", ""))
    params.setdefault("negative_types", config.get("ranking", {}).get("negative_types", []))
    return params


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


def _analysis_metrics(run_dir: Path, split: str) -> tuple[dict[str, Any], dict[str, Any]]:
    coherence = _load_json(run_dir / "eval" / split / "coherence_stratification" / "metrics.json")
    matrix = _load_json(run_dir / "eval" / split / "energy_matrix" / "metrics.json")
    return coherence, matrix


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_backbone_omc_rows(
    sweep_dir: Path,
    splits: list[str],
    ranking_checkpoint: str = "best_law_pair.pt",
    law_checkpoint: str = "best_law_pair.pt",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in _run_dirs(sweep_dir):
        params = _sweep_params(run_dir)
        for split in splits:
            try:
                ranking = collect_run(run_dir, split=split, checkpoint=ranking_checkpoint)
            except FileNotFoundError:
                continue
            law = _law_only_metrics(run_dir, split, law_checkpoint)
            coherence, matrix = _analysis_metrics(run_dir, split)
            missing = []
            if not law:
                missing.append("law_mismatch_only")
            if not coherence:
                missing.append("coherence_stratification")
            if not matrix:
                missing.append("energy_matrix")
            rows.append(
                {
                    "run_name": run_dir.name,
                    "run": str(run_dir),
                    "split": split,
                    "model_type": params.get("model_type", ""),
                    "variant": params.get("variant", ""),
                    "negative_set": params.get("negative_set", ""),
                    "negative_types": params.get("negative_types", []),
                    "seed": params.get("seed", ""),
                    "checkpoint": ranking.get("checkpoint", Path(ranking_checkpoint).stem),
                    "top1_acc": ranking.get("top1_acc", 0.0),
                    "law_pair": ranking.get("law_pair", 0.0),
                    "law_gap": ranking.get("law_gap", 0.0),
                    "law_only_top1": law.get("law_only_top1", law.get("top1_law_only")),
                    "fraction_same_lt_wrong": coherence.get("fraction_same_lt_wrong"),
                    "fraction_same_lt_wrong_lt_noise": coherence.get("fraction_same_lt_wrong_lt_noise"),
                    "gap_same_wrong": coherence.get("gap_same_wrong"),
                    "energy_matrix_mrr": matrix.get("mrr"),
                    "energy_matrix_diagonal_top1_accuracy": matrix.get("diagonal_top1_accuracy"),
                    "energy_matrix_diagonal_vs_offdiag_gap": matrix.get("diagonal_vs_offdiag_gap"),
                    "missing_outputs": ",".join(missing),
                }
            )
    return rows


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("split", "")), str(row.get("model_type", "")), str(row.get("variant", "")))
        groups.setdefault(key, []).append(row)
    summary: list[dict[str, Any]] = []
    for (split, model_type, variant), group in sorted(groups.items()):
        record: dict[str, Any] = {
            "split": split,
            "model_type": model_type,
            "variant": variant,
            "n": len(group),
            "seeds": ",".join(str(row.get("seed", "")) for row in sorted(group, key=lambda item: str(item.get("seed", "")))),
        }
        for metric in METRICS:
            values = [_float_or_none(row.get(metric)) for row in group]
            clean = [value for value in values if value is not None]
            mean, std = _mean_std(clean)
            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = std
            record[f"{metric}_n"] = len(clean)
        summary.append(record)
    return summary


def _write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = ["split", "model_type", "variant", "n", "seeds"]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_n"])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_runs_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "run_name",
        "split",
        "model_type",
        "variant",
        "negative_set",
        "seed",
        "checkpoint",
        *METRICS,
        "missing_outputs",
        "negative_types",
        "run",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    numeric = _float_or_none(value)
    return "" if numeric is None else f"{numeric:.4f}"


def _write_markdown(summary: list[dict[str, Any]], raw_rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "split",
        "variant",
        "n",
        "top1_acc",
        "law_pair",
        "law_gap",
        "law_only_top1",
        "same_lt_wrong",
        "stratified",
        "matrix_mrr",
        "matrix_top1",
        "matrix_gap",
    ]
    lines = [
        "# Paper 1 Direct-Context OMC Backbone Summary",
        "",
        "Mean +/- std across seeds.",
        "",
        "|" + "|".join(columns) + "|",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    metric_for_col = {
        "top1_acc": "top1_acc",
        "law_pair": "law_pair",
        "law_gap": "law_gap",
        "law_only_top1": "law_only_top1",
        "same_lt_wrong": "fraction_same_lt_wrong",
        "stratified": "fraction_same_lt_wrong_lt_noise",
        "matrix_mrr": "energy_matrix_mrr",
        "matrix_top1": "energy_matrix_diagonal_top1_accuracy",
        "matrix_gap": "energy_matrix_diagonal_vs_offdiag_gap",
    }
    for row in summary:
        values = [str(row.get("split", "")), str(row.get("variant", "")), str(row.get("n", ""))]
        for col in columns[3:]:
            metric = metric_for_col[col]
            values.append(f"{row.get(metric + '_mean', 0.0):.4f} +/- {row.get(metric + '_std', 0.0):.4f}")
        lines.append("|" + "|".join(values) + "|")

    missing = [row for row in raw_rows if row.get("missing_outputs")]
    if missing:
        lines.extend(["", "Missing analysis outputs:"])
        for row in missing:
            lines.append(f"- {row.get('run_name')} / {row.get('split')}: {row.get('missing_outputs')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_p1_backbone_omc(
    sweep_dir: Path,
    out_dir: Path,
    splits: list[str] | None = None,
    ranking_checkpoint: str = "best_law_pair.pt",
    law_checkpoint: str = "best_law_pair.pt",
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    split_names = splits or ["test_iid", "test_ood_param"]
    raw_rows = collect_backbone_omc_rows(sweep_dir, split_names, ranking_checkpoint, law_checkpoint)
    summary = summarize_rows(raw_rows)
    _write_summary_csv(summary, out_dir / "backbone_omc_summary.csv")
    _write_runs_csv(raw_rows, out_dir / "backbone_omc_runs.csv")
    _write_markdown(summary, raw_rows, out_dir / "backbone_omc_summary.md")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", type=str, nargs="*", default=None)
    parser.add_argument("--ranking-checkpoint", type=str, default="best_law_pair.pt")
    parser.add_argument("--law-checkpoint", type=str, default="best_law_pair.pt")
    args = parser.parse_args()
    rows = aggregate_p1_backbone_omc(
        args.sweep_dir,
        args.out,
        splits=args.splits,
        ranking_checkpoint=args.ranking_checkpoint,
        law_checkpoint=args.law_checkpoint,
    )
    print(f"aggregated {len(rows)} direct-context OMC summary rows into {args.out}")


if __name__ == "__main__":
    main()
