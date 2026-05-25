"""Aggregate CoPhy OMC/no-law ranking evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml


METRICS = (
    "generic_top1",
    "same_lt_wrong",
    "wrong_lt_noise",
    "full_stratification",
    "wrong_confounder_pair_acc",
    "confounder_only_top1",
    "energy_matrix_mrr",
    "energy_matrix_top1",
    "diagonal_vs_offdiag_gap",
    "gap_same_wrong",
    "gap_wrong_noise",
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


def _metadata(run_dir: Path) -> dict[str, Any]:
    config = _load_yaml(run_dir / "config.yaml")
    params = dict(config.get("sweep_params", {}))
    training = dict(config.get("training", {}))
    data = dict(config.get("data", {}))
    model_type = str(params.get("model_type", training.get("baseline", "lowm")))
    negative_types = params.get("negative_types", config.get("ranking", {}).get("negative_types", []))
    if not isinstance(negative_types, list):
        negative_types = []
    variant = str(params.get("variant", params.get("negative_set", run_dir.name)))
    objective = "OMC" if "law_mismatch" in {str(name) for name in negative_types} and "no_law" not in variant.lower() else "no_law_mismatch"
    return {
        "run_name": run_dir.name,
        "run": str(run_dir),
        "model_type": model_type,
        "backbone": "direct_context" if model_type == "direct_context_energy" else "LOWM" if model_type == "lowm" else model_type,
        "variant": variant,
        "objective": objective,
        "negative_types": negative_types,
        "seed": params.get("seed", training.get("seed", "")),
        "scenario": params.get("scenario", Path(str(data.get("root", ""))).name),
        "mode": params.get("mode", data.get("mode", "state/object-or-feature")),
    }


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if math.isfinite(value_f) else None


def collect_cophy_rows(sweep_dir: Path, splits: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in _run_dirs(sweep_dir):
        meta = _metadata(run_dir)
        for split in splits:
            metrics = _load_json(run_dir / "eval" / split / "cophy_ranking" / "metrics.json")
            missing = "" if metrics else "cophy_ranking"
            rows.append(
                {
                    **meta,
                    "split": split,
                    "scenario": metrics.get("scenario", meta["scenario"]),
                    **{metric: metrics.get(metric) for metric in METRICS},
                    "missing_outputs": missing,
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


def summarize_cophy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("scenario", "")), str(row.get("backbone", "")), str(row.get("objective", "")), str(row.get("split", "")))
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (scenario, backbone, objective, split), group in sorted(groups.items()):
        record: dict[str, Any] = {
            "scenario": scenario,
            "backbone": backbone,
            "objective": objective,
            "split": split,
            "n": len(group),
            "seeds": ",".join(str(row.get("seed", "")) for row in sorted(group, key=lambda item: str(item.get("seed", "")))),
            "mode": ",".join(sorted({str(row.get("mode", "")) for row in group})),
        }
        for metric in METRICS:
            clean = [value for value in (_float_or_none(row.get(metric)) for row in group) if value is not None]
            mean, std = _mean_std(clean)
            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = std
            record[f"{metric}_n"] = len(clean)
        record["missing_outputs"] = ",".join(sorted({str(row.get("missing_outputs")) for row in group if row.get("missing_outputs")}))
        out.append(record)
    return out


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = ["scenario", "backbone", "objective", "split", "n", "seeds", "mode", "missing_outputs"]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_n"])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(row: dict[str, Any], metric: str) -> str:
    if int(row.get(f"{metric}_n", 0) or 0) == 0:
        return "NA"
    return f"{row.get(metric + '_mean', 0.0):.4f} +/- {row.get(metric + '_std', 0.0):.4f}"


def _write_md(summary: list[dict[str, Any]], raw_rows: list[dict[str, Any]], path: Path) -> None:
    columns = ["scenario", "backbone", "objective", "split", "n", "generic_top1", "same_lt_wrong", "full_stratification", "matrix_mrr", "matrix_top1"]
    lines = ["# CoPhy OMC Summary", "", "|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in summary:
        values = []
        for col in columns:
            values.append(_fmt(row, "energy_matrix_mrr" if col == "matrix_mrr" else "energy_matrix_top1" if col == "matrix_top1" else col) if col in METRICS or col in {"matrix_mrr", "matrix_top1"} else str(row.get(col, "")))
        lines.append("|" + "|".join(values) + "|")
    missing = [row for row in raw_rows if row.get("missing_outputs")]
    if missing:
        lines.extend(["", "Missing outputs:"])
        for row in missing:
            lines.append(f"- {row.get('run_name')} / {row.get('split')}: {row.get('missing_outputs')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _delta(summary: list[dict[str, Any]], scenario: str, backbone: str, split: str, metric: str) -> float | None:
    omc = next((row for row in summary if row["scenario"] == scenario and row["backbone"] == backbone and row["split"] == split and row["objective"] == "OMC"), None)
    no_law = next((row for row in summary if row["scenario"] == scenario and row["backbone"] == backbone and row["split"] == split and row["objective"] == "no_law_mismatch"), None)
    if not omc or not no_law:
        return None
    if int(omc.get(f"{metric}_n", 0) or 0) == 0 or int(no_law.get(f"{metric}_n", 0) or 0) == 0:
        return None
    return float(omc.get(f"{metric}_mean", 0.0)) - float(no_law.get(f"{metric}_mean", 0.0))


def _any_positive(summary: list[dict[str, Any]], metric: str) -> bool:
    for scenario in sorted({row["scenario"] for row in summary}):
        for backbone in sorted({row["backbone"] for row in summary}):
            for split in sorted({row["split"] for row in summary}):
                delta = _delta(summary, scenario, backbone, split, metric)
                if delta is not None and delta > 0:
                    return True
    return False


def _write_checklist(summary: list[dict[str, Any]], raw_rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# CoPhy Claim Checklist",
        "",
        f"1. Does OMC improve same-confounder vs wrong-confounder discrimination? {'Supported' if _any_positive(summary, 'same_lt_wrong') else 'Not established from available files'}.",
        f"2. Does no_law_mismatch show generic top1 but fail wrong-confounder rejection? {'Check table: generic top1 and same_lt_wrong are reported side by side' if summary else 'No rows available'}.",
        f"3. Does OMC improve coherence stratification? {'Supported' if _any_positive(summary, 'full_stratification') else 'Not established from available files'}.",
        f"4. Does the result hold across scenarios? {'Multiple scenarios present' if len({row.get('scenario') for row in summary}) > 1 else 'Only one or zero scenarios present'}.",
        f"5. Does the result hold for LOWM and direct_context? {'Both backbones present' if {'LOWM', 'direct_context'}.issubset({row.get('backbone') for row in summary}) else 'Backbone coverage incomplete'}.",
        "6. Is this state/object-level, feature-level, or visual-level? See the `mode` column; this milestone does not train a raw visual model.",
        "7. What limitations remain? Raw visual-only CoPhy needs precomputed object states/features or a separate visual encoder; labels are used only for sampling/evaluation.",
    ]
    if any(row.get("missing_outputs") for row in raw_rows):
        lines.append("Some expected cophy_ranking outputs were missing and are marked in the summary.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_cophy_omc(sweep_dir: Path, out_dir: Path, splits: list[str] | None = None) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    split_names = splits or ["val", "test"]
    raw_rows = collect_cophy_rows(sweep_dir, split_names)
    summary = summarize_cophy_rows(raw_rows)
    _write_csv(summary, out_dir / "cophy_omc_summary.csv")
    _write_md(summary, raw_rows, out_dir / "cophy_omc_summary.md")
    _write_checklist(summary, raw_rows, out_dir / "cophy_claim_checklist.md")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", type=str, nargs="*", default=None)
    args = parser.parse_args()
    rows = aggregate_cophy_omc(args.sweep_dir, args.out, args.splits)
    print(f"aggregated {len(rows)} CoPhy summary rows into {args.out}")


if __name__ == "__main__":
    main()
