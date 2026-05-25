"""Aggregate current Paper 1 OMC results into paper-facing tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml

from lowm.eval.aggregate_results import collect_run


SPLITS = ("test_iid", "test_ood_param")
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
    "linear_op_id_probe_accuracy",
    "mlp_op_id_probe_accuracy",
    "linear_binned_param_accuracy",
    "mlp_binned_param_accuracy",
    "linear_op_param_r2",
    "mlp_op_param_r2",
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _run_dirs(root: Path) -> list[Path]:
    manifest = _load_json(root / "manifest.json")
    if isinstance(manifest.get("runs"), list):
        return [Path(path) for path in manifest["runs"]]
    runs_root = root / "runs"
    if runs_root.exists():
        return sorted(path for path in runs_root.iterdir() if path.is_dir())
    if (root / "config.yaml").exists():
        return [root]
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "config.yaml").exists()) if root.exists() else []


def _metadata(run_dir: Path) -> dict[str, Any]:
    config = _load_yaml(run_dir / "config.yaml")
    params = dict(config.get("sweep_params", {}))
    training = dict(config.get("training", {}))
    metadata = _load_json(run_dir / "metadata.json")
    model_type = str(
        params.get(
            "model_type",
            training.get("baseline", "lowm" if "LOWM" in str(metadata.get("model", "")) or "lambda_dim" in dict(config.get("model", {})) else ""),
        )
    )
    negative_types = params.get("negative_types", config.get("ranking", {}).get("negative_types", []))
    if isinstance(negative_types, tuple):
        negative_types = list(negative_types)
    if not isinstance(negative_types, list):
        negative_types = []
    variant = str(params.get("variant", params.get("component", params.get("negative_set", run_dir.name))))
    negative_set = str(params.get("negative_set", ""))
    lower = " ".join([variant, negative_set, run_dir.name]).lower()
    if "no_law" in lower or "law_mismatch" not in {str(item) for item in negative_types}:
        objective = "no_law_mismatch"
    else:
        objective = "OMC"
    if model_type == "direct_context_energy" or "direct_context" in run_dir.name.lower():
        backbone = "direct_context"
        model_type = "direct_context_energy"
    elif model_type == "lowm" or "lowm" in run_dir.name.lower():
        backbone = "LOWM"
        model_type = "lowm"
    else:
        backbone = model_type or run_dir.name
    return {
        "run_name": run_dir.name,
        "run": str(run_dir),
        "model_type": model_type,
        "backbone": backbone,
        "variant": variant,
        "objective": objective,
        "negative_set": negative_set,
        "negative_types": negative_types,
        "seed": params.get("seed", training.get("seed", "")),
    }


def _matches_source(meta: dict[str, Any], source: str) -> bool:
    if source == "lowm":
        return meta.get("model_type") == "lowm"
    if source == "direct":
        return meta.get("model_type") == "direct_context_energy"
    return True


def _law_only_metrics(run_dir: Path, split: str, checkpoint: str) -> dict[str, Any]:
    stem = Path(checkpoint).stem
    candidates = [
        run_dir / "eval" / split / f"law_mismatch_only_{stem}" / "law_mismatch_only_metrics.json",
        run_dir / "eval" / split / "law_mismatch_only_metrics.json",
    ]
    for path in candidates:
        metrics = _load_json(path)
        if metrics:
            return metrics
    return {}


def _probe_metrics(run_dir: Path, split: str) -> dict[str, Any]:
    metrics = _load_json(run_dir / "eval" / split / "operator_probe" / "metrics.json")
    if metrics:
        return metrics
    return _load_json(run_dir / "eval" / "val" / "operator_probe" / "metrics.json")


def _safe_collect_run(run_dir: Path, split: str, checkpoint: str) -> tuple[dict[str, Any], str | None]:
    try:
        return collect_run(run_dir, split=split, checkpoint=checkpoint), None
    except FileNotFoundError:
        try:
            return collect_run(run_dir, split=split, checkpoint=None), None
        except FileNotFoundError as exc:
            return {}, str(exc)


def _collect_root_rows(root: Path, source: str, splits: list[str], checkpoint: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in _run_dirs(root):
        meta = _metadata(run_dir)
        if not _matches_source(meta, source):
            continue
        for split in splits:
            ranking, ranking_error = _safe_collect_run(run_dir, split, checkpoint)
            law = _law_only_metrics(run_dir, split, checkpoint)
            coherence = _load_json(run_dir / "eval" / split / "coherence_stratification" / "metrics.json")
            matrix = _load_json(run_dir / "eval" / split / "energy_matrix" / "metrics.json")
            probe = _probe_metrics(run_dir, split)
            missing = []
            if ranking_error:
                missing.append("evaluate_all")
            if not law:
                missing.append("law_mismatch_only")
            if not coherence:
                missing.append("coherence_stratification")
            if not matrix:
                missing.append("energy_matrix")
            if not probe:
                missing.append("operator_probe")
            rows.append(
                {
                    **meta,
                    "source_root": str(root),
                    "split": split,
                    "checkpoint": ranking.get("checkpoint", Path(checkpoint).stem),
                    "top1_acc": ranking.get("top1_acc"),
                    "law_pair": ranking.get("law_pair"),
                    "law_gap": ranking.get("law_gap"),
                    "law_only_top1": law.get("law_only_top1", law.get("top1_law_only")),
                    "fraction_same_lt_wrong": coherence.get("fraction_same_lt_wrong"),
                    "fraction_same_lt_wrong_lt_noise": coherence.get("fraction_same_lt_wrong_lt_noise"),
                    "gap_same_wrong": coherence.get("gap_same_wrong"),
                    "energy_matrix_mrr": matrix.get("mrr"),
                    "energy_matrix_diagonal_top1_accuracy": matrix.get("diagonal_top1_accuracy"),
                    "energy_matrix_diagonal_vs_offdiag_gap": matrix.get("diagonal_vs_offdiag_gap"),
                    "linear_op_id_probe_accuracy": probe.get("linear_op_id_probe_accuracy", probe.get("op_id_probe_accuracy")),
                    "mlp_op_id_probe_accuracy": probe.get("mlp_op_id_probe_accuracy"),
                    "linear_binned_param_accuracy": probe.get("linear_binned_param_accuracy", probe.get("binned_param_accuracy")),
                    "mlp_binned_param_accuracy": probe.get("mlp_binned_param_accuracy"),
                    "linear_op_param_r2": probe.get("linear_op_param_r2", probe.get("op_param_r2")),
                    "mlp_op_param_r2": probe.get("mlp_op_param_r2"),
                    "missing_outputs": ",".join(missing),
                }
            )
    return rows


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if math.isfinite(value_f) else None


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("backbone", "")), str(row.get("objective", "")), str(row.get("split", "")))
        groups.setdefault(key, []).append(row)
    summary: list[dict[str, Any]] = []
    for (backbone, objective, split), group in sorted(groups.items()):
        record: dict[str, Any] = {
            "backbone": backbone,
            "objective": objective,
            "split": split,
            "n": len(group),
            "seeds": ",".join(str(row.get("seed", "")) for row in sorted(group, key=lambda item: str(item.get("seed", "")))),
            "variants": ",".join(sorted({str(row.get("variant", "")) for row in group})),
        }
        for metric in METRICS:
            clean = [value for value in (_float_or_none(row.get(metric)) for row in group) if value is not None]
            mean, std = _mean_std(clean)
            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = std
            record[f"{metric}_n"] = len(clean)
        missing = sorted({item for row in group for item in str(row.get("missing_outputs", "")).split(",") if item})
        record["missing_outputs"] = ",".join(missing)
        summary.append(record)
    return summary


def _write_summary_csv(summary: list[dict[str, Any]], path: Path) -> None:
    fieldnames = ["backbone", "objective", "split", "n", "seeds", "variants", "missing_outputs"]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_n"])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)


def _fmt(value: Any) -> str:
    numeric = _float_or_none(value)
    return "NA" if numeric is None else f"{numeric:.4f}"


def _fmt_mean(row: dict[str, Any], metric: str) -> str:
    n = int(row.get(f"{metric}_n", 0) or 0)
    if n == 0:
        return "NA"
    return f"{row.get(metric + '_mean', 0.0):.4f} +/- {row.get(metric + '_std', 0.0):.4f}"


def _write_table(path: Path, rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> None:
    lines = ["|" + "|".join(label for label, _ in columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        values: list[str] = []
        for _, key in columns:
            if key.endswith("_mean"):
                values.append(_fmt(row.get(key)))
            elif key in METRICS:
                values.append(_fmt_mean(row, key))
            else:
                values.append(str(row.get(key, "")))
        lines.append("|" + "|".join(values) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary_md(summary: list[dict[str, Any]], raw_rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Paper 1 Current Summary",
        "",
        "Current aggregation excludes LOWM-G and external datasets. Metrics are mean +/- std across available seeds.",
        "",
    ]
    key_cols = [
        ("backbone", "backbone"),
        ("objective", "objective"),
        ("split", "split"),
        ("n", "n"),
        ("top1", "top1_acc"),
        ("law_pair", "law_pair"),
        ("law_only_top1", "law_only_top1"),
        ("same_lt_wrong", "fraction_same_lt_wrong"),
        ("matrix_mrr", "energy_matrix_mrr"),
    ]
    lines.extend(["|" + "|".join(label for label, _ in key_cols) + "|", "|" + "|".join(["---"] * len(key_cols)) + "|"])
    for row in summary:
        values = []
        for _, key in key_cols:
            values.append(_fmt_mean(row, key) if key in METRICS else str(row.get(key, "")))
        lines.append("|" + "|".join(values) + "|")
    missing = [row for row in raw_rows if row.get("missing_outputs")]
    if missing:
        lines.extend(["", "Missing outputs were tolerated:"])
        for row in missing[:40]:
            lines.append(f"- {row.get('run_name')} / {row.get('split')}: {row.get('missing_outputs')}")
        if len(missing) > 40:
            lines.append(f"- ... {len(missing) - 40} more rows omitted")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _by_key(summary: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {(str(row.get("backbone")), str(row.get("objective")), str(row.get("split"))): row for row in summary}


def _delta(summary: list[dict[str, Any]], backbone: str, split: str, metric: str) -> float | None:
    by_key = _by_key(summary)
    omc = by_key.get((backbone, "OMC", split))
    no_law = by_key.get((backbone, "no_law_mismatch", split))
    if not omc or not no_law:
        return None
    if int(omc.get(f"{metric}_n", 0) or 0) == 0 or int(no_law.get(f"{metric}_n", 0) or 0) == 0:
        return None
    return float(omc.get(f"{metric}_mean", 0.0)) - float(no_law.get(f"{metric}_mean", 0.0))


def _any_positive_delta(summary: list[dict[str, Any]], metric: str, split: str | None = None) -> bool:
    splits = [split] if split else sorted({str(row.get("split")) for row in summary})
    for backbone in sorted({str(row.get("backbone")) for row in summary}):
        for split_name in splits:
            delta = _delta(summary, backbone, split_name, metric)
            if delta is not None and delta > 0:
                return True
    return False


def _all_available_backbone_positive(summary: list[dict[str, Any]], metric: str, split: str) -> bool:
    checked = 0
    positive = 0
    for backbone in sorted({str(row.get("backbone")) for row in summary}):
        delta = _delta(summary, backbone, split, metric)
        if delta is None:
            continue
        checked += 1
        positive += int(delta > 0)
    return checked > 0 and checked == positive


def _write_claim_checklist(summary: list[dict[str, Any]], raw_rows: list[dict[str, Any]], path: Path) -> None:
    has_missing = any(row.get("missing_outputs") for row in raw_rows)
    top1_hide = _any_positive_delta(summary, "top1_acc") and _any_positive_delta(summary, "law_pair")
    law_fix = _any_positive_delta(summary, "law_pair") or _any_positive_delta(summary, "law_only_top1")
    strat = _any_positive_delta(summary, "fraction_same_lt_wrong") or _any_positive_delta(summary, "fraction_same_lt_wrong_lt_noise")
    matrix = _any_positive_delta(summary, "energy_matrix_mrr") or _any_positive_delta(summary, "energy_matrix_diagonal_top1_accuracy")
    generalizes = _all_available_backbone_positive(summary, "law_pair", "test_iid") or _all_available_backbone_positive(
        summary, "fraction_same_lt_wrong", "test_iid"
    )
    ood = _any_positive_delta(summary, "law_pair", "test_ood_param") or _any_positive_delta(summary, "fraction_same_lt_wrong", "test_ood_param")
    lines = [
        "# Paper 1 Claim Checklist",
        "",
        "1. Does generic top1 hide operator-blindness?",
        f"   - {'Yes' if top1_hide or law_fix else 'Insufficient from available files'}: compare generic top1 against law-only and law-pair rows for no_law_mismatch.",
        "2. Does OMC fix law/operator coherence?",
        f"   - {'Supported' if law_fix else 'Not established from available files'}: OMC rows improve law/operator metrics when deltas are positive.",
        "3. Does OMC stratify coherent-here vs plausible-elsewhere vs noise?",
        f"   - {'Supported' if strat else 'Not established from available files'}: use same<wrong and same<wrong<noise metrics.",
        "4. Does OMC improve relational energy matrix alignment?",
        f"   - {'Supported' if matrix else 'Not established from available files'}: use energy-matrix MRR, diagonal top1, and diagonal/off-diagonal gap.",
        "5. Does OMC improve standalone lambda decodability?",
        "   - Not claimed. Current probe tables are diagnostic only; the Paper 1 claim should stay on relational E(tau, lambda) scoring unless probe improvements are clear and consistent.",
        "6. Does OMC generalize across LOWM and direct-context backbones?",
        f"   - {'Supported for available backbones' if generalizes else 'Mixed or incomplete'}: compare OMC vs no_law_mismatch within each backbone, not against each other.",
        "7. Is the effect stable in the OOD parameter split?",
        f"   - {'Supported in available OOD rows' if ood else 'Mixed or incomplete'}: check test_ood_param deltas separately from test_iid.",
        "8. What are current limitations?",
        "   - No external/CoPhy result yet.",
        "   - LOWM-G is intentionally excluded from Paper 1 current aggregation.",
        "   - Probe results do not justify a standalone lambda-representation claim.",
        "   - Direct-context results may have seed variance and should be interpreted within-backbone.",
        f"   - {'Some expected files were missing and are marked in the summary.' if has_missing else 'All scanned rows had the expected files for their available metrics.'}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_tables(summary: list[dict[str, Any]], out_dir: Path) -> None:
    common = [("backbone", "backbone"), ("objective", "objective"), ("split", "split"), ("n", "n")]
    _write_table(
        out_dir / "table_operator_blindness.md",
        summary,
        common + [("top1", "top1_acc"), ("law_pair", "law_pair"), ("law_gap", "law_gap"), ("law_only_top1", "law_only_top1")],
    )
    _write_table(
        out_dir / "table_coherence_stratification.md",
        summary,
        common
        + [
            ("same_lt_wrong", "fraction_same_lt_wrong"),
            ("same_lt_wrong_lt_noise", "fraction_same_lt_wrong_lt_noise"),
            ("gap_same_wrong", "gap_same_wrong"),
        ],
    )
    _write_table(
        out_dir / "table_energy_matrix.md",
        summary,
        common
        + [
            ("mrr", "energy_matrix_mrr"),
            ("diagonal_top1", "energy_matrix_diagonal_top1_accuracy"),
            ("diag_offdiag_gap", "energy_matrix_diagonal_vs_offdiag_gap"),
        ],
    )
    _write_table(
        out_dir / "table_probe.md",
        summary,
        common
        + [
            ("linear_op_id", "linear_op_id_probe_accuracy"),
            ("mlp_op_id", "mlp_op_id_probe_accuracy"),
            ("linear_binned", "linear_binned_param_accuracy"),
            ("mlp_binned", "mlp_binned_param_accuracy"),
            ("linear_r2", "linear_op_param_r2"),
            ("mlp_r2", "mlp_op_param_r2"),
        ],
    )


def aggregate_paper1_current(
    lowm_root: Path,
    direct_root: Path,
    out_dir: Path,
    splits: list[str] | None = None,
    checkpoint: str = "best_law_pair.pt",
) -> list[dict[str, Any]]:
    split_names = splits or list(SPLITS)
    raw_rows = _collect_root_rows(lowm_root, "lowm", split_names, checkpoint)
    raw_rows.extend(_collect_root_rows(direct_root, "direct", split_names, checkpoint))
    summary = _summary_rows(raw_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(summary, out_dir / "paper1_current_summary.csv")
    _write_summary_md(summary, raw_rows, out_dir / "paper1_current_summary.md")
    _write_tables(summary, out_dir)
    _write_claim_checklist(summary, raw_rows, out_dir / "claim_checklist.md")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lowm_root", type=Path, required=True)
    parser.add_argument("--direct_root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", type=str, nargs="*", default=None)
    parser.add_argument("--checkpoint", type=str, default="best_law_pair.pt")
    args = parser.parse_args()
    rows = aggregate_paper1_current(args.lowm_root, args.direct_root, args.out, args.splits, args.checkpoint)
    print(f"aggregated {len(rows)} Paper 1 current summary rows into {args.out}")


if __name__ == "__main__":
    main()
