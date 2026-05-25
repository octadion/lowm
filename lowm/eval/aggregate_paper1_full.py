"""Combine synthetic/OOD and CoPhy external Paper 1 summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from lowm.eval.aggregate_cophy_omc import aggregate_cophy_omc
from lowm.eval.aggregate_paper1_current import aggregate_paper1_current


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_md(rows: list[dict[str, Any]], path: Path) -> None:
    cols = ["domain", "scenario", "backbone", "objective", "split", "n", "primary_metric", "secondary_metric"]
    lines = ["# Paper 1 Full Summary", "", "|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(_fmt(row.get(col, "")) for col in cols) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_external_table(rows: list[dict[str, Any]], path: Path) -> None:
    cophy = [row for row in rows if row.get("domain") == "CoPhy"]
    cols = ["scenario", "backbone", "objective", "split", "same_lt_wrong", "full_stratification", "energy_matrix_mrr"]
    lines = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in cophy:
        lines.append("|" + "|".join(_fmt(row.get(col, "")) for col in cols) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_backbone_table(rows: list[dict[str, Any]], path: Path) -> None:
    cols = ["domain", "backbone", "objective", "split", "primary_metric", "secondary_metric"]
    lines = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(_fmt(row.get(col, "")) for col in cols) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_checklist(path: Path, has_cophy: bool) -> None:
    lines = [
        "# Final Paper 1 Claim Checklist",
        "",
        "1. Operator-blindness: synthetic/OOD rows compare generic top1 with law/operator metrics.",
        "2. OMC objective: compare OMC vs no_law_mismatch within each backbone.",
        "3. Relational scoring: energy-matrix and coherence-stratification tables are the central evidence.",
        "4. Standalone lambda decodability: not claimed.",
        f"5. External wrong-confounder rejection: {'included via CoPhy rows' if has_cophy else 'not available yet; run CoPhy pipeline first'}.",
        "6. Architecture scope: this is an objective/signal paper, not a new general world-model architecture claim.",
        "7. Human-like imagination: not claimed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_paper1_full(
    lowm_root: Path,
    direct_root: Path,
    cophy_sweep_dir: Path | None,
    out_dir: Path,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    synthetic = aggregate_paper1_current(lowm_root, direct_root, out_dir / "synthetic_current")
    rows: list[dict[str, Any]] = []
    for row in synthetic:
        rows.append(
            {
                "domain": "LOWM-Synth",
                "scenario": "synthetic",
                "backbone": row.get("backbone", ""),
                "objective": row.get("objective", ""),
                "split": row.get("split", ""),
                "n": row.get("n", 0),
                "primary_metric": row.get("law_pair_mean", 0.0),
                "secondary_metric": row.get("energy_matrix_mrr_mean", 0.0),
                "law_pair": row.get("law_pair_mean", 0.0),
                "energy_matrix_mrr": row.get("energy_matrix_mrr_mean", 0.0),
            }
        )
    cophy_rows: list[dict[str, Any]] = []
    if cophy_sweep_dir is not None and cophy_sweep_dir.exists():
        cophy_summary = aggregate_cophy_omc(cophy_sweep_dir, out_dir / "cophy")
        for row in cophy_summary:
            cophy_rows.append(
                {
                    "domain": "CoPhy",
                    "scenario": row.get("scenario", ""),
                    "backbone": row.get("backbone", ""),
                    "objective": row.get("objective", ""),
                    "split": row.get("split", ""),
                    "n": row.get("n", 0),
                    "primary_metric": row.get("same_lt_wrong_mean", 0.0),
                    "secondary_metric": row.get("energy_matrix_mrr_mean", 0.0),
                    "same_lt_wrong": row.get("same_lt_wrong_mean", 0.0),
                    "full_stratification": row.get("full_stratification_mean", 0.0),
                    "energy_matrix_mrr": row.get("energy_matrix_mrr_mean", 0.0),
                }
            )
    rows.extend(cophy_rows)
    _write_csv(rows, out_dir / "paper1_full_summary.csv")
    _write_md(rows, out_dir / "paper1_full_summary.md")
    _write_external_table(rows, out_dir / "table_external_cophy.md")
    _write_backbone_table(rows, out_dir / "table_backbone_generalization.md")
    _write_checklist(out_dir / "final_claim_checklist.md", has_cophy=bool(cophy_rows))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lowm_root", type=Path, required=True)
    parser.add_argument("--direct_root", type=Path, required=True)
    parser.add_argument("--cophy_sweep_dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = aggregate_paper1_full(args.lowm_root, args.direct_root, args.cophy_sweep_dir, args.out)
    print(f"aggregated {len(rows)} Paper 1 full rows into {args.out}")


if __name__ == "__main__":
    main()
