"""Aggregate hybrid EBTWM shaping sweep outputs."""

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


def _eval_summary(run: Path, split: str, checkpoint: str) -> dict[str, Any]:
    return _load_json(run / "eval" / split / Path(checkpoint).stem / "eval_summary.json")


def _law_only(run: Path, split: str, checkpoint: str) -> dict[str, Any]:
    stem = Path(checkpoint).stem
    return _load_json(run / "eval" / split / f"law_mismatch_only_{stem}" / "law_mismatch_only_metrics.json")


def _ebtwm(run: Path, split: str) -> dict[str, Any]:
    return _load_json(run / "eval" / split / "ebtwm_inference" / "metrics.json")


def collect_rows(sweep_dir: Path, split: str = "val", checkpoint: str = "best_law_pair.pt") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in _run_dirs(sweep_dir):
        config = _load_yaml(run / "config.yaml")
        params = dict(config.get("sweep_params", {}))
        training = dict(config.get("training", {}))
        metrics = _load_json(run / "metrics.json")
        final_val = metrics.get("final_val", {}) if isinstance(metrics.get("final_val", {}), dict) else {}
        loss_terms = final_val.get("loss_terms", {}) if isinstance(final_val.get("loss_terms", {}), dict) else {}
        summary = _eval_summary(run, split, checkpoint)
        ranking = summary.get("ranking", {}) if isinstance(summary.get("ranking", {}), dict) else {}
        law = _law_only(run, split, checkpoint)
        ebtwm = _ebtwm(run, split)
        row = {
            "run_name": run.name,
            "run": str(run),
            "split": split,
            "checkpoint": checkpoint,
            "alpha_dsm": params.get("alpha_dsm", training.get("alpha_dsm")),
            "alpha_denoise_rank": params.get("alpha_denoise_rank", training.get("alpha_denoise_rank")),
            "use_grad_reg": params.get("use_grad_reg", training.get("use_grad_reg")),
            "seed": params.get("seed", training.get("seed")),
            "top1_acc": ranking.get("top1_acc", final_val.get("top1_acc", 0.0)),
            "law_pair": ranking.get("law_pair", final_val.get("law_pair", 0.0)),
            "law_gap": ranking.get("law_gap", final_val.get("law_gap", 0.0)),
            "law_only_top1": law.get("law_only_top1", law.get("top1_law_only", 0.0)),
            "dsm_loss": loss_terms.get("dsm_loss", 0.0),
            "clean_noisy_pair_acc": loss_terms.get("clean_noisy_pair_acc", 0.0),
            "clean_noisy_gap": loss_terms.get("clean_noisy_gap", 0.0),
            "fraction_energy_decreased": ebtwm.get("fraction_energy_decreased", 0.0),
            "fraction_mse_improved": ebtwm.get("fraction_mse_improved", 0.0),
            "mean_mse_before": ebtwm.get("mean_mse_to_gt_before", 0.0),
            "mean_mse_after": ebtwm.get("mean_mse_to_gt_after", 0.0),
            "own_vs_wrong_pair_acc_after": ebtwm.get("own_vs_wrong_pair_acc_after", 0.0),
            "go_no_go_decision": ebtwm.get("go_no_go_decision", {}).get("verdict", "MISSING") if isinstance(ebtwm.get("go_no_go_decision", {}), dict) else "MISSING",
        }
        rows.append(row)
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "run_name",
        "alpha_dsm",
        "alpha_denoise_rank",
        "use_grad_reg",
        "seed",
        "top1_acc",
        "law_pair",
        "law_gap",
        "law_only_top1",
        "clean_noisy_pair_acc",
        "dsm_loss",
        "fraction_energy_decreased",
        "fraction_mse_improved",
        "mean_mse_before",
        "mean_mse_after",
        "own_vs_wrong_pair_acc_after",
        "go_no_go_decision",
        "run",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_md(rows: list[dict[str, Any]], path: Path) -> None:
    columns = ["run_name", "alpha_dsm", "law_pair", "fraction_mse_improved", "mean_mse_after", "own_vs_wrong_pair_acc_after", "go_no_go_decision"]
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            vals.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("|" + "|".join(vals) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(rows: list[dict[str, Any]], y_key: str, ylabel: str, path: Path) -> None:
    if not rows:
        return
    xs = [float(row.get("alpha_dsm", 0.0) or 0.0) for row in rows]
    ys = [float(row.get(y_key, 0.0) or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    ax.scatter(xs, ys)
    ax.plot(xs, ys, alpha=0.4)
    ax.set_xlabel("alpha_dsm")
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def aggregate_ebtwm_shaping(sweep_dir: Path, out_dir: Path, split: str = "val", checkpoint: str = "best_law_pair.pt") -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(sweep_dir, split=split, checkpoint=checkpoint)
    _write_csv(rows, out_dir / "ebtwm_shaping_summary.csv")
    _write_md(rows, out_dir / "ebtwm_shaping_summary.md")
    _plot(rows, "fraction_mse_improved", "Fraction MSE improved", out_dir / "alpha_vs_mse_improvement.png")
    _plot(rows, "law_pair", "Law pair", out_dir / "alpha_vs_law_pair.png")
    go_map = {"NO-GO": 0, "AMBIGUOUS": 1, "WEAK GO": 2, "STRONG GO": 3, "MISSING": -1}
    go_rows = [dict(row, go_score=go_map.get(str(row.get("go_no_go_decision", "MISSING")), -1)) for row in rows]
    _plot(go_rows, "go_score", "Go/no-go score", out_dir / "alpha_vs_go_no_go.png")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--checkpoint", type=str, default="best_law_pair.pt")
    args = parser.parse_args()
    rows = aggregate_ebtwm_shaping(args.sweep_dir, args.out, args.split, args.checkpoint)
    print(f"aggregated {len(rows)} EBTWM shaping runs into {args.out}")


if __name__ == "__main__":
    main()
