"""Compare training validation metrics with evaluate_all outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from lowm.eval.metrics import METRIC_VERSION


KEYS = (
    "top1_acc",
    "mean_rank",
    "mrr",
    "law_pair",
    "law_gap",
    "state_corrupted_pair_acc",
    "temporal_shuffled_pair_acc",
    "law_mismatch_pair_acc",
    "random_impossible_pair_acc",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return loaded


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return loaded if isinstance(loaded, dict) else {}


def _metric(metrics: Mapping[str, Any], key: str) -> float | None:
    if key in metrics:
        try:
            return float(metrics[key])
        except (TypeError, ValueError):
            return None
    if key == "law_pair":
        return _metric(metrics.get("law_mismatch", {}), "pairwise_acc") if isinstance(metrics.get("law_mismatch"), Mapping) else None
    if key == "law_gap":
        return _metric(metrics.get("law_mismatch", {}), "mean_energy_gap") if isinstance(metrics.get("law_mismatch"), Mapping) else None
    if key.endswith("_pair_acc"):
        type_name = key.removesuffix("_pair_acc")
        by_type = metrics.get("by_negative_type", {})
        if isinstance(by_type, Mapping) and isinstance(by_type.get(type_name), Mapping):
            return _metric(by_type[type_name], "pairwise_acc")
    return None


def _best_val_by_top1(history: list[Mapping[str, Any]]) -> tuple[int | None, Mapping[str, Any]]:
    best_epoch: int | None = None
    best_metrics: Mapping[str, Any] = {}
    best_top1 = float("-inf")
    for record in history:
        val = record.get("val", {})
        if not isinstance(val, Mapping):
            continue
        top1 = _metric(val, "top1_acc")
        if top1 is not None and top1 >= best_top1:
            best_top1 = top1
            best_metrics = val
            best_epoch = int(record.get("epoch", -1))
    return best_epoch, best_metrics


def compare_train_eval_metrics(run_dir: Path, split: str = "val") -> dict[str, Any]:
    train_path = run_dir / "metrics.json"
    eval_path = run_dir / "eval" / split / "eval_summary.json"
    if not train_path.exists():
        raise FileNotFoundError(f"missing training metrics: {train_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"missing evaluate_all summary: {eval_path}")

    train = _load_json(train_path)
    eval_summary = _load_json(eval_path)
    config = _load_yaml(run_dir / "config.yaml")
    history = train.get("history", [])
    history = history if isinstance(history, list) else []
    final_val = train.get("final_val", {})
    final_val = final_val if isinstance(final_val, Mapping) else {}
    best_epoch, best_val = _best_val_by_top1(history)
    eval_metrics = eval_summary.get("ranking", {})
    eval_metrics = eval_metrics if isinstance(eval_metrics, Mapping) else {}

    print(f"Run: {run_dir}")
    print(f"Split: {split}")
    print(f"evaluate_all checkpoint: requested={eval_summary.get('checkpoint_requested')} used={eval_summary.get('checkpoint_used')}")
    print(f"evaluate_all seed={eval_summary.get('ranking_seed')} num_samples={eval_summary.get('num_samples')}")
    print("")
    print("| metric | train_final | train_best_top1 | evaluate_all |")
    print("|---|---:|---:|---:|")
    rows: list[dict[str, Any]] = []
    for key in KEYS:
        final_value = _metric(final_val, key)
        best_value = _metric(best_val, key)
        eval_value = _metric(eval_metrics, key)
        rows.append({"metric": key, "train_final": final_value, "train_best_top1": best_value, "evaluate_all": eval_value})
        print(
            f"| {key} | "
            f"{final_value if final_value is not None else 'n/a'} | "
            f"{best_value if best_value is not None else 'n/a'} | "
            f"{eval_value if eval_value is not None else 'n/a'} |"
        )

    warnings: list[str] = []
    train_metric_version = final_val.get("metric_version") or best_val.get("metric_version")
    eval_metric_version = eval_summary.get("metric_version") or eval_metrics.get("metric_version")
    if train_metric_version and eval_metric_version and train_metric_version != eval_metric_version:
        warnings.append(f"metric_version differs: train={train_metric_version} eval={eval_metric_version}")
    if not train_metric_version:
        warnings.append("training metrics do not record metric_version; rerun training for exact metric provenance")
    if eval_metric_version != METRIC_VERSION:
        warnings.append(f"evaluate_all metric_version is {eval_metric_version}, expected {METRIC_VERSION}")

    config_seed = config.get("ranking", {}).get("seed") if isinstance(config.get("ranking"), Mapping) else None
    if config_seed is not None and eval_summary.get("ranking_seed") is not None and int(config_seed) != int(eval_summary["ranking_seed"]):
        warnings.append(f"ranking seed differs: training config={config_seed} eval={eval_summary['ranking_seed']}")
    val_samples = config.get("training", {}).get("val_samples") if isinstance(config.get("training"), Mapping) else None
    if val_samples is not None and eval_summary.get("num_samples") is not None and int(val_samples) != int(eval_summary["num_samples"]):
        warnings.append(f"num_samples differs: training val_samples={val_samples} eval={eval_summary['num_samples']}")
    if eval_summary.get("checkpoint_used") == "best.pt" and best_epoch is not None:
        warnings.append(f"evaluate_all used best.pt; compare against train_best_top1 epoch {best_epoch}, not necessarily final_val")
    if eval_summary.get("checkpoint_requested") != eval_summary.get("checkpoint_used"):
        warnings.append("requested checkpoint was not found; evaluate_all fell back to a different checkpoint")

    if warnings:
        print("")
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    return {"rows": rows, "warnings": warnings}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    args = parser.parse_args()
    compare_train_eval_metrics(args.run, args.split)


if __name__ == "__main__":
    main()
