"""Evaluate ranking with law-mismatch negatives only."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from lowm.data.dataset import LOWMSynthRankingDataset, make_ranking_dataloader, ranking_config_from_mapping
from lowm.eval.evaluate_all import _ensure_split, _move_batch_to_device, _resolve_checkpoint_path, _score_model, load_run_model
from lowm.eval.metrics import METRIC_VERSION, RankingMetricAccumulator
from lowm.training.losses import nce_ranking_loss


def evaluate_law_mismatch_only(
    run_dir: Path,
    split: str = "val",
    checkpoint_name: str = "best_law_pair.pt",
    model_type: str | None = None,
    batch_size: int | None = None,
    num_samples: int | None = None,
    seed: int | None = None,
    device_name: str = "auto",
) -> dict[str, Any]:
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    checkpoint_path = _resolve_checkpoint_path(run_dir, checkpoint_name)
    model, config, detected = load_run_model(run_dir, model_type=model_type, checkpoint_name=checkpoint_name, device=device)
    split_path = _ensure_split(config, split)
    ranking_cfg = ranking_config_from_mapping(config)
    ranking_cfg = replace(ranking_cfg, negative_types=("law_mismatch",), seed=int(seed if seed is not None else ranking_cfg.seed))
    eval_cfg = dict(config.get("evaluation", {}))
    sample_count = num_samples if num_samples is not None else eval_cfg.get("num_samples", config.get("training", {}).get("val_samples"))
    dataset = LOWMSynthRankingDataset(split_path, ranking_cfg, num_samples=int(sample_count) if sample_count else None)
    bs = int(batch_size or config.get("training", {}).get("batch_size", 64))

    acc = RankingMetricAccumulator()
    loader = make_ranking_dataloader(dataset, batch_size=bs, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            energies = _score_model(model, batch)
            loss = nce_ranking_loss(energies, batch["labels"])
            acc.update(energies, batch["labels"], batch["negative_types"], float(loss.item()))
    metrics = acc.compute()
    result = {
        "model_type": detected,
        "split": split,
        "checkpoint_requested": checkpoint_name,
        "checkpoint_used": checkpoint_path.name,
        "checkpoint_stem": checkpoint_path.stem,
        "ranking_seed": ranking_cfg.seed,
        "num_samples": len(dataset),
        "metric_version": METRIC_VERSION,
        "top1_law_only": metrics["top1_acc"],
        "pairwise_acc_law_only": metrics["law_pair"],
        "mean_law_gap": metrics["law_gap"],
        "mrr_law_only": metrics["mrr"],
        "loss": metrics.get("loss", 0.0),
        "raw": metrics,
    }

    out_dir = run_dir / "eval" / split / f"law_mismatch_only_{checkpoint_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "law_mismatch_only_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    legacy = run_dir / "eval" / split / "law_mismatch_only_metrics.json"
    legacy.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--checkpoint", type=str, default="best_law_pair.pt")
    parser.add_argument("--model_type", type=str, default=None, choices=["fixed_energy", "direct_context_energy", "lowm"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    result = evaluate_law_mismatch_only(
        args.run,
        split=args.split,
        checkpoint_name=args.checkpoint,
        model_type=args.model_type,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        seed=args.seed,
        device_name=args.device,
    )
    print(json.dumps({key: result[key] for key in ["top1_law_only", "pairwise_acc_law_only", "mean_law_gap", "mrr_law_only"]}, indent=2))


if __name__ == "__main__":
    main()
