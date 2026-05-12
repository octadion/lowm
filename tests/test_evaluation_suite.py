from pathlib import Path

import torch
import yaml

from lowm.data.generate_dataset import generate_dataset
from lowm.eval.aggregate_results import aggregate_results
from lowm.eval.compare_train_eval_metrics import compare_train_eval_metrics
from lowm.eval.evaluate_all import evaluate_run
from lowm.eval.evaluate_occl_alignment import evaluate_occl_alignment
from lowm.eval.evaluate_law_mismatch_only import evaluate_law_mismatch_only
from lowm.eval.metrics import METRIC_VERSION
from lowm.models.baselines import BaselineConfig, DirectContextEnergyModel, FixedEnergyModel
from lowm.models.lowm import LOWM, LOWMConfig


def _make_data(tmp_path: Path) -> tuple[Path, Path]:
    config = {
        "dataset": "LOWM-Synth",
        "version": "eval-test",
        "seed": 91,
        "simulation": {"T": 10, "nmax": 6},
        "splits": {
            "train": {"num_episodes": 48, "n_min": 3, "n_max": 5, "seed": 91},
            "val": {"num_episodes": 32, "n_min": 3, "n_max": 5, "seed": 92},
        },
    }
    config_path = tmp_path / "data_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    out = tmp_path / "dataset"
    generate_dataset(config_path, out, ["train", "val"])
    return out, config_path


def _run_config(tmp_path: Path, data_root: Path, data_config: Path) -> dict:
    return {
        "seed": 5,
        "data": {
            "root": str(data_root),
            "train_split": "train.npz",
            "val_split": "val.npz",
            "generate_if_missing": False,
            "dataset_config": str(data_config),
        },
        "ranking": {"K": 4, "H": 4, "M": 5, "seed": 21},
        "model": {
            "object_dim": 7,
            "action_dim": 2,
            "lambda_dim": 8,
            "hidden_dim": 32,
            "token_dim": 32,
            "context_dim": 32,
            "num_layers": 2,
            "use_mu_eval": True,
        },
        "training": {"batch_size": 8, "val_samples": 16, "device": "cpu"},
        "evaluation": {"num_samples": 16, "disentanglement_samples": 8, "retrieval_queries": 8, "retrieval_pool_size": 4},
    }


def _write_run(run_dir: Path, config: dict, model_type: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    if model_type == "lowm":
        model = LOWM(LOWMConfig(lambda_dim=8, hidden_dim=32, context_dim=32, use_mu_eval=True))
        checkpoint = {"model_state": model.state_dict(), "config": config}
        metadata = {"model": "LOWM"}
    elif model_type == "direct_context_energy":
        model = DirectContextEnergyModel(BaselineConfig(hidden_dim=32, token_dim=32, context_dim=32))
        checkpoint = {"model_state": model.state_dict(), "config": config, "baseline": model_type}
        metadata = {"baseline": model_type}
    else:
        model = FixedEnergyModel(BaselineConfig(hidden_dim=32, token_dim=32, context_dim=32))
        checkpoint = {"model_state": model.state_dict(), "config": config, "baseline": "fixed_energy"}
        metadata = {"baseline": "fixed_energy"}
    torch.save(checkpoint, run_dir / "checkpoints" / "best.pt")
    for name in ["best_top1.pt", "best_loss.pt", "best_law_pair.pt", "best_law_gap.pt", "best_occl_acc.pt", "last.pt"]:
        torch.save(checkpoint, run_dir / "checkpoints" / name)
    import json

    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    training_metrics = {
        "history": [
            {
                "epoch": 1,
                "val": {
                    "metric_version": METRIC_VERSION,
                    "top1_acc": 0.25,
                    "mean_rank": 2.0,
                    "mrr": 0.5,
                    "law_pair": 0.5,
                    "law_gap": 0.1,
                    "law_mismatch_pair_acc": 0.5,
                },
            }
        ],
        "best_val_top1_acc": 0.25,
        "final_val": {
            "metric_version": METRIC_VERSION,
            "top1_acc": 0.25,
            "mean_rank": 2.0,
            "mrr": 0.5,
            "law_pair": 0.5,
            "law_gap": 0.1,
            "law_mismatch_pair_acc": 0.5,
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(training_metrics), encoding="utf-8")


def test_evaluate_all_outputs_metrics_and_plots(tmp_path: Path) -> None:
    data_root, data_config = _make_data(tmp_path)
    config = _run_config(tmp_path, data_root, data_config)
    run_dir = tmp_path / "runs" / "lowm_seed5"
    _write_run(run_dir, config, "lowm")

    summary = evaluate_run(run_dir, split="val", checkpoint_name="best_top1.pt", device_name="cpu", num_samples=12, batch_size=4)
    eval_dir = run_dir / "eval" / "val"

    assert summary["model_type"] == "lowm"
    assert "top1_acc" in summary["ranking"]
    assert "law_pair" in summary["ranking"]
    assert "law_gap" in summary["ranking"]
    assert "retrieval_acc" in summary["retrieval"]
    assert (eval_dir / "ranking_metrics.json").exists()
    assert (eval_dir / "negative_type_breakdown.csv").exists()
    assert (eval_dir / "debug_energies.csv").exists()
    assert (eval_dir / "disentanglement_matrix.csv").exists()
    assert (eval_dir / "retrieval_metrics.json").exists()
    assert (eval_dir / "plots" / "disentanglement_heatmap.png").exists()
    assert (eval_dir / "plots" / "pairwise_accuracy_by_negative_type.png").exists()
    assert (eval_dir / "best_top1" / "ranking_metrics.json").exists()
    assert "occl_alignment" in summary

    comparison = compare_train_eval_metrics(run_dir, split="val")
    assert "rows" in comparison

    occl = evaluate_occl_alignment(run_dir, split="val", checkpoint_name="best_occl_acc.pt", device_name="cpu", num_samples=12, batch_size=4)
    assert "tau_to_lambda_acc" in occl
    assert (eval_dir / "best_occl_acc" / "occl_alignment_metrics.json").exists()
    assert (eval_dir / "best_occl_acc" / "occl_energy_matrix.csv").exists()
    assert (eval_dir / "best_occl_acc" / "plots" / "occl_energy_matrix_heatmap.png").exists()

    law_only = evaluate_law_mismatch_only(run_dir, split="val", checkpoint_name="best_law_pair.pt", device_name="cpu", num_samples=12, batch_size=4)
    assert "top1_law_only" in law_only
    assert "pairwise_acc_law_only" in law_only
    assert (eval_dir / "law_mismatch_only_best_law_pair" / "law_mismatch_only_metrics.json").exists()


def test_aggregate_results_outputs_tables_and_plots(tmp_path: Path) -> None:
    data_root, data_config = _make_data(tmp_path)
    config = _run_config(tmp_path, data_root, data_config)
    lowm_run = tmp_path / "runs" / "lowm_seed5"
    fixed_run = tmp_path / "runs" / "fixed_energy_seed5"
    _write_run(lowm_run, config, "lowm")
    _write_run(fixed_run, config, "fixed_energy")

    evaluate_run(lowm_run, split="val", checkpoint_name="best_top1.pt", device_name="cpu", num_samples=10, batch_size=5)
    evaluate_run(lowm_run, split="val", checkpoint_name="last.pt", device_name="cpu", num_samples=10, batch_size=5)
    evaluate_run(fixed_run, split="val", checkpoint_name="best_top1.pt", device_name="cpu", num_samples=10, batch_size=5)

    out = tmp_path / "summary"
    rows = aggregate_results([fixed_run, lowm_run], out, split="val", checkpoints=["best_top1.pt", "last.pt"])
    assert len(rows) == 3
    assert any(row["checkpoint"] == "last" for row in rows)
    assert (out / "summary_table.csv").exists()
    assert (out / "summary_table.md").exists()
    assert (out / "ranking_bar_by_model.png").exists()
    assert (out / "law_pair_gap_by_model.png").exists()
