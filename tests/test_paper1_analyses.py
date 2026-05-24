from pathlib import Path

import torch
import yaml

from lowm.data.generate_dataset import generate_dataset
from lowm.eval.evaluate_coherence_stratification import evaluate_coherence_stratification
from lowm.eval.evaluate_energy_matrix import evaluate_energy_matrix
from lowm.eval.probe_operator_representation import probe_operator_representation
from lowm.models.lowm import LOWM, LOWMConfig


def _make_data(tmp_path: Path) -> tuple[Path, Path]:
    config = {
        "dataset": "LOWM-Synth",
        "version": "paper1-analysis-test",
        "seed": 101,
        "simulation": {"T": 10, "nmax": 6},
        "splits": {
            "train": {"num_episodes": 40, "n_min": 3, "n_max": 5, "seed": 101},
            "val": {"num_episodes": 32, "n_min": 3, "n_max": 5, "seed": 102},
        },
    }
    config_path = tmp_path / "data_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    out = tmp_path / "dataset"
    generate_dataset(config_path, out, ["train", "val"])
    return out, config_path


def _write_lowm_run(run_dir: Path, data_root: Path, data_config: Path) -> None:
    config = {
        "seed": 11,
        "data": {
            "root": str(data_root),
            "train_split": "train.npz",
            "val_split": "val.npz",
            "generate_if_missing": False,
            "dataset_config": str(data_config),
        },
        "ranking": {"K": 3, "H": 4, "M": 5, "seed": 23},
        "model": {
            "object_dim": 7,
            "action_dim": 2,
            "lambda_dim": 8,
            "hidden_dim": 32,
            "context_dim": 32,
            "num_layers": 2,
            "use_mu_eval": True,
        },
        "training": {"batch_size": 4, "val_samples": 12, "device": "cpu"},
        "evaluation": {"num_samples": 12},
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    model = LOWM(LOWMConfig(lambda_dim=8, hidden_dim=32, context_dim=32, use_mu_eval=True))
    checkpoint = {"model_state": model.state_dict(), "config": config, "model_type": "lowm"}
    for name in ["best.pt", "best_law_pair.pt", "last.pt"]:
        torch.save(checkpoint, run_dir / "checkpoints" / name)
    (run_dir / "metadata.json").write_text('{"model":"LOWM"}', encoding="utf-8")


def test_paper1_analysis_outputs(tmp_path: Path) -> None:
    data_root, data_config = _make_data(tmp_path)
    run_dir = tmp_path / "runs" / "lowm_seed11"
    _write_lowm_run(run_dir, data_root, data_config)

    strat = evaluate_coherence_stratification(
        run_dir,
        split="val",
        checkpoint_name="best_law_pair.pt",
        device_name="cpu",
        num_samples=8,
        batch_size=4,
    )
    assert "fraction_same_lt_wrong" in strat
    assert "fraction_same_lt_wrong_lt_noise" in strat
    assert (run_dir / "eval" / "val" / "coherence_stratification" / "metrics.json").exists()
    assert (run_dir / "eval" / "val" / "coherence_stratification" / "per_sample.csv").exists()
    assert (run_dir / "eval" / "val" / "coherence_stratification" / "energy_by_type.png").exists()
    assert (run_dir / "eval" / "val" / "coherence_stratification" / "stratification_summary.md").exists()

    probe = probe_operator_representation(
        run_dir,
        split="val",
        checkpoint_name="best_law_pair.pt",
        device_name="cpu",
        num_samples=12,
        batch_size=4,
        probe_epochs=3,
    )
    assert "op_id_probe_accuracy" in probe
    assert "op_param_r2" in probe
    assert (run_dir / "eval" / "val" / "operator_probe" / "metrics.json").exists()
    assert (run_dir / "eval" / "val" / "operator_probe" / "lambda_embeddings.npy").exists()
    assert (run_dir / "eval" / "val" / "operator_probe" / "pca_lambda.png").exists()
    assert (run_dir / "eval" / "val" / "operator_probe" / "op_id_confusion.png").exists()
    assert (run_dir / "eval" / "val" / "operator_probe" / "probe_summary.md").exists()

    matrix = evaluate_energy_matrix(
        run_dir,
        split="val",
        checkpoint_name="best_law_pair.pt",
        device_name="cpu",
        matrix_size=4,
        max_batches=2,
        num_samples=8,
    )
    assert "diagonal_vs_offdiag_gap" in matrix
    assert "row_wise_correct_lambda_rank" in matrix
    assert "column_wise_correct_trajectory_rank" in matrix
    assert (run_dir / "eval" / "val" / "energy_matrix" / "metrics.json").exists()
    assert (run_dir / "eval" / "val" / "energy_matrix" / "energy_matrix.csv").exists()
    assert (run_dir / "eval" / "val" / "energy_matrix" / "energy_matrix_heatmap.png").exists()
    assert (run_dir / "eval" / "val" / "energy_matrix" / "diagonal_summary.md").exists()
