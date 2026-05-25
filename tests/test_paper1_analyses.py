from pathlib import Path

import torch
import yaml

from lowm.data.generate_dataset import generate_dataset
from lowm.eval.aggregate_p1_backbone_omc import aggregate_p1_backbone_omc
from lowm.eval.aggregate_paper1_current import aggregate_paper1_current
from lowm.eval.evaluate_coherence_stratification import evaluate_coherence_stratification
from lowm.eval.evaluate_energy_matrix import evaluate_energy_matrix
from lowm.eval.probe_operator_representation import probe_operator_representation
from lowm.models.lowm import LOWM, LOWMConfig
from lowm.training.run_sweep import run_sweep


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
        probe_type="both",
        probe_seed=7,
        probe_hidden_dim=16,
        probe_epochs=3,
    )
    assert "op_id_probe_accuracy" in probe
    assert "op_param_r2" in probe
    assert "linear_op_id_probe_accuracy" in probe
    assert "mlp_op_id_probe_accuracy" in probe
    assert "linear_binned_param_accuracy" in probe
    assert "mlp_binned_param_accuracy" in probe
    assert "linear_op_param_r2" in probe
    assert "mlp_op_param_r2" in probe
    assert (run_dir / "eval" / "val" / "operator_probe" / "metrics.json").exists()
    assert (run_dir / "eval" / "val" / "operator_probe" / "lambda_embeddings.npy").exists()
    assert (run_dir / "eval" / "val" / "operator_probe" / "pca_lambda.png").exists()
    assert (run_dir / "eval" / "val" / "operator_probe" / "op_id_confusion.png").exists()
    assert (run_dir / "eval" / "val" / "operator_probe" / "linear_op_id_confusion.png").exists()
    assert (run_dir / "eval" / "val" / "operator_probe" / "mlp_op_id_confusion.png").exists()
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_backbone_eval(run: Path, split: str, top1: float, law_pair: float, law_gap: float, law_only: float) -> None:
    eval_dir = run / "eval" / split / "best_law_pair"
    _write_json(
        eval_dir / "eval_summary.json",
        {
            "model_type": "direct_context_energy",
            "checkpoint_used": "best_law_pair.pt",
            "split": split,
            "ranking": {"top1_acc": top1, "law_pair": law_pair, "law_gap": law_gap, "mean_rank": 2.0, "mrr": 0.5},
            "retrieval": {},
            "occl_alignment": {},
        },
    )
    (eval_dir / "negative_type_breakdown.csv").write_text(
        "negative_type,pairwise_acc,mean_energy_gap,count,top1_acc_on_samples_with_type\n"
        f"law_mismatch,{law_pair},{law_gap},10,{top1}\n",
        encoding="utf-8",
    )
    _write_json(
        run / "eval" / split / "law_mismatch_only_best_law_pair" / "law_mismatch_only_metrics.json",
        {"law_only_top1": law_only, "pairwise_acc_law_only": law_pair, "mean_law_gap": law_gap},
    )
    _write_json(
        run / "eval" / split / "coherence_stratification" / "metrics.json",
        {
            "fraction_same_lt_wrong": law_pair,
            "fraction_same_lt_wrong_lt_noise": 0.5 * law_pair,
            "gap_same_wrong": law_gap,
        },
    )
    _write_json(
        run / "eval" / split / "energy_matrix" / "metrics.json",
        {
            "mrr": 0.4 + 0.1 * law_pair,
            "diagonal_top1_accuracy": law_pair,
            "diagonal_vs_offdiag_gap": law_gap,
        },
    )


def _write_current_run(root: Path, name: str, model_type: str, variant: str, objective: str, seed: int) -> Path:
    run = root / "runs" / name
    run.mkdir(parents=True, exist_ok=True)
    negative_types = ["state_corrupted", "temporal_shuffled", "random_impossible"]
    if objective == "OMC":
        negative_types.insert(2, "law_mismatch")
    config = {
        "sweep_params": {
            "variant": variant,
            "model_type": model_type,
            "seed": seed,
            "negative_set": "all" if objective == "OMC" else "no_law_mismatch",
            "negative_types": negative_types,
        },
        "ranking": {"negative_types": negative_types},
        "model": {"lambda_dim": 8} if model_type == "lowm" else {"hidden_dim": 32},
        "training": {"seed": seed, "baseline": model_type if model_type != "lowm" else None},
    }
    (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return run


def _write_current_eval(
    run: Path,
    split: str,
    model_type: str,
    top1: float,
    law_pair: float,
    law_gap: float,
    law_only: float,
    write_probe: bool = True,
    write_energy: bool = True,
) -> None:
    eval_dir = run / "eval" / split / "best_law_pair"
    _write_json(
        eval_dir / "eval_summary.json",
        {
            "model_type": model_type,
            "checkpoint_used": "best_law_pair.pt",
            "split": split,
            "ranking": {"top1_acc": top1, "law_pair": law_pair, "law_gap": law_gap, "mean_rank": 2.0, "mrr": 0.5},
            "retrieval": {},
            "occl_alignment": {},
        },
    )
    (eval_dir / "negative_type_breakdown.csv").write_text(
        "negative_type,pairwise_acc,mean_energy_gap,count,top1_acc_on_samples_with_type\n"
        f"law_mismatch,{law_pair},{law_gap},10,{top1}\n",
        encoding="utf-8",
    )
    _write_json(
        run / "eval" / split / "law_mismatch_only_best_law_pair" / "law_mismatch_only_metrics.json",
        {"law_only_top1": law_only, "pairwise_acc_law_only": law_pair, "mean_law_gap": law_gap},
    )
    _write_json(
        run / "eval" / split / "coherence_stratification" / "metrics.json",
        {
            "fraction_same_lt_wrong": law_pair,
            "fraction_same_lt_wrong_lt_noise": law_pair - 0.1,
            "gap_same_wrong": law_gap,
        },
    )
    if write_energy:
        _write_json(
            run / "eval" / split / "energy_matrix" / "metrics.json",
            {
                "mrr": 0.4 + 0.1 * law_pair,
                "diagonal_top1_accuracy": law_pair,
                "diagonal_vs_offdiag_gap": law_gap,
            },
        )
    if write_probe:
        _write_json(
            run / "eval" / split / "operator_probe" / "metrics.json",
            {
                "linear_op_id_probe_accuracy": 0.45,
                "mlp_op_id_probe_accuracy": 0.46,
                "linear_binned_param_accuracy": 0.35,
                "mlp_binned_param_accuracy": 0.34,
                "linear_op_param_r2": 0.05,
                "mlp_op_param_r2": 0.04,
            },
        )


def test_direct_context_omc_sweep_dry_run_and_aggregation(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "data": {"root": "data/lowm_synth_ood_param", "dataset_config": "configs/lowm_synth_ood_param.yaml"},
                "ranking": {"negative_types": ["law_mismatch"]},
                "model": {"hidden_dim": 32, "token_dim": 32, "context_dim": 32},
                "training": {"baseline": "direct_context_energy", "seed": 0, "selection_metric": "law_pair"},
            }
        ),
        encoding="utf-8",
    )
    sweep = tmp_path / "direct_context.yaml"
    sweep.write_text(
        yaml.safe_dump(
            {
                "base_config": str(base),
                "sweep_dir": str(tmp_path / "direct_context_sweep"),
                "variants": [
                    {
                        "variant": "direct_context_no_law_mismatch",
                        "model_type": "direct_context_energy",
                        "seed": 0,
                        "negative_set": "no_law_mismatch",
                        "selection_metric": "law_pair",
                        "negative_types": ["state_corrupted", "temporal_shuffled", "random_impossible"],
                    },
                    {
                        "variant": "direct_context_OMC",
                        "model_type": "direct_context_energy",
                        "seed": 0,
                        "negative_set": "all",
                        "selection_metric": "law_pair",
                        "negative_types": ["state_corrupted", "temporal_shuffled", "law_mismatch", "random_impossible"],
                    },
                ],
                "evaluation": {
                    "enabled": False,
                    "splits": ["test_iid", "test_ood_param"],
                    "coherence_stratification": True,
                    "energy_matrix": True,
                },
            }
        ),
        encoding="utf-8",
    )
    runs = run_sweep(sweep, dry_run=True)
    assert len(runs) == 2
    generated = sorted((tmp_path / "direct_context_sweep" / "configs").glob("*.yaml"))
    configs = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in generated]
    assert all(config["training"]["baseline"] == "direct_context_energy" for config in configs)
    assert any("law_mismatch" not in config["ranking"]["negative_types"] for config in configs)
    assert any("law_mismatch" in config["ranking"]["negative_types"] for config in configs)

    for run, variant, law_pair in [
        (runs[0], "direct_context_no_law_mismatch", 0.35),
        (runs[1], "direct_context_OMC", 0.75),
    ]:
        run.mkdir(parents=True, exist_ok=True)
        generated_config = tmp_path / "direct_context_sweep" / "configs" / f"{run.name}.yaml"
        config = yaml.safe_load(generated_config.read_text(encoding="utf-8"))
        config["sweep_params"]["variant"] = variant
        (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        _write_backbone_eval(run, "test_iid", 0.8, law_pair, 0.2 + law_pair, 0.3 + law_pair)
        _write_backbone_eval(run, "test_ood_param", 0.7, law_pair - 0.1, 0.1 + law_pair, 0.2 + law_pair)
    out = tmp_path / "summary"
    summary = aggregate_p1_backbone_omc(tmp_path / "direct_context_sweep", out)
    assert len(summary) == 4
    assert (out / "backbone_omc_summary.csv").exists()
    assert (out / "backbone_omc_summary.md").exists()
    assert (out / "backbone_omc_runs.csv").exists()


def test_aggregate_paper1_current_outputs_tables_and_tolerates_missing_files(tmp_path: Path) -> None:
    lowm_root = tmp_path / "lowm_main"
    direct_root = tmp_path / "direct_context"
    lowm_omc = _write_current_run(lowm_root, "lowm_omc_seed0", "lowm", "lowm_omc", "OMC", 0)
    lowm_no = _write_current_run(lowm_root, "lowm_no_law_seed0", "lowm", "no_law_mismatch", "no_law_mismatch", 0)
    direct_omc = _write_current_run(direct_root, "direct_context_OMC_seed0", "direct_context_energy", "direct_context_OMC", "OMC", 0)
    direct_no = _write_current_run(
        direct_root,
        "direct_context_no_law_mismatch_seed0",
        "direct_context_energy",
        "direct_context_no_law_mismatch",
        "no_law_mismatch",
        0,
    )
    _write_json(lowm_root / "manifest.json", {"runs": [str(lowm_omc), str(lowm_no)]})
    _write_json(direct_root / "manifest.json", {"runs": [str(direct_omc), str(direct_no)]})

    for split in ["test_iid", "test_ood_param"]:
        _write_current_eval(lowm_omc, split, "lowm", 0.98, 0.74, 2.1, 0.42)
        _write_current_eval(lowm_no, split, "lowm", 0.99, 0.50, 0.1, 0.21, write_probe=False)
        _write_current_eval(direct_omc, split, "direct_context_energy", 0.96, 0.70, 1.8, 0.40)
        _write_current_eval(direct_no, split, "direct_context_energy", 0.95, 0.52, 0.2, 0.22, write_energy=False)

    out = tmp_path / "paper1_current"
    summary = aggregate_paper1_current(lowm_root, direct_root, out)
    assert len(summary) == 8
    assert (out / "paper1_current_summary.md").exists()
    assert (out / "paper1_current_summary.csv").exists()
    assert (out / "table_operator_blindness.md").exists()
    assert (out / "table_coherence_stratification.md").exists()
    assert (out / "table_energy_matrix.md").exists()
    assert (out / "table_probe.md").exists()
    checklist = out / "claim_checklist.md"
    assert checklist.exists()
    text = checklist.read_text(encoding="utf-8")
    assert "Not claimed" in text
    assert "LOWM-G is intentionally excluded" in text
