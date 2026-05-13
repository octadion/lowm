import json
from pathlib import Path

import yaml

from lowm.eval.aggregate_sweep import aggregate_sweep
from lowm.training.run_sweep import build_run_config, expand_sweep, run_name_from_params, run_sweep


def test_sweep_config_expansion_and_run_naming(tmp_path: Path) -> None:
    sweep = {
        "parameters": {
            "alpha_occl": [0.0, 1.0],
            "lambda_dim": [4],
            "use_pairwise_energy": [True],
            "use_stability": [True, False],
            "beta_kl": [0.0],
            "seed": [0, 1],
        }
    }
    combos = expand_sweep(sweep)
    assert len(combos) == 8
    name = run_name_from_params(combos[0])
    assert name.startswith("lowm_occl_alpha")
    assert "_lambda4_" in name

    base = {
        "model": {"lambda_dim": 16, "use_pairwise_energy": True},
        "training": {"alpha_occl": 1.0, "beta_kl": 1e-4, "seed": 0},
    }
    config = build_run_config(base, combos[-1], tmp_path / "sweep")
    assert config["training"]["output_dir"] == str(tmp_path / "sweep" / "runs")
    assert config["training"]["use_occl"] is True
    assert config["sweep_params"] == combos[-1]


def test_variant_sweep_negative_types_config(tmp_path: Path) -> None:
    sweep = {
        "variants": [
            {
                "negative_set": "all",
                "alpha_occl": 0.0,
                "lambda_dim": 16,
                "use_pairwise_energy": True,
                "use_stability": True,
                "beta_kl": 1e-4,
                "seed": 0,
                "negative_types": ["state_corrupted", "law_mismatch"],
            }
        ]
    }
    combos = expand_sweep(sweep)
    assert len(combos) == 1
    assert run_name_from_params(combos[0]).startswith("lowm_negs_all")
    config = build_run_config({"ranking": {}, "model": {}, "training": {}}, combos[0], tmp_path / "sweep")
    assert config["ranking"]["negative_types"] == ["state_corrupted", "law_mismatch"]
    assert config["sweep_params"]["negative_set"] == "all"


def test_component_sweep_dry_run_generates_eight_configs(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "data": {"root": "data/lowm_synth_v0"},
                "ranking": {"negative_types": ["law_mismatch"]},
                "model": {"lambda_dim": 16, "use_pairwise_energy": True},
                "training": {"alpha_occl": 1.0, "beta_kl": 1e-4, "seed": 0},
            }
        ),
        encoding="utf-8",
    )
    variants = []
    for name in ["main_all", "no_pairwise", "no_stability", "lambda_dim_4", "lambda_dim_8", "lambda_dim_32", "no_kl", "high_kl"]:
        variants.append(
            {
                "component": name,
                "alpha_occl": 0.0,
                "lambda_dim": 4 if name == "lambda_dim_4" else 8 if name == "lambda_dim_8" else 32 if name == "lambda_dim_32" else 16,
                "use_pairwise_energy": name != "no_pairwise",
                "use_stability": name != "no_stability",
                "beta_kl": 0.0 if name == "no_kl" else 1e-3 if name == "high_kl" else 1e-4,
                "seed": 0,
                "negative_set": "all",
                "negative_types": ["state_corrupted", "temporal_shuffled", "law_mismatch", "random_impossible"],
            }
        )
    sweep = tmp_path / "component.yaml"
    sweep.write_text(
        yaml.safe_dump(
            {
                "base_config": str(base),
                "sweep_dir": str(tmp_path / "component_sweep"),
                "variants": variants,
                "overrides": {"training": {"epochs": 30, "train_samples_per_epoch": 5000, "val_samples": 1000}},
                "evaluation": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    runs = run_sweep(sweep, dry_run=True)
    assert len(runs) == 8
    generated = sorted((tmp_path / "component_sweep" / "configs").glob("*.yaml"))
    assert len(generated) == 8
    one = yaml.safe_load(generated[0].read_text(encoding="utf-8"))
    assert one["training"]["epochs"] == 30
    assert one["training"]["train_samples_per_epoch"] == 5000


def test_ood_sweep_dry_run_supports_baseline_and_multisplit(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "data": {"root": "data/lowm_synth_ood_param", "dataset_config": "configs/lowm_synth_ood_param.yaml"},
                "ranking": {"negative_types": ["law_mismatch"]},
                "model": {"lambda_dim": 16, "use_pairwise_energy": False},
                "training": {"alpha_occl": 0.0, "beta_kl": 1e-4, "seed": 0, "selection_metric": "law_pair"},
            }
        ),
        encoding="utf-8",
    )
    sweep = tmp_path / "ood.yaml"
    sweep.write_text(
        yaml.safe_dump(
            {
                "base_config": str(base),
                "sweep_dir": str(tmp_path / "ood_sweep"),
                "variants": [
                    {
                        "variant": "direct_context_energy",
                        "model_type": "direct_context_energy",
                        "alpha_occl": 0.0,
                        "lambda_dim": 16,
                        "use_pairwise_energy": False,
                        "use_stability": True,
                        "beta_kl": 1e-4,
                        "selection_metric": "law_pair",
                        "seed": 0,
                        "negative_set": "all",
                        "negative_types": ["law_mismatch"],
                    },
                    {
                        "variant": "lowm_omcr_no_pairwise",
                        "model_type": "lowm",
                        "alpha_occl": 0.0,
                        "lambda_dim": 16,
                        "use_pairwise_energy": False,
                        "use_stability": True,
                        "beta_kl": 1e-4,
                        "selection_metric": "law_pair",
                        "seed": 0,
                        "negative_set": "all",
                        "negative_types": ["law_mismatch"],
                    },
                ],
                "evaluation": {"enabled": False, "splits": ["val", "test_iid", "test_ood_param"]},
            }
        ),
        encoding="utf-8",
    )
    runs = run_sweep(sweep, dry_run=True)
    assert len(runs) == 2
    generated = sorted((tmp_path / "ood_sweep" / "configs").glob("*.yaml"))
    configs = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in generated]
    baseline_configs = [config for config in configs if config["sweep_params"]["model_type"] == "direct_context_energy"]
    assert baseline_configs[0]["training"]["baseline"] == "direct_context_energy"
    assert any(config["sweep_params"]["variant"] == "lowm_omcr_no_pairwise" for config in configs)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_dummy_run(sweep_dir: Path, name: str, params: dict) -> Path:
    run = sweep_dir / "runs" / name
    run.mkdir(parents=True, exist_ok=True)
    config = {
        "sweep_params": params,
        "model": {"lambda_dim": params["lambda_dim"], "use_pairwise_energy": params["use_pairwise_energy"]},
        "training": {
            "alpha_occl": params["alpha_occl"],
            "beta_kl": params["beta_kl"],
            "use_stability": params["use_stability"],
            "seed": params["seed"],
        },
        "ranking": {"negative_types": params.get("negative_types", ["law_mismatch"])},
    }
    (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    eval_dir = run / "eval" / "val" / "best_law_pair"
    _write_json(
        eval_dir / "eval_summary.json",
        {
            "model_type": "lowm",
            "checkpoint_used": "best_law_pair.pt",
            "split": "val",
            "ranking": {"top1_acc": 0.4, "law_pair": 0.7, "law_gap": 0.2, "mean_rank": 2.0, "mrr": 0.5},
            "retrieval": {},
            "occl_alignment": {},
        },
    )
    (eval_dir / "negative_type_breakdown.csv").write_text(
        "negative_type,pairwise_acc,mean_energy_gap,count,top1_acc_on_samples_with_type\n"
        "law_mismatch,0.7,0.2,10,0.4\n",
        encoding="utf-8",
    )
    _write_json(
        run / "eval" / "val" / "law_mismatch_only_best_law_pair" / "law_mismatch_only_metrics.json",
        {"law_only_top1": 0.6, "pairwise_acc_law_only": 0.8, "mean_law_gap": 0.3},
    )
    _write_json(
        run / "eval" / "val" / "best_occl_acc" / "occl_alignment_metrics.json",
        {"tau_to_lambda_acc": 0.5, "lambda_to_tau_acc": 0.45, "diagonal_vs_offdiag_gap": 0.1},
    )
    return run


def _write_split_metrics(run: Path, split: str, top1: float, law_pair: float, law_gap: float, law_only: float) -> None:
    eval_dir = run / "eval" / split / "best_law_pair"
    _write_json(
        eval_dir / "eval_summary.json",
        {
            "model_type": "lowm",
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


def test_aggregate_sweep_outputs_csv_markdown_and_plots(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "sweep"
    run = _write_dummy_run(
        sweep_dir,
        "lowm_occl_alpha1_lambda4_seed0_pair1_stab1_beta0",
        {
            "alpha_occl": 1.0,
            "lambda_dim": 4,
            "use_pairwise_energy": True,
            "use_stability": True,
            "beta_kl": 0.0,
            "seed": 0,
            "negative_set": "all",
            "component": "main_all",
            "negative_types": ["state_corrupted", "law_mismatch"],
        },
    )
    (sweep_dir / "manifest.json").write_text(json.dumps({"runs": [str(run)]}), encoding="utf-8")
    out = tmp_path / "out"
    rows = aggregate_sweep(sweep_dir, out)
    assert len(rows) == 1
    assert rows[0]["law_only_top1"] == 0.6
    assert rows[0]["negative_set"] == "all"
    assert rows[0]["component"] == "main_all"
    assert (out / "ablation_summary.csv").exists()
    assert (out / "ablation_summary.md").exists()
    assert (out / "plots" / "law_only_top1_vs_alpha.png").exists()
    assert (out / "plots" / "lambda_dim_ablation.png").exists()
    assert (out / "plots" / "negative_set_law_pair.png").exists()
    assert (out / "plots" / "component_ablation_law_pair.png").exists()
    assert (out / "plots" / "component_ablation_law_gap.png").exists()
    assert (out / "plots" / "component_ablation_law_only_top1.png").exists()


def test_aggregate_sweep_multisplit_ood_stats_and_plots(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "ood_sweep"
    params0 = {
        "variant": "lowm_omcr_no_pairwise",
        "model_type": "lowm",
        "alpha_occl": 0.0,
        "lambda_dim": 16,
        "use_pairwise_energy": False,
        "use_stability": True,
        "beta_kl": 1e-4,
        "seed": 0,
        "negative_set": "all",
        "negative_types": ["law_mismatch"],
    }
    params1 = dict(params0, seed=1)
    run0 = _write_dummy_run(sweep_dir, "lowm_lowm_omcr_no_pairwise_seed0", params0)
    run1 = _write_dummy_run(sweep_dir, "lowm_lowm_omcr_no_pairwise_seed1", params1)
    for run, offset in [(run0, 0.0), (run1, 0.1)]:
        _write_split_metrics(run, "test_iid", 0.8 + offset, 0.7 + offset, 1.0 + offset, 0.65 + offset)
        _write_split_metrics(run, "test_ood_param", 0.7 + offset, 0.5 + offset, 0.6 + offset, 0.45 + offset)
    (sweep_dir / "manifest.json").write_text(json.dumps({"runs": [str(run0), str(run1)]}), encoding="utf-8")
    out = tmp_path / "out"
    rows = aggregate_sweep(sweep_dir, out, split=["val", "test_iid", "test_ood_param"])
    assert len(rows) == 6
    assert (out / "ablation_summary_stats.csv").exists()
    assert (out / "ablation_summary_stats.md").exists()
    assert (out / "plots" / "iid_vs_ood_law_pair.png").exists()
    assert (out / "plots" / "iid_vs_ood_law_only_top1.png").exists()
    assert (out / "plots" / "ood_degradation.png").exists()
