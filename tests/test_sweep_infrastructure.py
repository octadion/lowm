import json
from pathlib import Path

import yaml

from lowm.eval.aggregate_sweep import aggregate_sweep
from lowm.training.run_sweep import build_run_config, expand_sweep, run_name_from_params


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
        },
    )
    (sweep_dir / "manifest.json").write_text(json.dumps({"runs": [str(run)]}), encoding="utf-8")
    out = tmp_path / "out"
    rows = aggregate_sweep(sweep_dir, out)
    assert len(rows) == 1
    assert rows[0]["law_only_top1"] == 0.6
    assert (out / "ablation_summary.csv").exists()
    assert (out / "ablation_summary.md").exists()
    assert (out / "plots" / "law_only_top1_vs_alpha.png").exists()
    assert (out / "plots" / "lambda_dim_ablation.png").exists()
