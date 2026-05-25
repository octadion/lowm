from pathlib import Path
import json

import numpy as np
import torch
import yaml

from lowm.data.cophy_adapter import CoPhyAdapterConfig, build_cophy_dataset
from lowm.data.inspect_cophy import inspect_cophy
from lowm.eval.aggregate_cophy_omc import aggregate_cophy_omc
from lowm.eval.evaluate_cophy_ranking import evaluate_cophy_ranking
from lowm.models.lowm import LOWM, LOWMConfig
from lowm.training.run_sweep import run_sweep


def _write_raw_cophy(root: Path, scenario: str = "BallsCF") -> None:
    scenario_dir = root / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    for split, episodes in [("train", 12), ("val", 10), ("test", 10)]:
        states = rng.normal(size=(episodes, 8, 4, 7)).astype(np.float32)
        actions = rng.normal(scale=0.1, size=(episodes, 7, 4, 2)).astype(np.float32)
        mask = np.ones((episodes, 8, 4), dtype=np.float32)
        confounder_id = np.asarray([idx % 2 for idx in range(episodes)], dtype=np.int64)
        physical_params = np.zeros((episodes, 5), dtype=np.float32)
        physical_params[:, 0] = confounder_id.astype(np.float32)
        np.savez_compressed(
            scenario_dir / f"{split}.npz",
            object_states=states,
            actions=actions,
            mask=mask,
            confounder_id=confounder_id,
            physical_params=physical_params,
            sample_id=np.arange(episodes, dtype=np.int64),
        )


def _write_run(run_dir: Path, data_root: Path, scenario: str = "BallsCF") -> None:
    config = {
        "seed": 0,
        "data": {
            "root": str(data_root / scenario),
            "train_split": "train.npz",
            "val_split": "val.npz",
            "test_split": "test.npz",
            "generate_if_missing": False,
            "dataset": "CoPhy",
            "scenario": scenario,
        },
        "ranking": {"K": 2, "H": 3, "M": 5, "seed": 17, "negative_types": ["state_corrupted", "temporal_shuffled", "law_mismatch", "random_impossible"]},
        "model": {"object_dim": 7, "action_dim": 2, "lambda_dim": 8, "hidden_dim": 32, "context_dim": 32, "num_layers": 2, "use_mu_eval": True},
        "training": {"batch_size": 4, "val_samples": 8, "seed": 0},
        "evaluation": {"num_samples": 8},
        "sweep_params": {"variant": "lowm_cophy_OMC", "model_type": "lowm", "seed": 0, "scenario": scenario, "mode": "state", "negative_types": ["law_mismatch"]},
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    model = LOWM(LOWMConfig(object_dim=7, lambda_dim=8, hidden_dim=32, context_dim=32, use_mu_eval=True))
    checkpoint = {"model_state": model.state_dict(), "config": config, "model_type": "lowm"}
    torch.save(checkpoint, run_dir / "checkpoints" / "best_law_pair.pt")
    torch.save(checkpoint, run_dir / "checkpoints" / "best.pt")
    (run_dir / "metadata.json").write_text('{"model":"LOWM"}', encoding="utf-8")


def test_inspect_cophy_missing_root_writes_report(tmp_path: Path) -> None:
    out = tmp_path / "audit"
    report = inspect_cophy(tmp_path / "missing_cophy", out)
    assert report["available"] is False
    assert (out / "cophy_data_report.md").exists()
    assert (out / "cophy_data_report.json").exists()
    assert "Expected state/feature layout" in (out / "cophy_data_report.md").read_text(encoding="utf-8")


def test_cophy_adapter_eval_and_aggregate(tmp_path: Path) -> None:
    raw = tmp_path / "raw_cophy"
    processed = tmp_path / "processed"
    _write_raw_cophy(raw)

    audit = inspect_cophy(raw, tmp_path / "audit")
    assert audit["available"] is True
    metadata = build_cophy_dataset(CoPhyAdapterConfig(root=raw, out=processed, scenario="BallsCF", splits=("train", "val", "test")))
    assert sorted(metadata["splits"]) == ["test", "train", "val"]
    assert (processed / "BallsCF" / "train.npz").exists()

    sweep_dir = tmp_path / "cophy_sweep"
    run_dir = sweep_dir / "runs" / "lowm_cophy_OMC_seed0"
    _write_run(run_dir, processed)
    (sweep_dir / "manifest.json").write_text(json.dumps({"runs": [str(run_dir)]}), encoding="utf-8")

    metrics = evaluate_cophy_ranking(
        run_dir,
        split="val",
        scenario="BallsCF",
        checkpoint_name="best_law_pair.pt",
        device_name="cpu",
        num_samples=6,
        batch_size=3,
        matrix_size=3,
        max_batches=1,
    )
    assert "same_lt_wrong" in metrics
    assert "energy_matrix_mrr" in metrics
    assert (run_dir / "eval" / "val" / "cophy_ranking" / "metrics.json").exists()
    assert (run_dir / "eval" / "val" / "cophy_ranking" / "per_sample.csv").exists()
    assert (run_dir / "eval" / "val" / "cophy_ranking" / "summary.md").exists()

    summary = aggregate_cophy_omc(sweep_dir, tmp_path / "summary", splits=["val"])
    assert len(summary) == 1
    assert (tmp_path / "summary" / "cophy_omc_summary.csv").exists()
    assert (tmp_path / "summary" / "cophy_omc_summary.md").exists()
    assert (tmp_path / "summary" / "cophy_claim_checklist.md").exists()


def test_cophy_sweep_dry_run_accepts_metadata_params(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "data": {"root": "data/cophy_omc/BallsCF", "train_split": "train.npz", "val_split": "val.npz", "test_split": "test.npz"},
                "ranking": {"negative_types": ["law_mismatch"]},
                "model": {"lambda_dim": 8, "object_dim": 7},
                "training": {"seed": 0},
            }
        ),
        encoding="utf-8",
    )
    sweep = tmp_path / "sweep.yaml"
    sweep.write_text(
        yaml.safe_dump(
            {
                "base_config": str(base),
                "sweep_dir": str(tmp_path / "sweep"),
                "variants": [
                    {
                        "variant": "lowm_cophy_OMC",
                        "model_type": "lowm",
                        "seed": 0,
                        "scenario": "BallsCF",
                        "mode": "state",
                        "negative_set": "all",
                        "negative_types": ["law_mismatch"],
                    }
                ],
                "evaluation": {"enabled": False, "cophy_ranking": True},
            }
        ),
        encoding="utf-8",
    )
    runs = run_sweep(sweep, dry_run=True)
    assert len(runs) == 1
    generated = yaml.safe_load((tmp_path / "sweep" / "configs" / f"{runs[0].name}.yaml").read_text(encoding="utf-8"))
    assert generated["sweep_params"]["scenario"] == "BallsCF"
