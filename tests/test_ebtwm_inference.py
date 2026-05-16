import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from lowm.data.dataset import LOWMSynthRankingDataset, RankingConfig
from lowm.data.simulate import SimulationConfig, simulate_split
from lowm.eval import ebtwm_inference
from lowm.eval.ebtwm_inference import evaluate_ebtwm_inference, optimize_trajectory_energy
from lowm.models.lowm import LOWM, lowm_config_from_mapping


def _make_dummy_run(tmp_path: Path) -> tuple[Path, LOWMSynthRankingDataset, LOWM]:
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True)
    arrays = simulate_split(seed=41, cfg=SimulationConfig(num_episodes=28, T=6, n_min=3, n_max=5, nmax=6))
    for split in ["val", "test_iid"]:
        np.savez_compressed(data_root / f"{split}.npz", **arrays)

    dataset_config = tmp_path / "dataset.yaml"
    dataset_config.write_text(
        yaml.safe_dump(
            {
                "dataset": "LOWM-Synth",
                "simulation": {"T": 6, "nmax": 6, "dt": 0.05, "world_size": 1.0},
                "splits": {"val": {"num_episodes": 28}, "test_iid": {"num_episodes": 28}},
            }
        ),
        encoding="utf-8",
    )
    config = {
        "data": {
            "root": str(data_root),
            "val_split": "val.npz",
            "test_iid_split": "test_iid.npz",
            "generate_if_missing": False,
            "dataset_config": str(dataset_config),
        },
        "ranking": {"K": 3, "H": 2, "M": 4, "seed": 11, "min_law_param_distance": 0.05},
        "model": {
            "object_dim": 7,
            "action_dim": 2,
            "lambda_dim": 4,
            "hidden_dim": 16,
            "context_dim": 16,
            "num_layers": 1,
            "use_pairwise_energy": False,
            "use_mu_eval": True,
        },
        "training": {"batch_size": 4, "seed": 0},
        "evaluation": {"num_samples": 4},
    }
    model = LOWM(lowm_config_from_mapping(config))
    run = tmp_path / "lowm_dummy"
    (run / "checkpoints").mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "config": config}, run / "checkpoints" / "best_law_pair.pt")
    dataset = LOWMSynthRankingDataset(data_root / "test_iid.npz", RankingConfig(K=3, H=2, M=4, seed=11, min_law_param_distance=0.05))
    return run, dataset, model


def test_optimize_trajectory_energy_tiny_batch_preserves_model_and_s0(tmp_path: Path) -> None:
    _, dataset, model = _make_dummy_run(tmp_path)
    item = dataset[0]
    init = item["pos_states"].clone()
    init[1:, :, 0:4] += torch.randn_like(init[1:, :, 0:4]) * 0.001
    before_params = {name: value.detach().clone() for name, value in model.state_dict().items()}

    result = optimize_trajectory_energy(
        model,
        item["context_states"],
        item["context_actions"],
        item["context_mask"],
        init,
        item["pos_actions"],
        item["pos_mask"],
        num_steps=3,
        step_size=1e-4,
        gt_states=item["pos_states"],
        data_range=1.0,
    )

    opt = result["optimized_states"].squeeze(0)
    assert opt.shape == init.shape
    assert torch.isfinite(opt).all()
    assert torch.allclose(opt[0], init[0])
    assert result["energy_before"].shape == result["energy_after"].shape == (1,)
    assert "mse_to_gt_before" in result and "mse_to_gt_after" in result
    for name, value in model.state_dict().items():
        assert torch.allclose(value, before_params[name])


def test_ebtwm_eval_creates_metrics_and_plots(tmp_path: Path) -> None:
    run, _, _ = _make_dummy_run(tmp_path)
    metrics = evaluate_ebtwm_inference(
        run,
        split="test_iid",
        checkpoint="best_law_pair.pt",
        num_samples=2,
        num_steps=2,
        step_size=1e-4,
        noise_std=0.01,
        horizon=2,
        device="cpu",
        seed=3,
        skip_preflight=True,
    )
    out = run / "eval" / "test_iid" / "ebtwm_inference"
    assert metrics["num_samples"] == 2
    assert "go_no_go_decision" in metrics
    assert (out / "metrics.json").exists()
    assert (out / "per_sample_metrics.csv").exists()
    assert (out / "optimization_curves.csv").exists()
    assert (out / "cross_operator_energy.csv").exists()
    assert (out / "optimization_curves.png").exists()
    assert (out / "before_after_examples.png").exists()
    assert (out / "README.txt").exists()


def test_ebtwm_cli_creates_metrics_on_tiny_run(tmp_path: Path, monkeypatch) -> None:
    run, _, _ = _make_dummy_run(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ebtwm_inference",
            "--run",
            str(run),
            "--split",
            "test_iid",
            "--checkpoint",
            "best_law_pair.pt",
            "--num-samples",
            "1",
            "--num-steps",
            "1",
            "--step-size",
            "1e-4",
            "--horizon",
            "2",
            "--device",
            "cpu",
            "--skip-preflight",
        ],
    )
    ebtwm_inference.main()
    assert (run / "eval" / "test_iid" / "ebtwm_inference" / "metrics.json").exists()
