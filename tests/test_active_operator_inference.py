import json
from pathlib import Path

import numpy as np
import torch
import yaml

from lowm.data.simulate import SimulationConfig, simulate_split
from lowm.eval.active_operator_inference import (
    entropy,
    evaluate_active_operator_inference,
    posterior_from_energies,
)
from lowm.eval.aggregate_aoi import aggregate_aoi
from lowm.models.lowm import LOWM, lowm_config_from_mapping


def test_posterior_normalizes_and_entropy_drops_for_clear_true_energy() -> None:
    uniform = posterior_from_energies(np.array([1.0, 1.0, 1.0]), temperature=1.0)
    clear = posterior_from_energies(np.array([0.0, 8.0, 9.0]), temperature=1.0)
    assert np.isclose(uniform.sum(), 1.0)
    assert np.isclose(clear.sum(), 1.0)
    assert int(np.argmax(clear)) == 0
    assert entropy(clear) < entropy(uniform)


def _write_dummy_lowm_run(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True)
    arrays = simulate_split(seed=33, cfg=SimulationConfig(num_episodes=24, T=6, n_min=3, n_max=5, nmax=6))
    for split in ["val", "test_iid"]:
        np.savez_compressed(data_root / f"{split}.npz", **arrays)

    dataset_config = tmp_path / "dataset.yaml"
    dataset_config.write_text(
        yaml.safe_dump(
            {
                "dataset": "LOWM-Synth",
                "simulation": {"T": 6, "nmax": 6, "dt": 0.05, "world_size": 1.0},
                "splits": {"val": {"num_episodes": 24}, "test_iid": {"num_episodes": 24}},
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
        "ranking": {"K": 2, "H": 3, "M": 4, "seed": 5, "min_law_param_distance": 0.05},
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
    ckpt = run / "checkpoints"
    ckpt.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "config": config}, ckpt / "best_law_pair.pt")
    return run


def test_active_operator_inference_outputs_files_and_random_metrics(tmp_path: Path) -> None:
    run = _write_dummy_lowm_run(tmp_path)
    metrics = evaluate_active_operator_inference(
        run,
        split="test_iid",
        checkpoint_name="best_law_pair.pt",
        num_episodes=2,
        num_operator_hypotheses=3,
        num_actions=5,
        horizon=2,
        seed=7,
        device_name="cpu",
    )
    assert "random_action" in metrics["methods"]
    random_metrics = metrics["methods"]["random_action"]
    assert 0.0 <= random_metrics["identification_accuracy"] <= 1.0
    out = run / "eval" / "test_iid" / "aoi"
    assert (out / "aoi_metrics.json").exists()
    assert (out / "aoi_per_episode.csv").exists()
    assert (out / "action_score_examples.csv").exists()
    assert (out / "plots" / "entropy_reduction_by_method.png").exists()
    saved = json.loads((out / "aoi_metrics.json").read_text(encoding="utf-8"))
    assert saved["temperature"] > 0


def test_aggregate_aoi_outputs_summary(tmp_path: Path) -> None:
    run = _write_dummy_lowm_run(tmp_path)
    evaluate_active_operator_inference(
        run,
        split="test_iid",
        checkpoint_name="best_law_pair.pt",
        num_episodes=2,
        num_operator_hypotheses=3,
        num_actions=5,
        horizon=2,
        seed=8,
        device_name="cpu",
        temperature=1.0,
    )
    out = tmp_path / "aoi_summary"
    rows = aggregate_aoi([run], out, split="test_iid")
    assert rows
    assert (out / "aoi_summary.csv").exists()
    assert (out / "aoi_summary.md").exists()
    assert (out / "aoi_identification_accuracy.png").exists()
