from pathlib import Path

import torch
import yaml

from lowm.data.dataset import LOWMSynthRankingDataset, RankingConfig, make_ranking_dataloader
from lowm.data.generate_dataset import generate_dataset
from lowm.eval.metrics import RankingMetricAccumulator
from lowm.models.baselines import BaselineConfig, DirectContextEnergyModel, FixedEnergyModel
from lowm.training.losses import nce_ranking_loss
from lowm.training.train_baseline import train_baseline


def _make_split(tmp_path: Path, episodes: int = 48) -> Path:
    config = {
        "dataset": "LOWM-Synth",
        "version": "baseline-test",
        "seed": 51,
        "simulation": {"T": 10, "nmax": 6},
        "splits": {
            "train": {
                "num_episodes": episodes,
                "n_min": 3,
                "n_max": 5,
                "parameter_split": "iid",
                "seed": 51,
            }
        },
    }
    config_path = tmp_path / "data_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    out = tmp_path / "dataset"
    generate_dataset(config_path, out, ["train"])
    return out / "train.npz"


def test_fixed_and_direct_context_forward_shapes(tmp_path: Path) -> None:
    path = _make_split(tmp_path)
    ranking_cfg = RankingConfig(K=3, H=4, M=5, seed=1)
    dataset = LOWMSynthRankingDataset(path, ranking_cfg, num_samples=8)
    loader = make_ranking_dataloader(dataset, batch_size=4)
    batch = next(iter(loader))
    model_cfg = BaselineConfig(hidden_dim=32, token_dim=32, context_dim=32)

    for model in [FixedEnergyModel(model_cfg), DirectContextEnergyModel(model_cfg)]:
        energies = model(batch)
        assert tuple(energies.shape) == (4, 5)
        loss = nce_ranking_loss(energies, batch["labels"])
        loss.backward()
        assert torch.isfinite(loss)
        model.zero_grad(set_to_none=True)


def test_ranking_metric_accumulator_law_breakdown() -> None:
    energies = torch.tensor([[0.1, 0.5, 0.6], [0.2, 0.4, 0.9]])
    labels = torch.tensor([0, 1])
    types = [["positive", "law_mismatch", "state_corrupted"], ["law_mismatch", "positive", "random_impossible"]]
    acc = RankingMetricAccumulator()
    acc.update(energies, labels, types, loss=0.7)
    metrics = acc.compute()
    assert metrics["top1_acc"] == 0.5
    assert metrics["law_mismatch"]["pairwise_acc"] == 0.5
    assert metrics["law_pair"] == 0.5
    assert abs(metrics["law_gap"] - 0.1) < 1e-6
    assert metrics["law_mismatch_pair_acc"] == 0.5
    assert abs(metrics["law_mismatch_gap"] - 0.1) < 1e-6
    assert metrics["by_negative_type"]["state_corrupted"]["count"] == 1


def test_train_baseline_one_epoch(tmp_path: Path) -> None:
    data_config = {
        "dataset": "LOWM-Synth",
        "version": "baseline-train-test",
        "seed": 61,
        "simulation": {"T": 10, "nmax": 6},
        "splits": {
            "train": {"num_episodes": 48, "n_min": 3, "n_max": 5, "seed": 61},
            "val": {"num_episodes": 24, "n_min": 3, "n_max": 5, "seed": 62},
        },
    }
    data_config_path = tmp_path / "lowm_synth_test.yaml"
    data_config_path.write_text(yaml.safe_dump(data_config), encoding="utf-8")

    train_config = {
        "seed": 2,
        "data": {
            "root": str(tmp_path / "dataset"),
            "train_split": "train.npz",
            "val_split": "val.npz",
            "generate_if_missing": True,
            "dataset_config": str(data_config_path),
        },
        "ranking": {"K": 3, "H": 4, "M": 5, "seed": 7},
        "model": {"hidden_dim": 32, "token_dim": 32, "context_dim": 32, "num_layers": 2},
        "training": {
            "baseline": "fixed_energy",
            "seed": 2,
            "output_dir": str(tmp_path / "runs"),
            "epochs": 1,
            "batch_size": 8,
            "lr": 0.001,
            "weight_decay": 0.0,
            "train_samples_per_epoch": 24,
            "val_samples": 16,
            "max_train_steps_per_epoch": 2,
            "device": "cpu",
        },
    }
    train_config_path = tmp_path / "train_baseline.yaml"
    train_config_path.write_text(yaml.safe_dump(train_config), encoding="utf-8")

    metrics = train_baseline(train_config_path)
    assert "final_val" in metrics
    assert (tmp_path / "runs" / "fixed_energy_seed2" / "metrics.json").exists()
    assert (tmp_path / "runs" / "fixed_energy_seed2" / "checkpoints" / "best.pt").exists()
