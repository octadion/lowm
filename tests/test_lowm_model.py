from pathlib import Path

import torch
import yaml

from lowm.data.dataset import LOWMSynthRankingDataset, RankingConfig, make_ranking_dataloader
from lowm.data.generate_dataset import generate_dataset
from lowm.eval.metrics import RankingMetricAccumulator
from lowm.models.lowm import LOWM, LOWMConfig
from lowm.training.losses import law_stability_loss, lowm_total_loss
from lowm.training.train_lowm import train_lowm


def _make_split(tmp_path: Path, episodes: int = 64, horizon: int = 10) -> Path:
    config = {
        "dataset": "LOWM-Synth",
        "version": "lowm-test",
        "seed": 71,
        "simulation": {"T": horizon, "nmax": 6},
        "splits": {
            "train": {
                "num_episodes": episodes,
                "n_min": 3,
                "n_max": 5,
                "parameter_split": "iid",
                "seed": 71,
            }
        },
    }
    config_path = tmp_path / "data_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    out = tmp_path / "dataset"
    generate_dataset(config_path, out, ["train"])
    return out / "train.npz"


def _batch(tmp_path: Path, batch_size: int = 4, m: int = 5):
    path = _make_split(tmp_path)
    ranking_cfg = RankingConfig(K=4, H=4, M=m, seed=9)
    dataset = LOWMSynthRankingDataset(path, ranking_cfg, num_samples=32)
    loader = make_ranking_dataloader(dataset, batch_size=batch_size, shuffle=False)
    return next(iter(loader)), dataset


def test_lowm_forward_shape(tmp_path: Path) -> None:
    batch, _ = _batch(tmp_path)
    model = LOWM(LOWMConfig(lambda_dim=16, hidden_dim=32, context_dim=32))
    output = model(batch)

    assert tuple(output["energies"].shape) == (4, 5)
    assert tuple(output["mu"].shape) == (4, 16)
    assert tuple(output["logvar"].shape) == (4, 16)
    assert tuple(output["lambda"].shape) == (4, 16)


def test_lowm_backward_pass(tmp_path: Path) -> None:
    batch, _ = _batch(tmp_path)
    model = LOWM(LOWMConfig(lambda_dim=8, hidden_dim=32, context_dim=32))
    output = model(batch)
    mid = batch["context_states"].shape[1] // 2
    mu_a, _ = model.context_encoder(batch["context_states"][:, :mid], batch["context_actions"][:, :mid], batch["context_mask"][:, :mid])
    mu_b, _ = model.context_encoder(batch["context_states"][:, mid:], batch["context_actions"][:, mid:], batch["context_mask"][:, mid:])
    losses = lowm_total_loss(
        output["energies"],
        batch["labels"],
        output["mu"],
        output["logvar"],
        beta_kl=1e-4,
        stability=law_stability_loss(mu_a, mu_b),
        alpha_stable=0.1,
    )
    losses["total"].backward()
    grad_norm = sum(float(p.grad.abs().sum().item()) for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0.0


def test_lowm_masks_ignore_padded_objects(tmp_path: Path) -> None:
    batch, _ = _batch(tmp_path, batch_size=2)
    model = LOWM(LOWMConfig(lambda_dim=8, hidden_dim=32, context_dim=32, use_mu_eval=True))
    model.eval()
    with torch.no_grad():
        original = model(batch)["energies"]
        modified = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
        cand_pad = modified["cand_mask"] == 0
        modified["cand_states"][cand_pad] = 999.0
        context_pad = modified["context_mask"] == 0
        context_pad_states = context_pad.unsqueeze(-1).expand_as(modified["context_states"])
        modified["context_states"][context_pad_states] = -999.0
        changed = model(modified)["energies"]
    assert torch.allclose(original, changed, atol=1e-5)


def test_lowm_eval_mode_with_mu_is_deterministic(tmp_path: Path) -> None:
    batch, _ = _batch(tmp_path)
    model = LOWM(LOWMConfig(lambda_dim=8, hidden_dim=32, context_dim=32, use_mu_eval=True))
    model.eval()
    with torch.no_grad():
        first = model(batch)
        second = model(batch)
    assert torch.allclose(first["lambda"], first["mu"])
    assert torch.allclose(first["energies"], second["energies"])


def test_lowm_small_overfit_better_than_random(tmp_path: Path) -> None:
    torch.manual_seed(0)
    path = _make_split(tmp_path, episodes=48)
    ranking_cfg = RankingConfig(K=4, H=4, M=2, seed=13, negative_types=("random_impossible",))
    dataset = LOWMSynthRankingDataset(path, ranking_cfg, num_samples=32)
    loader = make_ranking_dataloader(dataset, batch_size=16, shuffle=True)
    eval_loader = make_ranking_dataloader(dataset, batch_size=16, shuffle=False)
    model = LOWM(LOWMConfig(lambda_dim=4, hidden_dim=48, context_dim=48, use_pairwise_energy=False, use_mu_eval=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)

    for _ in range(12):
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            losses = lowm_total_loss(output["energies"], batch["labels"], output["mu"], output["logvar"], beta_kl=0.0)
            losses["total"].backward()
            optimizer.step()

    model.eval()
    acc = RankingMetricAccumulator()
    with torch.no_grad():
        for batch in eval_loader:
            output = model(batch)
            acc.update(output["energies"], batch["labels"], batch["negative_types"])
    metrics = acc.compute()
    assert metrics["top1_acc"] > 0.5


def test_train_lowm_one_epoch(tmp_path: Path) -> None:
    data_config = {
        "dataset": "LOWM-Synth",
        "version": "lowm-train-test",
        "seed": 81,
        "simulation": {"T": 10, "nmax": 6},
        "splits": {
            "train": {"num_episodes": 48, "n_min": 3, "n_max": 5, "seed": 81},
            "val": {"num_episodes": 24, "n_min": 3, "n_max": 5, "seed": 82},
        },
    }
    data_config_path = tmp_path / "lowm_synth_test.yaml"
    data_config_path.write_text(yaml.safe_dump(data_config), encoding="utf-8")

    train_config = {
        "seed": 3,
        "data": {
            "root": str(tmp_path / "dataset"),
            "train_split": "train.npz",
            "val_split": "val.npz",
            "generate_if_missing": True,
            "dataset_config": str(data_config_path),
        },
        "ranking": {"K": 4, "H": 4, "M": 5, "seed": 17},
        "model": {"lambda_dim": 8, "hidden_dim": 32, "context_dim": 32, "num_layers": 2, "use_mu_eval": True},
        "training": {
            "seed": 3,
            "output_dir": str(tmp_path / "runs"),
            "run_name": "lowm_test_seed3",
            "epochs": 1,
            "batch_size": 8,
            "lr": 0.001,
            "weight_decay": 0.0,
            "beta_kl": 1e-4,
            "alpha_stable": 0.1,
            "use_stability": True,
            "train_samples_per_epoch": 24,
            "val_samples": 16,
            "max_train_steps_per_epoch": 2,
            "device": "cpu",
        },
    }
    train_config_path = tmp_path / "train_lowm.yaml"
    train_config_path.write_text(yaml.safe_dump(train_config), encoding="utf-8")

    metrics = train_lowm(train_config_path)
    assert "final_val" in metrics
    run_dir = tmp_path / "runs" / "lowm_test_seed3"
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "checkpoints" / "best.pt").exists()
    assert (run_dir / "checkpoints" / "best_top1.pt").exists()
    assert (run_dir / "checkpoints" / "best_loss.pt").exists()
    assert (run_dir / "checkpoints" / "best_law_pair.pt").exists()
    assert (run_dir / "checkpoints" / "best_law_gap.pt").exists()
    assert (run_dir / "checkpoints" / "last.pt").exists()
    assert metrics["selection_metric"] == "law_pair"
