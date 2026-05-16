import json
from pathlib import Path

import numpy as np
import torch
import yaml

from lowm.data.dataset import LOWMSynthRankingDataset, RankingConfig, ranking_collate
from lowm.data.simulate import SimulationConfig, simulate_split
from lowm.eval.aggregate_ebtwm_shaping import aggregate_ebtwm_shaping
from lowm.eval.energy_gradient_diagnostic import evaluate_gradient_diagnostic
from lowm.models.lowm import LOWM, lowm_config_from_mapping
from lowm.training.losses import denoising_energy_shaping_loss
from lowm.training.run_sweep import run_sweep


def _tiny_batch(tmp_path: Path):
    data_path = tmp_path / "tiny.npz"
    arrays = simulate_split(seed=51, cfg=SimulationConfig(num_episodes=24, T=6, n_min=3, n_max=5, nmax=6))
    np.savez_compressed(data_path, **arrays)
    dataset = LOWMSynthRankingDataset(data_path, RankingConfig(K=3, H=2, M=4, seed=9))
    batch = ranking_collate([dataset[0], dataset[1], dataset[2]])
    config = {
        "model": {
            "object_dim": 7,
            "action_dim": 2,
            "lambda_dim": 4,
            "hidden_dim": 16,
            "context_dim": 16,
            "num_layers": 1,
            "use_pairwise_energy": False,
            "use_mu_eval": True,
        }
    }
    return batch, LOWM(lowm_config_from_mapping(config)), data_path, config


def test_dsm_loss_runs_finite_and_target_shape_future_only(tmp_path: Path) -> None:
    batch, model, _, _ = _tiny_batch(tmp_path)
    out = model(batch)
    losses = denoising_energy_shaping_loss(
        model,
        batch,
        out["lambda"],
        torch.ones(4),
        noise_stds=[0.05],
        future_only=True,
        create_graph=True,
    )
    assert losses["dsm_loss"].ndim == 0
    assert torch.isfinite(losses["dsm_loss"])
    assert torch.isfinite(losses["dsm_target_norm"])
    assert torch.isfinite(losses["dsm_cosine_to_clean_direction"])


def test_training_step_with_dsm_updates_without_nan(tmp_path: Path) -> None:
    batch, model, _, _ = _tiny_batch(tmp_path)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    opt.zero_grad(set_to_none=True)
    out = model(batch)
    losses = denoising_energy_shaping_loss(model, batch, out["lambda"], torch.ones(4), noise_stds=[0.05], create_graph=True)
    total = out["energies"].mean() + losses["dsm_loss"] + 0.1 * losses["denoise_rank_loss"]
    total.backward()
    opt.step()
    assert torch.isfinite(total)
    assert any(not torch.allclose(value, before[name]) for name, value in model.state_dict().items())
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())


def _write_run(tmp_path: Path) -> Path:
    batch, model, data_path, config = _tiny_batch(tmp_path)
    data_root = tmp_path / "data"
    data_root.mkdir()
    for split in ["val", "test_iid"]:
        (data_root / f"{split}.npz").write_bytes(data_path.read_bytes())
    dataset_config = tmp_path / "dataset.yaml"
    dataset_config.write_text(yaml.safe_dump({"simulation": {"T": 6, "nmax": 6}, "splits": {"val": {"num_episodes": 24}, "test_iid": {"num_episodes": 24}}}), encoding="utf-8")
    run_config = {
        "data": {"root": str(data_root), "val_split": "val.npz", "test_iid_split": "test_iid.npz", "generate_if_missing": False, "dataset_config": str(dataset_config)},
        "ranking": {"K": 3, "H": 2, "M": 4, "seed": 9, "min_law_param_distance": 0.05},
        **config,
        "training": {"batch_size": 4, "seed": 0},
    }
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump(run_config), encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "config": run_config}, run / "checkpoints" / "best_law_pair.pt")
    return run


def test_energy_gradient_diagnostic_outputs_files(tmp_path: Path) -> None:
    run = _write_run(tmp_path)
    metrics = evaluate_gradient_diagnostic(run, split="test_iid", checkpoint="best_law_pair.pt", num_samples=2, noise_std=0.01, device_name="cpu")
    out = run / "eval" / "test_iid" / "gradient_diagnostic"
    assert "fraction_cosine_positive" in metrics
    assert (out / "metrics.json").exists()
    assert (out / "per_sample.csv").exists()
    assert (out / "cosine_histogram.png").exists()


def test_debug_sweep_config_loads(tmp_path: Path) -> None:
    cfg = yaml.safe_load(Path("configs/sweeps/ebtwm_shaping_alpha_debug.yaml").read_text(encoding="utf-8"))
    cfg["sweep_dir"] = str(tmp_path / "debug_sweep")
    sweep = tmp_path / "debug.yaml"
    sweep.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    paths = run_sweep(sweep, dry_run=True, max_runs=1)
    assert len(paths) == 1


def test_aggregate_ebtwm_shaping_handles_missing_runs(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    run = sweep / "runs" / "missing_eval"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump({"training": {"alpha_dsm": 1.0}, "sweep_params": {"alpha_dsm": 1.0}}), encoding="utf-8")
    (sweep / "manifest.json").write_text(json.dumps({"runs": [str(run)]}), encoding="utf-8")
    out = tmp_path / "summary"
    rows = aggregate_ebtwm_shaping(sweep, out)
    assert rows
    assert (out / "ebtwm_shaping_summary.csv").exists()
    assert (out / "ebtwm_shaping_summary.md").exists()
