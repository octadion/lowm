import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from lowm.data.dataset import LOWMSynthRankingDataset, RankingConfig, ranking_collate
from lowm.data.simulate import SimulationConfig, simulate_split
from lowm.eval import evaluate_lowm_g as eval_lowm_g_module
from lowm.eval.evaluate_lowm_g import evaluate_lowm_g
from lowm.models.lowm import LOWM, lowm_config_from_mapping
from lowm.models.lowm_g import OperatorConditionedProposalModel, lowm_g_config_from_mapping, masked_rollout_mse
from lowm.training.train_lowm_g import train_one_epoch


def _dataset(tmp_path: Path) -> tuple[Path, LOWMSynthRankingDataset]:
    data = tmp_path / "data.npz"
    arrays = simulate_split(seed=61, cfg=SimulationConfig(num_episodes=28, T=6, n_min=3, n_max=5, nmax=6))
    np.savez_compressed(data, **arrays)
    return data, LOWMSynthRankingDataset(data, RankingConfig(K=3, H=2, M=4, seed=4, min_law_param_distance=0.05))


def _proposal_config(data_root: Path, dataset_config: Path) -> dict:
    return {
        "data": {"root": str(data_root), "train_split": "train.npz", "val_split": "val.npz", "test_iid_split": "test_iid.npz", "generate_if_missing": False, "dataset_config": str(dataset_config)},
        "ranking": {"K": 3, "H": 2, "M": 4, "seed": 4, "min_law_param_distance": 0.05},
        "model": {"object_dim": 7, "action_dim": 2, "lambda_dim": 4, "context_dim": 16, "proposal_hidden_dim": 16, "proposal_num_layers": 1, "proposal_noise_dim": 3, "proposal_use_noise": True},
        "training": {"batch_size": 4, "seed": 0, "alpha_delta": 0.0, "alpha_smooth": 0.0},
    }


def _write_runs(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_root = tmp_path / "data"
    data_root.mkdir()
    arrays = simulate_split(seed=71, cfg=SimulationConfig(num_episodes=32, T=6, n_min=3, n_max=5, nmax=6))
    for split in ["train", "val", "test_iid"]:
        np.savez_compressed(data_root / f"{split}.npz", **arrays)
    dataset_config = tmp_path / "dataset.yaml"
    dataset_config.write_text(yaml.safe_dump({"simulation": {"T": 6, "nmax": 6}, "splits": {"train": {"num_episodes": 32}, "val": {"num_episodes": 32}, "test_iid": {"num_episodes": 32}}}), encoding="utf-8")
    proposal_config = _proposal_config(data_root, dataset_config)
    proposal = OperatorConditionedProposalModel(lowm_g_config_from_mapping(proposal_config))
    proposal_run = tmp_path / "proposal"
    (proposal_run / "checkpoints").mkdir(parents=True)
    (proposal_run / "config.yaml").write_text(yaml.safe_dump(proposal_config), encoding="utf-8")
    torch.save({"model_state": proposal.state_dict(), "config": proposal_config}, proposal_run / "checkpoints" / "best_pred.pt")

    critic_config = {
        "data": proposal_config["data"],
        "ranking": proposal_config["ranking"],
        "model": {"object_dim": 7, "action_dim": 2, "lambda_dim": 4, "hidden_dim": 16, "context_dim": 16, "num_layers": 1, "use_pairwise_energy": False, "use_mu_eval": True},
        "training": {"batch_size": 4, "seed": 0},
    }
    critic = LOWM(lowm_config_from_mapping(critic_config))
    critic_run = tmp_path / "critic"
    (critic_run / "checkpoints").mkdir(parents=True)
    (critic_run / "config.yaml").write_text(yaml.safe_dump(critic_config), encoding="utf-8")
    torch.save({"model_state": critic.state_dict(), "config": critic_config}, critic_run / "checkpoints" / "best_law_pair.pt")
    critic2_run = tmp_path / "critic2"
    (critic2_run / "checkpoints").mkdir(parents=True)
    (critic2_run / "config.yaml").write_text(yaml.safe_dump(critic_config), encoding="utf-8")
    torch.save({"model_state": critic.state_dict(), "config": critic_config}, critic2_run / "checkpoints" / "best_law_pair.pt")
    return proposal_run, critic_run, critic2_run


def test_proposal_forward_shape_and_s0_fixed(tmp_path: Path) -> None:
    _, dataset = _dataset(tmp_path)
    batch = ranking_collate([dataset[0], dataset[1]])
    model = OperatorConditionedProposalModel(lowm_g_config_from_mapping({"model": {"lambda_dim": 4, "context_dim": 16, "proposal_hidden_dim": 16, "proposal_num_layers": 1, "proposal_noise_dim": 3}}))
    out = model(batch["context_states"], batch["context_actions"], batch["context_mask"], batch["pos_states"][:, 0], batch["pos_actions"], batch["pos_mask"])
    assert out["pred_states"].shape == batch["pos_states"].shape
    assert torch.allclose(out["pred_states"][:, 0], batch["pos_states"][:, 0])


def test_train_lowm_g_one_step_without_nan(tmp_path: Path) -> None:
    _, dataset = _dataset(tmp_path)
    loader = torch.utils.data.DataLoader([dataset[0], dataset[1], dataset[2], dataset[3]], batch_size=2, collate_fn=ranking_collate)
    model = OperatorConditionedProposalModel(lowm_g_config_from_mapping({"model": {"lambda_dim": 4, "context_dim": 16, "proposal_hidden_dim": 16, "proposal_num_layers": 1, "proposal_noise_dim": 3}}))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = train_one_epoch(model, loader, opt, torch.device("cpu"), alpha_delta=0.0, alpha_smooth=0.0, noise_scale=1.0, max_steps=1)
    assert np.isfinite(metrics["pred_mse"])
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_evaluate_lowm_g_outputs_and_candidates(tmp_path: Path) -> None:
    proposal_run, critic_run, _ = _write_runs(tmp_path)
    metrics = evaluate_lowm_g(proposal_run, critic_run, split="test_iid", num_samples=2, num_candidates=4, context_length=2, candidate_noise_scale=0.1, device_name="cpu")
    out = proposal_run / "eval" / "test_iid" / "lowm_g_rerank"
    assert metrics["num_samples"] == 2
    assert (out / "metrics.json").exists()
    assert (out / "per_sample_candidates.csv").exists()
    assert (out / "selected_candidates.csv").exists()
    rows = (out / "per_sample_candidates.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1 + 2 * 4


def test_cross_critic_comparison_uses_candidate_set(tmp_path: Path) -> None:
    proposal_run, critic_run, critic2_run = _write_runs(tmp_path)
    out = tmp_path / "cross"
    metrics = evaluate_lowm_g(proposal_run, None, split="test_iid", num_samples=2, num_candidates=4, context_length=2, candidate_noise_scale=0.1, compare_critics=[critic_run, critic2_run], out_dir=out, device_name="cpu")
    assert "cross_critic" in metrics
    assert (out / "cross_critic_rerank.csv").exists()
    assert (out / "cross_critic_summary.json").exists()


def test_evaluate_lowm_g_cli_creates_metrics(tmp_path: Path, monkeypatch) -> None:
    proposal_run, critic_run, _ = _write_runs(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_lowm_g",
            "--proposal-run",
            str(proposal_run),
            "--critic-run",
            str(critic_run),
            "--split",
            "test_iid",
            "--num-samples",
            "1",
            "--num-candidates",
            "3",
            "--context-length",
            "2",
            "--candidate-noise-scale",
            "0.1",
            "--device",
            "cpu",
        ],
    )
    eval_lowm_g_module.main()
    assert (proposal_run / "eval" / "test_iid" / "lowm_g_rerank" / "metrics.json").exists()


def test_counterfactual_mode_runs_or_skips_gracefully(tmp_path: Path) -> None:
    proposal_run, critic_run, _ = _write_runs(tmp_path)
    metrics = evaluate_lowm_g(proposal_run, critic_run, split="test_iid", num_samples=1, num_candidates=3, context_length=2, candidate_noise_scale=0.1, mode="counterfactual", device_name="cpu")
    assert "num_samples" in metrics
