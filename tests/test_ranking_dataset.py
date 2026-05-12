from pathlib import Path

import torch
import yaml

from lowm.data.dataset import (
    LOWMSynthRankingDataset,
    RankingConfig,
    make_ranking_dataloader,
    validate_ranking_sample,
)
from lowm.data.generate_dataset import generate_dataset
from lowm.data.negatives import REQUIRED_NEGATIVE_TYPES, is_law_mismatch


def _make_npz(tmp_path: Path) -> Path:
    config = {
        "dataset": "LOWM-Synth",
        "version": "ranking-test",
        "seed": 41,
        "simulation": {"T": 12, "nmax": 6},
        "splits": {
            "train": {
                "num_episodes": 80,
                "n_min": 3,
                "n_max": 5,
                "parameter_split": "iid",
                "seed": 41,
            }
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    out = tmp_path / "dataset"
    generate_dataset(config_path, out, ["train"])
    return out / "train.npz"


def test_dataset_item_shapes(tmp_path: Path) -> None:
    path = _make_npz(tmp_path)
    cfg = RankingConfig(K=4, H=5, M=5, seed=3)
    dataset = LOWMSynthRankingDataset(path, cfg, num_samples=16)
    sample = dataset[0]

    assert tuple(sample["context_states"].shape) == (4, 2, 6, 7)
    assert tuple(sample["context_actions"].shape) == (4, 6, 2)
    assert tuple(sample["context_mask"].shape) == (4, 2, 6)
    assert tuple(sample["cand_states"].shape) == (5, 6, 6, 7)
    assert tuple(sample["cand_actions"].shape) == (5, 5, 6, 2)
    assert tuple(sample["cand_mask"].shape) == (5, 6, 6)

    report = validate_ranking_sample(sample, cfg)
    assert report["ok"], report["errors"]


def test_dataloader_batch_shapes(tmp_path: Path) -> None:
    path = _make_npz(tmp_path)
    cfg = RankingConfig(K=3, H=4, M=5, seed=4)
    dataset = LOWMSynthRankingDataset(path, cfg, num_samples=12)
    loader = make_ranking_dataloader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    assert tuple(batch["context_states"].shape) == (4, 3, 2, 6, 7)
    assert tuple(batch["context_actions"].shape) == (4, 3, 6, 2)
    assert tuple(batch["context_mask"].shape) == (4, 3, 2, 6)
    assert tuple(batch["cand_states"].shape) == (4, 5, 5, 6, 7)
    assert tuple(batch["cand_actions"].shape) == (4, 5, 4, 6, 2)
    assert tuple(batch["cand_mask"].shape) == (4, 5, 5, 6)
    assert tuple(batch["labels"].shape) == (4,)
    assert len(batch["negative_types"]) == 4


def test_distinct_operator_batch_sampler_spreads_ops_when_possible(tmp_path: Path) -> None:
    path = _make_npz(tmp_path)
    cfg = RankingConfig(K=3, H=4, M=5, seed=4, ensure_distinct_operators_in_batch=True)
    dataset = LOWMSynthRankingDataset(path, cfg, num_samples=16)
    loader = make_ranking_dataloader(dataset, batch_size=4, shuffle=True, ensure_distinct_operators_in_batch=True)
    batch = next(iter(loader))

    assert tuple(batch["pos_states"].shape) == (4, 5, 6, 7)
    assert torch.unique(batch["query_op_id"]).numel() >= 2


def test_negative_sampler_produces_all_required_types(tmp_path: Path) -> None:
    path = _make_npz(tmp_path)
    cfg = RankingConfig(K=4, H=5, M=5, seed=5)
    dataset = LOWMSynthRankingDataset(path, cfg, num_samples=20)
    seen: set[str] = set()
    for idx in range(12):
        sample = dataset[idx]
        seen.update(t for t in sample["negative_types"] if t != "positive")
    assert set(REQUIRED_NEGATIVE_TYPES).issubset(seen)


def test_candidate_labels_are_valid_and_law_mismatch_is_different(tmp_path: Path) -> None:
    path = _make_npz(tmp_path)
    cfg = RankingConfig(K=4, H=5, M=5, seed=6)
    dataset = LOWMSynthRankingDataset(path, cfg, num_samples=20)

    for idx in range(10):
        sample = dataset[idx]
        label = int(sample["labels"].item())
        assert 0 <= label < cfg.M
        assert sample["is_positive"].sum().item() == 1
        assert sample["is_positive"][label].item()
        assert sample["negative_types"].count("positive") == 1

        query_op_id = int(sample["query_op_id"].item())
        query_params = sample["query_op_params"].numpy()
        candidate_op_id = sample["candidate_op_id"].numpy()
        candidate_params = sample["candidate_op_params"].numpy()
        for cand_idx, neg_type in enumerate(sample["negative_types"]):
            if neg_type == "law_mismatch":
                assert is_law_mismatch(
                    query_op_id,
                    query_params,
                    int(candidate_op_id[cand_idx]),
                    candidate_params[cand_idx],
                    cfg.min_law_param_distance,
                )
