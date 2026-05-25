"""PyTorch ranking dataset for LOWM-Synth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset

from lowm.data.negatives import (
    Candidate,
    REQUIRED_NEGATIVE_TYPES,
    choose_negative_types,
    is_law_mismatch,
    make_random_impossible,
    make_state_corrupted,
    make_temporal_shuffled,
)


@dataclass(frozen=True)
class RankingConfig:
    K: int = 8
    H: int = 6
    M: int = 5
    seed: int = 123
    negative_types: tuple[str, ...] = REQUIRED_NEGATIVE_TYPES
    min_law_param_distance: float = 0.15
    ensure_distinct_operators_in_batch: bool = False


def ranking_config_from_mapping(config: Mapping[str, Any]) -> RankingConfig:
    ranking = dict(config.get("ranking", {}))
    negative_types = ranking.get("negative_types", REQUIRED_NEGATIVE_TYPES)
    return RankingConfig(
        K=int(ranking.get("K", RankingConfig.K)),
        H=int(ranking.get("H", RankingConfig.H)),
        M=int(ranking.get("M", RankingConfig.M)),
        seed=int(ranking.get("seed", RankingConfig.seed)),
        negative_types=tuple(str(x) for x in negative_types),
        min_law_param_distance=float(ranking.get("min_law_param_distance", RankingConfig.min_law_param_distance)),
        ensure_distinct_operators_in_batch=bool(
            ranking.get(
                "ensure_distinct_operators_in_batch",
                config.get("training", {}).get("ensure_distinct_operators_in_batch", RankingConfig.ensure_distinct_operators_in_batch)
                if isinstance(config.get("training", {}), Mapping)
                else RankingConfig.ensure_distinct_operators_in_batch,
            )
        ),
    )


class LOWMSynthRankingDataset(Dataset[dict[str, Any]]):
    """Context-query ranking samples from raw LOWM-Synth episodes.

    Returned item tensor shapes:
    context_states: [K, 2, Nmax, D]
    context_actions: [K, Nmax, 2]
    context_mask: [K, 2, Nmax]
    cand_states: [M, H+1, Nmax, D]
    cand_actions: [M, H, Nmax, 2]
    cand_mask: [M, H+1, Nmax]
    labels: scalar index of the positive candidate
    """

    def __init__(
        self,
        path: str | Path,
        ranking_config: RankingConfig | None = None,
        num_samples: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.cfg = ranking_config or RankingConfig()
        if self.cfg.K <= 0:
            raise ValueError("K must be positive")
        if self.cfg.H <= 0:
            raise ValueError("H must be positive")
        if self.cfg.M < 2:
            raise ValueError("M must include one positive and at least one negative")

        with np.load(self.path) as data:
            self.is_paired_context = "context_states" in data.files and "positive_states" in data.files
            if self.is_paired_context:
                self.context_source_states = data["context_states"].astype(np.float32)
                self.states = data["positive_states"].astype(np.float32)
                self.context_source_mask = data["context_mask"].astype(np.float32) if "context_mask" in data.files else np.ones(
                    self.context_source_states.shape[:3], dtype=np.float32
                )
                self.mask = data["positive_mask"].astype(np.float32) if "positive_mask" in data.files else (
                    data["mask"].astype(np.float32) if "mask" in data.files else np.ones(self.states.shape[:3], dtype=np.float32)
                )
                if "context_actions" in data.files:
                    self.context_source_actions = data["context_actions"].astype(np.float32)
                else:
                    self.context_source_actions = np.zeros(
                        (self.context_source_states.shape[0], self.context_source_states.shape[1] - 1, self.context_source_states.shape[2], 2),
                        dtype=np.float32,
                    )
                if "positive_actions" in data.files:
                    self.actions = data["positive_actions"].astype(np.float32)
                elif "actions" in data.files:
                    self.actions = data["actions"].astype(np.float32)
                else:
                    self.actions = np.zeros((self.states.shape[0], self.states.shape[1] - 1, self.states.shape[2], 2), dtype=np.float32)
            else:
                self.context_source_states = None
                self.context_source_actions = None
                self.context_source_mask = None
                self.states = data["states"].astype(np.float32)
                self.actions = data["actions"].astype(np.float32)
                self.mask = data["mask"].astype(np.float32)
            self.op_id = data["op_id"].astype(np.int64)
            self.op_params = data["op_params"].astype(np.float32)
            self.num_objects = data["num_objects"].astype(np.int64)

        self.num_episodes, self.tp1, self.nmax, self.d_object = self.states.shape
        self.T = self.tp1 - 1
        if self.is_paired_context:
            if self.context_source_states is None or self.context_source_actions is None or self.context_source_mask is None:
                raise ValueError("paired context dataset did not initialize context arrays")
            if self.context_source_states.shape != self.states.shape:
                raise ValueError("context_states and positive_states must have the same shape")
            if self.context_source_actions.shape != (self.num_episodes, self.T, self.nmax, 2):
                raise ValueError("context_actions shape does not match context_states")
            if self.context_source_mask.shape != (self.num_episodes, self.tp1, self.nmax):
                raise ValueError("context_mask shape does not match context_states")
        if self.actions.shape != (self.num_episodes, self.T, self.nmax, 2):
            raise ValueError("actions shape does not match states")
        if self.mask.shape != (self.num_episodes, self.tp1, self.nmax):
            raise ValueError("mask shape does not match states")
        if self.cfg.H > self.T:
            raise ValueError(f"H={self.cfg.H} exceeds episode horizon T={self.T}")

        self.num_samples = int(num_samples or self.num_episodes)
        self._by_op: dict[int, np.ndarray] = {
            int(op): np.where(self.op_id == op)[0] for op in np.unique(self.op_id)
        }
        self._all_indices = np.arange(self.num_episodes)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = np.random.default_rng(self.cfg.seed + int(idx))
        query_ep = int(idx % self.num_episodes)
        query_t0 = int(rng.integers(0, self.T - self.cfg.H + 1))

        context_states, context_actions, context_mask, context_episodes, context_times = self._sample_context(
            rng, query_ep, query_t0
        )
        positive = self._positive_candidate(query_ep, query_t0)
        candidates = [positive]
        for neg_type in choose_negative_types(self.cfg.M - 1, rng, self.cfg.negative_types):
            candidates.append(self._make_negative(neg_type, positive, rng))

        order = rng.permutation(len(candidates))
        shuffled = [candidates[int(i)] for i in order]
        label = int(np.where(order == 0)[0][0])

        return {
            "context_states": torch.from_numpy(context_states),
            "context_actions": torch.from_numpy(context_actions),
            "context_mask": torch.from_numpy(context_mask),
            "cand_states": torch.from_numpy(np.stack([c.states for c in shuffled]).astype(np.float32)),
            "cand_actions": torch.from_numpy(np.stack([c.actions for c in shuffled]).astype(np.float32)),
            "cand_mask": torch.from_numpy(np.stack([c.mask for c in shuffled]).astype(np.float32)),
            "pos_states": torch.from_numpy(positive.states.astype(np.float32)),
            "pos_actions": torch.from_numpy(positive.actions.astype(np.float32)),
            "pos_mask": torch.from_numpy(positive.mask.astype(np.float32)),
            "labels": torch.tensor(label, dtype=torch.long),
            "negative_types": [c.candidate_type for c in shuffled],
            "is_positive": torch.tensor([c.is_positive for c in shuffled], dtype=torch.bool),
            "candidate_op_id": torch.tensor([c.op_id for c in shuffled], dtype=torch.long),
            "candidate_op_params": torch.from_numpy(np.stack([c.op_params for c in shuffled]).astype(np.float32)),
            "candidate_source_episode": torch.tensor([c.source_episode for c in shuffled], dtype=torch.long),
            "query_episode": torch.tensor(query_ep, dtype=torch.long),
            "query_t0": torch.tensor(query_t0, dtype=torch.long),
            "query_op_id": torch.tensor(int(self.op_id[query_ep]), dtype=torch.long),
            "query_op_params": torch.from_numpy(self.op_params[query_ep].copy()),
            "context_episode": torch.tensor(context_episodes, dtype=torch.long),
            "context_t": torch.tensor(context_times, dtype=torch.long),
        }

    def _sample_context(
        self,
        rng: np.random.Generator,
        query_ep: int,
        query_t0: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        context_states = np.zeros((self.cfg.K, 2, self.nmax, self.d_object), dtype=np.float32)
        context_actions = np.zeros((self.cfg.K, self.nmax, 2), dtype=np.float32)
        context_mask = np.zeros((self.cfg.K, 2, self.nmax), dtype=np.float32)
        context_episodes = np.zeros((self.cfg.K,), dtype=np.int64)
        context_times = np.zeros((self.cfg.K,), dtype=np.int64)

        if self.is_paired_context:
            if self.context_source_states is None or self.context_source_actions is None or self.context_source_mask is None:
                raise ValueError("paired context dataset did not initialize context arrays")
            valid_times = np.arange(self.T)
            for k in range(self.cfg.K):
                t = int(rng.choice(valid_times))
                context_states[k, 0] = self.context_source_states[query_ep, t]
                context_states[k, 1] = self.context_source_states[query_ep, t + 1]
                context_actions[k] = self.context_source_actions[query_ep, t]
                context_mask[k, 0] = self.context_source_mask[query_ep, t]
                context_mask[k, 1] = self.context_source_mask[query_ep, t + 1]
                context_episodes[k] = query_ep
                context_times[k] = t
            return context_states, context_actions, context_mask, context_episodes, context_times

        same_op = self._by_op.get(int(self.op_id[query_ep]), self._all_indices)
        preferred = same_op[same_op != query_ep]
        episode_pool = preferred if len(preferred) > 0 else same_op
        if len(episode_pool) == 0:
            episode_pool = self._all_indices

        forbidden = set(range(query_t0, query_t0 + self.cfg.H))
        for k in range(self.cfg.K):
            ep = int(rng.choice(episode_pool))
            valid_times = np.arange(self.T)
            if ep == query_ep:
                valid_times = np.asarray([t for t in valid_times if t not in forbidden], dtype=np.int64)
            if len(valid_times) == 0:
                valid_times = np.arange(self.T)
            t = int(rng.choice(valid_times))
            context_states[k, 0] = self.states[ep, t]
            context_states[k, 1] = self.states[ep, t + 1]
            context_actions[k] = self.actions[ep, t]
            context_mask[k, 0] = self.mask[ep, t]
            context_mask[k, 1] = self.mask[ep, t + 1]
            context_episodes[k] = ep
            context_times[k] = t
        return context_states, context_actions, context_mask, context_episodes, context_times

    def _segment(self, ep: int, t0: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.states[ep, t0 : t0 + self.cfg.H + 1].copy(),
            self.actions[ep, t0 : t0 + self.cfg.H].copy(),
            self.mask[ep, t0 : t0 + self.cfg.H + 1].copy(),
        )

    def _positive_candidate(self, ep: int, t0: int) -> Candidate:
        states, actions, mask = self._segment(ep, t0)
        return Candidate(
            states=states,
            actions=actions,
            mask=mask,
            candidate_type="positive",
            source_episode=ep,
            op_id=int(self.op_id[ep]),
            op_params=self.op_params[ep].copy(),
            is_positive=True,
        )

    def _make_negative(self, neg_type: str, positive: Candidate, rng: np.random.Generator) -> Candidate:
        if neg_type == "state_corrupted":
            states, actions, mask = make_state_corrupted(positive.states, positive.actions, positive.mask, rng)
            return Candidate(states, actions, mask, neg_type, positive.source_episode, positive.op_id, positive.op_params.copy())
        if neg_type == "temporal_shuffled":
            states, actions, mask = make_temporal_shuffled(positive.states, positive.actions, positive.mask, rng)
            return Candidate(states, actions, mask, neg_type, positive.source_episode, positive.op_id, positive.op_params.copy())
        if neg_type == "random_impossible":
            states, actions, mask = make_random_impossible(positive.states, positive.actions, positive.mask, rng)
            return Candidate(states, actions, mask, neg_type, positive.source_episode, positive.op_id, positive.op_params.copy())
        if neg_type == "law_mismatch":
            ep = self._sample_law_mismatch_episode(positive, rng)
            t0 = int(rng.integers(0, self.T - self.cfg.H + 1))
            states, actions, mask = self._segment(ep, t0)
            return Candidate(
                states=states,
                actions=actions,
                mask=mask,
                candidate_type=neg_type,
                source_episode=ep,
                op_id=int(self.op_id[ep]),
                op_params=self.op_params[ep].copy(),
            )
        raise ValueError(f"unknown negative type '{neg_type}'")

    def _sample_law_mismatch_episode(self, positive: Candidate, rng: np.random.Generator) -> int:
        different_op = self._all_indices[self.op_id != positive.op_id]
        if len(different_op) > 0:
            return int(rng.choice(different_op))
        candidates = [
            int(i)
            for i in self._all_indices
            if i != positive.source_episode
            and is_law_mismatch(
                positive.op_id,
                positive.op_params,
                int(self.op_id[i]),
                self.op_params[i],
                self.cfg.min_law_param_distance,
            )
        ]
        if candidates:
            return int(rng.choice(candidates))
        fallback = self._all_indices[self._all_indices != positive.source_episode]
        if len(fallback) == 0:
            raise ValueError("cannot sample law_mismatch from a single-episode dataset")
        return int(rng.choice(fallback))


def ranking_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = [
        "context_states",
        "context_actions",
        "context_mask",
        "cand_states",
        "cand_actions",
        "cand_mask",
        "pos_states",
        "pos_actions",
        "pos_mask",
        "labels",
        "is_positive",
        "candidate_op_id",
        "candidate_op_params",
        "candidate_source_episode",
        "query_episode",
        "query_t0",
        "query_op_id",
        "query_op_params",
        "context_episode",
        "context_t",
    ]
    out = {key: torch.stack([item[key] for item in batch], dim=0) for key in tensor_keys}
    out["negative_types"] = [item["negative_types"] for item in batch]
    return out


class DistinctOperatorBatchSampler(BatchSampler):
    """Best-effort sampler that spreads different operator ids across a batch."""

    def __init__(self, dataset: LOWMSynthRankingDataset, batch_size: int, shuffle: bool = True) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def _episode(self, idx: int) -> int:
        return int(idx % self.dataset.num_episodes)

    def _compatible_with_batch(self, idx: int, batch: list[int]) -> bool:
        ep = self._episode(idx)
        op_id = int(self.dataset.op_id[ep])
        params = self.dataset.op_params[ep]
        for existing_idx in batch:
            existing_ep = self._episode(existing_idx)
            existing_op = int(self.dataset.op_id[existing_ep])
            if op_id != existing_op:
                continue
            distance = float(np.linalg.norm(params - self.dataset.op_params[existing_ep]))
            if distance <= self.dataset.cfg.min_law_param_distance:
                return False
        return True

    def __iter__(self):
        rng = np.random.default_rng(self.dataset.cfg.seed)
        indices = list(self.indices)
        if self.shuffle:
            rng.shuffle(indices)

        remaining = indices
        while remaining:
            batch: list[int] = []
            next_remaining: list[int] = []
            for idx in remaining:
                if len(batch) < self.batch_size and self._compatible_with_batch(idx, batch):
                    batch.append(idx)
                else:
                    next_remaining.append(idx)
            if len(batch) < self.batch_size and next_remaining:
                needed = self.batch_size - len(batch)
                batch.extend(next_remaining[:needed])
                next_remaining = next_remaining[needed:]
            remaining = next_remaining
            yield batch

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def make_ranking_dataloader(
    dataset: LOWMSynthRankingDataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    ensure_distinct_operators_in_batch: bool | None = None,
) -> DataLoader[dict[str, Any]]:
    ensure_distinct = dataset.cfg.ensure_distinct_operators_in_batch if ensure_distinct_operators_in_batch is None else ensure_distinct_operators_in_batch
    if ensure_distinct:
        return DataLoader(
            dataset,
            batch_sampler=DistinctOperatorBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle),
            num_workers=num_workers,
            collate_fn=ranking_collate,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=ranking_collate,
    )


def validate_ranking_sample(sample: Mapping[str, Any], cfg: RankingConfig, strict_law_mismatch: bool = True) -> dict[str, Any]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(tuple(sample["context_states"].shape) == (cfg.K, 2, sample["cand_states"].shape[-2], sample["cand_states"].shape[-1]), "bad context_states shape")
    require(tuple(sample["context_actions"].shape) == (cfg.K, sample["cand_actions"].shape[-2], 2), "bad context_actions shape")
    require(tuple(sample["context_mask"].shape) == (cfg.K, 2, sample["cand_mask"].shape[-1]), "bad context_mask shape")
    require(sample["cand_states"].shape[0] == cfg.M, "bad candidate count")
    require(sample["cand_states"].shape[1] == cfg.H + 1, "bad candidate horizon")
    require(sample["cand_actions"].shape[1] == cfg.H, "bad action horizon")

    is_positive = sample["is_positive"].detach().cpu().numpy().astype(bool)
    label = int(sample["labels"].item())
    require(0 <= label < cfg.M, "label out of range")
    require(int(is_positive.sum()) == 1, "sample must contain exactly one positive candidate")
    if 0 <= label < len(is_positive):
        require(bool(is_positive[label]), "label does not point to the positive candidate")

    neg_types = list(sample["negative_types"])
    require(neg_types.count("positive") == 1, "negative_types must include exactly one positive marker")
    type_counts = {name: neg_types.count(name) for name in ["positive", *REQUIRED_NEGATIVE_TYPES]}

    if strict_law_mismatch and "law_mismatch" in neg_types:
        query_op_id = int(sample["query_op_id"].item())
        query_params = sample["query_op_params"].detach().cpu().numpy()
        candidate_op_id = sample["candidate_op_id"].detach().cpu().numpy()
        candidate_params = sample["candidate_op_params"].detach().cpu().numpy()
        for i, name in enumerate(neg_types):
            if name == "law_mismatch":
                require(
                    is_law_mismatch(
                        query_op_id,
                        query_params,
                        int(candidate_op_id[i]),
                        candidate_params[i],
                        cfg.min_law_param_distance,
                    ),
                    "law_mismatch candidate is not a different law",
                )

    return {"ok": not errors, "errors": errors, "negative_type_counts": type_counts}
