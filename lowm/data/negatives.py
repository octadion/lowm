"""Negative sampling utilities for LOWM-Synth ranking tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


REQUIRED_NEGATIVE_TYPES = (
    "state_corrupted",
    "temporal_shuffled",
    "law_mismatch",
    "random_impossible",
)


@dataclass(frozen=True)
class Candidate:
    states: np.ndarray
    actions: np.ndarray
    mask: np.ndarray
    candidate_type: str
    source_episode: int
    op_id: int
    op_params: np.ndarray
    is_positive: bool = False


def _active_future(mask: np.ndarray) -> np.ndarray:
    return mask > 0.5


def make_state_corrupted(
    states: np.ndarray,
    actions: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    position_scale: float = 0.06,
    velocity_scale: float = 0.18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neg_states = states.copy()
    active = _active_future(mask[1:])
    if active.any():
        pos_noise = rng.normal(0.0, position_scale, size=neg_states[1:, :, 0:2].shape).astype(np.float32)
        vel_noise = rng.normal(0.0, velocity_scale, size=neg_states[1:, :, 2:4].shape).astype(np.float32)
        neg_states[1:, :, 0:2][active] += pos_noise[active]
        neg_states[1:, :, 2:4][active] += vel_noise[active]
        neg_states[1:, :, 0:2] = np.clip(neg_states[1:, :, 0:2], 0.0, 1.0)
    return neg_states, actions.copy(), mask.copy()


def make_temporal_shuffled(
    states: np.ndarray,
    actions: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neg_states = states.copy()
    neg_actions = actions.copy()
    horizon = actions.shape[0]
    if horizon <= 1:
        return neg_states, neg_actions, mask.copy()
    if rng.random() < 0.5:
        neg_states[1:] = neg_states[1:][::-1]
        neg_actions[:] = neg_actions[::-1]
    else:
        perm = rng.permutation(np.arange(1, horizon + 1))
        if np.all(perm == np.arange(1, horizon + 1)):
            perm = perm[::-1]
        neg_states[1:] = states[perm]
        neg_actions[:] = actions[rng.permutation(horizon)]
    return neg_states, neg_actions, mask.copy()


def make_random_impossible(
    states: np.ndarray,
    actions: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neg_states = states.copy()
    neg_actions = actions.copy()
    active = _active_future(mask[1:])
    if active.any():
        neg_states[1:, :, 0:2][active] = rng.uniform(0.0, 1.0, size=neg_states[1:, :, 0:2][active].shape)
        neg_states[1:, :, 2:4][active] = rng.uniform(-4.0, 4.0, size=neg_states[1:, :, 2:4][active].shape)
        if states.shape[1] >= 2:
            center = rng.uniform(0.35, 0.65, size=2).astype(np.float32)
            active_first = mask[1] > 0.5
            neg_states[1, active_first, 0:2] = center
    neg_actions[:] = rng.uniform(-1.5, 1.5, size=neg_actions.shape)
    return neg_states.astype(np.float32), neg_actions.astype(np.float32), mask.copy()


def law_distance(op_id_a: int, params_a: np.ndarray, op_id_b: int, params_b: np.ndarray) -> float:
    if int(op_id_a) != int(op_id_b):
        return float("inf")
    return float(np.linalg.norm(np.asarray(params_a, dtype=np.float32) - np.asarray(params_b, dtype=np.float32)))


def is_law_mismatch(
    op_id_a: int,
    params_a: np.ndarray,
    op_id_b: int,
    params_b: np.ndarray,
    min_param_distance: float = 0.15,
) -> bool:
    return int(op_id_a) != int(op_id_b) or law_distance(op_id_a, params_a, op_id_b, params_b) >= min_param_distance


def choose_negative_types(num_negatives: int, rng: np.random.Generator, types: Sequence[str] = REQUIRED_NEGATIVE_TYPES) -> list[str]:
    if num_negatives < 1:
        return []
    chosen: list[str] = []
    base = list(types)
    while len(chosen) < num_negatives:
        order = [str(name) for name in rng.permutation(base)]
        chosen.extend(order)
    return chosen[:num_negatives]
