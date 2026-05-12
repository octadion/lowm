"""NumPy simulator for LOWM-Synth v0."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from lowm.data.operators import OperatorRanges, ranges_from_config, sample_operator


@dataclass(frozen=True)
class SimulationConfig:
    num_episodes: int = 1000
    T: int = 20
    n_min: int = 3
    n_max: int = 5
    nmax: int = 6
    d_object: int = 7
    world_size: float = 1.0
    dt: float = 0.05
    impulse_prob: float = 0.15
    force_min: float = -0.5
    force_max: float = 0.5
    radius_min: float = 0.025
    radius_max: float = 0.055
    mass_min: float = 0.5
    mass_max: float = 2.0
    velocity_min: float = -0.25
    velocity_max: float = 0.25
    num_types: int = 3
    eps: float = 1e-4
    max_speed: float = 5.0
    wall_damping: float = 0.5
    parameter_split: str = "iid"
    operator_ranges: OperatorRanges = OperatorRanges()


def config_from_mapping(config: Mapping[str, object]) -> SimulationConfig:
    sim = dict(config.get("simulation", {}))
    actions = dict(config.get("actions", {}))
    objects = dict(config.get("objects", {}))
    operators = ranges_from_config(config.get("operators", {}))

    kwargs = {
        "T": sim.get("T", SimulationConfig.T),
        "world_size": sim.get("world_size", SimulationConfig.world_size),
        "dt": sim.get("dt", SimulationConfig.dt),
        "nmax": sim.get("nmax", SimulationConfig.nmax),
        "max_speed": sim.get("max_speed", SimulationConfig.max_speed),
        "wall_damping": sim.get("wall_damping", SimulationConfig.wall_damping),
        "impulse_prob": actions.get("impulse_prob", SimulationConfig.impulse_prob),
        "force_min": actions.get("force_min", SimulationConfig.force_min),
        "force_max": actions.get("force_max", SimulationConfig.force_max),
        "radius_min": objects.get("radius_min", SimulationConfig.radius_min),
        "radius_max": objects.get("radius_max", SimulationConfig.radius_max),
        "mass_min": objects.get("mass_min", SimulationConfig.mass_min),
        "mass_max": objects.get("mass_max", SimulationConfig.mass_max),
        "velocity_min": objects.get("velocity_min", SimulationConfig.velocity_min),
        "velocity_max": objects.get("velocity_max", SimulationConfig.velocity_max),
        "num_types": objects.get("num_types", SimulationConfig.num_types),
        "operator_ranges": operators,
    }
    return SimulationConfig(**kwargs)


def _sample_initial_state(rng: np.random.Generator, cfg: SimulationConfig, n: int) -> np.ndarray:
    if n > cfg.nmax:
        raise ValueError(f"n={n} exceeds nmax={cfg.nmax}")

    state = np.zeros((cfg.nmax, cfg.d_object), dtype=np.float32)
    radii = rng.uniform(cfg.radius_min, cfg.radius_max, size=n).astype(np.float32)
    masses = rng.uniform(cfg.mass_min, cfg.mass_max, size=n).astype(np.float32)
    types = rng.integers(0, cfg.num_types, size=n).astype(np.float32)

    positions: list[np.ndarray] = []
    for i in range(n):
        radius = float(radii[i])
        for _ in range(512):
            pos = rng.uniform(radius, cfg.world_size - radius, size=2)
            if all(np.linalg.norm(pos - prev) >= 1.15 * (radius + float(radii[j])) for j, prev in enumerate(positions)):
                positions.append(pos)
                break
        else:
            # Dense samples are rare for v0, but falling back to any valid
            # in-bounds position keeps generation from hanging.
            positions.append(rng.uniform(radius, cfg.world_size - radius, size=2))

    velocities = rng.uniform(cfg.velocity_min, cfg.velocity_max, size=(n, 2)).astype(np.float32)
    state[:n, 0:2] = np.asarray(positions, dtype=np.float32)
    state[:n, 2:4] = velocities
    state[:n, 4] = radii
    state[:n, 5] = masses
    state[:n, 6] = types
    return state


def _sample_actions(rng: np.random.Generator, cfg: SimulationConfig, n: int) -> np.ndarray:
    actions = np.zeros((cfg.T, cfg.nmax, 2), dtype=np.float32)
    for t in range(cfg.T):
        if rng.random() < cfg.impulse_prob:
            idx = int(rng.integers(0, n))
            actions[t, idx] = rng.uniform(cfg.force_min, cfg.force_max, size=2)
    return actions


def _apply_pairwise_attraction(
    pos: np.ndarray,
    masses: np.ndarray,
    k: float,
    cfg: SimulationConfig,
) -> np.ndarray:
    n = pos.shape[0]
    acc = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            delta = pos[j] - pos[i]
            dist2 = float(np.dot(delta, delta) + cfg.eps)
            force = k * delta / dist2
            acc[i] += force / max(float(masses[i]), cfg.eps)
    return acc


def _resolve_collisions(
    pos: np.ndarray,
    vel: np.ndarray,
    radii: np.ndarray,
    masses: np.ndarray,
    restitution: float,
    cfg: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    n = pos.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            delta = pos[j] - pos[i]
            dist = float(np.linalg.norm(delta))
            min_dist = float(radii[i] + radii[j])
            if dist >= min_dist:
                continue

            if dist < cfg.eps:
                normal = np.array([1.0, 0.0], dtype=np.float32)
                dist = cfg.eps
            else:
                normal = (delta / dist).astype(np.float32)

            overlap = min_dist - dist
            inv_mi = 1.0 / max(float(masses[i]), cfg.eps)
            inv_mj = 1.0 / max(float(masses[j]), cfg.eps)
            inv_sum = inv_mi + inv_mj
            pos[i] -= normal * overlap * (inv_mi / inv_sum)
            pos[j] += normal * overlap * (inv_mj / inv_sum)

            rel_vel = vel[i] - vel[j]
            closing_speed = float(np.dot(rel_vel, normal))
            if closing_speed > 0.0:
                impulse = -(1.0 + restitution) * closing_speed / inv_sum
                vel[i] += impulse * inv_mi * normal
                vel[j] -= impulse * inv_mj * normal
    return pos, vel


def _apply_boundary(
    pos: np.ndarray,
    vel: np.ndarray,
    radii: np.ndarray,
    mode: int,
    cfg: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    world = cfg.world_size
    if mode == 1:
        pos[:] = np.mod(pos, world)
        return pos, vel

    damping = cfg.wall_damping if mode == 2 else 1.0
    for i in range(pos.shape[0]):
        r = float(radii[i])
        lo = r
        hi = world - r
        for axis in range(2):
            if pos[i, axis] < lo:
                pos[i, axis] = lo
                vel[i, axis] = abs(vel[i, axis]) * damping
            elif pos[i, axis] > hi:
                pos[i, axis] = hi
                vel[i, axis] = -abs(vel[i, axis]) * damping
    return pos, vel


def step_state(
    state: np.ndarray,
    action: np.ndarray,
    n: int,
    op_id: int,
    op_params: np.ndarray,
    cfg: SimulationConfig,
) -> np.ndarray:
    """Advance one time step for active objects; padded rows remain zero."""

    next_state = np.zeros_like(state)
    active = state[:n].copy()
    pos = active[:, 0:2].copy()
    vel = active[:, 2:4].copy()
    radii = active[:, 4].copy()
    masses = active[:, 5].copy()

    force_acc = action[:n] / np.maximum(masses[:, None], cfg.eps)
    vel += force_acc * cfg.dt

    if op_id == 0:
        gravity_y = float(op_params[0])
        damping = float(op_params[1])
        vel *= damping
        vel[:, 1] += gravity_y * cfg.dt
    elif op_id == 1:
        k = float(op_params[2])
        vel += _apply_pairwise_attraction(pos, masses, k, cfg) * cfg.dt

    speeds = np.linalg.norm(vel, axis=1, keepdims=True)
    scale = np.minimum(1.0, cfg.max_speed / np.maximum(speeds, cfg.eps))
    vel *= scale
    pos += vel * cfg.dt

    restitution = float(op_params[3]) if op_id == 2 else 0.8
    pos, vel = _resolve_collisions(pos, vel, radii, masses, restitution, cfg)

    boundary_mode = int(round(float(op_params[4]))) if op_id == 3 else 0
    pos, vel = _apply_boundary(pos, vel, radii, boundary_mode, cfg)

    next_state[:n] = active
    next_state[:n, 0:2] = pos
    next_state[:n, 2:4] = vel
    return next_state


def simulate_episode(
    rng: np.random.Generator,
    cfg: SimulationConfig,
    n: int | None = None,
    op_id: int | None = None,
) -> dict[str, np.ndarray | int]:
    """Simulate one LOWM-Synth episode."""

    if n is None:
        n = int(rng.integers(cfg.n_min, cfg.n_max + 1))
    op_id, op_params = sample_operator(rng, cfg.operator_ranges, cfg.parameter_split, op_id)

    states = np.zeros((cfg.T + 1, cfg.nmax, cfg.d_object), dtype=np.float32)
    actions = _sample_actions(rng, cfg, n)
    mask = np.zeros((cfg.T + 1, cfg.nmax), dtype=np.float32)

    states[0] = _sample_initial_state(rng, cfg, n)
    mask[:, :n] = 1.0
    for t in range(cfg.T):
        states[t + 1] = step_state(states[t], actions[t], n, op_id, op_params, cfg)

    return {
        "states": states,
        "actions": actions,
        "mask": mask,
        "op_id": op_id,
        "op_params": op_params,
        "num_objects": n,
    }


def simulate_split(
    seed: int,
    cfg: SimulationConfig,
) -> dict[str, np.ndarray]:
    """Generate a complete split as stacked NumPy arrays."""

    rng = np.random.default_rng(seed)
    states = np.zeros((cfg.num_episodes, cfg.T + 1, cfg.nmax, cfg.d_object), dtype=np.float32)
    actions = np.zeros((cfg.num_episodes, cfg.T, cfg.nmax, 2), dtype=np.float32)
    mask = np.zeros((cfg.num_episodes, cfg.T + 1, cfg.nmax), dtype=np.float32)
    op_id = np.zeros((cfg.num_episodes,), dtype=np.int64)
    op_params = np.zeros((cfg.num_episodes, 5), dtype=np.float32)
    num_objects = np.zeros((cfg.num_episodes,), dtype=np.int64)

    for idx in range(cfg.num_episodes):
        episode = simulate_episode(rng, cfg)
        states[idx] = episode["states"]
        actions[idx] = episode["actions"]
        mask[idx] = episode["mask"]
        op_id[idx] = episode["op_id"]
        op_params[idx] = episode["op_params"]
        num_objects[idx] = episode["num_objects"]

    return {
        "states": states,
        "actions": actions,
        "mask": mask,
        "op_id": op_id,
        "op_params": op_params,
        "num_objects": num_objects,
    }


def with_split_overrides(base: SimulationConfig, split_cfg: Mapping[str, object]) -> SimulationConfig:
    n_value = split_cfg.get("N")
    if n_value is not None:
        n_min = n_max = int(n_value)
    else:
        n_min = int(split_cfg.get("n_min", base.n_min))
        n_max = int(split_cfg.get("n_max", base.n_max))
    return replace(
        base,
        num_episodes=int(split_cfg.get("num_episodes", base.num_episodes)),
        n_min=n_min,
        n_max=n_max,
        parameter_split=str(split_cfg.get("parameter_split", base.parameter_split)),
    )
