import numpy as np

from lowm.data.operators import ranges_from_config, sample_operator
from lowm.data.simulate import SimulationConfig, simulate_episode, simulate_split


def test_simulator_shapes() -> None:
    cfg = SimulationConfig(num_episodes=8, T=5, n_min=3, n_max=5, nmax=6)
    arrays = simulate_split(seed=0, cfg=cfg)

    assert arrays["states"].shape == (8, 6, 6, 7)
    assert arrays["actions"].shape == (8, 5, 6, 2)
    assert arrays["mask"].shape == (8, 6, 6)
    assert arrays["op_id"].shape == (8,)
    assert arrays["op_params"].shape == (8, 5)
    assert arrays["num_objects"].shape == (8,)


def test_simulator_no_nan() -> None:
    cfg = SimulationConfig(num_episodes=32, T=12, n_min=3, n_max=5, nmax=6)
    arrays = simulate_split(seed=1, cfg=cfg)
    for key in ["states", "actions", "op_params"]:
        assert np.isfinite(arrays[key]).all()


def test_boundary_modes_keep_positions_valid() -> None:
    rng = np.random.default_rng(4)
    cfg = SimulationConfig(T=24, n_min=4, n_max=4, nmax=6, velocity_min=-1.0, velocity_max=1.0)
    for _ in range(12):
        episode = simulate_episode(rng, cfg, n=4, op_id=3)
        active = episode["mask"].astype(bool)
        positions = episode["states"][..., 0:2]
        assert np.all(positions[active] >= -1e-6)
        assert np.all(positions[active] <= 1.0 + 1e-6)


def test_operator_families_can_be_forced() -> None:
    rng = np.random.default_rng(7)
    cfg = SimulationConfig(T=4, n_min=3, n_max=3, nmax=6)
    for op_id in range(4):
        episode = simulate_episode(rng, cfg, n=3, op_id=op_id)
        assert episode["op_id"] == op_id
        assert episode["states"].shape == (5, 6, 7)


def test_ood_parameter_split_uses_held_out_continuous_ranges() -> None:
    ranges = ranges_from_config(
        {
            "gravity": [-1.0, -0.5],
            "gravity_ood": [[-1.5, -1.1], [-0.4, -0.2]],
            "damping": [0.96, 0.99],
            "damping_ood": [[0.90, 0.94], [0.995, 1.0]],
            "attraction_k": [-0.015, 0.015],
            "attraction_k_ood": [[-0.05, -0.025], [0.025, 0.05]],
            "restitution": [0.45, 0.75],
            "restitution_ood": [[0.05, 0.25], [0.85, 1.0]],
        }
    )
    rng = np.random.default_rng(17)
    for op_id in [0, 1, 2]:
        _, params = sample_operator(rng, ranges, parameter_split="ood_param", op_id=op_id)
        if op_id == 0:
            assert not (-1.0 <= params[0] <= -0.5)
            assert not (0.96 <= params[1] <= 0.99)
        elif op_id == 1:
            assert abs(params[2]) >= 0.025
        elif op_id == 2:
            assert not (0.45 <= params[3] <= 0.75)
