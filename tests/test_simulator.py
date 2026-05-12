import numpy as np

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
