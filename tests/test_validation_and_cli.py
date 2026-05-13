from pathlib import Path
import json

import numpy as np

from lowm.data.generate_dataset import generate_dataset
from lowm.data.simulate import SimulationConfig, simulate_split
from lowm.data.validate_dataset import validate_arrays, validate_file
from lowm.data.visualize import visualize_dataset


def test_validate_generated_arrays() -> None:
    cfg = SimulationConfig(num_episodes=64, T=8, n_min=3, n_max=5, nmax=6)
    arrays = simulate_split(seed=9, cfg=cfg)
    result = validate_arrays(arrays, strict=True)
    assert result["ok"], result["errors"]
    assert sum(result["op_counts"]) == 64


def test_generate_validate_visualize_small_config(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
dataset: LOWM-Synth
version: v0-test
seed: 11
simulation:
  T: 6
  nmax: 6
splits:
  train:
    num_episodes: 12
    n_min: 3
    n_max: 4
    seed: 11
  n_extrapolation:
    num_episodes: 6
    N: 6
    seed: 12
""",
        encoding="utf-8",
    )

    out = tmp_path / "dataset"
    generate_dataset(config, out)

    train_path = out / "train.npz"
    assert train_path.exists()
    assert (out / "metadata.json").exists()

    result = validate_file(train_path, strict=False)
    assert result["ok"], result["errors"]

    fig_dir = tmp_path / "figures"
    written = visualize_dataset(train_path, fig_dir)
    assert written
    assert all(path.exists() for path in written)

    with np.load(out / "n_extrapolation.npz") as data:
        assert data["states"].shape[2] == 6
        assert np.all(data["num_objects"] == 6)


def test_generate_ood_param_split_metadata(tmp_path: Path) -> None:
    config = tmp_path / "ood.yaml"
    config.write_text(
        """
dataset: LOWM-Synth
version: ood-test
seed: 21
simulation:
  T: 4
  nmax: 6
operators:
  gravity: [-1.0, -0.5]
  gravity_ood: [[-1.5, -1.1], [-0.4, -0.2]]
  damping: [0.96, 0.99]
  damping_ood: [[0.90, 0.94], [0.995, 1.0]]
  attraction_k: [-0.015, 0.015]
  attraction_k_ood: [[-0.05, -0.025], [0.025, 0.05]]
  restitution: [0.45, 0.75]
  restitution_ood: [[0.05, 0.25], [0.85, 1.0]]
splits:
  test_ood_param:
    num_episodes: 10
    n_min: 3
    n_max: 4
    parameter_split: ood_param
    split_type: ood_param_test
    is_ood: true
    seed: 22
""",
        encoding="utf-8",
    )
    out = tmp_path / "dataset"
    generate_dataset(config, out)
    with np.load(out / "test_ood_param.npz") as data:
        assert "is_ood" in data.files
        assert np.all(data["is_ood"] == 1)
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    split = metadata["splits"]["test_ood_param"]
    assert split["split_type"] == "ood_param_test"
    assert split["is_ood"] is True
    assert split["operator_ranges"]["gravity_ood"] == [[-1.5, -1.1], [-0.4, -0.2]]
