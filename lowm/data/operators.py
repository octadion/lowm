"""Hidden operator families for LOWM-Synth.

The simulator stores these labels and parameters for validation, analysis, and
oracle baselines. They are not intended as model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


OP_FAMILIES = {
    0: "gravity_damping",
    1: "attraction_repulsion",
    2: "collision_elasticity",
    3: "boundary_behavior",
}

OP_PARAM_NAMES = ["gravity_y", "damping", "k", "restitution", "boundary_mode"]


@dataclass(frozen=True)
class OperatorRanges:
    gravity: tuple[float, float] = (-1.5, -0.3)
    gravity_ood_low: tuple[float, float] = (-1.5, -1.1)
    gravity_ood_high: tuple[float, float] = (-0.4, -0.2)
    damping: tuple[float, float] = (0.94, 1.0)
    attraction_k: tuple[float, float] = (-0.03, 0.03)
    restitution: tuple[float, float] = (0.2, 1.0)


def ranges_from_config(config: Mapping[str, object] | None) -> OperatorRanges:
    if not config:
        return OperatorRanges()
    defaults = OperatorRanges()

    def pair(name: str) -> tuple[float, float]:
        value = config.get(name, getattr(defaults, name))
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"operator range '{name}' must contain two numbers")
        return (float(value[0]), float(value[1]))

    return OperatorRanges(
        gravity=pair("gravity"),
        gravity_ood_low=pair("gravity_ood_low"),
        gravity_ood_high=pair("gravity_ood_high"),
        damping=pair("damping"),
        attraction_k=pair("attraction_k"),
        restitution=pair("restitution"),
    )


def sample_operator(
    rng: np.random.Generator,
    ranges: OperatorRanges,
    parameter_split: str = "iid",
    op_id: int | None = None,
) -> tuple[int, np.ndarray]:
    """Sample a hidden operator family and parameter vector.

    Parameters are stored in a fixed vector:
    [gravity_y, damping, k, restitution, boundary_mode].
    Inactive parameters are set to conservative defaults rather than NaN so the
    dataset can be validated with a simple finite check.
    """

    if op_id is None:
        op_id = int(rng.integers(0, len(OP_FAMILIES)))
    if op_id not in OP_FAMILIES:
        raise ValueError(f"unknown op_id {op_id}")

    params = np.array([0.0, 1.0, 0.0, 0.8, 0.0], dtype=np.float32)

    if op_id == 0:
        if parameter_split == "ood_gravity_low":
            gravity_range = ranges.gravity_ood_low
        elif parameter_split == "ood_gravity_high":
            gravity_range = ranges.gravity_ood_high
        else:
            gravity_range = ranges.gravity
        params[0] = rng.uniform(*gravity_range)
        params[1] = rng.uniform(*ranges.damping)
    elif op_id == 1:
        k_min, k_max = ranges.attraction_k
        # Avoid values too close to zero; they make the operator visually and
        # dynamically ambiguous in small debug datasets.
        for _ in range(32):
            k = rng.uniform(k_min, k_max)
            if abs(k) >= 0.004:
                break
        params[2] = k
    elif op_id == 2:
        params[3] = rng.uniform(*ranges.restitution)
    elif op_id == 3:
        params[4] = float(rng.integers(0, 3))

    return op_id, params


def operator_metadata() -> dict[str, object]:
    return {
        "op_families": {str(k): v for k, v in OP_FAMILIES.items()},
        "op_param_names": OP_PARAM_NAMES,
    }
