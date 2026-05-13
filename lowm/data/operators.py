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

RangeSpec = tuple[float, float] | tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class OperatorRanges:
    gravity: tuple[float, float] = (-1.5, -0.3)
    gravity_ood_low: tuple[float, float] = (-1.5, -1.1)
    gravity_ood_high: tuple[float, float] = (-0.4, -0.2)
    gravity_ood: RangeSpec = ((-1.5, -1.1), (-0.4, -0.2))
    damping: tuple[float, float] = (0.94, 1.0)
    damping_ood: RangeSpec = ((0.90, 0.94), (0.995, 1.0))
    attraction_k: tuple[float, float] = (-0.03, 0.03)
    attraction_k_ood: RangeSpec = ((-0.05, -0.035), (0.035, 0.05))
    restitution: tuple[float, float] = (0.2, 1.0)
    restitution_ood: RangeSpec = ((0.05, 0.2), (0.9, 1.0))


def _is_pair(value: object) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
    )


def _range_spec_from_value(name: str, value: object) -> RangeSpec:
    if _is_pair(value):
        seq = value  # type: ignore[assignment]
        return (float(seq[0]), float(seq[1]))  # type: ignore[index]
    if isinstance(value, (list, tuple)) and value and all(_is_pair(item) for item in value):
        return tuple((float(item[0]), float(item[1])) for item in value)  # type: ignore[index]
    raise ValueError(f"operator range '{name}' must be [low, high] or a list of [low, high] ranges")


def _single_range_from_value(name: str, value: object) -> tuple[float, float]:
    spec = _range_spec_from_value(name, value)
    if len(spec) != 2 or not isinstance(spec[0], float):
        raise ValueError(f"operator range '{name}' must contain exactly two numbers")
    return spec  # type: ignore[return-value]


def _sample_range(rng: np.random.Generator, spec: RangeSpec) -> float:
    if len(spec) == 2 and isinstance(spec[0], float):
        low, high = spec  # type: ignore[misc]
        return float(rng.uniform(low, high))
    segments = spec  # type: ignore[assignment]
    low, high = segments[int(rng.integers(0, len(segments)))]
    return float(rng.uniform(low, high))


def _range_spec_to_jsonable(spec: RangeSpec) -> list[float] | list[list[float]]:
    if len(spec) == 2 and isinstance(spec[0], float):
        return [float(spec[0]), float(spec[1])]  # type: ignore[index]
    return [[float(low), float(high)] for low, high in spec]  # type: ignore[misc]


def ranges_from_config(config: Mapping[str, object] | None) -> OperatorRanges:
    if not config:
        return OperatorRanges()
    defaults = OperatorRanges()

    def pair(name: str) -> tuple[float, float]:
        value = config.get(name, getattr(defaults, name))
        return _single_range_from_value(name, value)

    def spec(name: str) -> RangeSpec:
        return _range_spec_from_value(name, config.get(name, getattr(defaults, name)))

    return OperatorRanges(
        gravity=pair("gravity"),
        gravity_ood_low=pair("gravity_ood_low"),
        gravity_ood_high=pair("gravity_ood_high"),
        gravity_ood=spec("gravity_ood"),
        damping=pair("damping"),
        damping_ood=spec("damping_ood"),
        attraction_k=pair("attraction_k"),
        attraction_k_ood=spec("attraction_k_ood"),
        restitution=pair("restitution"),
        restitution_ood=spec("restitution_ood"),
    )


def ranges_to_metadata(ranges: OperatorRanges) -> dict[str, object]:
    return {
        "gravity": list(ranges.gravity),
        "gravity_ood_low": list(ranges.gravity_ood_low),
        "gravity_ood_high": list(ranges.gravity_ood_high),
        "gravity_ood": _range_spec_to_jsonable(ranges.gravity_ood),
        "damping": list(ranges.damping),
        "damping_ood": _range_spec_to_jsonable(ranges.damping_ood),
        "attraction_k": list(ranges.attraction_k),
        "attraction_k_ood": _range_spec_to_jsonable(ranges.attraction_k_ood),
        "restitution": list(ranges.restitution),
        "restitution_ood": _range_spec_to_jsonable(ranges.restitution_ood),
    }


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
            damping_range: RangeSpec = ranges.damping
        elif parameter_split == "ood_gravity_high":
            gravity_range = ranges.gravity_ood_high
            damping_range = ranges.damping
        elif parameter_split in {"ood", "ood_param"}:
            gravity_range = ranges.gravity_ood
            damping_range = ranges.damping_ood
        else:
            gravity_range = ranges.gravity
            damping_range = ranges.damping
        params[0] = _sample_range(rng, gravity_range)
        params[1] = _sample_range(rng, damping_range)
    elif op_id == 1:
        k_range = ranges.attraction_k_ood if parameter_split in {"ood", "ood_param"} else ranges.attraction_k
        # Avoid values too close to zero; they make the operator visually and
        # dynamically ambiguous in small debug datasets.
        for _ in range(32):
            k = _sample_range(rng, k_range)
            if abs(k) >= 0.004:
                break
        params[2] = k
    elif op_id == 2:
        restitution_range = ranges.restitution_ood if parameter_split in {"ood", "ood_param"} else ranges.restitution
        params[3] = _sample_range(rng, restitution_range)
    elif op_id == 3:
        params[4] = float(rng.integers(0, 3))

    return op_id, params


def operator_metadata() -> dict[str, object]:
    return {
        "op_families": {str(k): v for k, v in OP_FAMILIES.items()},
        "op_param_names": OP_PARAM_NAMES,
    }
