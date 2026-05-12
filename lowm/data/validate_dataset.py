"""Validation checks for LOWM-Synth .npz files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_arrays(arrays: dict[str, np.ndarray], strict: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    required = ["states", "actions", "mask", "op_id", "op_params", "num_objects"]
    for key in required:
        _require(key in arrays, f"missing required array '{key}'", errors)
    if errors:
        return {"ok": False, "errors": errors}

    states = arrays["states"]
    actions = arrays["actions"]
    mask = arrays["mask"]
    op_id = arrays["op_id"]
    op_params = arrays["op_params"]
    num_objects = arrays["num_objects"]

    _require(states.ndim == 4, "states must have shape [E,T+1,Nmax,D]", errors)
    _require(actions.ndim == 4, "actions must have shape [E,T,Nmax,2]", errors)
    _require(mask.ndim == 3, "mask must have shape [E,T+1,Nmax]", errors)
    if states.ndim == 4 and actions.ndim == 4 and mask.ndim == 3:
        e, tp1, nmax, d = states.shape
        _require(d == 7, f"object dimension must be 7, got {d}", errors)
        _require(actions.shape == (e, tp1 - 1, nmax, 2), "actions shape does not match states", errors)
        _require(mask.shape == (e, tp1, nmax), "mask shape does not match states", errors)
        _require(op_id.shape == (e,), "op_id shape must be [E]", errors)
        _require(op_params.shape == (e, 5), "op_params shape must be [E,5]", errors)
        _require(num_objects.shape == (e,), "num_objects shape must be [E]", errors)

    _require(np.isfinite(states).all(), "states contain NaN or Inf", errors)
    _require(np.isfinite(actions).all(), "actions contain NaN or Inf", errors)
    _require(np.isfinite(op_params).all(), "op_params contain NaN or Inf", errors)
    _require(np.all((mask == 0.0) | (mask == 1.0)), "mask must be binary", errors)
    _require(np.all((op_id >= 0) & (op_id <= 3)), "op_id values must be in [0,3]", errors)

    active = mask.astype(bool)
    if states.size and active.any():
        pos = states[..., 0:2]
        vel = states[..., 2:4]
        radii = states[..., 4]
        masses = states[..., 5]
        types = states[..., 6]
        _require(np.all(pos[active] >= -1e-5) and np.all(pos[active] <= 1.0 + 1e-5), "active positions must remain in [0,1]", errors)
        _require(float(np.max(np.linalg.norm(vel[active], axis=-1))) <= 5.5, "velocity exceeded stability threshold", errors)
        _require(np.all(radii[active] > 0), "active radii must be positive", errors)
        _require(np.all(masses[active] > 0), "active masses must be positive", errors)
        _require(np.all((types[active] >= 0) & (types[active] < 16)), "active type ids look invalid", errors)
        _require(np.all(states[~active] == 0.0), "padded object states must be zero", errors)

    if mask.ndim == 3:
        _require(np.all(mask == mask[:, :1, :]), "object masks must be stable across time", errors)

    severe_overlaps = 0
    if states.ndim == 4 and mask.ndim == 3:
        for ep in range(states.shape[0]):
            n = int(num_objects[ep])
            for t in range(states.shape[1]):
                for i in range(n):
                    for j in range(i + 1, n):
                        dist = float(np.linalg.norm(states[ep, t, i, 0:2] - states[ep, t, j, 0:2]))
                        rsum = float(states[ep, t, i, 4] + states[ep, t, j, 4])
                        if dist < 0.25 * rsum:
                            severe_overlaps += 1
        _require(severe_overlaps == 0, f"found {severe_overlaps} severe overlaps", errors)

    op_counts = np.bincount(op_id.astype(np.int64), minlength=4) if op_id.size else np.zeros(4, dtype=int)
    if strict and op_id.size >= 16:
        _require(np.all(op_counts > 0), f"operator distribution missing a family: {op_counts.tolist()}", errors)

    return {
        "ok": not errors,
        "errors": errors,
        "num_episodes": int(states.shape[0]) if states.ndim == 4 else None,
        "shape": {
            "states": list(states.shape),
            "actions": list(actions.shape),
            "mask": list(mask.shape),
            "op_params": list(op_params.shape),
        },
        "op_counts": op_counts.astype(int).tolist(),
        "num_objects_min": int(np.min(num_objects)) if num_objects.size else None,
        "num_objects_max": int(np.max(num_objects)) if num_objects.size else None,
    }


def validate_file(path: Path, strict: bool = True) -> dict[str, Any]:
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    return validate_arrays(arrays, strict=strict)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args()
    result = validate_file(args.path, strict=not args.no_strict)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
