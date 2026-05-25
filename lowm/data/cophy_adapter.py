"""Convert state/object/feature CoPhy-style data into LOWM ranking arrays."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


STATE_KEYS = ("states", "object_states", "objects", "features", "trajectories", "object_features")
ACTION_KEYS = ("actions", "action", "controls")
MASK_KEYS = ("mask", "object_mask", "masks", "valid")
OP_ID_KEYS = ("op_id", "confounder_id", "operator_id", "world_id", "physical_id")
PARAM_KEYS = ("op_params", "confounders", "physical_params", "params", "properties")
SAMPLE_ID_KEYS = ("sample_id", "source_sample_id", "experiment_id", "id")


@dataclass(frozen=True)
class CoPhyAdapterConfig:
    root: Path
    out: Path
    scenario: str
    splits: tuple[str, ...] = ("train", "val", "test")
    mode: str = "state"
    max_episodes: int | None = None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _candidate_split_paths(root: Path, scenario: str, split: str) -> list[Path]:
    scenario_dir = root / scenario
    return [
        scenario_dir / f"{split}.npz",
        scenario_dir / split / f"{split}.npz",
        root / f"{scenario}_{split}.npz",
        root / f"{split}.npz",
    ]


def _find_npz_files(root: Path, scenario: str, split: str) -> list[Path]:
    direct = [path for path in _candidate_split_paths(root, scenario, split) if path.exists()]
    if direct:
        return direct
    split_dir = root / scenario / split
    if split_dir.exists():
        return sorted(split_dir.rglob("*.npz"))
    return []


def _first_key(data: dict[str, np.ndarray], keys: tuple[str, ...]) -> str | None:
    return next((key for key in keys if key in data), None)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _concat_dicts(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        return {}
    keys = set.intersection(*(set(part) for part in parts))
    out: dict[str, np.ndarray] = {}
    for key in keys:
        try:
            out[key] = np.concatenate([part[key] for part in parts], axis=0)
        except ValueError:
            continue
    return out


def _load_split(root: Path, scenario: str, split: str) -> tuple[dict[str, np.ndarray], list[str]]:
    files = _find_npz_files(root, scenario, split)
    if not files:
        return {}, []
    parts = [_load_npz(path) for path in files]
    data = parts[0] if len(parts) == 1 else _concat_dicts(parts)
    return data, [str(path) for path in files]


def _normalize_states(states: np.ndarray) -> np.ndarray:
    arr = np.asarray(states, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, None, :]
    if arr.ndim != 4:
        raise ValueError("states/features must have shape [episodes,time,objects,dims] or [episodes,time,dims]")
    return arr


def _normalize_mask(data: dict[str, np.ndarray], states: np.ndarray) -> np.ndarray:
    key = _first_key(data, MASK_KEYS)
    if key is None:
        return np.ones(states.shape[:3], dtype=np.float32)
    mask = np.asarray(data[key], dtype=np.float32)
    if mask.ndim == 2:
        mask = mask[:, :, None]
    if mask.shape != states.shape[:3]:
        raise ValueError(f"mask shape {mask.shape} does not match states prefix {states.shape[:3]}")
    return (mask > 0).astype(np.float32)


def _normalize_actions(data: dict[str, np.ndarray], states: np.ndarray) -> np.ndarray:
    key = _first_key(data, ACTION_KEYS)
    episodes, time, objects, _ = states.shape
    if key is None:
        return np.zeros((episodes, time - 1, objects, 2), dtype=np.float32)
    actions = np.asarray(data[key], dtype=np.float32)
    if actions.ndim == 3:
        actions = actions[:, :, None, :]
    if actions.shape[1] == time:
        actions = actions[:, :-1]
    if actions.ndim != 4:
        raise ValueError("actions must have shape [episodes,time-1,objects,dims] or [episodes,time-1,dims]")
    if actions.shape[2] == 1 and objects > 1:
        actions = np.repeat(actions, objects, axis=2)
    if actions.shape[:3] != (episodes, time - 1, objects):
        raise ValueError(f"actions shape {actions.shape} is incompatible with states shape {states.shape}")
    if actions.shape[-1] < 2:
        actions = np.pad(actions, ((0, 0), (0, 0), (0, 0), (0, 2 - actions.shape[-1])))
    elif actions.shape[-1] > 2:
        actions = actions[..., :2]
    return actions.astype(np.float32)


def _param_matrix(values: np.ndarray, episodes: int) -> np.ndarray:
    params = np.asarray(values, dtype=np.float32)
    if params.ndim == 1:
        params = params[:, None]
    params = params.reshape(episodes, -1)
    if params.shape[1] < 5:
        params = np.pad(params, ((0, 0), (0, 5 - params.shape[1])))
    elif params.shape[1] > 5:
        params = params[:, :5]
    return params.astype(np.float32)


def _hash_rows(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values).reshape(values.shape[0], -1)
    ids: dict[bytes, int] = {}
    out = np.zeros((flat.shape[0],), dtype=np.int64)
    for idx, row in enumerate(flat):
        key = np.asarray(row).tobytes()
        if key not in ids:
            ids[key] = len(ids)
        out[idx] = ids[key]
    return out


def _operator_arrays(data: dict[str, np.ndarray], episodes: int) -> tuple[np.ndarray, np.ndarray, str]:
    op_key = _first_key(data, OP_ID_KEYS)
    param_key = _first_key(data, PARAM_KEYS)
    if op_key is not None:
        op_id = np.asarray(data[op_key]).reshape(episodes, -1)[:, 0].astype(np.int64)
        source = op_key
    elif param_key is not None:
        op_id = _hash_rows(np.asarray(data[param_key])[:episodes])
        source = f"hashed_{param_key}"
    else:
        raise ValueError("CoPhy adapter requires op_id/confounder_id/operator_id or physical parameter arrays")
    if param_key is not None:
        params = _param_matrix(np.asarray(data[param_key])[:episodes], episodes)
    else:
        params = np.zeros((episodes, 5), dtype=np.float32)
        params[:, 0] = op_id.astype(np.float32)
    return op_id, params, source


def _num_objects(mask: np.ndarray) -> np.ndarray:
    return np.maximum(1, np.rint(mask[:, 0].sum(axis=1))).astype(np.int64)


def _sample_ids(data: dict[str, np.ndarray], episodes: int) -> np.ndarray:
    key = _first_key(data, SAMPLE_ID_KEYS)
    if key is None:
        return np.arange(episodes, dtype=np.int64)
    values = np.asarray(data[key])
    if np.issubdtype(values.dtype, np.number):
        return values.reshape(episodes, -1)[:, 0].astype(np.int64)
    return np.arange(episodes, dtype=np.int64)


def _convert_split(data: dict[str, np.ndarray], max_episodes: int | None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    state_key = _first_key(data, STATE_KEYS)
    if state_key is None:
        raise ValueError("no state/object/feature array found; raw visual-only conversion is not implemented")
    states = _normalize_states(data[state_key])
    if max_episodes is not None:
        limit = min(int(max_episodes), states.shape[0])
        original_episodes = states.shape[0]
        states = states[:limit]
        data = {
            key: value[:limit] if hasattr(value, "shape") and value.shape[:1] == (original_episodes,) else value
            for key, value in data.items()
        }
    episodes = states.shape[0]
    mask = _normalize_mask(data, states)
    actions = _normalize_actions(data, states)
    op_id, op_params, op_source = _operator_arrays(data, episodes)
    arrays = {
        "states": states.astype(np.float32),
        "actions": actions.astype(np.float32),
        "mask": mask.astype(np.float32),
        "op_id": op_id.astype(np.int64),
        "op_params": op_params.astype(np.float32),
        "num_objects": _num_objects(mask),
        "source_sample_id": _sample_ids(data, episodes),
    }
    meta = {
        "state_key": state_key,
        "operator_source": op_source,
        "num_episodes": int(episodes),
        "T": int(states.shape[1] - 1),
        "nmax": int(states.shape[2]),
        "object_dim": int(states.shape[3]),
        "op_counts": np.bincount(op_id.astype(np.int64)).astype(int).tolist() if op_id.size else [],
    }
    return arrays, meta


def build_cophy_dataset(config: CoPhyAdapterConfig) -> dict[str, Any]:
    out_scenario = config.out / config.scenario
    out_scenario.mkdir(parents=True, exist_ok=True)
    split_meta: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for split in config.splits:
        data, files = _load_split(config.root, config.scenario, split)
        if not data:
            errors[split] = "missing split npz files"
            continue
        try:
            arrays, meta = _convert_split(data, config.max_episodes)
        except ValueError as exc:
            errors[split] = str(exc)
            continue
        arrays["is_external"] = np.ones((arrays["states"].shape[0],), dtype=np.int64)
        np.savez_compressed(out_scenario / f"{split}.npz", **arrays)
        split_meta[split] = {**meta, "source_files": files, "output": str(out_scenario / f"{split}.npz")}
    metadata = {
        "dataset": "CoPhy",
        "scenario": config.scenario,
        "mode": config.mode,
        "root": str(config.root),
        "output_dir": str(out_scenario),
        "splits": split_meta,
        "errors": errors,
        "format": "LOWM-compatible episode arrays; negatives are sampled by the ranking dataset at train/eval time",
        "limitations": [
            "Raw visual-only CoPhy is not converted without provided object states or encoder features.",
            "Actions are zero-filled when source actions are unavailable.",
            "Confounder labels are stored only for sampling/evaluation and are not model inputs.",
        ],
    }
    with (out_scenario / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonable(metadata), f, indent=2, sort_keys=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/cophy_omc"))
    parser.add_argument("--scenario", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="*", default=["train", "val", "test"])
    parser.add_argument("--mode", type=str, default="state", choices=["state", "feature"])
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()
    metadata = build_cophy_dataset(
        CoPhyAdapterConfig(
            root=args.root,
            out=args.out,
            scenario=args.scenario,
            splits=tuple(args.splits),
            mode=args.mode,
            max_episodes=args.max_episodes,
        )
    )
    print(json.dumps({"scenario": metadata["scenario"], "splits": sorted(metadata["splits"]), "errors": metadata["errors"]}, indent=2))


if __name__ == "__main__":
    main()
