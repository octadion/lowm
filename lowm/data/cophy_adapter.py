"""Convert CoPhy state/feature arrays or original videos into LOWM ranking arrays."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
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
FEATURE_D = 7


@dataclass(frozen=True)
class CoPhyAdapterConfig:
    root: Path
    out: Path
    scenario: str
    splits: tuple[str, ...] = ("train", "val", "test")
    mode: str = "segm_features"
    max_episodes: int | None = None
    num_frames: int = 20
    nmax: int = 9
    split_seed: int = 0
    background_threshold: int = 8
    object_counts: tuple[int, ...] | None = None
    progress_every: int = 100
    save_shards: bool = False
    shard_size: int = 1000
    resume: bool = False
    skip_existing: bool = False
    num_workers: int = 1
    no_final_merge: bool = False
    force_full: bool = False


@dataclass(frozen=True)
class OriginalEpisode:
    path: Path
    scenario: str
    episode_id: str
    num_objects: int
    confounders_path: Path
    explanations_path: Path | None
    ab_video: Path
    cd_video: Path


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


def _scenario_dir(root: Path, scenario: str) -> Path:
    direct = root / scenario
    if direct.exists():
        return direct
    lower = scenario.lower()
    for path in root.iterdir() if root.exists() else []:
        if path.is_dir() and path.name.lower() == lower:
            return path
    return direct


def _candidate_split_paths(root: Path, scenario: str, split: str) -> list[Path]:
    scenario_dir = _scenario_dir(root, scenario)
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
    split_dir = _scenario_dir(root, scenario) / split
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
    return params.reshape(episodes, -1).astype(np.float32)


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
        params = np.zeros((episodes, 1), dtype=np.float32)
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
        raise ValueError("no state/object/feature array found; raw visual-only conversion requires segm_features/rgb_features mode")
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


def _read_video(path: Path) -> np.ndarray:
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if frames:
            return np.asarray(frames, dtype=np.uint8)
    except Exception:
        pass
    try:
        import imageio.v3 as iio  # type: ignore

        frames = iio.imread(path)
        if frames.ndim == 3:
            frames = frames[:, :, :, None]
        if frames.shape[-1] == 4:
            frames = frames[..., :3]
        return np.asarray(frames, dtype=np.uint8)
    except Exception as exc:
        raise RuntimeError(
            f"could not decode video {path}; install opencv-python or imageio with ffmpeg support"
        ) from exc


def _sample_video_frames(frames: np.ndarray, num_frames: int) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[-1] < 3:
        raise ValueError("video frames must be [frames,height,width,channels]")
    if frames.shape[0] == 0:
        raise ValueError("video has no frames")
    indices = np.linspace(0, frames.shape[0] - 1, int(num_frames)).round().astype(np.int64)
    return frames[indices, :, :, :3]


def _segmentation_colors(frames: np.ndarray, threshold: int, nmax: int) -> list[tuple[int, int, int]]:
    flat = frames.reshape(-1, 3)
    colors = np.unique(flat, axis=0)
    keep = []
    for color in colors:
        if int(np.max(color)) <= threshold:
            continue
        keep.append(tuple(int(x) for x in color.tolist()))
    return sorted(keep)[: int(nmax)]


def extract_segmentation_features_from_frames(
    frames: np.ndarray,
    colors: list[tuple[int, int, int]] | None = None,
    nmax: int = 9,
    background_threshold: int = 8,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int]]]:
    sampled = np.asarray(frames, dtype=np.uint8)
    if colors is None:
        colors = _segmentation_colors(sampled, background_threshold, nmax)
    height, width = sampled.shape[1], sampled.shape[2]
    features = np.zeros((sampled.shape[0], int(nmax), FEATURE_D), dtype=np.float32)
    mask = np.zeros((sampled.shape[0], int(nmax)), dtype=np.float32)
    previous_xy = np.zeros((int(nmax), 2), dtype=np.float32)
    previous_visible = np.zeros((int(nmax),), dtype=bool)
    for t, frame in enumerate(sampled):
        for slot, color in enumerate(colors[: int(nmax)]):
            match = np.all(frame == np.asarray(color, dtype=np.uint8), axis=-1)
            if not match.any():
                continue
            ys, xs = np.nonzero(match)
            x = (float(xs.mean()) / max(1.0, width - 1.0)) * 2.0 - 1.0
            y = (float(ys.mean()) / max(1.0, height - 1.0)) * 2.0 - 1.0
            area = float(match.mean())
            vx = x - float(previous_xy[slot, 0]) if previous_visible[slot] else 0.0
            vy = y - float(previous_xy[slot, 1]) if previous_visible[slot] else 0.0
            color_id = slot / max(1, int(nmax) - 1)
            features[t, slot] = np.asarray([x, y, area, 1.0, vx, vy, color_id], dtype=np.float32)
            mask[t, slot] = 1.0
            previous_xy[slot] = (x, y)
            previous_visible[slot] = True
    return features, mask, colors


def extract_rgb_features_from_frames(frames: np.ndarray, nmax: int = 9) -> tuple[np.ndarray, np.ndarray]:
    sampled = np.asarray(frames, dtype=np.float32) / 255.0
    features = np.zeros((sampled.shape[0], int(nmax), FEATURE_D), dtype=np.float32)
    mask = np.zeros((sampled.shape[0], int(nmax)), dtype=np.float32)
    mean = sampled.mean(axis=(1, 2))
    std = sampled.std(axis=(1, 2))
    features[:, 0, :] = np.concatenate([mean, std, np.ones((sampled.shape[0], 1), dtype=np.float32)], axis=1)
    mask[:, 0] = 1.0
    return features, mask


def _parse_num_objects(path: Path) -> int:
    try:
        return int(path.parent.name)
    except ValueError:
        return 0


def _collect_original_episodes(root: Path, scenario: str, mode: str) -> list[OriginalEpisode]:
    scenario_dir = _scenario_dir(root, scenario)
    if not scenario_dir.exists():
        return []
    video_name = "segm.mp4" if mode == "segm_features" else "rgb.mp4"
    episodes: list[OriginalEpisode] = []
    for confounders in sorted(scenario_dir.rglob("confounders.npy")):
        ep = confounders.parent
        ab_video = ep / "ab" / video_name
        cd_video = ep / "cd" / video_name
        if not ab_video.exists() or not cd_video.exists():
            continue
        episodes.append(
            OriginalEpisode(
                path=ep,
                scenario=scenario_dir.name,
                episode_id=ep.name,
                num_objects=_parse_num_objects(ep),
                confounders_path=confounders,
                explanations_path=ep / "explanations.txt" if (ep / "explanations.txt").exists() else None,
                ab_video=ab_video,
                cd_video=cd_video,
            )
        )
    return episodes


def _split_episodes(episodes: list[OriginalEpisode], seed: int) -> dict[str, list[OriginalEpisode]]:
    rng = np.random.default_rng(seed)
    order = np.arange(len(episodes))
    rng.shuffle(order)
    shuffled = [episodes[int(i)] for i in order]
    n = len(shuffled)
    n_train = int(round(0.70 * n))
    n_val = int(round(0.15 * n))
    if n >= 3:
        n_train = max(1, min(n - 2, n_train))
        n_val = max(1, min(n - n_train - 1, n_val))
    n_test = max(0, n - n_train - n_val)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val : n_train + n_val + n_test],
    }


def _episode_features(ep: OriginalEpisode, config: CoPhyAdapterConfig) -> dict[str, Any]:
    confounders = np.asarray(np.load(ep.confounders_path, allow_pickle=False), dtype=np.float32)
    if confounders.ndim == 1:
        confounders = confounders[:, None]
    if confounders.shape[0] < config.nmax:
        confounders = np.pad(confounders, ((0, config.nmax - confounders.shape[0]), (0, 0)))
    elif confounders.shape[0] > config.nmax:
        confounders = confounders[: config.nmax]
    ab_frames = _sample_video_frames(_read_video(ep.ab_video), config.num_frames)
    cd_frames = _sample_video_frames(_read_video(ep.cd_video), config.num_frames)
    if config.mode == "segm_features":
        colors = _segmentation_colors(np.concatenate([ab_frames, cd_frames], axis=0), config.background_threshold, config.nmax)
        ab_states, ab_mask, colors = extract_segmentation_features_from_frames(ab_frames, colors, config.nmax, config.background_threshold)
        cd_states, cd_mask, _ = extract_segmentation_features_from_frames(cd_frames, colors, config.nmax, config.background_threshold)
    elif config.mode == "rgb_features":
        ab_states, ab_mask = extract_rgb_features_from_frames(ab_frames, config.nmax)
        cd_states, cd_mask = extract_rgb_features_from_frames(cd_frames, config.nmax)
        colors = []
    else:
        raise ValueError("original CoPhy video mode must be segm_features or rgb_features")
    return {
        "context_states": ab_states,
        "context_mask": ab_mask,
        "positive_states": cd_states,
        "positive_mask": cd_mask,
        "confounders": confounders.astype(np.float32),
        "episode_id": ep.episode_id,
        "num_objects": int(ep.num_objects or max(1, np.rint(cd_mask[0].sum()))),
        "colors": colors,
        "path": str(ep.path),
    }


def _hash_confounders(confounders: np.ndarray) -> np.ndarray:
    if confounders.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    return _hash_rows(confounders.reshape(confounders.shape[0], -1))


def _records_to_original_arrays(records: list[dict[str, Any]], scenario_id: int) -> dict[str, np.ndarray]:
    context_states = np.stack([record["context_states"] for record in records]).astype(np.float32)
    positive_states = np.stack([record["positive_states"] for record in records]).astype(np.float32)
    context_mask = np.stack([record["context_mask"] for record in records]).astype(np.float32)
    positive_mask = np.stack([record["positive_mask"] for record in records]).astype(np.float32)
    confounders = np.stack([record["confounders"] for record in records]).astype(np.float32)
    op_params = confounders.reshape(confounders.shape[0], -1).astype(np.float32)
    op_id = _hash_confounders(confounders).astype(np.int64)
    num_objects = np.asarray([record["num_objects"] for record in records], dtype=np.int64)
    actions = np.zeros((positive_states.shape[0], positive_states.shape[1] - 1, positive_states.shape[2], 2), dtype=np.float32)
    context_actions = np.zeros_like(actions)
    source_sample_id = np.arange(positive_states.shape[0], dtype=np.int64)
    return {
        "context_states": context_states,
        "positive_states": positive_states,
        "states": positive_states,
        "context_actions": context_actions,
        "positive_actions": actions,
        "actions": actions,
        "context_mask": context_mask,
        "positive_mask": positive_mask,
        "mask": positive_mask,
        "confounders": confounders,
        "op_id": op_id,
        "op_params": op_params,
        "episode_ids": np.asarray([record["episode_id"] for record in records]),
        "source_paths": np.asarray([record["path"] for record in records]),
        "scenario_ids": np.full((positive_states.shape[0],), scenario_id, dtype=np.int64),
        "num_objects": num_objects,
        "source_sample_id": source_sample_id,
        "is_external": np.ones((positive_states.shape[0],), dtype=np.int64),
    }


def _refresh_original_arrays(arrays: dict[str, np.ndarray], scenario_id: int) -> dict[str, np.ndarray]:
    if not arrays:
        return arrays
    n = int(arrays["positive_states"].shape[0])
    out = dict(arrays)
    out["states"] = out["positive_states"].astype(np.float32)
    out["mask"] = out["positive_mask"].astype(np.float32)
    out["actions"] = out["positive_actions"].astype(np.float32)
    param_dim = int(np.prod(out["confounders"].shape[1:])) if out["confounders"].ndim > 1 else 1
    out["op_params"] = out["confounders"].reshape(n, param_dim).astype(np.float32)
    out["op_id"] = _hash_confounders(out["confounders"]).astype(np.int64)
    out["scenario_ids"] = np.full((n,), scenario_id, dtype=np.int64)
    out["source_sample_id"] = np.arange(n, dtype=np.int64)
    out["is_external"] = np.ones((n,), dtype=np.int64)
    return out


def _write_original_arrays_npz(out_path: Path, arrays: dict[str, np.ndarray]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)


def _original_arrays_summary(out_path: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    if not arrays:
        return {"num_episodes": 0, "output": str(out_path), "skipped": True}
    positive_states = arrays["positive_states"]
    op_id = arrays["op_id"].astype(np.int64)
    return {
        "num_episodes": int(positive_states.shape[0]),
        "T": int(positive_states.shape[1] - 1),
        "nmax": int(positive_states.shape[2]),
        "object_dim": int(positive_states.shape[3]),
        "output": str(out_path),
        "op_counts": np.bincount(op_id).astype(int).tolist() if op_id.size else [],
    }


def _write_original_split(out_path: Path, records: list[dict[str, Any]], scenario_id: int) -> dict[str, Any]:
    if not records:
        return {"num_episodes": 0, "output": str(out_path), "skipped": True}
    arrays = _records_to_original_arrays(records, scenario_id)
    _write_original_arrays_npz(out_path, arrays)
    return _original_arrays_summary(out_path, arrays)


def _episode_sort_key(ep: OriginalEpisode) -> tuple[str, int, str]:
    return (ep.scenario.lower(), int(ep.num_objects), str(ep.path))


def _select_original_episodes(
    episodes: list[OriginalEpisode],
    config: CoPhyAdapterConfig,
) -> tuple[list[OriginalEpisode], list[int], int]:
    all_sorted = sorted(episodes, key=_episode_sort_key)
    total_available = len(all_sorted)
    allowed = set(config.object_counts or [])
    filtered = [ep for ep in all_sorted if not allowed or ep.num_objects in allowed]
    rng = np.random.default_rng(config.split_seed)
    order = np.arange(len(filtered))
    rng.shuffle(order)
    selected = [filtered[int(i)] for i in order]
    if config.max_episodes is not None:
        selected = selected[: max(0, int(config.max_episodes))]
    object_counts_used = sorted({int(ep.num_objects) for ep in selected})
    return selected, object_counts_used, total_available


def _split_ranges(n: int, splits: tuple[str, ...]) -> dict[str, tuple[int, int]]:
    if not splits:
        return {}
    if len(splits) == 1:
        return {splits[0]: (0, n)}
    n_train = int(round(0.70 * n))
    n_val = int(round(0.15 * n))
    if n >= 3:
        n_train = max(1, min(n - 2, n_train))
        n_val = max(1, min(n - n_train - 1, n_val))
    n_test = max(0, n - n_train - n_val)
    canonical = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n_train + n_val + n_test),
    }
    if set(splits).issubset(canonical):
        return {split: canonical[split] for split in splits}
    boundaries = np.linspace(0, n, len(splits) + 1).round().astype(int)
    return {split: (int(boundaries[i]), int(boundaries[i + 1])) for i, split in enumerate(splits)}


def _slice_original_arrays(arrays: dict[str, np.ndarray], start: int, end: int) -> dict[str, np.ndarray]:
    return {key: value[start:end] for key, value in arrays.items()}


def _concat_original_arrays(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        return {}
    keys = set.intersection(*(set(part) for part in parts))
    arrays: dict[str, np.ndarray] = {}
    for key in keys:
        arrays[key] = np.concatenate([part[key] for part in parts], axis=0)
    return arrays


def _load_original_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _manifest_path(out_scenario: Path) -> Path:
    return out_scenario / "shards" / "manifest.json"


def _load_shard_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"shards": [], "processed_episode_paths": [], "errors": {}}
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest.setdefault("shards", [])
    manifest.setdefault("processed_episode_paths", [])
    manifest.setdefault("errors", {})
    return manifest


def _write_shard_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(manifest), f, indent=2, sort_keys=True)


def _processed_paths_from_manifest(manifest: dict[str, Any]) -> set[str]:
    paths = set(str(path) for path in manifest.get("processed_episode_paths", []))
    for shard in manifest.get("shards", []):
        paths.update(str(path) for path in shard.get("episode_paths", []))
    return paths


def _next_shard_index(manifest: dict[str, Any], shards_dir: Path) -> int:
    indices = []
    for shard in manifest.get("shards", []):
        try:
            indices.append(int(shard.get("shard_index", -1)))
        except (TypeError, ValueError):
            pass
    for path in shards_dir.glob("shard_*.npz") if shards_dir.exists() else []:
        try:
            indices.append(int(path.stem.split("_")[-1]))
        except ValueError:
            pass
    return max(indices, default=-1) + 1


def _initial_shard_manifest(out_scenario: Path, config: CoPhyAdapterConfig) -> dict[str, Any]:
    path = _manifest_path(out_scenario)
    if config.resume or config.skip_existing:
        return _load_shard_manifest(path)
    if path.exists():
        raise RuntimeError(
            f"existing shard manifest found at {path}; pass --resume to continue or choose a new output directory"
        )
    shards_dir = path.parent
    if shards_dir.exists() and any(shards_dir.glob("shard_*.npz")):
        raise RuntimeError(
            f"existing shard files found in {shards_dir}; pass --resume to continue or choose a new output directory"
        )
    return {"shards": [], "processed_episode_paths": [], "errors": {}}


def _save_shard(
    records: list[dict[str, Any]],
    out_scenario: Path,
    manifest: dict[str, Any],
    scenario_id: int,
    shard_index: int,
) -> int:
    if not records:
        return shard_index
    shards_dir = out_scenario / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    while True:
        shard_path = shards_dir / f"shard_{shard_index:05d}.npz"
        if not shard_path.exists():
            break
        shard_index += 1
    arrays = _records_to_original_arrays(records, scenario_id)
    _write_original_arrays_npz(shard_path, arrays)
    episode_paths = [str(record["path"]) for record in records]
    manifest.setdefault("shards", []).append(
        {
            "shard_index": int(shard_index),
            "file": shard_path.name,
            "path": str(shard_path),
            "num_episodes": int(len(records)),
            "episode_ids": [str(record["episode_id"]) for record in records],
            "episode_paths": episode_paths,
        }
    )
    manifest["processed_episode_paths"] = list(dict.fromkeys([*manifest.get("processed_episode_paths", []), *episode_paths]))
    _write_shard_manifest(_manifest_path(out_scenario), manifest)
    return shard_index + 1


def _load_arrays_from_manifest(out_scenario: Path, manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    parts: list[dict[str, np.ndarray]] = []
    for shard in sorted(manifest.get("shards", []), key=lambda item: int(item.get("shard_index", 0))):
        shard_path = Path(shard.get("path", ""))
        if not shard_path.exists():
            shard_path = out_scenario / "shards" / str(shard.get("file", ""))
        if shard_path.exists():
            parts.append(_load_original_arrays(shard_path))
    return _concat_original_arrays(parts)


def _process_episode_batch(
    episodes: list[OriginalEpisode],
    config: CoPhyAdapterConfig,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    def one(ep: OriginalEpisode) -> tuple[dict[str, Any] | None, tuple[str, str] | None]:
        try:
            return _episode_features(ep, config), None
        except Exception as exc:
            return None, (str(ep.path), str(exc))

    if config.num_workers > 1 and len(episodes) > 1:
        with ThreadPoolExecutor(max_workers=int(config.num_workers)) as executor:
            results = list(executor.map(one, episodes))
    else:
        results = [one(ep) for ep in episodes]
    records = [record for record, error in results if record is not None]
    errors = {str(path): str(message) for record, error in results if error is not None for path, message in [error]}
    return records, errors


def _format_eta(seconds: float) -> str:
    if seconds == float("inf") or seconds < 0:
        return "unknown"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def _print_progress(done: int, total: int, start_time: float, current_path: str) -> None:
    elapsed = max(1e-9, time.monotonic() - start_time)
    rate = done / elapsed
    eta = (total - done) / rate if rate > 0 else float("inf")
    print(
        f"[cophy_adapter] processed {done}/{total} | elapsed {_format_eta(elapsed)} | "
        f"{rate:.3f} episodes/sec | ETA {_format_eta(eta)} | current {current_path}",
        flush=True,
    )


def _write_final_splits(
    out_scenario: Path,
    arrays: dict[str, np.ndarray],
    scenario_id: int,
    splits: tuple[str, ...],
) -> dict[str, Any]:
    if not arrays:
        return {split: {"num_episodes": 0, "output": str(out_scenario / f"{split}.npz"), "skipped": True} for split in splits}
    arrays = _refresh_original_arrays(arrays, scenario_id)
    split_meta: dict[str, Any] = {}
    n = int(arrays["positive_states"].shape[0])
    for split, (start, end) in _split_ranges(n, splits).items():
        out_path = out_scenario / f"{split}.npz"
        split_arrays = _refresh_original_arrays(_slice_original_arrays(arrays, start, end), scenario_id)
        if end <= start:
            split_meta[split] = {"num_episodes": 0, "output": str(out_path), "skipped": True}
            continue
        _write_original_arrays_npz(out_path, split_arrays)
        split_meta[split] = _original_arrays_summary(out_path, split_arrays)
    return split_meta


def _build_original_video_dataset(config: CoPhyAdapterConfig) -> dict[str, Any]:
    episodes = _collect_original_episodes(config.root, config.scenario, config.mode)
    selected_episodes, object_counts_used, total_available = _select_original_episodes(episodes, config)
    out_scenario = config.out / config.scenario
    out_scenario.mkdir(parents=True, exist_ok=True)
    if len(selected_episodes) > 10000 and config.max_episodes is None and not config.save_shards:
        print(
            "Large preprocessing without shards may take many hours and lose progress in Colab.",
            flush=True,
        )
        if not config.force_full:
            time.sleep(5)
    split_meta: dict[str, Any] = {}
    errors: dict[str, str] = {}
    scenario_id = 0
    records_for_merge: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {"shards": [], "processed_episode_paths": [], "errors": {}}
    completed_paths: set[str] = set()
    next_shard = 0
    if config.save_shards:
        manifest = _initial_shard_manifest(out_scenario, config)
        completed_paths = _processed_paths_from_manifest(manifest) if (config.resume or config.skip_existing) else set()
        next_shard = _next_shard_index(manifest, out_scenario / "shards")
        if completed_paths:
            print(f"[cophy_adapter] resume/skip-existing: {len(completed_paths)} episode(s) already processed", flush=True)

    start_time = time.monotonic()
    shard_records: list[dict[str, Any]] = []
    selected_paths = [str(ep.path) for ep in selected_episodes]
    already_done = len([path for path in selected_paths if path in completed_paths])
    attempted = already_done
    batch_size = max(1, int(config.progress_every)) if config.num_workers > 1 else 1
    if config.save_shards:
        batch_size = max(1, min(int(config.shard_size), max(1, int(config.progress_every))))
    remaining: list[OriginalEpisode] = [ep for ep in selected_episodes if str(ep.path) not in completed_paths]
    for offset in range(0, len(remaining), batch_size):
        batch = remaining[offset : offset + batch_size]
        batch_records, batch_errors = _process_episode_batch(batch, config)
        errors.update(batch_errors)
        attempted += len(batch)
        if config.save_shards:
            shard_records.extend(batch_records)
            if len(shard_records) >= int(config.shard_size):
                next_shard = _save_shard(shard_records, out_scenario, manifest, scenario_id, next_shard)
                shard_records = []
        else:
            records_for_merge.extend(batch_records)
        if attempted == len(selected_episodes) or attempted % max(1, int(config.progress_every)) == 0:
            current = str(batch[-1].path) if batch else ""
            _print_progress(attempted, len(selected_episodes), start_time, current)

    if config.save_shards and shard_records:
        next_shard = _save_shard(shard_records, out_scenario, manifest, scenario_id, next_shard)
        shard_records = []
    if config.save_shards:
        errors = {**manifest.get("errors", {}), **errors}
        manifest["errors"] = errors
        _write_shard_manifest(_manifest_path(out_scenario), manifest)

    arrays: dict[str, np.ndarray] = {}
    if not config.no_final_merge:
        if config.save_shards:
            manifest = _load_shard_manifest(_manifest_path(out_scenario))
            arrays = _load_arrays_from_manifest(out_scenario, manifest)
        elif records_for_merge:
            arrays = _records_to_original_arrays(records_for_merge, scenario_id)
        split_meta = _write_final_splits(out_scenario, arrays, scenario_id, config.splits)
    elif config.save_shards:
        manifest = _load_shard_manifest(_manifest_path(out_scenario))

    processed_episodes = int(arrays["positive_states"].shape[0]) if arrays else len(records_for_merge)
    if config.save_shards:
        processed_episodes = len(_processed_paths_from_manifest(manifest))
    shards_written = len(manifest.get("shards", [])) if config.save_shards else 0
    completed = processed_episodes >= len(selected_episodes) and not errors
    metadata = {
        "dataset": "CoPhy",
        "scenario": config.scenario,
        "mode": config.mode,
        "root": str(config.root),
        "output_dir": str(out_scenario),
        "total_available_episodes": int(total_available),
        "selected_episodes": int(len(selected_episodes)),
        "processed_episodes": int(processed_episodes),
        "object_counts_used": object_counts_used,
        "max_episodes": config.max_episodes,
        "num_frames": config.num_frames,
        "nmax": config.nmax,
        "feature_dim": FEATURE_D,
        "split_seed": config.split_seed,
        "progress_every": config.progress_every,
        "save_shards": config.save_shards,
        "shard_size": config.shard_size,
        "shards_written": int(shards_written),
        "resume": config.resume,
        "skip_existing": config.skip_existing,
        "num_workers": config.num_workers,
        "no_final_merge": config.no_final_merge,
        "completed": bool(completed),
        "final_merge_completed": bool(not config.no_final_merge),
        "shard_manifest": str(_manifest_path(out_scenario)) if config.save_shards else None,
        "shards": manifest.get("shards", []) if config.save_shards else [],
        "splits": split_meta,
        "errors": errors,
        "format": "paired AB/CD feature arrays; AB is context, CD is positive trajectory",
        "confounder_usage": "confounders.npy is saved only as metadata for sampling/evaluation; it is not a model input",
        "limitations": [
            "Segmentation color identity is assumed stable enough within an episode.",
            "Actions are zero-filled because CoPhy original videos do not expose actions.",
            "RGB feature mode is global-frame only and should be treated as a fallback.",
        ],
    }
    with (out_scenario / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonable(metadata), f, indent=2, sort_keys=True)
    return metadata


def _build_npz_dataset(config: CoPhyAdapterConfig) -> dict[str, Any]:
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
            "Raw visual-only CoPhy requires segm_features or rgb_features mode.",
            "Actions are zero-filled when source actions are unavailable.",
            "Confounder labels are stored only for sampling/evaluation and are not model inputs.",
        ],
    }
    with (out_scenario / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonable(metadata), f, indent=2, sort_keys=True)
    return metadata


def build_cophy_dataset(config: CoPhyAdapterConfig) -> dict[str, Any]:
    if config.mode in {"segm_features", "rgb_features"}:
        return _build_original_video_dataset(config)
    if config.mode in {"state", "feature"}:
        return _build_npz_dataset(config)
    raise ValueError("mode must be one of state, feature, segm_features, rgb_features")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/cophy_omc"))
    parser.add_argument("--scenario", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="*", default=["train", "val", "test"])
    parser.add_argument("--mode", type=str, default="segm_features", choices=["state", "feature", "segm_features", "rgb_features"])
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--object-counts", type=int, nargs="*", default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--save-shards", action="store_true")
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--no-final-merge", action="store_true")
    parser.add_argument("--force-full", action="store_true")
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--nmax", type=int, default=9)
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()
    metadata = build_cophy_dataset(
        CoPhyAdapterConfig(
            root=args.root,
            out=args.out,
            scenario=args.scenario,
            splits=tuple(args.splits),
            mode=args.mode,
            max_episodes=args.max_episodes,
            num_frames=args.num_frames,
            nmax=args.nmax,
            split_seed=args.split_seed,
            object_counts=tuple(args.object_counts) if args.object_counts is not None else None,
            progress_every=args.progress_every,
            save_shards=args.save_shards,
            shard_size=args.shard_size,
            resume=args.resume,
            skip_existing=args.skip_existing,
            num_workers=args.num_workers,
            no_final_merge=args.no_final_merge,
            force_full=args.force_full,
        )
    )
    print(
        json.dumps(
            {
                "scenario": metadata["scenario"],
                "splits": sorted(metadata.get("splits", {})),
                "processed_episodes": metadata.get("processed_episodes"),
                "shards_written": metadata.get("shards_written"),
                "completed": metadata.get("completed"),
                "errors": metadata["errors"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
