"""Convert CoPhy state/feature arrays or original videos into LOWM ranking arrays."""

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
    return _hash_rows(confounders.reshape(confounders.shape[0], -1))


def _write_original_split(
    out_path: Path,
    records: list[dict[str, Any]],
    scenario_id: int,
) -> dict[str, Any]:
    if not records:
        return {"num_episodes": 0, "output": str(out_path), "skipped": True}
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
    np.savez_compressed(
        out_path,
        context_states=context_states,
        positive_states=positive_states,
        states=positive_states,
        context_actions=context_actions,
        positive_actions=actions,
        actions=actions,
        context_mask=context_mask,
        positive_mask=positive_mask,
        mask=positive_mask,
        confounders=confounders,
        op_id=op_id,
        op_params=op_params,
        episode_ids=np.asarray([record["episode_id"] for record in records]),
        source_paths=np.asarray([record["path"] for record in records]),
        scenario_ids=np.full((positive_states.shape[0],), scenario_id, dtype=np.int64),
        num_objects=num_objects,
        source_sample_id=source_sample_id,
        is_external=np.ones((positive_states.shape[0],), dtype=np.int64),
    )
    return {
        "num_episodes": int(positive_states.shape[0]),
        "T": int(positive_states.shape[1] - 1),
        "nmax": int(positive_states.shape[2]),
        "object_dim": int(positive_states.shape[3]),
        "output": str(out_path),
        "op_counts": np.bincount(op_id).astype(int).tolist() if op_id.size else [],
    }


def _build_original_video_dataset(config: CoPhyAdapterConfig) -> dict[str, Any]:
    episodes = _collect_original_episodes(config.root, config.scenario, config.mode)
    if config.max_episodes is not None:
        episodes = episodes[: int(config.max_episodes)]
    out_scenario = config.out / config.scenario
    out_scenario.mkdir(parents=True, exist_ok=True)
    split_eps = _split_episodes(episodes, config.split_seed)
    split_meta: dict[str, Any] = {}
    errors: dict[str, str] = {}
    scenario_id = 0
    for split in config.splits:
        records = []
        for ep in split_eps.get(split, []):
            try:
                records.append(_episode_features(ep, config))
            except Exception as exc:
                errors[str(ep.path)] = str(exc)
        split_meta[split] = _write_original_split(out_scenario / f"{split}.npz", records, scenario_id)
    metadata = {
        "dataset": "CoPhy",
        "scenario": config.scenario,
        "mode": config.mode,
        "root": str(config.root),
        "output_dir": str(out_scenario),
        "num_frames": config.num_frames,
        "nmax": config.nmax,
        "feature_dim": FEATURE_D,
        "split_seed": config.split_seed,
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
        )
    )
    print(json.dumps({"scenario": metadata["scenario"], "splits": sorted(metadata["splits"]), "errors": metadata["errors"]}, indent=2))


if __name__ == "__main__":
    main()
