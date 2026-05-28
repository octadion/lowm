"""Merge multi-instance CoPhy preprocessing shard outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from lowm.data.cophy_adapter import (
    _concat_original_arrays,
    _jsonable,
    _load_original_arrays,
    _load_shard_manifest,
    _refresh_original_arrays,
    _slice_original_arrays,
    _split_ranges,
    _write_original_arrays_npz,
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_shard_path(part: Path, shard: dict[str, Any]) -> Path:
    local = part / "shards" / str(shard.get("file", ""))
    if local.exists():
        return local
    absolute = Path(str(shard.get("path", "")))
    if absolute.exists():
        return absolute
    raise FileNotFoundError(f"missing shard {shard.get('file')} for part {part}")


def _array_episode_paths(arrays: dict[str, np.ndarray]) -> list[str]:
    if "episode_paths" in arrays:
        return [str(value) for value in arrays["episode_paths"].tolist()]
    if "source_paths" in arrays:
        return [str(value) for value in arrays["source_paths"].tolist()]
    return []


def _ensure_merged_fields(arrays: dict[str, np.ndarray], metadata: dict[str, Any], episode_paths: list[str]) -> dict[str, np.ndarray]:
    n = int(arrays["positive_states"].shape[0])
    out = dict(arrays)
    if "episode_paths" not in out:
        paths = episode_paths or _array_episode_paths(out)
        out["episode_paths"] = np.asarray(paths if paths else ["" for _ in range(n)])
    if "source_paths" not in out:
        out["source_paths"] = out["episode_paths"]
    if "scenario" not in out:
        out["scenario"] = np.asarray([str(metadata.get("scenario", "")) for _ in range(n)])
    if "object_count" not in out and "num_objects" in out:
        out["object_count"] = out["num_objects"]
    if "masks" not in out and "positive_mask" in out:
        out["masks"] = out["positive_mask"]
    return out


def _permute_arrays(arrays: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    if not arrays:
        return arrays
    n = int(arrays["positive_states"].shape[0])
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    return {key: value[order] for key, value in arrays.items()}


def _write_splits(
    out: Path,
    arrays: dict[str, np.ndarray],
    split_seed: int,
    splits: tuple[str, ...],
    compression: str,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    shuffled = _refresh_original_arrays(_permute_arrays(arrays, split_seed), scenario_id=0)
    n = int(shuffled["positive_states"].shape[0]) if shuffled else 0
    split_meta: dict[str, Any] = {}
    for split, (start, end) in _split_ranges(n, splits).items():
        path = out / f"{split}.npz"
        if end <= start:
            split_meta[split] = {"num_episodes": 0, "output": str(path), "skipped": True}
            continue
        split_arrays = _refresh_original_arrays(_slice_original_arrays(shuffled, start, end), scenario_id=0)
        _write_original_arrays_npz(path, split_arrays, compression=compression)
        split_meta[split] = {
            "num_episodes": int(end - start),
            "output": str(path),
            "T": int(split_arrays["positive_states"].shape[1] - 1),
            "nmax": int(split_arrays["positive_states"].shape[2]),
            "object_dim": int(split_arrays["positive_states"].shape[3]),
        }
    return split_meta


def merge_cophy_partitions(
    parts: list[Path],
    out: Path,
    split_seed: int = 0,
    splits: tuple[str, ...] = ("train", "val", "test"),
    compression: str = "none",
) -> dict[str, Any]:
    if not parts:
        raise ValueError("at least one part directory is required")
    out.mkdir(parents=True, exist_ok=True)
    source_metadata: list[dict[str, Any]] = []
    merged_manifest: dict[str, Any] = {"parts": [], "shards": [], "episode_paths": []}
    arrays_parts: list[dict[str, np.ndarray]] = []
    seen: dict[str, str] = {}
    duplicates: list[tuple[str, str, str]] = []

    for part in parts:
        metadata = _load_json(part / "metadata.json")
        manifest_path = part / "shards" / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing {manifest_path}")
        manifest = _load_shard_manifest(manifest_path)
        source_metadata.append(metadata)
        part_entry = {
            "part": str(part),
            "metadata": str(part / "metadata.json"),
            "manifest": str(manifest_path),
            "num_shards": len(manifest.get("shards", [])),
            "num_episodes": 0,
        }
        for shard in sorted(manifest.get("shards", []), key=lambda item: int(item.get("shard_index", 0))):
            shard_path = _resolve_shard_path(part, shard)
            arrays = _load_original_arrays(shard_path)
            episode_paths = [str(path) for path in shard.get("episode_paths", [])] or _array_episode_paths(arrays)
            arrays = _ensure_merged_fields(arrays, metadata, episode_paths)
            episode_paths = _array_episode_paths(arrays)
            for episode_path in episode_paths:
                if episode_path in seen:
                    duplicates.append((episode_path, seen[episode_path], str(shard_path)))
                else:
                    seen[episode_path] = str(shard_path)
            part_entry["num_episodes"] += int(arrays["positive_states"].shape[0])
            merged_manifest["shards"].append(
                {
                    "part": str(part),
                    "source_file": str(shard_path),
                    "num_episodes": int(arrays["positive_states"].shape[0]),
                    "episode_paths": episode_paths,
                }
            )
            merged_manifest["episode_paths"].extend(episode_paths)
            arrays_parts.append(arrays)
        merged_manifest["parts"].append(part_entry)

    if duplicates:
        examples = "; ".join(f"{path} in {first} and {second}" for path, first, second in duplicates[:5])
        raise ValueError(f"duplicate CoPhy episode paths detected while merging: {examples}")

    arrays = _concat_original_arrays(arrays_parts)
    total_episodes = int(arrays["positive_states"].shape[0]) if arrays else 0
    split_meta = _write_splits(out, arrays, split_seed=split_seed, splits=splits, compression=compression)
    merged_manifest.update(
        {
            "total_episodes": total_episodes,
            "duplicate_check_passed": True,
            "split_seed": int(split_seed),
            "splits": split_meta,
        }
    )
    with (out / "merged_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonable(merged_manifest), f, indent=2, sort_keys=True)

    metadata = {
        "dataset": "CoPhy",
        "scenario": source_metadata[0].get("scenario", out.name) if source_metadata else out.name,
        "mode": source_metadata[0].get("mode", "segm_features") if source_metadata else "segm_features",
        "output_dir": str(out),
        "merged_from_parts": [str(part) for part in parts],
        "total_merged_episodes": total_episodes,
        "duplicate_check_passed": True,
        "split_seed": int(split_seed),
        "splits": split_meta,
        "merged_manifest": str(out / "merged_manifest.json"),
        "source_partition_metadata": source_metadata,
        "confounder_usage": "confounders are preserved as metadata for sampling/evaluation; they are not model inputs",
    }
    with (out / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonable(metadata), f, indent=2, sort_keys=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--splits", type=str, nargs="*", default=["train", "val", "test"])
    parser.add_argument("--compression", type=str, default="none", choices=["compressed", "none"])
    args = parser.parse_args()
    metadata = merge_cophy_partitions(
        parts=list(args.parts),
        out=args.out,
        split_seed=args.split_seed,
        splits=tuple(args.splits),
        compression=args.compression,
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "total_merged_episodes": metadata["total_merged_episodes"],
                "splits": sorted(metadata.get("splits", {})),
                "duplicate_check_passed": metadata["duplicate_check_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
