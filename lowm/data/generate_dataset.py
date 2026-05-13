"""Command-line generation for LOWM-Synth v0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from lowm.data.operators import operator_metadata, ranges_to_metadata
from lowm.data.simulate import config_from_mapping, simulate_split, with_split_overrides


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"config {path} must contain a YAML mapping")
    return loaded


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def generate_dataset(config_path: Path, out_dir: Path, selected_splits: list[str] | None = None) -> None:
    config = _load_config(config_path)
    base_seed = int(config.get("seed", 0))
    base_cfg = config_from_mapping(config)
    splits = config.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("config must define a 'splits' mapping")

    out_dir.mkdir(parents=True, exist_ok=True)
    split_summaries: dict[str, Any] = {}
    names = selected_splits or list(splits.keys())
    for offset, name in enumerate(names):
        if name not in splits:
            raise ValueError(f"split '{name}' not present in config")
        split_cfg = with_split_overrides(base_cfg, splits[name])
        raw_split_cfg = dict(splits[name])
        seed = int(splits[name].get("seed", base_seed + offset))
        arrays = simulate_split(seed=seed, cfg=split_cfg)
        split_type = str(raw_split_cfg.get("split_type", "ood_param" if split_cfg.parameter_split in {"ood", "ood_param"} else "iid"))
        is_ood = bool(raw_split_cfg.get("is_ood", split_type.startswith("ood") or split_cfg.parameter_split.startswith("ood")))
        arrays["is_ood"] = np.full((split_cfg.num_episodes,), int(is_ood), dtype=np.int64)
        np.savez_compressed(out_dir / f"{name}.npz", **arrays)
        counts = np.bincount(arrays["op_id"], minlength=4)
        split_summaries[name] = {
            "seed": seed,
            "num_episodes": int(split_cfg.num_episodes),
            "T": int(split_cfg.T),
            "n_min": int(split_cfg.n_min),
            "n_max": int(split_cfg.n_max),
            "nmax": int(split_cfg.nmax),
            "split_type": split_type,
            "is_ood": is_ood,
            "parameter_split": split_cfg.parameter_split,
            "operator_ranges": ranges_to_metadata(split_cfg.operator_ranges),
            "op_counts": counts.astype(int).tolist(),
        }
        print(f"wrote {out_dir / f'{name}.npz'} with {split_cfg.num_episodes} episodes")

    metadata = {
        "dataset": "LOWM-Synth",
        "version": config.get("version", "v0"),
        "config_path": str(config_path),
        "config": _jsonable(config),
        "splits": split_summaries,
        **operator_metadata(),
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    print(f"wrote {out_dir / 'metadata.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", nargs="*", default=None, help="Optional subset of split names to generate.")
    args = parser.parse_args()
    generate_dataset(args.config, args.out, args.splits)


if __name__ == "__main__":
    main()
