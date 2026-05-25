"""Inspect a local CoPhy/Filtered-CoPhy-style dataset directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


KNOWN_SCENARIOS = ("BlocktowerCF", "BallsCF", "CollisionCF")
SPLIT_NAMES = ("train", "val", "valid", "validation", "test")
STATE_KEYS = ("states", "object_states", "objects", "features", "trajectories", "object_features")
CONFOUNDER_KEYS = ("op_id", "confounder_id", "operator_id", "world_id", "physical_id")
PARAM_KEYS = ("op_params", "confounders", "physical_params", "params", "properties")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".gif"}
ARRAY_SUFFIXES = {".npz", ".npy", ".h5", ".hdf5", ".pkl", ".pickle"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _safe_npz_summary(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"path": str(path), "keys": [], "arrays": {}}
    try:
        with np.load(path, allow_pickle=False) as data:
            summary["keys"] = list(data.files)
            for key in data.files[:32]:
                arr = data[key]
                summary["arrays"][key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    except Exception as exc:
        summary["error"] = str(exc)
    return summary


def _scenario_dirs(root: Path) -> list[Path]:
    dirs = [root / name for name in KNOWN_SCENARIOS if (root / name).exists()]
    if dirs:
        return dirs
    return [path for path in root.iterdir() if path.is_dir()] if root.exists() else []


def _split_files(scenario_dir: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for split in SPLIT_NAMES:
        candidates = []
        for suffix in ARRAY_SUFFIXES:
            candidates.extend([scenario_dir / f"{split}{suffix}", scenario_dir / split / f"{split}{suffix}"])
        split_dir = scenario_dir / split
        if split_dir.exists():
            candidates.extend(path for path in split_dir.rglob("*") if path.suffix.lower() in ARRAY_SUFFIXES)
        found = sorted({path for path in candidates if path.exists()})
        if found:
            canonical = "val" if split in {"valid", "validation"} else split
            out.setdefault(canonical, []).extend(found)
    return out


def _count_suffixes(paths: list[Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in paths:
        suffix = path.suffix.lower() or "<dir>"
        counts[suffix] = counts.get(suffix, 0) + 1
    return dict(sorted(counts.items()))


def _scenario_report(root: Path, scenario_dir: Path) -> dict[str, Any]:
    files = [path for path in scenario_dir.rglob("*") if path.is_file()]
    suffix_counts = _count_suffixes(files)
    split_files = _split_files(scenario_dir)
    array_files = [path for path in files if path.suffix.lower() in ARRAY_SUFFIXES]
    image_count = sum(1 for path in files if path.suffix.lower() in IMAGE_SUFFIXES)
    video_count = sum(1 for path in files if path.suffix.lower() in VIDEO_SUFFIXES)
    npz_examples = [_safe_npz_summary(path) for path in array_files if path.suffix.lower() == ".npz"][:3]
    keys = set()
    shapes: dict[str, Any] = {}
    for example in npz_examples:
        for key in example.get("keys", []):
            keys.add(str(key))
        for key, value in example.get("arrays", {}).items():
            shapes[key] = value
    has_state = any(key in keys for key in STATE_KEYS)
    has_confounder = any(key in keys for key in CONFOUNDER_KEYS)
    has_params = any(key in keys for key in PARAM_KEYS)
    states_shape = next((shapes[key]["shape"] for key in STATE_KEYS if key in shapes), None)
    trajectory_length = int(states_shape[1]) if states_shape and len(states_shape) >= 2 else None
    mode = "state/object-level LOWM" if has_state and (has_confounder or has_params) else "feature-level LOWM" if has_state else "visual encoder mode"
    feasibility = {
        "state_object_level_lowm": bool(has_state and (has_confounder or has_params)),
        "feature_level_lowm": bool(has_state),
        "visual_encoder_mode": bool(image_count or video_count),
    }
    return {
        "name": scenario_dir.name,
        "path": _relative(scenario_dir, root),
        "num_files": len(files),
        "suffix_counts": suffix_counts,
        "splits": {split: [_relative(path, root) for path in paths] for split, paths in split_files.items()},
        "npz_examples": npz_examples,
        "has_object_or_state_trajectories": has_state,
        "has_raw_images": image_count > 0,
        "has_raw_videos": video_count > 0,
        "has_confounder_or_operator_metadata": bool(has_confounder or has_params),
        "trajectory_length": trajectory_length,
        "intervention_counterfactual_structure": "unknown_from_filesystem_audit",
        "feasibility": feasibility,
        "recommended_first_mode": mode,
    }


def _missing_report(root: Path) -> dict[str, Any]:
    expected = {
        "state_or_feature_npz": {
            "layout": "<cophy_root>/<scenario>/<split>.npz",
            "scenarios": list(KNOWN_SCENARIOS),
            "splits": ["train", "val", "test"],
            "required_arrays": [
                "states or object_states or features: [episodes, time, objects, dims] or [episodes, time, dims]",
                "op_id or confounder_id or operator_id: [episodes]",
            ],
            "optional_arrays": [
                "actions: [episodes, time-1, objects, 2]",
                "mask: [episodes, time, objects]",
                "op_params or confounders or physical_params: [episodes, param_dim]",
                "num_objects: [episodes]",
                "sample_id/source_sample_id: [episodes]",
            ],
        },
        "raw_visual_only": {
            "layout": "<cophy_root>/<scenario>/<split>/... image or video files plus metadata",
            "status": "inspection supported; conversion requires provided state/object/features or precomputed encoder features",
        },
    }
    return {
        "root": str(root),
        "available": False,
        "scenarios": [],
        "expected_format": expected,
        "recommended_next_step": "Place downloaded CoPhy or Filtered-CoPhy files under the expected root, then rerun inspect_cophy.",
    }


def inspect_cophy(root: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not root.exists():
        report = _missing_report(root)
    else:
        scenario_dirs = _scenario_dirs(root)
        scenarios = [_scenario_report(root, path) for path in scenario_dirs]
        report = {
            "root": str(root),
            "available": bool(scenarios),
            "known_scenarios": list(KNOWN_SCENARIOS),
            "scenarios": scenarios,
            "recommended_mode_for_first_run": next(
                (scenario["recommended_first_mode"] for scenario in scenarios if scenario["feasibility"]["state_object_level_lowm"]),
                "feature-level LOWM if features are present; otherwise provide encoder features before training",
            ),
            "expected_format": _missing_report(root)["expected_format"],
        }
        if not scenarios:
            report["recommended_next_step"] = "No scenario directories were found. Place CoPhy scenario folders under the root."
    with (out_dir / "cophy_data_report.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonable(report), f, indent=2, sort_keys=True)
    _write_markdown(report, out_dir / "cophy_data_report.md")
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = ["# CoPhy Data Audit", "", f"- Root: `{report.get('root', '')}`", f"- Available: {report.get('available', False)}"]
    if not report.get("available"):
        lines.extend(
            [
                "",
                "Data was not found or no scenario directories were detected.",
                "",
                "Expected state/feature layout:",
                "- `<cophy_root>/<scenario>/<split>.npz`",
                "- Scenarios: `BlocktowerCF`, `BallsCF`, `CollisionCF`",
                "- Splits: `train`, `val`, `test`",
                "- Required arrays: `states`/`object_states`/`features` and `op_id`/`confounder_id`/`operator_id`.",
                "- Optional arrays: `actions`, `mask`, `op_params`/`confounders`/`physical_params`, `num_objects`, `sample_id`.",
                "",
                "Raw visual-only directories are inspectable but not directly converted in this milestone; provide object states or encoder features first.",
            ]
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    lines.append(f"- Recommended first mode: {report.get('recommended_mode_for_first_run', '')}")
    for scenario in report.get("scenarios", []):
        lines.extend(
            [
                "",
                f"## {scenario.get('name')}",
                f"- Path: `{scenario.get('path')}`",
                f"- Splits: {', '.join(sorted(scenario.get('splits', {}).keys())) or 'none detected'}",
                f"- File suffixes: {scenario.get('suffix_counts', {})}",
                f"- Object/state trajectories: {scenario.get('has_object_or_state_trajectories')}",
                f"- Raw images: {scenario.get('has_raw_images')}",
                f"- Raw videos: {scenario.get('has_raw_videos')}",
                f"- Confounder/operator metadata: {scenario.get('has_confounder_or_operator_metadata')}",
                f"- Trajectory length: {scenario.get('trajectory_length')}",
                f"- Feasibility: {scenario.get('feasibility')}",
                f"- Recommended mode: {scenario.get('recommended_first_mode')}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_cophy(args.root, args.out)
    print(json.dumps({"available": report.get("available", False), "num_scenarios": len(report.get("scenarios", []))}, indent=2))


if __name__ == "__main__":
    main()
