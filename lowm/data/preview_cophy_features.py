"""Preview CoPhy video feature extraction for one original episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lowm.data.cophy_adapter import (
    _read_video,
    _sample_video_frames,
    _segmentation_colors,
    extract_rgb_features_from_frames,
    extract_segmentation_features_from_frames,
)


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


def _save_frame(frame: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, frame)


def _plot_tracks(features: np.ndarray, mask: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    for slot in range(features.shape[1]):
        visible = mask[:, slot] > 0.5
        if not visible.any():
            continue
        ax.plot(features[visible, slot, 0], features[visible, slot, 1], marker="o", markersize=2, label=f"slot {slot}")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(1.05, -1.05)
    ax.set_xlabel("x center")
    ax.set_ylabel("y center")
    ax.set_title("Extracted centroid tracks")
    if features.shape[1] <= 12:
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def preview_cophy_features(
    episode: Path,
    out_dir: Path,
    mode: str = "segm_features",
    num_frames: int = 20,
    nmax: int = 9,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = episode / "ab" / "rgb.mp4"
    segm_path = episode / "ab" / "segm.mp4"
    source_path = segm_path if mode == "segm_features" else rgb_path
    if not source_path.exists():
        raise FileNotFoundError(f"missing source video {source_path}")
    if rgb_path.exists():
        rgb_frames = _sample_video_frames(_read_video(rgb_path), num_frames)
        _save_frame(rgb_frames[0], out_dir / "first_rgb_frame.png")
    if segm_path.exists():
        segm_frames = _sample_video_frames(_read_video(segm_path), num_frames)
        _save_frame(segm_frames[0], out_dir / "first_segm_frame.png")
    frames = _sample_video_frames(_read_video(source_path), num_frames)
    if mode == "segm_features":
        colors = _segmentation_colors(frames, threshold=8, nmax=nmax)
        features, mask, colors = extract_segmentation_features_from_frames(frames, colors=colors, nmax=nmax)
    elif mode == "rgb_features":
        features, mask = extract_rgb_features_from_frames(frames, nmax=nmax)
        colors = []
    else:
        raise ValueError("mode must be segm_features or rgb_features")
    _plot_tracks(features, mask, out_dir / "centroid_tracks.png")
    confounders_path = episode / "confounders.npy"
    confounders_shape = None
    if confounders_path.exists():
        confounders = np.load(confounders_path, allow_pickle=False)
        confounders_shape = list(confounders.shape)
    summary = {
        "episode": str(episode),
        "mode": mode,
        "num_frames": int(num_frames),
        "nmax": int(nmax),
        "feature_shape": list(features.shape),
        "mask_shape": list(mask.shape),
        "visible_slots": int((mask.sum(axis=0) > 0).sum()),
        "colors": colors,
        "confounders_shape": confounders_shape,
        "outputs": {
            "first_rgb_frame": str(out_dir / "first_rgb_frame.png") if rgb_path.exists() else None,
            "first_segm_frame": str(out_dir / "first_segm_frame.png") if segm_path.exists() else None,
            "centroid_tracks": str(out_dir / "centroid_tracks.png"),
        },
    }
    with (out_dir / "feature_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, indent=2, sort_keys=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True, help="Path to one CoPhy episode directory.")
    parser.add_argument("--out", type=Path, default=Path("runs/cophy_feature_preview"))
    parser.add_argument("--mode", type=str, default="segm_features", choices=["segm_features", "rgb_features"])
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--nmax", type=int, default=9)
    args = parser.parse_args()
    summary = preview_cophy_features(args.episode, args.out, args.mode, args.num_frames, args.nmax)
    print(json.dumps({"feature_shape": summary["feature_shape"], "visible_slots": summary["visible_slots"]}, indent=2))


if __name__ == "__main__":
    main()
