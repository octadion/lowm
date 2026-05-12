"""Plot LOWM-Synth trajectory examples."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lowm.data.operators import OP_FAMILIES


def _plot_episode(ax: plt.Axes, states: np.ndarray, mask: np.ndarray, title: str) -> None:
    active = np.where(mask[0] > 0.5)[0]
    for obj_idx in active:
        xy = states[:, obj_idx, 0:2]
        radius = float(states[0, obj_idx, 4])
        ax.plot(xy[:, 0], xy[:, 1], marker="o", markersize=2.5, linewidth=1.2, label=f"obj {obj_idx}")
        ax.scatter(xy[0, 0], xy[0, 1], s=1800 * radius, facecolors="none", edgecolors="black", linewidths=0.8)
        ax.scatter(xy[-1, 0], xy[-1, 1], s=28, marker="x", color=ax.lines[-1].get_color())
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.45)


def visualize_dataset(path: Path, out_dir: Path, num_per_op: int = 1) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(path) as data:
        states = data["states"]
        mask = data["mask"]
        op_id = data["op_id"]
        op_params = data["op_params"]

    written: list[Path] = []
    for op in sorted(OP_FAMILIES):
        indices = np.where(op_id == op)[0][:num_per_op]
        if len(indices) == 0:
            continue
        fig, axes = plt.subplots(1, len(indices), figsize=(4.0 * len(indices), 4.0), squeeze=False)
        for col, idx in enumerate(indices):
            params = ", ".join(f"{v:.3g}" for v in op_params[idx])
            _plot_episode(axes[0, col], states[idx], mask[idx], f"{OP_FAMILIES[op]}\nparams=[{params}]")
        fig.tight_layout()
        out_path = out_dir / f"op_{op}_{OP_FAMILIES[op]}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        written.append(out_path)

    if len(written) > 1:
        fig, axes = plt.subplots(1, len(written), figsize=(4.0 * len(written), 4.0), squeeze=False)
        for col, op in enumerate(sorted(OP_FAMILIES)):
            indices = np.where(op_id == op)[0]
            if len(indices) == 0:
                continue
            idx = int(indices[0])
            _plot_episode(axes[0, col], states[idx], mask[idx], OP_FAMILIES[op])
        fig.tight_layout()
        overview = out_dir / "operator_examples.png"
        fig.savefig(overview, dpi=180)
        plt.close(fig)
        written.append(overview)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-per-op", type=int, default=1)
    args = parser.parse_args()
    written = visualize_dataset(args.path, args.out, args.num_per_op)
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
