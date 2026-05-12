"""Debug LOWM-Synth ranking samples and visualize positives vs negatives."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from lowm.data.dataset import LOWMSynthRankingDataset, make_ranking_dataloader, ranking_config_from_mapping, validate_ranking_sample
from lowm.data.generate_dataset import generate_dataset


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError("config must contain a YAML mapping")
    return loaded


def _plot_candidate(ax: plt.Axes, states, mask, title: str) -> None:
    active = (mask[0] > 0.5).nonzero()[0]
    for obj_idx in active:
        xy = states[:, obj_idx, 0:2]
        ax.plot(xy[:, 0], xy[:, 1], marker="o", markersize=2, linewidth=1)
        ax.scatter(xy[0, 0], xy[0, 1], s=14, color="black")
        ax.scatter(xy[-1, 0], xy[-1, 1], s=18, marker="x", color="black")
    ax.set_title(title, fontsize=8)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.25, alpha=0.4)


def visualize_sample(sample: dict[str, Any], out_path: Path) -> None:
    cand_states = sample["cand_states"].numpy()
    cand_mask = sample["cand_mask"].numpy()
    neg_types = sample["negative_types"]
    label = int(sample["labels"].item())
    m = cand_states.shape[0]
    cols = min(m, 5)
    rows = (m + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.1 * rows), squeeze=False)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i >= m:
            ax.axis("off")
            continue
        title = f"{i}: {neg_types[i]}"
        if i == label:
            title += " (label)"
        _plot_candidate(ax, cand_states[i], cand_mask[i], title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--path", type=Path, default=None, help="Optional existing .npz split.")
    parser.add_argument("--out", type=Path, default=Path("figures/ranking_debug"))
    parser.add_argument("--num-samples", type=int, default=3)
    args = parser.parse_args()

    config = _load_config(args.config)
    ranking_cfg = ranking_config_from_mapping(config)
    dataset_path = args.path
    if dataset_path is None:
        debug_dir = Path("data/lowm_synth_ranking_debug")
        dataset_path = debug_dir / "train.npz"
        if not dataset_path.exists():
            debug_config = dict(config)
            debug_config["splits"] = {
                "train": {
                    "num_episodes": 64,
                    "n_min": 3,
                    "n_max": 5,
                    "parameter_split": "iid",
                    "seed": int(config.get("seed", 0)) + 991,
                }
            }
            temp_config = debug_dir / "debug_config.yaml"
            debug_dir.mkdir(parents=True, exist_ok=True)
            with temp_config.open("w", encoding="utf-8") as f:
                yaml.safe_dump(debug_config, f, sort_keys=False)
            generate_dataset(temp_config, debug_dir, ["train"])

    dataset = LOWMSynthRankingDataset(dataset_path, ranking_cfg, num_samples=max(args.num_samples, 8))
    loader = make_ranking_dataloader(dataset, batch_size=min(2, len(dataset)), shuffle=False)
    batch = next(iter(loader))
    print("Batch shapes:")
    for key in ["context_states", "context_actions", "context_mask", "cand_states", "cand_actions", "cand_mask", "labels"]:
        value = batch[key]
        print(f"  {key}: {tuple(value.shape)}")
    print(f"  negative_types[0]: {batch['negative_types'][0]}")

    for i in range(args.num_samples):
        sample = dataset[i]
        report = validate_ranking_sample(sample, ranking_cfg)
        print(f"sample {i}: ok={report['ok']} label={int(sample['labels'])} counts={report['negative_type_counts']}")
        if not report["ok"]:
            raise RuntimeError(report["errors"])
        visualize_sample(sample, args.out / f"ranking_sample_{i}.png")
    print(f"wrote visualizations to {args.out}")


if __name__ == "__main__":
    main()
