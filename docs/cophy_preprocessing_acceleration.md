# CoPhy Preprocessing Acceleration

Run a benchmark first:

```bash
python -m lowm.data.benchmark_cophy_adapter \
  --root /content/drive/MyDrive/LOWM/raw_cophy \
  --out /content/drive/MyDrive/LOWM/data/cophy_benchmark \
  --scenario ballsCF \
  --mode segm_features \
  --benchmark-episodes 200
```

Fast 10k pilot across 5 runtimes, one `--partition-index` per runtime:

```bash
python -m lowm.data.cophy_adapter \
  --root /content/drive/MyDrive/LOWM/raw_cophy \
  --out /content/drive/MyDrive/LOWM/data/cophy_parts_10k/part0 \
  --scenario ballsCF \
  --mode segm_features \
  --object-counts 2 \
  --num-partitions 5 \
  --partition-index 0 \
  --num-frames 20 \
  --save-shards \
  --no-final-merge \
  --worker-backend process \
  --num-workers auto \
  --compression none \
  --local-tmp /content/cophy_part0 \
  --copy-final-to-out \
  --resume
```

Faster pilot mode with explicit feature tradeoffs:

```bash
python -m lowm.data.cophy_adapter \
  --root /content/drive/MyDrive/LOWM/raw_cophy \
  --out /content/drive/MyDrive/LOWM/data/cophy_parts_10k/part0 \
  --scenario ballsCF \
  --mode segm_features \
  --object-counts 2 \
  --num-partitions 5 \
  --partition-index 0 \
  --num-frames 12 \
  --segm-resize 112 \
  --save-shards \
  --no-final-merge \
  --worker-backend process \
  --num-workers auto \
  --compression none \
  --local-tmp /content/cophy_part0 \
  --copy-final-to-out \
  --resume
```

Full ballsCF across many runtimes:

```bash
python -m lowm.data.cophy_adapter \
  --root /content/drive/MyDrive/LOWM/raw_cophy \
  --out /content/drive/MyDrive/LOWM/data/cophy_full_parts/partK \
  --scenario ballsCF \
  --mode segm_features \
  --num-partitions N \
  --partition-index K \
  --num-frames 20 \
  --save-shards \
  --resume \
  --no-final-merge \
  --worker-backend process \
  --num-workers auto \
  --compression none \
  --local-tmp /content/cophy_partK \
  --copy-final-to-out \
  --force-full
```

Merge completed parts:

```bash
python -m lowm.data.merge_cophy_partitions \
  --parts \
    /content/drive/MyDrive/LOWM/data/cophy_parts_10k/part0/ballsCF \
    /content/drive/MyDrive/LOWM/data/cophy_parts_10k/part1/ballsCF \
    /content/drive/MyDrive/LOWM/data/cophy_parts_10k/part2/ballsCF \
    /content/drive/MyDrive/LOWM/data/cophy_parts_10k/part3/ballsCF \
    /content/drive/MyDrive/LOWM/data/cophy_parts_10k/part4/ballsCF \
  --out /content/drive/MyDrive/LOWM/data/cophy_merged_10k/ballsCF \
  --split-seed 0
```

`confounders.npy` is preserved only as metadata for sampling/evaluation. It is not used as model input.
