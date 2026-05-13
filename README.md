# LOWM: Latent Operator World Model

This repository currently includes LOWM-Synth v0 data generation, validation, visualization, and a PyTorch ranking `Dataset`/`DataLoader` with context sampling and negative candidates.

## Setup

```bash
pip install -e .[dev]
```

## Generate LOWM-Synth v0

```bash
python -m lowm.data.generate_dataset --config configs/lowm_synth_v0.yaml --out data/lowm_synth_v0
```

Generate only a subset while debugging:

```bash
python -m lowm.data.generate_dataset --config configs/lowm_synth_v0.yaml --out data/lowm_synth_v0 --splits train
```

## Validate

```bash
python -m lowm.data.validate_dataset --path data/lowm_synth_v0/train.npz
```

## Visualize

```bash
python -m lowm.data.visualize --path data/lowm_synth_v0/train.npz --out figures/dataset_examples
```

## Debug Ranking Dataset

```bash
python -m lowm.data.debug_ranking_dataset --config configs/lowm_synth_v0.yaml
```

The ranking dataset yields:

- `context_states`: `[B, K, 2, Nmax, D]`
- `context_actions`: `[B, K, Nmax, 2]`
- `context_mask`: `[B, K, 2, Nmax]`
- `cand_states`: `[B, M, H+1, Nmax, D]`
- `cand_actions`: `[B, M, H, Nmax, 2]`
- `cand_mask`: `[B, M, H+1, Nmax]`
- `labels`: `[B]`
- `negative_types`: per-sample metadata with one `positive` marker and shuffled negatives

## Test

```bash
pytest
```

## Train Milestone 3 Baselines

```bash
python -m lowm.training.train_baseline --config configs/train_baselines.yaml --baseline fixed_energy
python -m lowm.training.train_baseline --config configs/train_baselines.yaml --baseline direct_context_energy
```

Training logs include NCE loss, ranking top-1 accuracy, mean rank, MRR, and law-mismatch pairwise accuracy/energy gap. Checkpoints and `metrics.json` are written under `runs/lowm_synth_v0/`.

## Train LOWM v0

```bash
python -m lowm.training.train_lowm --config configs/train_lowm.yaml
python -m lowm.training.train_lowm --config configs/train_lowm_occl.yaml
```

LOWM infers `q(lambda | C)` from context transitions, scores candidates with self and pairwise object-centric energy, and logs NCE/KL/stability terms plus validation ranking and law-mismatch metrics.

## Evaluate And Aggregate

```bash
python -m lowm.eval.evaluate_all --run runs/lowm_synth_v0/lowm_seed0 --split val --checkpoint best.pt --num-samples 200 --seed 123
python -m lowm.eval.evaluate_occl_alignment --run runs/lowm_synth_v0/lowm_occl_seed0 --split val --checkpoint best_occl_acc.pt
python -m lowm.eval.evaluate_law_mismatch_only --run runs/lowm_synth_v0/lowm_seed0 --split val --checkpoint best_law_pair.pt
python -m lowm.eval.compare_train_eval_metrics --run runs/lowm_synth_v0/lowm_seed0 --split val
python -m lowm.eval.aggregate_results --runs runs/lowm_synth_v0/fixed_energy_seed0 runs/lowm_synth_v0/direct_context_energy_seed0 runs/lowm_synth_v0/lowm_seed0 --checkpoints best_top1.pt best_law_pair.pt last.pt --out runs/lowm_synth_v0/summary
```

Evaluation writes ranking metrics, negative-type breakdowns, `debug_energies.csv`, state-vs-law energy matrices, retrieval metrics, and plots under `<run>/eval/<split>/`.
LOWM training saves `best_top1.pt`, `best_loss.pt`, `best_law_pair.pt`, `best_law_gap.pt`, `last.pt`, and keeps `best.pt` as a `best_top1.pt` alias.

## Run Ablation Sweep

```bash
python -m lowm.training.run_sweep --config configs/sweeps/lowm_occl_ablation.yaml
python -m lowm.eval.aggregate_sweep --sweep_dir runs/lowm_synth_v0/lowm_occl_ablation --out runs/lowm_synth_v0/lowm_occl_ablation/summary
```

The sweep runner writes generated configs, trains each LOWM-OCCL ablation, evaluates ranking/law-only/OCCL alignment, and the sweep aggregator produces ablation CSV/Markdown tables plus plots.

Component ablations:

```bash
python -m lowm.training.run_sweep --config configs/sweeps/lowm_component_ablation.yaml
python -m lowm.eval.aggregate_sweep --sweep_dir runs/lowm_synth_v0/lowm_component_ablation --out runs/lowm_synth_v0/lowm_component_ablation/summary
```

## Stored Arrays

Each split is a compressed `.npz` file with:

- `states`: `[num_episodes, T+1, Nmax, 7]`
- `actions`: `[num_episodes, T, Nmax, 2]`
- `mask`: `[num_episodes, T+1, Nmax]`
- `op_id`: `[num_episodes]`
- `op_params`: `[num_episodes, 5]`
- `num_objects`: `[num_episodes]`

The hidden `op_id` and `op_params` are stored for validation, visualization, and future oracle analysis only. They are not model inputs.
