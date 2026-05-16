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

OOD parameter generalization:

```bash
python -m lowm.data.generate_dataset --config configs/lowm_synth_ood_param.yaml --out data/lowm_synth_ood_param
python -m lowm.training.run_sweep --config configs/sweeps/lowm_ood_param_main.yaml
python -m lowm.eval.evaluate_all --run runs/lowm_synth_ood_param/main/runs/lowm_lowm_omcr_no_pairwise_seed0 --split test_iid --checkpoint best_law_pair.pt
python -m lowm.eval.evaluate_all --run runs/lowm_synth_ood_param/main/runs/lowm_lowm_omcr_no_pairwise_seed0 --split test_ood_param --checkpoint best_law_pair.pt
python -m lowm.eval.evaluate_law_mismatch_only --run runs/lowm_synth_ood_param/main/runs/lowm_lowm_omcr_no_pairwise_seed0 --split test_ood_param --checkpoint best_law_pair.pt
python -m lowm.eval.aggregate_sweep --sweep_dir runs/lowm_synth_ood_param/main --out runs/lowm_synth_ood_param/main/summary --splits val test_iid test_ood_param
```

Active Operator Inference:

```bash
python -m lowm.eval.active_operator_inference --run runs/lowm_synth_ood_param/main/runs/lowm_lowm_omcr_no_pairwise_seed0 --split test_iid --checkpoint best_law_pair.pt --num-episodes 200 --num-operator-hypotheses 4 --num-actions 8 --horizon 3
python -m lowm.eval.active_operator_inference --run runs/lowm_synth_ood_param/main/runs/lowm_no_law_mismatch_seed0 --split test_iid --checkpoint best_law_pair.pt --num-episodes 200 --num-operator-hypotheses 4 --num-actions 8 --horizon 3
python -m lowm.eval.active_operator_inference --run runs/lowm_synth_ood_param/main/runs/lowm_lowm_omcr_no_pairwise_seed0 --split test_ood_param --checkpoint best_law_pair.pt --num-episodes 200 --num-operator-hypotheses 4 --num-actions 8 --horizon 3
python -m lowm.eval.aggregate_aoi --runs runs/lowm_synth_ood_param/main/runs/lowm_lowm_omcr_no_pairwise_seed0 runs/lowm_synth_ood_param/main/runs/lowm_no_law_mismatch_seed0 --out runs/lowm_synth_ood_param/main/aoi_summary --split test_iid
```

AOI writes `aoi_metrics.json`, per-episode decisions, action score examples, and plots under `<run>/eval/<split>/aoi/`. The primary diagnostic is AOI with an OMC critic versus the no-law-mismatch critic.

EBTWM inference pilot:

```bash
python -m lowm.eval.ebtwm_inference --run runs/lowm_synth_ood_param/main/runs/lowm_lowm_omcr_no_pairwise_seed0 --split test_iid --checkpoint best_law_pair.pt --num-samples 100 --num-steps 100 --step-size 1e-2 --noise-std 0.05 --corruption-type gaussian
python -m lowm.eval.ebtwm_inference --run runs/lowm_synth_ood_param/main/runs/lowm_lowm_omcr_no_pairwise_seed0 --split test_iid --checkpoint best_law_pair.pt --mode counterfactual --num-samples 100 --num-steps 100 --step-size 1e-2
python -m lowm.eval.ebtwm_inference --compare-runs runs/lowm_synth_ood_param/main/runs/lowm_lowm_omcr_no_pairwise_seed0 runs/lowm_synth_ood_param/main/runs/lowm_no_law_mismatch_seed0 --split test_iid --compare-out runs/lowm_synth_ood_param/main/ebtwm_cross_critic
```

EBTWM writes `metrics.json`, per-sample metrics, optimization curves, cross-operator energies, and before/after trajectory plots under `<run>/eval/<split>/ebtwm_inference/`. The `go_no_go_decision` field reports the strict pilot verdict.

Hybrid EBTWM shaping:

```bash
python -m lowm.eval.energy_gradient_diagnostic --run runs/lowm_synth_ood_param/main/runs/lowm_lowm_omcr_no_pairwise_seed0 --split test_iid --checkpoint best_law_pair.pt --num-samples 100 --noise-std 0.05
python -m lowm.training.train_lowm --config configs/train_lowm_ebtwm_shaped.yaml
python -m lowm.training.run_sweep --config configs/sweeps/ebtwm_shaping_alpha_debug.yaml
python -m lowm.training.run_sweep --config configs/sweeps/ebtwm_shaping_alpha.yaml
python -m lowm.eval.aggregate_ebtwm_shaping --sweep_dir runs/lowm_synth_ood_param/ebtwm_shaping --out runs/lowm_synth_ood_param/ebtwm_shaping/summary
```

Hybrid shaping adds DSM, clean-vs-noisy ranking, and optional gradient regularization to the OMC ranking objective. OCCL remains disabled in these configs.

## Stored Arrays

Each split is a compressed `.npz` file with:

- `states`: `[num_episodes, T+1, Nmax, 7]`
- `actions`: `[num_episodes, T, Nmax, 2]`
- `mask`: `[num_episodes, T+1, Nmax]`
- `op_id`: `[num_episodes]`
- `op_params`: `[num_episodes, 5]`
- `num_objects`: `[num_episodes]`
- `is_ood`: `[num_episodes]` when generated by configs with IID/OOD split metadata

The hidden `op_id` and `op_params` are stored for validation, visualization, and future oracle analysis only. They are not model inputs.
