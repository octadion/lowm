# Paper 1 C-SWM-Style Baseline Plan

Purpose: test whether OMC works as a semantic hard-negative principle for a contrastive object-centric world-model baseline, without turning Paper 1 into an architecture paper.

## Baseline Shape

Use a lightweight C-SWM-style transition scorer:

- Object encoder: per-object MLP over state/object features.
- Context encoder: DeepSets pooling over observed transitions.
- Transition head: predicts next object latent from current object latent, action, and context embedding.
- Energy: masked prediction error between predicted next object latents and candidate next object latents, summed or averaged over objects and time.

## Training Variants

Standard negatives:

- `state_corrupted`
- `temporal_shuffled`
- `random_impossible`

OMC negatives:

- `state_corrupted`
- `temporal_shuffled`
- `law_mismatch` / `wrong_confounder`
- `random_impossible`

The comparison must be within the same C-SWM-style backbone.

## Metrics

Report the same Paper 1 diagnostics:

- generic top1
- law_pair / wrong-confounder pair accuracy
- law_only_top1 / confounder_only_top1
- same_lt_wrong
- same_lt_wrong_lt_noise
- energy_matrix_mrr
- energy_matrix_diagonal_top1_accuracy
- OOD parameter split

## Implementation Notes

This should live as a baseline model rather than a new main architecture:

- Add `cswm_energy` to `lowm/models/baselines.py`.
- Reuse `train_baseline.py`.
- Reuse `evaluate_all`, `evaluate_coherence_stratification`, `evaluate_energy_matrix`, and `evaluate_cophy_ranking`.
- Add a sweep config mirroring `direct_context_omc_ablation.yaml`.

## Paper Framing

Use this baseline only to support the claim that OMC is not LOWM-specific. Do not claim to beat full C-SWM, Dreamer, DALI, or JEPA systems.
