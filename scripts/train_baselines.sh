#!/usr/bin/env bash
set -euo pipefail

python -m lowm.training.train_baseline --config configs/train_baselines.yaml --baseline fixed_energy
python -m lowm.training.train_baseline --config configs/train_baselines.yaml --baseline direct_context_energy
