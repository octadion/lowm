#!/usr/bin/env bash
set -euo pipefail

python -m lowm.training.train_lowm --config configs/train_lowm.yaml
