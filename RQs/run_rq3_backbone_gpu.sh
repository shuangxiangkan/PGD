#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

./.venv/bin/python RQs/run_rq3_backbone.py \
  --profile gpu \
  --device cuda \
  --runs 3 \
  --fault-rate 0.3 \
  --rounds 2 \
  --gamma-min 0.7 \
  --gamma-max 1.0 \
  --beta-min 0.0 \
  --beta-max 0.3 \
  --train-samples 100 \
  --test-samples 100 \
  --epochs 60 \
  --hidden-dim 64 \
  --layers 2 \
  --eval-every 5 \
  --batch-size 16 \
  --amp
