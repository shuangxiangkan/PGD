#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

./.venv/bin/python RQs/run_rq1.py \
  --profile gpu \
  --device cuda \
  --runs 3 \
  --fault-rate 0.3 \
  --rounds 2 \
  --gamma-min 0.7 \
  --gamma-max 1.0 \
  --beta-min 0.0 \
  --beta-max 0.3 \
  --train-samples 10 \
  --test-samples 5 \
  --epochs 20 \
  --hidden-dim 64 \
  --layers 2 \
  --backbone graphsage \
  --batch-size 16 \
  --amp \
  --force
