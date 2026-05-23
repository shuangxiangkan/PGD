#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

./.venv/bin/python RQs/run_rq2.py \
  --profile gpu \
  --device cuda \
  --runs 3 \
  --fault-rates 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 \
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
