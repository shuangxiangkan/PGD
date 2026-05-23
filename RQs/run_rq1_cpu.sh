#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

./.venv/bin/python RQs/run_rq1.py \
  --profile cpu \
  --device cpu \
  --runs 3 \
  --fault-rate 0.3 \
  --rounds 2 \
  --gamma-min 0.7 \
  --gamma-max 1.0 \
  --beta-min 0.0 \
  --beta-max 0.3 \
  --train-samples 120 \
  --test-samples 60 \
  --epochs 80 \
  --hidden-dim 64 \
  --layers 2 \
  --backbone graphsage
