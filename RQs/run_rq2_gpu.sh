#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

./.venv/bin/python RQs/run_rq2.py \
  --profile gpu \
  --device cuda \
  --runs 3 \
  --fault-rates 0.1 0.2 0.3 0.4 0.5 \
  --rounds 2 \
  --gamma-min 0.7 \
  --gamma-max 1.0 \
  --beta-min 0.0 \
  --beta-max 0.3 \
  --train-samples 200 \
  --test-samples 100 \
  --epochs 100 \
  --hidden-dim 64 \
  --layers 2 \
  --backbone graphsage
