#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfd.dataset import load_dataset
from pfd.decision import refine_ambiguous_posteriors
from pfd.metrics import binary_metrics, brier_score
from pfd.train import train_single_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the reliability-aware GNN on one sample.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--theta-low", type=float, default=0.4)
    parser.add_argument("--theta-high", type=float, default=0.6)
    parser.add_argument("--theta-c", type=float, default=0.5)
    parser.add_argument("--refine-top-pct", type=float, default=0.1)
    args = parser.parse_args()

    sample = load_dataset(args.dataset)[args.sample_index]
    _, probs, _ = train_single_sample(sample, epochs=args.epochs)
    refined, pred = refine_ambiguous_posteriors(
        sample, probs, args.theta_low, args.theta_high, args.theta_c, args.refine_top_pct
    )
    metrics = binary_metrics(sample.labels, pred)
    metrics["brier"] = brier_score(sample.labels, refined)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
