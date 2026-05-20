#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pfd.baselines import clustered_pmc_labels, majority_mismatch_scores
from pfd.dataset import load_dataset
from pfd.decision import refine_ambiguous_posteriors
from pfd.metrics import binary_metrics, brier_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reusable heuristic baseline on a generated dataset.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=["majority-mismatch", "clustered-pmc"],
        default="majority-mismatch",
    )
    parser.add_argument("--theta-low", type=float, default=0.4)
    parser.add_argument("--theta-high", type=float, default=0.6)
    parser.add_argument("--theta-c", type=float, default=0.5)
    parser.add_argument("--refine-top-pct", type=float, default=0.1)
    args = parser.parse_args()

    samples = load_dataset(args.dataset)
    rows = []
    for sample in samples:
        if args.method == "majority-mismatch":
            probs = majority_mismatch_scores(sample)
            refined, pred = refine_ambiguous_posteriors(
                sample, probs, args.theta_low, args.theta_high, args.theta_c, args.refine_top_pct
            )
        else:
            pred = clustered_pmc_labels(sample)
            refined = pred.astype(np.float32)
        m = binary_metrics(sample.labels, pred)
        m["brier"] = brier_score(sample.labels, refined)
        rows.append(m)
    keys = rows[0].keys()
    avg = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    print("samples:", len(samples))
    for key, value in avg.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
