#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pfd.dataset import generate_sample, save_dataset
from pfd.topologies import augmented_kary_n_cube, hypercube, kary_n_cube


def build_graph(name: str, n: int, k: int | None):
    if name == "hypercube":
        return hypercube(n)
    if name == "kary":
        if k is None:
            raise ValueError("--k is required for kary topology")
        return kary_n_cube(n, k)
    if name == "augmented_kary":
        if k is None:
            raise ValueError("--k is required for augmented_kary topology")
        return augmented_kary_n_cube(n, k)
    raise ValueError(f"unknown topology: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reusable probabilistic PMC datasets.")
    parser.add_argument("--topology", choices=["hypercube", "kary", "augmented_kary"], required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--fault-rate", type=float, default=0.1)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=None, help="Fix gamma for all samples.")
    parser.add_argument("--beta", type=float, default=None, help="Fix beta for all samples.")
    parser.add_argument("--gamma-min", type=float, default=0.8)
    parser.add_argument("--gamma-max", type=float, default=1.0)
    parser.add_argument("--beta-min", type=float, default=0.0)
    parser.add_argument("--beta-max", type=float, default=0.2)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    graph = build_graph(args.topology, args.n, args.k)
    if args.gamma is None and not (0.0 <= args.gamma_min <= args.gamma_max <= 1.0):
        raise ValueError("--gamma-min/--gamma-max must satisfy 0 <= min <= max <= 1")
    if args.beta is None and not (0.0 <= args.beta_min <= args.beta_max <= 1.0):
        raise ValueError("--beta-min/--beta-max must satisfy 0 <= min <= max <= 1")
    if args.gamma is not None and not (0.0 <= args.gamma <= 1.0):
        raise ValueError("--gamma must be in [0, 1]")
    if args.beta is not None and not (0.0 <= args.beta <= 1.0):
        raise ValueError("--beta must be in [0, 1]")

    rng = np.random.default_rng(args.seed)
    samples = [
        generate_sample(
            graph=graph,
            fault_rate=args.fault_rate,
            rounds=args.rounds,
            gamma=float(
                args.gamma
                if args.gamma is not None
                else rng.uniform(args.gamma_min, args.gamma_max)
            ),
            beta=float(
                args.beta
                if args.beta is not None
                else rng.uniform(args.beta_min, args.beta_max)
            ),
            seed=args.seed + i,
        )
        for i in range(args.num_samples)
    ]
    save_dataset(samples, args.output)
    print(f"saved {len(samples)} samples to {args.output}")
    print(f"topology={graph.name}, nodes={graph.num_nodes}, edges={len(graph.edges)}")
    gammas = [sample.gamma for sample in samples]
    betas = [sample.beta for sample in samples]
    print(f"gamma: min={min(gammas):.4f}, max={max(gammas):.4f}")
    print(f"beta: min={min(betas):.4f}, max={max(betas):.4f}")


if __name__ == "__main__":
    main()
