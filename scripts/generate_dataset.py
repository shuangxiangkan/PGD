#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    graph = build_graph(args.topology, args.n, args.k)
    samples = [
        generate_sample(
            graph=graph,
            fault_rate=args.fault_rate,
            rounds=args.rounds,
            gamma=args.gamma,
            beta=args.beta,
            seed=args.seed + i,
        )
        for i in range(args.num_samples)
    ]
    save_dataset(samples, args.output)
    print(f"saved {len(samples)} samples to {args.output}")
    print(f"topology={graph.name}, nodes={graph.num_nodes}, edges={len(graph.edges)}")


if __name__ == "__main__":
    main()
