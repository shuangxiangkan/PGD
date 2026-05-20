from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np

from pfd.topologies import Graph


@dataclass(frozen=True)
class DatasetConfig:
    topology: str
    num_nodes: int
    num_edges: int
    fault_rate: float
    rounds: int
    gamma: float
    beta: float
    seed: int


@dataclass
class PMCSample:
    graph_name: str
    num_nodes: int
    edges: np.ndarray
    labels: np.ndarray
    s_uv: np.ndarray
    s_vu: np.ndarray
    gamma: float
    beta: float
    fault_rate: float
    seed: int


def _sample_directed_outcomes(
    tester_state: int,
    tested_state: int,
    rounds: int,
    gamma: float,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if tester_state == 0 and tested_state == 0:
        return np.zeros(rounds, dtype=np.int8)
    if tester_state != tested_state:
        return (rng.random(rounds) < gamma).astype(np.int8)
    match = rng.random(rounds) < beta
    return (~match).astype(np.int8)


def generate_sample(
    graph: Graph,
    fault_rate: float,
    rounds: int,
    gamma: float,
    beta: float,
    seed: int,
) -> PMCSample:
    """Generate one reusable probabilistic PMC diagnosis instance."""
    if not (0.0 <= fault_rate <= 1.0):
        raise ValueError("fault_rate must be in [0, 1]")
    rng = np.random.default_rng(seed)
    num_faults = max(1, int(round(graph.num_nodes * fault_rate)))
    faulty = rng.choice(graph.num_nodes, size=num_faults, replace=False)
    labels = np.zeros(graph.num_nodes, dtype=np.int8)
    labels[faulty] = 1

    s_uv = np.zeros((len(graph.edges), rounds), dtype=np.int8)
    s_vu = np.zeros((len(graph.edges), rounds), dtype=np.int8)
    for i, (u, v) in enumerate(graph.edges):
        u = int(u)
        v = int(v)
        s_uv[i] = _sample_directed_outcomes(labels[u], labels[v], rounds, gamma, beta, rng)
        s_vu[i] = _sample_directed_outcomes(labels[v], labels[u], rounds, gamma, beta, rng)

    return PMCSample(
        graph_name=graph.name,
        num_nodes=graph.num_nodes,
        edges=graph.edges.copy(),
        labels=labels,
        s_uv=s_uv,
        s_vu=s_vu,
        gamma=gamma,
        beta=beta,
        fault_rate=fault_rate,
        seed=seed,
    )


def save_dataset(samples: list[PMCSample], output_dir: str | Path) -> None:
    output = Path(output_dir)
    sample_dir = output / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for idx, sample in enumerate(samples):
        name = f"sample_{idx:05d}.npz"
        np.savez_compressed(
            sample_dir / name,
            graph_name=sample.graph_name,
            num_nodes=sample.num_nodes,
            edges=sample.edges,
            labels=sample.labels,
            s_uv=sample.s_uv,
            s_vu=sample.s_vu,
            gamma=sample.gamma,
            beta=sample.beta,
            fault_rate=sample.fault_rate,
            seed=sample.seed,
        )
        manifest.append(
            {
                "file": f"samples/{name}",
                "graph_name": sample.graph_name,
                "num_nodes": int(sample.num_nodes),
                "num_edges": int(len(sample.edges)),
                "fault_rate": float(sample.fault_rate),
                "rounds": int(sample.s_uv.shape[1]),
                "gamma": float(sample.gamma),
                "beta": float(sample.beta),
                "seed": int(sample.seed),
            }
        )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_dataset(dataset_dir: str | Path) -> list[PMCSample]:
    root = Path(dataset_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    samples: list[PMCSample] = []
    for row in manifest:
        arr = np.load(root / row["file"], allow_pickle=False)
        samples.append(
            PMCSample(
                graph_name=str(arr["graph_name"]),
                num_nodes=int(arr["num_nodes"]),
                edges=arr["edges"].astype(np.int64),
                labels=arr["labels"].astype(np.int8),
                s_uv=arr["s_uv"].astype(np.int8),
                s_vu=arr["s_vu"].astype(np.int8),
                gamma=float(arr["gamma"]),
                beta=float(arr["beta"]),
                fault_rate=float(arr["fault_rate"]),
                seed=int(arr["seed"]),
            )
        )
    return samples
