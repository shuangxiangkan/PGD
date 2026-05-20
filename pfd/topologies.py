from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class Graph:
    name: str
    num_nodes: int
    edges: np.ndarray

    def neighbors(self) -> list[list[int]]:
        adj = [[] for _ in range(self.num_nodes)]
        for u, v in self.edges:
            adj[int(u)].append(int(v))
            adj[int(v)].append(int(u))
        return adj


def _unique_edges(edges: set[tuple[int, int]]) -> np.ndarray:
    ordered = sorted({(min(u, v), max(u, v)) for u, v in edges if u != v})
    return np.asarray(ordered, dtype=np.int64)


def hypercube(n: int) -> Graph:
    """Return the n-dimensional hypercube Q_n."""
    if n <= 0:
        raise ValueError("n must be positive")
    num_nodes = 2**n
    edges: set[tuple[int, int]] = set()
    for u in range(num_nodes):
        for bit in range(n):
            v = u ^ (1 << bit)
            if u < v:
                edges.add((u, v))
    return Graph(name=f"Q_{n}", num_nodes=num_nodes, edges=_unique_edges(edges))


def _coord_to_id(coord: tuple[int, ...], k: int) -> int:
    node_id = 0
    for x in coord:
        node_id = node_id * k + x
    return node_id


def kary_n_cube(n: int, k: int) -> Graph:
    """Return the k-ary n-cube with wrap-around links."""
    if n <= 0 or k <= 1:
        raise ValueError("n must be positive and k must be greater than one")
    edges: set[tuple[int, int]] = set()
    for coord in product(range(k), repeat=n):
        u = _coord_to_id(coord, k)
        for dim in range(n):
            for delta in (-1, 1):
                nxt = list(coord)
                nxt[dim] = (nxt[dim] + delta) % k
                v = _coord_to_id(tuple(nxt), k)
                edges.add((u, v))
    return Graph(name=f"K_{n},{k}", num_nodes=k**n, edges=_unique_edges(edges))


def augmented_kary_n_cube(n: int, k: int) -> Graph:
    """Return the augmented k-ary n-cube AQ_{n,k}.

    Coordinates are stored as (x_1, ..., x_n). The generator follows the
    standard definition for k >= 3: each vertex is adjacent to vertices obtained
    by changing one coordinate by +/-1, and to vertices obtained by changing
    the prefix (x_1, ..., x_i) by +/-1 for every 2 <= i <= n.
    """
    if n <= 0 or k < 3:
        raise ValueError("n must be positive and k must be at least three")
    edges: set[tuple[int, int]] = set()
    for coord in product(range(k), repeat=n):
        u = _coord_to_id(coord, k)
        for dim in range(n):
            for delta in (-1, 1):
                nxt = list(coord)
                nxt[dim] = (nxt[dim] + delta) % k
                v = _coord_to_id(tuple(nxt), k)
                edges.add((u, v))
        for prefix in range(2, n + 1):
            for delta in (-1, 1):
                nxt = list(coord)
                for dim in range(prefix):
                    nxt[dim] = (nxt[dim] + delta) % k
                v = _coord_to_id(tuple(nxt), k)
                edges.add((u, v))
    return Graph(name=f"AQ_{n},{k}", num_nodes=k**n, edges=_unique_edges(edges))
