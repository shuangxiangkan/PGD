from __future__ import annotations

import numpy as np

from pfd.dataset import PMCSample


def build_features(sample: PMCSample) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build node features, edge features, and bidirectional mismatch ratios."""
    n = sample.num_nodes
    edges = sample.edges
    rounds = sample.s_uv.shape[1]
    mismatch_counts = sample.s_uv.sum(axis=1) + sample.s_vu.sum(axis=1)
    match_counts = 2 * rounds - mismatch_counts
    r_uv = mismatch_counts.astype(np.float32) / float(2 * rounds)

    neighbor_count = np.zeros(n, dtype=np.float32)
    match_sum = np.zeros(n, dtype=np.float32)
    mismatch_sum = np.zeros(n, dtype=np.float32)
    ratios: list[list[float]] = [[] for _ in range(n)]

    for i, (u, v) in enumerate(edges):
        u = int(u)
        v = int(v)
        neighbor_count[u] += 1
        neighbor_count[v] += 1
        match_sum[u] += match_counts[i]
        match_sum[v] += match_counts[i]
        mismatch_sum[u] += mismatch_counts[i]
        mismatch_sum[v] += mismatch_counts[i]
        ratios[u].append(float(r_uv[i]))
        ratios[v].append(float(r_uv[i]))

    denom = np.maximum(neighbor_count, 1.0) * float(2 * rounds)
    f_match = match_sum / denom
    f_mismatch = mismatch_sum / denom
    f_dispersion = np.zeros(n, dtype=np.float32)
    for u in range(n):
        if ratios[u]:
            vals = np.asarray(ratios[u], dtype=np.float32)
            f_dispersion[u] = np.sqrt(np.mean((vals - f_mismatch[u]) ** 2))

    x = np.stack([f_match, f_mismatch, f_dispersion], axis=1).astype(np.float32)
    raw_syndrome = np.concatenate([sample.s_uv, sample.s_vu], axis=1).astype(np.float32)
    edge_attr = np.concatenate([raw_syndrome, r_uv[:, None]], axis=1).astype(np.float32)
    return x, edge_attr, r_uv
