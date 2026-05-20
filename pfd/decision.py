from __future__ import annotations

import numpy as np

from pfd.dataset import PMCSample


def partition_predictions(
    probs: np.ndarray,
    theta_low: float,
    theta_high: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    v0 = probs <= theta_low
    v1 = probs >= theta_high
    va = ~(v0 | v1)
    labels = np.full(probs.shape, -1, dtype=np.int8)
    labels[v0] = 0
    labels[v1] = 1
    return v0, v1, va, labels


def select_refinement_candidates(
    probs: np.ndarray,
    theta_low: float,
    theta_high: float,
    refine_top_pct: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select posterior-refinement candidates.

    Candidates include interval-ambiguous nodes and the class-wise least-confident
    top-q fraction among nodes directly predicted as fault-free or faulty.
    """
    v0, v1, va, labels = partition_predictions(probs, theta_low, theta_high)
    candidates = va.copy()
    if refine_top_pct > 0:
        for mask in (v0, v1):
            nodes = np.where(mask)[0]
            if len(nodes) == 0:
                continue
            confidence = np.maximum(probs[nodes], 1.0 - probs[nodes])
            k = max(1, int(np.ceil(len(nodes) * refine_top_pct)))
            selected = nodes[np.argsort(confidence)[:k]]
            candidates[selected] = True
    return v0, v1, va, labels, candidates


def _pmc_prob(s: int, zu: int, zv: int, gamma: float, beta: float, eps: float = 1e-12) -> float:
    if zu == 0 and zv == 0:
        p = 1.0 if s == 0 else 0.0
    elif zu != zv:
        p = gamma if s == 1 else 1.0 - gamma
    else:
        p = beta if s == 0 else 1.0 - beta
    return max(float(p), eps)


def refine_ambiguous_posteriors(
    sample: PMCSample,
    probs: np.ndarray,
    theta_low: float,
    theta_high: float,
    theta_c: float,
    refine_top_pct: float = 0.1,
    return_candidates: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refine selected nodes using high-confidence-neighbor syndrome likelihoods."""
    v0, v1, va, labels, candidates = select_refinement_candidates(
        probs, theta_low, theta_high, refine_top_pct
    )
    refined = probs.astype(np.float64).copy()
    incident: list[list[tuple[int, bool]]] = [[] for _ in range(sample.num_nodes)]
    for edge_id, (u, v) in enumerate(sample.edges):
        incident[int(u)].append((edge_id, True))
        incident[int(v)].append((edge_id, False))

    reference_mask = (v0 | v1) & (~candidates)
    for u in np.where(candidates)[0]:
        log_l = np.zeros(2, dtype=np.float64)
        has_reference = False
        for edge_id, u_is_first in incident[int(u)]:
            a, b = map(int, sample.edges[edge_id])
            v = b if u_is_first else a
            if not reference_mask[v]:
                continue
            has_reference = True
            zv = int(labels[v])
            for h in (0, 1):
                for suv, svu in zip(sample.s_uv[edge_id], sample.s_vu[edge_id], strict=True):
                    if u_is_first:
                        log_l[h] += np.log(_pmc_prob(int(suv), h, zv, sample.gamma, sample.beta))
                        log_l[h] += np.log(_pmc_prob(int(svu), zv, h, sample.gamma, sample.beta))
                    else:
                        log_l[h] += np.log(_pmc_prob(int(suv), zv, h, sample.gamma, sample.beta))
                        log_l[h] += np.log(_pmc_prob(int(svu), h, zv, sample.gamma, sample.beta))
        if not has_reference:
            continue
        prior1 = np.log(max(float(probs[u]), 1e-12))
        prior0 = np.log(max(float(1.0 - probs[u]), 1e-12))
        score1 = prior1 + log_l[1]
        score0 = prior0 + log_l[0]
        m = max(score0, score1)
        p1 = np.exp(score1 - m)
        p0 = np.exp(score0 - m)
        refined[u] = p1 / (p0 + p1)

    final = labels.copy()
    final[va] = (probs[va] >= theta_c).astype(np.int8)
    final[candidates] = (refined[candidates] >= theta_c).astype(np.int8)
    if return_candidates:
        return refined.astype(np.float32), final, candidates
    return refined.astype(np.float32), final
