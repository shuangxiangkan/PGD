from __future__ import annotations

import numpy as np

from pfd.dataset import PMCSample


def majority_mismatch_scores(sample: PMCSample) -> np.ndarray:
    """Local baseline: score each node by its average incident mismatch ratio."""
    n = sample.num_nodes
    rounds = sample.s_uv.shape[1]
    scores = np.zeros(n, dtype=np.float32)
    degree = np.zeros(n, dtype=np.float32)
    mismatch = sample.s_uv.sum(axis=1) + sample.s_vu.sum(axis=1)
    ratio = mismatch.astype(np.float32) / float(2 * rounds)
    for edge_id, (u, v) in enumerate(sample.edges):
        u = int(u)
        v = int(v)
        scores[u] += ratio[edge_id]
        scores[v] += ratio[edge_id]
        degree[u] += 1
        degree[v] += 1
    return scores / np.maximum(degree, 1.0)


def last_round_directed_syndrome(sample: PMCSample) -> list[set[int]]:
    """Use the last-round directed syndrome as outgoing match-neighbor sets.

    The clustered-fault diagnosis algorithm in Sun et al. assumes one PMC
    syndrome value per directed test. When a reusable dataset contains multiple
    rounds, the baseline is evaluated on the final collected syndrome.
    """
    n = sample.num_nodes
    outgoing_matches = [set() for _ in range(n)]
    for edge_id, (u, v) in enumerate(sample.edges):
        u = int(u)
        v = int(v)
        if int(sample.s_uv[edge_id, -1]) == 0:
            outgoing_matches[u].add(v)
        if int(sample.s_vu[edge_id, -1]) == 0:
            outgoing_matches[v].add(u)
    return outgoing_matches


def clustered_pmc_labels(sample: PMCSample, threshold: int = 3) -> np.ndarray:
    """Clustered-fault probabilistic PMC baseline with threshold k=3.

    This implements the local faction rule from Sun et al., "Probabilistic
    Fault Diagnosis of Clustered Faults for Multiprocessor Systems". A node is
    diagnosed as fault-free when its outgoing match neighborhood gives evidence
    that it belongs to a faction larger than the threshold; otherwise it is
    diagnosed as faulty.
    """
    if threshold != 3:
        raise ValueError("clustered_pmc_labels currently implements the k=3 rule")

    matches = last_round_directed_syndrome(sample)
    pred = np.ones(sample.num_nodes, dtype=np.int8)
    for a in range(sample.num_nodes):
        gamma_a = matches[a]
        fault_free = False
        if len(gamma_a) >= 3:
            fault_free = True
        elif len(gamma_a) == 2:
            fault_free = any(len(matches[a_i] - {a}) >= 1 for a_i in gamma_a)
        elif len(gamma_a) == 1:
            a_1 = next(iter(gamma_a))
            next_matches = matches[a_1] - {a}
            if len(next_matches) >= 2:
                fault_free = True
            elif len(next_matches) == 1:
                b_1 = next(iter(next_matches))
                fault_free = len(matches[b_1] - {a_1}) >= 1
        pred[a] = 0 if fault_free else 1
    return pred
