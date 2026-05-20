from __future__ import annotations

import numpy as np


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = float(np.logical_and(y_true, y_pred).sum())
    tn = float(np.logical_and(~y_true, ~y_pred).sum())
    fp = float(np.logical_and(~y_true, y_pred).sum())
    fn = float(np.logical_and(y_true, ~y_pred).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean((probs.astype(float) - y_true.astype(float)) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    num_bins: int = 10,
) -> float:
    y_true = y_true.astype(float)
    probs = probs.astype(float)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (probs >= lo) & (probs < hi if hi < 1.0 else probs <= hi)
        if not np.any(mask):
            continue
        confidence = float(np.mean(probs[mask]))
        accuracy = float(np.mean(y_true[mask]))
        ece += float(np.mean(mask)) * abs(confidence - accuracy)
    return ece


def topk_localization(y_true: np.ndarray, probs: np.ndarray, k: int | None = None) -> float:
    y_true = y_true.astype(int)
    num_faults = int(y_true.sum())
    if num_faults == 0:
        return 1.0
    k = num_faults if k is None else min(k, len(y_true))
    order = np.argsort(-probs)[:k]
    return float(y_true[order].sum() / num_faults)
