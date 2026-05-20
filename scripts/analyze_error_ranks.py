#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from pfd.dataset import load_dataset
from scripts.run_method import choose_device, sample_to_tensors, train_model


@torch.no_grad()
def predict(model, sample, device):
    model.eval()
    x, edge_index, edge_attr, _ = sample_to_tensors(sample, device)
    logits, _ = model(x, edge_index, edge_attr)
    return torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)


def percentile_rank(sorted_scores: np.ndarray, value: float) -> float:
    if len(sorted_scores) <= 1:
        return 1.0
    pos = int(np.searchsorted(sorted_scores, value, side="right"))
    return pos / len(sorted_scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze confidence ranks of raw classification errors.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--test-dataset", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--brier-weight", type=float, default=0.1)
    parser.add_argument("--theta-low", type=float, default=0.4)
    parser.add_argument("--theta-high", type=float, default=0.6)
    parser.add_argument("--theta-c", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--output", type=Path, default=Path("runs/error_rank_analysis.csv"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    train_samples = load_dataset(args.dataset)
    test_samples = load_dataset(args.test_dataset)
    print(f"device: {device}")
    print(f"train={len(train_samples)} test={len(test_samples)}")
    model = train_model(args, train_samples, device)

    rows = []
    coverage = {1: 0, 2: 0, 3: 0, 5: 0, 10: 0, 20: 0}
    total_errors = 0
    for sample_id, sample in enumerate(test_samples):
        probs = predict(model, sample, device)
        pred = (probs >= 0.5).astype(np.int8)
        conf = np.where(pred == 1, probs, 1.0 - probs)
        errors = np.where(pred != sample.labels)[0]
        total_errors += len(errors)

        normal_nodes = np.where(pred == 0)[0]
        faulty_nodes = np.where(pred == 1)[0]
        normal_conf_asc = normal_nodes[np.argsort(conf[normal_nodes])]
        faulty_conf_asc = faulty_nodes[np.argsort(conf[faulty_nodes])]
        all_conf_asc = np.argsort(conf)

        top_sets = {k: set(all_conf_asc[: max(1, int(round(len(all_conf_asc) * k / 100)))]) for k in coverage}
        for u in errors:
            for k, selected in top_sets.items():
                coverage[k] += int(int(u) in selected)

            group = "pred_faulty" if pred[u] == 1 else "pred_fault_free"
            group_nodes = faulty_nodes if pred[u] == 1 else normal_nodes
            group_sorted = faulty_conf_asc if pred[u] == 1 else normal_conf_asc
            group_rank = int(np.where(group_sorted == u)[0][0]) + 1
            all_rank = int(np.where(all_conf_asc == u)[0][0]) + 1
            rows.append(
                {
                    "sample": sample_id,
                    "node": int(u),
                    "true_label": int(sample.labels[u]),
                    "raw_p": float(probs[u]),
                    "raw_pred": int(pred[u]),
                    "pred_group": group,
                    "confidence": float(conf[u]),
                    "group_rank_least_confident": group_rank,
                    "group_size": int(len(group_nodes)),
                    "group_percentile_least_confident": group_rank / max(len(group_nodes), 1),
                    "overall_rank_least_confident": all_rank,
                    "num_nodes": int(sample.num_nodes),
                    "overall_percentile_least_confident": all_rank / sample.num_nodes,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    print(f"total_errors: {total_errors}")
    for k, hit in coverage.items():
        rate = hit / max(total_errors, 1)
        print(f"covered_by_overall_least_confident_top_{k}%: {hit}/{total_errors} = {rate:.4f}")
    if rows:
        group_pct = np.asarray([r["group_percentile_least_confident"] for r in rows], dtype=float)
        overall_pct = np.asarray([r["overall_percentile_least_confident"] for r in rows], dtype=float)
        conf_vals = np.asarray([r["confidence"] for r in rows], dtype=float)
        print(f"error_confidence_mean: {conf_vals.mean():.4f}")
        print(f"error_confidence_min/max: {conf_vals.min():.4f}/{conf_vals.max():.4f}")
        print(f"group_percentile_mean: {group_pct.mean():.4f}")
        print(f"group_percentile_median: {np.median(group_pct):.4f}")
        print(f"overall_percentile_mean: {overall_pct.mean():.4f}")
        print(f"overall_percentile_median: {np.median(overall_pct):.4f}")
    print(f"saved error rank table to {args.output}")


if __name__ == "__main__":
    main()

