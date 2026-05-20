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
from pfd.decision import partition_predictions, refine_ambiguous_posteriors
from pfd.features import build_features
from pfd.model import ReliabilityAwareGNN, make_bidirectional_edges
from scripts.run_method import choose_device, sample_to_tensors, train_model


@torch.no_grad()
def predict_sample(model, sample, device):
    model.eval()
    x, edge_index, edge_attr, _ = sample_to_tensors(sample, device)
    logits, _ = model(x, edge_index, edge_attr)
    return torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)


def bucket_counts(values: np.ndarray) -> dict[str, int]:
    return {
        "[0,0.1)": int(np.sum((values >= 0.0) & (values < 0.1))),
        "[0.1,0.4)": int(np.sum((values >= 0.1) & (values < 0.4))),
        "[0.4,0.6)": int(np.sum((values >= 0.4) & (values < 0.6))),
        "[0.6,0.9)": int(np.sum((values >= 0.6) & (values < 0.9))),
        "[0.9,1]": int(np.sum((values >= 0.9) & (values <= 1.0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect node-level confidence distributions.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--test-dataset", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--brier-weight", type=float, default=0.1)
    parser.add_argument("--theta-low", type=float, default=0.4)
    parser.add_argument("--theta-high", type=float, default=0.6)
    parser.add_argument("--theta-c", type=float, default=0.5)
    parser.add_argument("--refine-top-pct", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("runs/confidence_10_samples.csv"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    train_samples = load_dataset(args.dataset)
    test_samples = load_dataset(args.test_dataset)[: args.num_samples]

    print(f"device: {device}")
    print(f"train={len(train_samples)} inspect={len(test_samples)}")
    model = train_model(args, train_samples, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample",
                "node",
                "true_label",
                "raw_p",
                "refined_p",
                "raw_pred",
                "final_pred",
                "region",
            ]
        )
        for sample_id, sample in enumerate(test_samples):
            probs = predict_sample(model, sample, device)
            refined, final_pred, candidates = refine_ambiguous_posteriors(
                sample,
                probs,
                args.theta_low,
                args.theta_high,
                args.theta_c,
                args.refine_top_pct,
                return_candidates=True,
            )
            v0, v1, va, raw_pred = partition_predictions(probs, args.theta_low, args.theta_high)
            raw_pred[va] = (probs[va] >= args.theta_c).astype(np.int8)
            counts = bucket_counts(probs)
            print(
                f"sample {sample_id:02d}: "
                f"min={probs.min():.4f} max={probs.max():.4f} "
                f"mean={probs.mean():.4f} ambiguous={int(va.sum())}/{len(probs)} "
                f"refine_candidates={int(candidates.sum())}/{len(probs)} "
                f"buckets={counts}"
            )
            for node_id in range(sample.num_nodes):
                region = "Vr" if candidates[node_id] else ("Va" if va[node_id] else ("V0" if v0[node_id] else "V1"))
                writer.writerow(
                    [
                        sample_id,
                        node_id,
                        int(sample.labels[node_id]),
                        f"{float(probs[node_id]):.6f}",
                        f"{float(refined[node_id]):.6f}",
                        int(raw_pred[node_id]),
                        int(final_pred[node_id]),
                        region,
                    ]
                )
    print(f"saved node-level confidence scores to {args.output}")


if __name__ == "__main__":
    main()
