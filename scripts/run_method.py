#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn

from pfd.dataset import load_dataset
from pfd.decision import partition_predictions, refine_ambiguous_posteriors
from pfd.metrics import (
    binary_metrics,
    brier_score,
    expected_calibration_error,
    topk_localization,
)
from pfd.features import build_features
from pfd.model import ReliabilityAwareGNN, make_bidirectional_edges


def choose_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_samples(samples, train_ratio: float, seed: int):
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_train = max(1, int(round(len(samples) * train_ratio)))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:] or idx[:1]
    return [samples[i] for i in train_idx], [samples[i] for i in test_idx]


def sample_to_tensors(sample, device: torch.device):
    x, edge_attr, _ = build_features(sample)
    edge_index, edge_attr_bi = make_bidirectional_edges(sample.edges, edge_attr)
    return (
        torch.as_tensor(x, dtype=torch.float32, device=device),
        edge_index.to(device),
        edge_attr_bi.to(device),
        torch.as_tensor(sample.labels.astype(np.float32), dtype=torch.float32, device=device),
    )


def train_model(args, train_samples, device: torch.device):
    x0, _, edge_attr0, _ = sample_to_tensors(train_samples[0], device)
    model = ReliabilityAwareGNN(
        node_dim=x0.shape[1],
        edge_dim=edge_attr0.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_samples)
        losses = []
        for sample in train_samples:
            x, edge_index, edge_attr, y = sample_to_tensors(sample, device)
            optimizer.zero_grad()
            logits, _ = model(x, edge_index, edge_attr)
            bce = loss_fn(logits, y)
            probs = torch.sigmoid(logits)
            brier = torch.mean((probs - y) ** 2)
            loss = bce + args.brier_weight * brier
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch {epoch:04d} loss={np.mean(losses):.4f}")
    return model


@torch.no_grad()
def evaluate_model(model, samples, device: torch.device, args):
    model.eval()
    raw_rows = []
    refined_rows = []
    ambiguous_rows = []
    changed_rates = []
    corrected_rates = []
    wrong_flip_rates = []
    for sample in samples:
        x, edge_index, edge_attr, _ = sample_to_tensors(sample, device)
        logits, aux = model(x, edge_index, edge_attr)
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        _, _, va, raw_pred = partition_predictions(probs, args.theta_low, args.theta_high)
        raw_pred[va] = (probs[va] >= args.theta_c).astype(np.int8)
        refined, pred, candidates = refine_ambiguous_posteriors(
            sample,
            probs,
            args.theta_low,
            args.theta_high,
            args.theta_c,
            args.refine_top_pct,
            return_candidates=True,
        )
        raw = binary_metrics(sample.labels, raw_pred)
        raw["brier"] = brier_score(sample.labels, probs)
        raw["ece"] = expected_calibration_error(sample.labels, probs)
        raw["topk"] = topk_localization(sample.labels, probs)
        raw_rows.append(raw)

        row = binary_metrics(sample.labels, pred)
        row["brier"] = brier_score(sample.labels, refined)
        row["ece"] = expected_calibration_error(sample.labels, refined)
        row["topk"] = topk_localization(sample.labels, refined)
        row["avg_rho"] = float(aux["rho"].detach().cpu().mean())
        row["ambiguous_ratio"] = float(np.mean(va))
        row["refinement_candidate_ratio"] = float(np.mean(candidates))
        refined_rows.append(row)

        if np.any(candidates):
            amb_pred = pred[candidates]
            ambiguous_rows.append(binary_metrics(sample.labels[candidates], amb_pred))
            changed = raw_pred[candidates] != pred[candidates]
            changed_rates.append(float(np.mean(changed)))
            if np.any(changed):
                before_correct = raw_pred[candidates][changed] == sample.labels[candidates][changed]
                after_correct = pred[candidates][changed] == sample.labels[candidates][changed]
                corrected_rates.append(float(np.mean((~before_correct) & after_correct)))
                wrong_flip_rates.append(float(np.mean(before_correct & (~after_correct))))
            else:
                corrected_rates.append(0.0)
                wrong_flip_rates.append(0.0)
        else:
            changed_rates.append(0.0)
            corrected_rates.append(0.0)
            wrong_flip_rates.append(0.0)

    raw_avg = {f"raw_{k}": float(np.mean([row[k] for row in raw_rows])) for k in raw_rows[0]}
    refined_avg = {
        f"refined_{k}": float(np.mean([row[k] for row in refined_rows])) for k in refined_rows[0]
    }
    avg = {**raw_avg, **refined_avg}
    for key in raw_rows[0]:
        avg[f"delta_{key}"] = refined_avg[f"refined_{key}"] - raw_avg[f"raw_{key}"]
    if ambiguous_rows:
        for key in ambiguous_rows[0]:
            avg[f"candidate_{key}"] = float(np.mean([row[key] for row in ambiguous_rows]))
    avg["refinement_change_rate"] = float(np.mean(changed_rates))
    avg["refinement_corrected_rate"] = float(np.mean(corrected_rates))
    avg["refinement_wrong_flip_rate"] = float(np.mean(wrong_flip_rates))
    return avg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the proposed PFD method only.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--test-dataset", type=Path, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
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
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    print(f"device: {device}")

    samples = load_dataset(args.dataset)
    if args.test_dataset is None:
        train_samples, test_samples = split_samples(samples, args.train_ratio, args.seed)
    else:
        train_samples = samples
        test_samples = load_dataset(args.test_dataset)
    print(f"samples: train={len(train_samples)} test={len(test_samples)}")
    model = train_model(args, train_samples, device)
    metrics = evaluate_model(model, test_samples, device, args)
    print("test metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
