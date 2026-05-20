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
from pfd.features import build_features
from pfd.metrics import binary_metrics, brier_score, expected_calibration_error
from pfd.model import VanillaGNNClassifier, make_bidirectional_edges


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
    edge_index, _ = make_bidirectional_edges(sample.edges, edge_attr)
    return (
        torch.as_tensor(x, dtype=torch.float32, device=device),
        edge_index.to(device),
        torch.as_tensor(sample.labels.astype(np.float32), dtype=torch.float32, device=device),
    )


def train_model(args, train_samples, device: torch.device):
    x0, _, _ = sample_to_tensors(train_samples[0], device)
    model = VanillaGNNClassifier(
        node_dim=x0.shape[1],
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
            x, edge_index, y = sample_to_tensors(sample, device)
            optimizer.zero_grad()
            logits = model(x, edge_index)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch {epoch:04d} loss={np.mean(losses):.4f}")
    return model


@torch.no_grad()
def evaluate_model(model, samples, device: torch.device, threshold: float):
    model.eval()
    rows = []
    for sample in samples:
        x, edge_index, _ = sample_to_tensors(sample, device)
        logits = model(x, edge_index)
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        pred = (probs >= threshold).astype(np.int8)
        row = binary_metrics(sample.labels, pred)
        row["brier"] = brier_score(sample.labels, probs)
        row["ece"] = expected_calibration_error(sample.labels, probs)
        rows.append(row)
    return {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a vanilla GNN classifier baseline.")
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
    parser.add_argument("--threshold", type=float, default=0.5)
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
    metrics = evaluate_model(model, test_samples, device, args.threshold)
    print("test metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
