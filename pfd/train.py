from __future__ import annotations

import numpy as np

from pfd.features import build_features


def train_single_sample(sample, epochs: int = 200, lr: float = 1e-3, hidden_dim: int = 64):
    """Small training helper for debugging the proposed model on one sample."""
    import torch
    from torch import nn

    from pfd.model import GNNPosteriorEstimator, make_bidirectional_edges

    x, edge_attr, _ = build_features(sample)
    edge_index, edge_attr_bi = make_bidirectional_edges(sample.edges, edge_attr)
    y = torch.as_tensor(sample.labels.astype(np.float32))
    model = GNNPosteriorEstimator(x.shape[1], hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    x_t = torch.as_tensor(x, dtype=torch.float32)

    for _ in range(epochs):
        optimizer.zero_grad()
        logits, _ = model(x_t, edge_index, edge_attr_bi)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits, aux = model(x_t, edge_index, edge_attr_bi)
        probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
    return model, probs, aux
