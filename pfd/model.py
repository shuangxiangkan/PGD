from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReliabilityAwareLayer(nn.Module):
    """Edge-aware message passing with a separate syndrome-reliability gate."""

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.0):
        super().__init__()
        self.message = MLP(hidden_dim * 2 + edge_dim, hidden_dim, hidden_dim, dropout)
        self.attn = nn.Linear(hidden_dim * 2 + edge_dim, 1)
        self.reliability = nn.Linear(edge_dim, 1)
        self.update = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        pair = torch.cat([h[dst], h[src], edge_attr], dim=-1)
        msg = self.message(pair)
        attn_logits = torch.nn.functional.leaky_relu(self.attn(pair).squeeze(-1), negative_slope=0.2)
        rho = torch.sigmoid(self.reliability(edge_attr)).squeeze(-1)

        alpha = torch.zeros_like(attn_logits)
        for u in torch.unique(dst):
            mask = dst == u
            alpha[mask] = torch.softmax(attn_logits[mask], dim=0)

        weighted = msg * (alpha * rho).unsqueeze(-1)
        out = torch.zeros_like(h)
        out.index_add_(0, dst, weighted)
        out = torch.relu(self.update(out) + h)
        return self.dropout(out), rho


class ReliabilityAwareGNN(nn.Module):
    """Backbone-agnostic reliability-aware probabilistic node classifier."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_encoder = MLP(node_dim, hidden_dim, hidden_dim, dropout)
        self.edge_encoder = MLP(edge_dim, hidden_dim, hidden_dim, dropout)
        self.layers = nn.ModuleList(
            ReliabilityAwareLayer(hidden_dim, hidden_dim, dropout) for _ in range(num_layers)
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        rho_values = []
        for layer in self.layers:
            h, rho = layer(h, edge_index, e)
            rho_values.append(rho)
        logits = self.head(h).squeeze(-1)
        return logits, {"rho": torch.stack(rho_values)}


class VanillaGNNLayer(nn.Module):
    """Plain mean-aggregation GNN layer without edge features or reliability gates."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.update = MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, h[src])
        deg = torch.zeros(h.shape[0], dtype=h.dtype, device=h.device)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
        agg = agg / deg.clamp_min(1.0).unsqueeze(-1)
        out = torch.relu(self.update(torch.cat([h, agg], dim=-1)) + h)
        return self.dropout(out)


class VanillaGNNClassifier(nn.Module):
    """Standard node-level GNN classifier with direct binary supervision."""

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_encoder = MLP(node_dim, hidden_dim, hidden_dim, dropout)
        self.layers = nn.ModuleList(VanillaGNNLayer(hidden_dim, dropout) for _ in range(num_layers))
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.node_encoder(x)
        for layer in self.layers:
            h = layer(h, edge_index)
        return self.head(h).squeeze(-1)


def make_bidirectional_edges(edges, edge_attr):
    """Duplicate undirected edge features for both message-passing directions."""
    src = torch.as_tensor(edges[:, 0], dtype=torch.long)
    dst = torch.as_tensor(edges[:, 1], dtype=torch.long)
    edge_index = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)
    edge_attr_t = torch.as_tensor(edge_attr, dtype=torch.float32)
    edge_attr_bi = torch.cat([edge_attr_t, edge_attr_t], dim=0)
    return edge_index, edge_attr_bi
