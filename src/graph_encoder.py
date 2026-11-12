"""Graph encoder branch of the semantic gap model (Module 2).

A Relational Graph Attention Network (R-GAT) that operates on the JSON
graph produced by Module 1 (``sql_to_ast_config.parse_sql`` -> ConfigBuilder).

The JSON graph has three element kinds that all become graph nodes for the
GNN:
    * data-source nodes (squares) -> "node" elements,
    * operation edges (arrows)    -> "edge" elements,
    * predicate labels             -> attribute features attached to edges.

Each graph element u receives a feature vector H_u in R^{1 x d} after
message passing, which is then used as Key/Value in cross-modal attention.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


class RelationalGATLayer(nn.Module):
    """One layer of relational graph attention.

    For each relation type r, the layer computes attention coefficients
    a_{ij}^r = LeakyReLU(a^T [W_r h_i || W_r h_j]) and aggregates
    h_i' = sigma(sum_{j in N(i)} sum_r alpha_{ij}^r W_r h_j).

    Relation types follow the JSON graph vocabulary: "from", "to",
    "join_on", "group_by", "where", "select", etc.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int,
        num_heads: int = config.GRAPH_NUM_HEADS,
        dropout: float = config.GRAPH_DROPOUT,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.W = nn.ParameterDict({
            str(r): nn.Linear(in_dim, out_dim, bias=False)
            for r in range(num_relations)
        })
        self.a = nn.ParameterDict({
            str(r): nn.Parameter(torch.empty(1, num_heads, 2 * self.head_dim))
            for r in range(num_relations)
        })
        for r in range(num_relations):
            nn.init.xavier_uniform_(self.a[str(r)])
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """h: (N, in_dim); edge_index: (2, E); edge_type: (E,)."""
        N = h.size(0)
        out = torch.zeros(N, self.num_heads * self.head_dim, device=h.device)
        for r in range(len(self.W)):
            mask = edge_type == r
            if not mask.any():
                continue
            e_idx = edge_index[:, mask]
            h_r = self.W[str(r)](h).view(N, self.num_heads, self.head_dim)
            src, dst = e_idx[0], e_idx[1]
            # attention: a^T [h_src || h_dst]
            pair = torch.cat([h_r[src], h_r[dst]], dim=-1)  # (E_r, heads, 2*hd)
            attn = (self.a[str(r)] * pair).sum(dim=-1)  # (E_r, heads)
            attn = self.leaky_relu(attn)
            # softmax over incoming edges per destination node
            attn = attn - attn.max(dim=0, keepdim=True).values
            attn = torch.exp(attn)
            denom = torch.zeros(N, self.num_heads, device=h.device)
            denom.index_add_(0, dst, attn)
            alpha = attn / (denom[dst] + 1e-12)
            alpha = self.dropout(alpha)
            msg = h_r[src] * alpha.unsqueeze(-1)
            out.index_add_(0, dst, msg.view(-1, self.num_heads * self.head_dim))
        return F.elu(out)


class GraphEncoder(nn.Module):
    """Encode a JSON graph into per-element hidden states via R-GAT.

    Args:
        num_relations: number of distinct relation types in the graph.
    """

    def __init__(self, num_relations: int, in_dim: int = config.NODE_FEATURE_DIM):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, config.GRAPH_HIDDEN_DIM)
        self.layers = nn.ModuleList([
            RelationalGATLayer(
                config.GRAPH_HIDDEN_DIM,
                config.GRAPH_HIDDEN_DIM,
                num_relations,
            )
            for _ in range(config.GRAPH_NUM_LAYERS)
        ])

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-element hidden states (N, d)."""
        h = self.input_proj(node_features)
        for layer in self.layers:
            h = layer(h, edge_index, edge_type)
        return h
