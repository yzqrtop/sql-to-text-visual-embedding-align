"""Cross-modal alignment scorer (Module 2 core).

Combines the text encoder (H_q) and graph encoder (H_u) through
multi-head cross-modal attention, mean-pools the fused sequence, and
passes the result through an MLP scoring head to produce P_align(u),
the probability that graph element u is aligned with the user's intent.

    C_u' = MultiHeadAttention(H_q, H_u, H_u)
    C_u  = MeanPooling(C_u')  in R^{1 x d}
    P_align(u) = sigmoid(MLP(C_u))
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from . import config
from .graph_encoder import GraphEncoder
from .text_encoder import TextEncoder


class CrossModalAttention(nn.Module):
    """Multi-head attention where text is Query and graph is Key/Value."""

    def __init__(self, dim: int = config.HIDDEN_DIM, num_heads: int = config.ATTENTION_NUM_HEADS):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h_q: torch.Tensor, h_u: torch.Tensor) -> torch.Tensor:
        """h_q: (1, L_q, d); h_u: (1, L_u, d) -> (1, L_q, d)."""
        out, _ = self.attn(h_q, h_u, h_u)
        return self.norm(out + h_q)


class AlignmentScorer(nn.Module):
    """End-to-end element-level alignment model.

    Args:
        num_relations: number of relation types in the JSON graph.
        node_feature_dim: dimensionality of the raw node/edge feature vectors.
    """

    def __init__(
        self,
        num_relations: int,
        node_feature_dim: int = config.NODE_FEATURE_DIM,
        text_model_name: Optional[str] = None,
    ):
        super().__init__()
        self.text_encoder = TextEncoder(model_name=text_model_name, freeze=True)
        # Project graph hidden states to the text hidden dimension if needed.
        graph_dim = config.GRAPH_HIDDEN_DIM
        text_dim = self.text_encoder.output_dim
        self.graph_proj = (
            nn.Linear(graph_dim, text_dim) if graph_dim != text_dim else nn.Identity()
        )
        self.graph_encoder = GraphEncoder(num_relations, in_dim=node_feature_dim)
        self.cross_attention = CrossModalAttention(dim=text_dim)
        self.score_head = nn.Sequential(
            nn.Linear(text_dim, config.MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.MLP_HIDDEN_DIM, 1),
        )

    def encode_graph(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """Return projected graph element hidden states (1, L_u, d)."""
        h_u = self.graph_encoder(node_features, edge_index, edge_type)  # (N, d)
        h_u = self.graph_proj(h_u)
        return h_u.unsqueeze(0)

    def forward(
        self,
        question: str,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """Return P_align(u) for every graph element u.

        Returns:
            scores: tensor of shape (N,) with values in [0, 1].
        """
        h_q = self.text_encoder(question)  # (1, L_q, d)
        h_u = self.encode_graph(node_features, edge_index, edge_type)  # (1, L_u, d)
        c_prime = self.cross_attention(h_q, h_u)  # (1, L_q, d)
        # Mean pool over the sequence dimension.
        c_u = c_prime.mean(dim=1)  # (1, d) -- one vector per graph (broadcast)
        # Per-element scoring: project each element's fused representation.
        # The paper pools C_u' to a single C_u; we use it to score each
        # element by combining C_u with the element's own hidden state.
        h_u_sq = h_u.squeeze(0)  # (N, d)
        fused = h_u_sq + c_u  # broadcast add
        logits = self.score_head(fused).squeeze(-1)  # (N,)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict_scores(
        self,
        question: str,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """Return S_u = 1 - P_align(u), the semantic-gap score in [0, 1]."""
        p_align = self.forward(question, node_features, edge_index, edge_type)
        return 1.0 - p_align
