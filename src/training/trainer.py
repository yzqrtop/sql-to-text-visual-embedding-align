"""Training loop for the alignment scorer (Module 2).

Implements the joint objective from the paper:

    L_train = L_BCE + lambda * L_contrast

where L_BCE is element-level binary cross-entropy over P_align(u) and
L_contrast is a triplet margin loss pulling correct graph elements
toward the question representation and pushing bug elements away.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .. import config
from ..models.aligner import AlignmentScorer
from ..data.graph_features import build_graph_tensors


class AlignmentDataset(Dataset):
    """Dataset of (question, json_graph, element_labels) triples.

    element_labels maps element_id -> {0 (bug), 1 (correct)}.
    """

    def __init__(self, samples: List[Tuple[str, Dict, Dict[str, int]]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def bce_loss(
    p_align: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Element-level binary cross-entropy loss."""
    return F.binary_cross_entropy(p_align, labels.float())


def contrastive_loss(
    c_q: torch.Tensor,
    c_pos: torch.Tensor,
    c_neg: torch.Tensor,
    margin: float = config.CONTRASTIVE_MARGIN,
) -> torch.Tensor:
    """Triplet contrastive loss with cosine similarity.

    sim(c_q, c_pos) should be larger than sim(c_q, c_neg) by at least
    ``margin``.
    """
    sim_pos = F.cosine_similarity(c_q, c_pos, dim=-1)
    sim_neg = F.cosine_similarity(c_q, c_neg, dim=-1)
    return torch.clamp(margin - sim_pos + sim_neg, min=0.0).mean()


class AlignmentTrainer:
    """Trainer wrapping BCE + triplet contrastive optimization."""

    def __init__(
        self,
        model: AlignmentScorer,
        device: str = config.DEVICE,
        lr: float = config.LEARNING_RATE,
        weight_decay: float = config.WEIGHT_DECAY,
        loss_lambda: float = config.LOSS_LAMBDA,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )
        self.loss_lambda = loss_lambda

    def _build_labels(
        self,
        element_ids: List[str],
        label_map: Dict[str, int],
    ) -> torch.Tensor:
        labels = [float(label_map.get(eid, 1)) for eid in element_ids]
        return torch.tensor(labels, device=self.device)

    def train_epoch(self, samples: List[Tuple[str, Dict, Dict[str, int]]]) -> float:
        """Run one training epoch; return average loss."""
        self.model.train()
        total_loss = 0.0
        for question, json_graph, label_map in samples:
            node_feats, edge_index, edge_type, element_ids = build_graph_tensors(json_graph)
            node_feats = node_feats.to(self.device)
            edge_index = edge_index.to(self.device)
            edge_type = edge_type.to(self.device)
            labels = self._build_labels(element_ids, label_map)

            self.optimizer.zero_grad()
            p_align = self.model(question, node_feats, edge_index, edge_type)

            # BCE over all elements.
            loss = bce_loss(p_align, labels)

            # Contrastive: separate positive (y=1) and negative (y=0) elements.
            pos_mask = labels >= 0.5
            neg_mask = ~pos_mask
            if pos_mask.any() and neg_mask.any():
                with torch.no_grad():
                    h_q = self.model.text_encoder(question).mean(dim=1)
                h_u = self.model.encode_graph(node_feats, edge_index, edge_type).squeeze(0)
                c_pos = h_u[pos_mask].mean(dim=0, keepdim=True)
                c_neg = h_u[neg_mask].mean(dim=0, keepdim=True)
                loss = loss + self.loss_lambda * contrastive_loss(h_q, c_pos, c_neg)

            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                config.GRAD_CLIP,
            )
            self.optimizer.step()
            total_loss += float(loss.item())
        return total_loss / max(len(samples), 1)

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
