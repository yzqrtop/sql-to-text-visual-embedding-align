"""Joint optimization (paper Section: Joint Optimization Objective).

Exposes the BCE + triplet contrastive losses and the ``AlignmentTrainer``
training loop.
"""

from .trainer import AlignmentDataset, AlignmentTrainer, bce_loss, contrastive_loss

__all__ = ["AlignmentDataset", "AlignmentTrainer", "bce_loss", "contrastive_loss"]
