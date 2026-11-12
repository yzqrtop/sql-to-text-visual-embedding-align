"""Semantic gap modeling module for SQL-to-diagram alignment.

This package implements Module 2 (Phi) of the framework: a cross-modal
alignment scorer that predicts element-level alignment probabilities between
a natural-language question and the JSON graph built from a generated SQL.

Subpackages:
    - models.encoders: text encoder (Qwen2.5-Coder-7B) producing H_q and the
      Relational Graph Attention Network producing H_u.
    - models.aligner: cross-modal attention + scoring head producing P_align(u).
    - data.graph_alignment: GED-based graph alignment for training triples.
    - data.graph_features: JSON graph tensorization and score mapping.
    - training.trainer: BCE + triplet contrastive training loop.
"""

__version__ = "0.1.0"
