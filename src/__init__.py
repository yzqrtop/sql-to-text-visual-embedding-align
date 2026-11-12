"""Semantic gap modeling module for SQL-to-diagram alignment.

This package implements Module 2 (Phi) of the framework: a cross-modal
alignment scorer that predicts element-level alignment probabilities between
a natural-language question and the JSON graph built from a generated SQL.

Components:
    - text_encoder: text encoder (Qwen2.5-Coder-7B) producing H_q.
    - graph_encoder: Relational Graph Attention Network over the JSON graph.
    - aligner: cross-modal attention + scoring head producing P_align(u).
    - graph_alignment: GED-based graph alignment for training triples.
    - trainer: BCE + triplet contrastive training loop.
"""

__version__ = "0.1.0"
