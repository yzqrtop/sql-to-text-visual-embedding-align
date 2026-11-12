"""Data utilities for triplet construction and graph tensorization.

- ``graph_alignment``: GED-based alignment of (q, s_gt, s_bug) graph pairs.
- ``graph_features``: JSON graph -> tensor features and score mapping.
"""

from .graph_features import RELATION_TYPES, apply_scores_to_graph, build_graph_tensors

__all__ = ["RELATION_TYPES", "apply_scores_to_graph", "build_graph_tensors"]
