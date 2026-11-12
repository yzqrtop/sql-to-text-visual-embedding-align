"""Convert a JSON graph (Module 1 output) into tensor inputs for Module 2.

The JSON graph produced by ``ConfigBuilder`` is turned into:
    * node_features: (N, F) tensor -- one row per graph element
      (data-source nodes, operation edges, and predicate labels are all
      treated as graph elements for the GNN).
    * edge_index: (2, E) tensor -- structural adjacency between elements.
    * edge_type: (E,) tensor -- integer relation type per adjacency.

A fixed vocabulary of relation types and a simple bag-of-words feature
over element labels is used so the graph encoder can be trained without
an external embedding table.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from . import config

# Canonical relation-type vocabulary used across the JSON graph.
RELATION_TYPES: List[str] = [
    "from",
    "to",
    "join_on",
    "where",
    "group_by",
    "having",
    "select",
    "order_by",
    "limit",
    "union",
    "subquery",
    "attribute",
]
_RELATION_TO_ID = {r: i for i, r in enumerate(RELATION_TYPES)}


def _label_bow(label: str, vocab_size: int = 128) -> List[float]:
    """Hashed bag-of-words feature vector for an element label."""
    vec = [0.0] * vocab_size
    if not label:
        return vec
    for token in label.lower().split():
        h = hash(token) % vocab_size
        vec[h] += 1.0
    # L2 normalize
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def build_graph_tensors(json_graph: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Build (node_features, edge_index, edge_type, element_ids).

    element_ids is the ordered list of graph element identifiers so that
    predicted scores can be mapped back to the JSON graph.
    """
    elements: List[Tuple[str, str]] = []  # (element_id, label)
    id_to_idx: Dict[str, int] = {}

    # 1. data-source nodes
    for nid, node in json_graph.get("nodes", {}).items():
        id_to_idx[nid] = len(elements)
        elements.append((nid, node.get("text", nid)))

    # 2. operation edges (become graph elements)
    for idx, edge in enumerate(json_graph.get("edges", [])):
        eid = f"__edge_{idx}"
        id_to_idx[eid] = len(elements)
        elements.append((eid, edge.get("label", "")))

    # Build node features.
    feats = [_label_bow(lbl, config.NODE_FEATURE_DIM) for _, lbl in elements]
    node_features = torch.tensor(feats, dtype=torch.float32)

    # Build structural adjacency (graph edges between elements).
    src_list: List[int] = []
    dst_list: List[int] = []
    type_list: List[int] = []

    for idx, edge in enumerate(json_graph.get("edges", [])):
        eid = f"__edge_{idx}"
        e_idx = id_to_idx[eid]
        from_id = edge.get("from")
        to_id = edge.get("to")
        if from_id in id_to_idx:
            src_list.append(id_to_idx[from_id])
            dst_list.append(e_idx)
            type_list.append(_RELATION_TO_ID.get("from", 0))
            src_list.append(e_idx)
            dst_list.append(id_to_idx[from_id])
            type_list.append(_RELATION_TO_ID.get("from", 0))
        if to_id in id_to_idx:
            src_list.append(e_idx)
            dst_list.append(id_to_idx[to_id])
            type_list.append(_RELATION_TO_ID.get("to", 0))
            src_list.append(id_to_idx[to_id])
            dst_list.append(e_idx)
            type_list.append(_RELATION_TO_ID.get("to", 0))

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_type = torch.tensor(type_list, dtype=torch.long)
    element_ids = [eid for eid, _ in elements]
    return node_features, edge_index, edge_type, element_ids


def apply_scores_to_graph(
    json_graph: Dict,
    scores: torch.Tensor,
    element_ids: List[str],
    threshold: float = config.HIGHLIGHT_THRESHOLD,
) -> Dict:
    """Map predicted semantic-gap scores back onto the JSON graph.

    Elements with S_u >= threshold receive ``highlight = True``;
    attribute rows of highlighted edges receive ``highlight_rows``.
    """
    score_map = {eid: float(scores[i]) for i, eid in enumerate(element_ids)}
    for nid, node in json_graph.get("nodes", {}).items():
        s = score_map.get(nid, 0.0)
        node["score"] = s
        node["highlight"] = s >= threshold
    for idx, edge in enumerate(json_graph.get("edges", [])):
        eid = f"__edge_{idx}"
        s = score_map.get(eid, 0.0)
        edge["score"] = s
        edge["highlight"] = s >= threshold
    return json_graph
