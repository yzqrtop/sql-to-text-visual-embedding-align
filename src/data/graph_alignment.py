"""Graph alignment via Graph Edit Distance (GED).

Used to build dense supervision for (q, s_gt, s_bug) triples. Because
real LLM errors cause structural drift, J_gt and J_bug are not
topologically isomorphic. We first anchor shared skeleton elements
(table/column names, clause skeleton) and then run GED on the drift
region to find the minimum edit path. Elements in J_bug that are
inserted or substituted become negative samples (y=0); everything else
is positive (y=1).

Implementation notes:
    * ``networkx.optimize_edit_paths`` yields ``(vertex_path, edge_path,
      cost)`` triples; ``vertex_path`` is a list of ``(u, v)`` node pairs
      where ``u``/``v`` may be ``None`` for deletions/insertions.
    * GED requires scipy (Hungarian assignment). If scipy is missing or
      GED fails/times out, a deterministic label-multiset fallback still
      flags bug elements whose (kind, label) does not occur in J_gt.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import networkx as nx
from networkx.algorithms.similarity import optimize_edit_paths

# A vertex path is a list of (gt_id, bug_id) pairs; either side may be None.
VertexPath = List[Tuple[Optional[str], Optional[str]]]


def build_networkx_graph(json_graph: Dict) -> nx.Graph:
    """Convert a JSON graph (from ConfigBuilder) into a NetworkX graph.

    Nodes represent graph elements (data sources, operations, predicates);
    edges represent structural adjacency with relation-type attributes.
    """
    g = nx.Graph()
    # Data-source nodes.
    for node_id, node in json_graph.get("nodes", {}).items():
        g.add_node(node_id, kind="node", label=node.get("text", node_id))
    # Operation edges become graph nodes too (so they can be aligned).
    for idx, edge in enumerate(json_graph.get("edges", [])):
        eid = f"e{idx}"
        g.add_node(eid, kind="edge", label=edge.get("label", ""))
        g.add_edge(edge.get("from"), eid, relation="from")
        g.add_edge(eid, edge.get("to"), relation="to")
    return g


def node_match(a: Dict, b: Dict) -> bool:
    """Two elements match if they have the same kind and similar label."""
    if a.get("kind") != b.get("kind"):
        return False
    return a.get("label", "").lower() == b.get("label", "").lower()


def edge_match(a: Dict, b: Dict) -> bool:
    return a.get("relation") == b.get("relation")


def _element_label_map(json_graph: Dict) -> Dict[str, Tuple[str, str]]:
    """Map element id -> (kind, label) using the same ids as the NX graph."""
    labels: Dict[str, Tuple[str, str]] = {}
    for node_id, node in json_graph.get("nodes", {}).items():
        labels[node_id] = ("node", node.get("text", node_id))
    for idx, edge in enumerate(json_graph.get("edges", [])):
        labels[f"e{idx}"] = ("edge", edge.get("label", "") or "")
    return labels


def align_graphs(
    j_gt: Dict,
    j_bug: Dict,
    timeout: float = 5.0,
) -> Tuple[nx.Graph, nx.Graph, Optional[VertexPath]]:
    """Compute the minimum-cost GED vertex path between J_gt and J_bug.

    Returns:
        (g_gt, g_bug, vertex_path) where vertex_path is a list of
        (gt_id, bug_id) pairs (None on either side marks deletions and
        insertions), or None when GED is unavailable or times out.
    """
    g_gt = build_networkx_graph(j_gt)
    g_bug = build_networkx_graph(j_bug)
    best_path: Optional[VertexPath] = None
    best_cost: Optional[float] = None
    try:
        for vertex_path, _, cost in optimize_edit_paths(
            g_gt,
            g_bug,
            node_match=node_match,
            edge_match=edge_match,
            timeout=timeout,
        ):
            if best_cost is None or cost < best_cost:
                best_path, best_cost = vertex_path, cost
    except Exception:
        best_path = None
    return g_gt, g_bug, best_path


def _fallback_labels(
    gt_map: Dict[str, Tuple[str, str]],
    bug_map: Dict[str, Tuple[str, str]],
    bug_labels: Dict[str, int],
) -> None:
    """Deterministic fallback when no GED path is available.

    A bug element is positive only while a matching (kind, label) element
    remains unmatched in J_gt (multiset matching); everything else is a
    substituted/inserted element and therefore negative.
    """
    remaining = Counter(gt_map.values())
    for eid in bug_labels:
        key = bug_map.get(eid)
        if key is not None and remaining[key] > 0:
            remaining[key] -= 1
            bug_labels[eid] = 1
        else:
            bug_labels[eid] = 0


def label_elements(
    j_gt: Dict,
    j_bug: Dict,
    edit_path: Optional[VertexPath],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Assign y=0 (bug) / y=1 (correct) to elements of J_bug.

    Inserted bug elements (``(None, v)`` pairs) and substituted bug
    elements whose (kind, label) differs from the aligned gold element
    are labeled 0; all other J_bug elements and all J_gt elements are 1.
    """
    gt_labels: Dict[str, int] = {nid: 1 for nid in j_gt.get("nodes", {})}
    bug_labels: Dict[str, int] = {nid: 1 for nid in j_bug.get("nodes", {})}
    for i, _ in enumerate(j_bug.get("edges", [])):
        bug_labels[f"e{i}"] = 1

    gt_map = _element_label_map(j_gt)
    bug_map = _element_label_map(j_bug)

    if edit_path is None:
        _fallback_labels(gt_map, bug_map, bug_labels)
        return gt_labels, bug_labels

    for pair in edit_path:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        u, v = pair
        if v is None or v not in bug_labels:
            continue  # deletion of a gold element: nothing to label in J_bug
        if u is None:
            bug_labels[v] = 0  # inserted bug element
        else:
            bug_labels[v] = 1 if gt_map.get(u) == bug_map.get(v) else 0
    return gt_labels, bug_labels
