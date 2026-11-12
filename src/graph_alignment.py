"""Graph alignment via Graph Edit Distance (GED).

Used to build dense supervision for (q, s_gt, s_bug) triples. Because
real LLM errors cause structural drift, J_gt and J_bug are not
topologically isomorphic. We first anchor shared skeleton elements
(table/column names, clause skeleton) and then run GED on the drift
region to find the minimum edit path. Elements in J_bug that are
inserted or substituted become negative samples (y=0); everything else
is positive (y=1).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
from networkx.algorithms.similarity import graph_edit_distance


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


def align_graphs(
    j_gt: Dict,
    j_bug: Dict,
    timeout: float = 5.0,
) -> Tuple[nx.Graph, nx.Graph, Optional[List[Tuple]]]:
    """Compute GED between J_gt and J_bug and return the edit path.

    Returns:
        (g_gt, g_bug, edit_path) where edit_path is a list of
        (operation, ...) tuples describing the minimum edits.
    """
    g_gt = build_networkx_graph(j_gt)
    g_bug = build_networkx_graph(j_bug)
    try:
        edit_path = graph_edit_distance(
            g_gt,
            g_bug,
            node_match=node_match,
            edge_match=edge_match,
            timeout=timeout,
        )
    except Exception:
        edit_path = None
    return g_gt, g_bug, edit_path


def label_elements(
    j_gt: Dict,
    j_bug: Dict,
    edit_path: Optional[List[Tuple]],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Assign y=0 (bug) / y=1 (correct) to elements of J_bug.

    Elements targeted by insert/substitute edits in J_bug are labeled 0;
    all other J_bug elements and all J_gt elements are labeled 1.
    """
    gt_labels: Dict[str, int] = {nid: 1 for nid in j_gt.get("nodes", {})}
    bug_labels: Dict[str, int] = {nid: 1 for nid in j_bug.get("nodes", {})}
    for i, e in enumerate(j_bug.get("edges", [])):
        bug_labels[f"e{i}"] = 1
    if edit_path is None:
        return gt_labels, bug_labels
    for op in edit_path:
        if not isinstance(op, (list, tuple)) or len(op) < 1:
            continue
        kind = op[0]
        if kind == "insert" and len(op) >= 3:
            target = op[2]
            if target in bug_labels:
                bug_labels[target] = 0
        elif kind == "substitute" and len(op) >= 4:
            target = op[3]
            if target in bug_labels:
                bug_labels[target] = 0
    return gt_labels, bug_labels
