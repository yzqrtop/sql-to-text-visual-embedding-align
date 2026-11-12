#!/usr/bin/env python3
"""
SQL intent visualisation metaphor-diagram renderer

Graph semantics (unified conventions):
  - all entities (base table / temp table / derived table / virtual table / result set) are always rendered as rectangles
    the header shows the entity name (including the original SQL alias, e.g. user_events e), rows show attribute columns
  - clauses and operations (FROM / JOIN / WHERE / GROUP BY / SELECT ...) are not drawn as nodes,
    but written on the arrow label: JOIN ON ..., WHERE ..., GROUP BY ..., etc.
  - arrow direction = data-flow direction (from input entity to output entity)

Usage:
  python tool/sql_metaphor_diagram.py --out tool/output/diagram.svg
  python tool/sql_metaphor_diagram.py --config tool/config_spatial_events.json --out tool/output/spatial.svg
  python tool/sql_metaphor_diagram.py --table "user_events=user_events e|user_id,event_id" --out tool/output/diagram.svg

Two layout modes (layout.mode in the config):
  slot     - arrange by semantic slots (default fallback example)
  layered  - auto-layer by edge dependency, arrange CTE chains/DAGs left->right, works for any multi-level query

Attribute-level wiring (optional edge fields):
  from_attr / to_attr - explicitly specify the source/target anchor attribute row name; the line starts from that row centre
  attr_link: true     - automatically find same-named attribute rows of two tables and draw pairwise lines (JOIN key visualisation)
  layout.attr_link    - global switch; when true all straight edges enable attr_link by default
  label               - edge label text (clause/operation description), rendered with a white background box
  falls back automatically to whole-table boundary wiring when no attribute row matches; never errors

Python usage:
  from tool.sql_metaphor_diagram import render_diagram
  render_diagram({"nodes": {"user_events": {"rows": ["user_id", "event_id"]}}}, "out.svg")
"""

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ==================== Default shape definitions (overridable by config) ====================

DEFAULT_SPEC: Dict[str, object] = {
    "canvas": {"width": 720, "height": 452, "background": "#ffffff"},
    "baseline_y": 222,
    "style": {
        "font_family": "Arial, Helvetica, 'Microsoft YaHei', sans-serif",
        "font_size": 16,
        "label_size": 17,
        "label_color": "#e01b1b",
        "stroke": "#111111",
        "stroke_width": 1.6,
        "header_height": 24,
        "row_height": 27,
        "min_width": 90,
        "text_padding": 11,
        "arrow_size": 7,
        "label_offset": 14,
        "highlight_color": "#e01b1b",
        "highlight_fill": "#ffe9e9",
        "highlight_width": 2.6,
    },
    "gate": {"size": 18},
    "nodes": {
        "result": {
            "slot": "source",
            "kind": "table",
            "header": "SELECT",
            "header_fill": "#cccccc",
            "header_color": "#666666",
            "rows": ["drinker"],
        },
        "L1": {
            "slot": "main",
            "kind": "table",
            "label": "L1",
            "header": "Likes",
            "rows": ["drinker"],
        },
        "gate": {"slot": "gate", "kind": "gate"},
        "L2": {
            "slot": "center",
            "kind": "dashed",
            "label": "L2",
            "header": "Likes",
            "rows": ["drinker"],
        },
        "L4": {
            "slot": "upper",
            "kind": "table",
            "label": "L4",
            "header": "Likes",
            "rows": ["drinker", "beer"],
        },
        "L3": {
            "slot": "upper_right",
            "kind": "stacked",
            "label": "L3",
            "header": "Likes",
            "rows": ["beer", "drinker"],
        },
        "L5": {
            "slot": "lower",
            "kind": "stacked",
            "label": "L5",
            "header": "Likes",
            "rows": ["drinker", "beer"],
        },
        "L6": {
            "slot": "lower_right",
            "kind": "table",
            "label": "L6",
            "header": "Likes",
            "rows": ["drinker", "beer"],
        },
    },
    "edges": [
        {"from": "result", "to": "L1", "kind": "line"},
        {"from": "L1", "to": "gate", "kind": "line"},
        {"from": "gate", "to": "L2", "kind": "line"},
        {"from": "gate", "to": "L4", "kind": "curve", "bend": [-45, -35]},
        {"from": "gate", "to": "L5", "kind": "curve", "bend": [-45, 30]},
        {"from": "L4", "to": "L3", "kind": "curve", "to_t": 0.35, "bend": [10, 20]},
        {"from": "L2", "to": "L3", "kind": "curve", "to_t": 0.68, "bend": [25, -15]},
        {"from": "L5", "to": "L6", "kind": "curve", "bend": [15, -20]},
    ],
}


# ==================== Utility functions ====================

def deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge configs: dicts merge key by key, lists are replaced wholesale."""
    if not isinstance(base, dict) or not isinstance(override, dict):
        return copy.deepcopy(override)
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def text_width(text: object, size: float, bold: bool = False) -> float:
    """Rough character width: ASCII 0.58em, CJK 1.0em."""
    units = 0.0
    for ch in str(text):
        units += 1.0 if ord(ch) > 127 else 0.58
    return units * size * (1.05 if bold else 1.0)


def build_spec(
    overrides: Optional[Dict] = None,
    table_overrides: Optional[Sequence[str]] = None,
) -> Dict:
    """Default config + file config + command-line --table overrides."""
    base = copy.deepcopy(DEFAULT_SPEC)
    # layered mode uses an independent node/edge set to avoid default-example nodes (e.g. gate) leaking in
    if overrides and str(overrides.get("layout", {}).get("mode", "")).strip() == "layered":
        base.pop("nodes", None)
        base.pop("edges", None)

    spec = deep_merge(base, overrides) if overrides else base

    for item in table_overrides or []:
        if "=" not in item:
            raise ValueError(f"--table format should be 'node=header|row1,row2', got: {item}")
        node_id, payload = item.split("=", 1)
        node_id = node_id.strip()
        if "|" in payload:
            header, rows_text = payload.split("|", 1)
            rows = [r.strip() for r in rows_text.split(",") if r.strip()]
        else:
            header, rows = payload, []
        node = spec["nodes"].setdefault(node_id, {"slot": "main", "kind": "table"})
        node["header"] = header.strip()
        node["rows"] = rows
    return spec


# ==================== Sizing and layout ====================

def measure_node(node: Dict, style: Dict, gate: Dict) -> Tuple[float, float]:
    if node.get("kind") == "gate":
        size = float(gate.get("size", 18))
        return size, size

    font = float(style["font_size"])
    padding = float(style["text_padding"])
    min_w = float(style["min_width"])

    texts = [node.get("header", "")] + [str(r) for r in node.get("rows", [])]
    width = max(
        [text_width(node.get("header", ""), font, bold=True)]
        + [text_width(t, font) for t in texts]
    )
    width = max(min_w, width + 2 * padding)

    height = float(style["header_height"]) + float(style["row_height"]) * len(
        node.get("rows", [])
    )
    return round(width, 1), round(height, 1)


def wrapper_rect(node: Dict) -> Tuple[float, float, float, float]:
    """All entities (base/temp/derived/virtual/result tables) are unified as rectangles."""
    return node["x"], node["y"], node["w"], node["h"]


def rect_center(rect: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x, y, w, h = rect
    return x + w / 2, y + h / 2


def point_on_rect(
    rect: Tuple[float, float, float, float], side: str, frac: float = 0.5
) -> Tuple[float, float]:
    x, y, w, h = rect
    frac = max(0.0, min(1.0, float(frac)))
    if side == "left":
        return x, y + h * frac
    if side == "right":
        return x + w, y + h * frac
    if side == "top":
        return x + w * frac, y
    if side == "bottom":
        return x + w * frac, y + h
    return x + w / 2, y + h / 2


def auto_sides(rect_a, rect_b) -> Tuple[str, str]:
    ax, ay = rect_center(rect_a)
    bx, by = rect_center(rect_b)
    if abs(bx - ax) >= abs(by - ay):
        return ("right", "left") if bx >= ax else ("left", "right")
    return ("bottom", "top") if by >= ay else ("top", "bottom")


def _node_rank(node: Dict) -> Optional[int]:
    rank = node.get("rank")
    return rank if isinstance(rank, int) else None


def default_edge_sides(node_a: Dict, node_b: Dict) -> Tuple[str, str]:
    """Choose the default connection side by column relation (lr mode):
    - target in the right column: left table right side -> right table left side (right-side wires only connect to the right column)
    - same column (vertically connected): both ends use the table left side to avoid wires crossing the table body
    - target in the left column (back edge): left table left side -> right table right side
    tb mode (layout.direction="tb", rank is the row):
    - target in a lower row: top table bottom -> bottom table top; same row: both ends use the top side
    falls back to geometric-centre judgement when there is no rank info (slot mode).
    """
    ra, rb = _node_rank(node_a), _node_rank(node_b)
    if ra is not None and rb is not None:
        if node_a.get("_dir") == "tb" or node_b.get("_dir") == "tb":
            if ra == rb:
                return "top", "top"
            return ("bottom", "top") if rb > ra else ("top", "bottom")
        if ra == rb:
            return "left", "left"
        return ("right", "left") if rb > ra else ("left", "right")
    rect_a, rect_b = wrapper_rect(node_a), wrapper_rect(node_b)
    ax, bx = rect_center(rect_a)[0], rect_center(rect_b)[0]
    if abs(ax - bx) < 1:
        return "left", "left"
    return ("right", "left") if bx > ax else ("left", "right")


# ==================== Attribute-level anchors ====================

def _row_index(node: Dict, attr: object) -> Optional[int]:
    """Find an attribute row index in a node's rows: exact match first, then case-insensitive."""
    rows = [str(r) for r in node.get("rows", [])]
    target = str(attr)
    for i, row in enumerate(rows):
        if row == target:
            return i
    for i, row in enumerate(rows):
        if row.lower() == target.lower():
            return i
    return None


def attr_row_y(node: Dict, attr: object, style: Dict) -> Optional[float]:
    """Vertical centre coordinate of an attribute row; None if not found."""
    idx = _row_index(node, attr)
    if idx is None:
        return None
    header_h = float(style["header_height"])
    row_h = float(style["row_height"])
    return node["y"] + header_h + idx * row_h + row_h / 2


def _attr_point(node: Dict, side: str, attr: object, style: Dict) -> Optional[Tuple[float, float]]:
    """Anchor point of an attribute row on the table border; only meaningful for left/right sides, other sides return None as fallback."""
    y = attr_row_y(node, attr, style)
    if y is None:
        return None
    if side == "left":
        return node["x"], y
    if side == "right":
        return node["x"] + node["w"], y
    return None


def _shared_attrs(a: Dict, b: Dict) -> List[str]:
    """Attribute row names shared by two tables (preserving the order in table a)."""
    rows_b = {str(r) for r in b.get("rows", [])}
    return [str(r) for r in a.get("rows", []) if str(r) in rows_b]


def expand_edges(spec: Dict) -> List[Dict]:
    """Expand attr_link edges into per-attribute edges; explicit from_attr/to_attr and other edges are kept as-is."""
    nodes = spec.get("nodes", {})
    global_link = bool(spec.get("layout", {}).get("attr_link", False))
    out = []
    for edge in spec.get("edges", []):
        a, b = nodes.get(edge.get("from")), nodes.get(edge.get("to"))
        link = edge.get("attr_link", global_link and edge.get("kind", "line") == "line")
        if link and a and b and "from_attr" not in edge and "to_attr" not in edge:
            shared = _shared_attrs(a, b)
            if shared:
                for i, attr in enumerate(shared):
                    derived = dict(edge)
                    derived["from_attr"] = attr
                    derived["to_attr"] = attr
                    # multiple attribute wires show the relation label only once on the first to avoid duplication
                    if i > 0:
                        derived.pop("label", None)
                    out.append(derived)
                continue
        out.append(edge)
    return out


def edge_geometry(nodes: Dict, edge: Dict, style: Dict) -> Tuple[Tuple[float, float], Optional[Tuple[float, float]], Tuple[float, float]]:
    """Return (start, control_point or None, end) of an edge; shared by rendering and canvas-boundary computation."""
    node_a = nodes[edge["from"]]
    node_b = nodes[edge["to"]]
    rect_a = wrapper_rect(node_a)
    rect_b = wrapper_rect(node_b)
    from_side, to_side = default_edge_sides(node_a, node_b)
    ra, rb = _node_rank(node_a), _node_rank(node_b)
    same_col = ra is not None and rb is not None and ra == rb
    tb_mode = node_a.get("_dir") == "tb" or node_b.get("_dir") == "tb"
    explicit_from, explicit_to = edge.get("from_side"), edge.get("to_side")
    if same_col:
        # rule: tables connected in the same column (row) use the same side to avoid wires crossing the table body
        from_side, to_side = ("top", "top") if tb_mode else ("left", "left")
    else:
        # compatibility with the old "right-out right-back" key-chain style: switch to left-in when the target is already placed in the right column
        if (explicit_from, explicit_to) == ("right", "right") and rb is not None and ra is not None and rb > ra:
            explicit_to = "left"
        from_side = explicit_from or from_side
        to_side = explicit_to or to_side

    # attribute-level anchors take priority; fall back to whole-table boundary when unmatched
    start = None
    if edge.get("from_attr") is not None:
        start = _attr_point(node_a, from_side, edge["from_attr"], style)
    if start is None:
        start = point_on_rect(rect_a, from_side, edge.get("from_t", 0.5))

    end = None
    if edge.get("to_attr") is not None:
        end = _attr_point(node_b, to_side, edge["to_attr"], style)
    if end is None:
        end = point_on_rect(rect_b, to_side, edge.get("to_t", 0.5))

    control = None
    if edge.get("kind") == "curve":
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        bx, by = edge.get("bend", [0, 0])
        bx, by = float(bx), float(by)
        # curves exiting on the left automatically arch leftward (bend's x direction follows the connection side to avoid the arc pressing back into the table body)
        if from_side == "left" and to_side == "left" and bx > 0:
            bx = -bx
        elif from_side == "right" and to_side == "right" and bx < 0:
            bx = -bx
        # tb-mode same-row top-side wires arch upward
        if from_side == "top" and to_side == "top" and by > 0:
            by = -by
        control = (mx + bx, my + by)
    return start, control, end


def _curve_points(nodes: Dict, spec: Dict, style: Dict) -> List[Tuple[float, float]]:
    points = []
    for edge in expand_edges(spec):
        if edge.get("kind") != "curve":
            continue
        if edge.get("from") not in nodes or edge.get("to") not in nodes:
            continue
        _, control, _ = edge_geometry(nodes, edge, style)
        if control:
            points.append(control)
    return points


def _assign_layered(nodes: Dict, spec: Dict, base_y: float, style: Dict) -> None:
    """Layer by longest path: sources on the left, dependants progressively to the right, same-layer nodes stacked vertically and centred.
    Column gaps are auto-widened by the width of the longest edge label between adjacent layers, so labels do not press onto tables."""
    cfg = spec.get("layout", {})
    gap_x = float(cfg.get("gap_x", 42))
    gap_y = float(cfg.get("gap_y", 24))
    margin_x = float(cfg.get("margin_x", 34))
    direction = str(cfg.get("direction", "lr")).lower()

    ids = list(nodes.keys())
    preds = {node_id: set() for node_id in ids}
    for edge in spec.get("edges", []):
        if edge.get("no_rank"):
            continue  # annotation wires (e.g. same-column JOIN key pairs) do not participate in layering
        a, b = edge.get("from"), edge.get("to")
        if a in nodes and b in nodes:
            preds[b].add(a)

    # same-name attribute direct-wire rule: when a.x -> b.x and a.x has only this one outgoing edge,
    # a is fixed one column to the left of b. Except no_rank key-pair edges -- they stay in the same column,
    # otherwise the base-table key chain would fan out in steps and projection wires would be forced to cross intermediate tables.
    expanded = expand_edges(spec)
    attr_out: Dict[Tuple[str, str], int] = {}
    for edge in expanded:
        a, fa = edge.get("from"), edge.get("from_attr")
        if a in nodes and fa is not None:
            key = (a, str(fa).lower())
            attr_out[key] = attr_out.get(key, 0) + 1
    for edge in expanded:
        if edge.get("no_rank"):
            continue  # key-pair edges only do same-column left-side wiring, do not participate in layering
        a, b = edge.get("from"), edge.get("to")
        fa, ta = edge.get("from_attr"), edge.get("to_attr")
        if a not in nodes or b not in nodes or a == b:
            continue
        if fa is None or ta is None or str(fa).lower() != str(ta).lower():
            continue
        if attr_out.get((a, str(fa).lower()), 0) == 1:
            preds[b].add(a)

    rank = {node_id: 0 for node_id in ids}
    for _ in range(len(ids) + 1):
        changed = False
        for node_id in ids:
            if preds[node_id]:
                value = 1 + max(rank[p] for p in preds[node_id])
                if value > rank[node_id]:
                    rank[node_id] = value
                    changed = True
        if not changed:
            break

    columns: Dict[int, List[str]] = {}
    for node_id in ids:
        columns.setdefault(rank[node_id], []).append(node_id)

    # record each node's layer number and direction for edge-label positioning and connection-side selection
    for node_id in ids:
        nodes[node_id]["rank"] = rank[node_id]
        nodes[node_id]["_dir"] = direction

    levels = sorted(columns)

    if direction == "tb":
        # top->bottom layering: rank is the row (sources on top), arranged horizontally within the row;
        # row spacing is raised by the tallest edge label between rows
        row_gap_label: Dict[Tuple[int, int], float] = {}
        for edge in expand_edges(spec):
            a, b = edge.get("from"), edge.get("to")
            label = edge.get("label")
            if not label or a not in nodes or b not in nodes:
                continue
            la, lb = rank[a], rank[b]
            if lb >= la + 1 and edge.get("kind", "line") != "curve":
                _, lh = _label_size(label, style)
                row_gap_label[(la, la + 1)] = max(
                    row_gap_label.get((la, la + 1), 0.0), lh)
        y = 30.0
        for idx, level in enumerate(levels):
            members = columns[level]
            row_h = max(nodes[node_id]["h"] for node_id in members)
            x = margin_x
            for node_id in members:
                nodes[node_id]["x"] = x
                nodes[node_id]["y"] = y
                x += nodes[node_id]["w"] + gap_x
            y += row_h
            if idx < len(levels) - 1:
                nxt = levels[idx + 1]
                y += max(gap_y + 16, row_gap_label.get((level, nxt), 0.0) + 56)
        return

    # column gap = the actual width of the longest label in that gap (label text width + padding):
    # - cross-layer straight edges: label placed in the first gap after the source layer
    # - same-column left-side curves: label at the control point, i.e. the gap to the left of that column
    # layout.max_label_gap can serve as an explicit upper bound; uncapped when not set
    font = float(style.get("font_size", 16))
    max_gap = cfg.get("max_label_gap")
    gap_needed: Dict[Tuple[int, int], float] = {}
    for edge in expand_edges(spec):
        a, b = edge.get("from"), edge.get("to")
        label = edge.get("label")
        if not label or a not in nodes or b not in nodes:
            continue
        la, lb = rank[a], rank[b]
        # multi-line labels count width by the longest line (line-splitting rules see _label_lines)
        need = max(text_width(t, font * 0.85) for t in _label_lines(label)) + 24
        if max_gap is not None:
            need = min(need, float(max_gap))
        if lb >= la + 1 and edge.get("kind", "line") != "curve":
            gap_needed[(la, la + 1)] = max(gap_needed.get((la, la + 1), 0.0), need)
        elif la == lb and edge.get("kind") == "curve" and la >= 1:
            gap_needed[(la - 1, la)] = max(gap_needed.get((la - 1, la), 0.0), need)

    levels = sorted(columns)
    x = margin_x
    for idx, level in enumerate(levels):
        members = columns[level]
        col_w = max(nodes[node_id]["w"] for node_id in members)
        total_h = sum(nodes[node_id]["h"] for node_id in members) + gap_y * (len(members) - 1)
        y = base_y - total_h / 2
        for node_id in members:
            nodes[node_id]["w"] = col_w  # tables in the same column are unified to the widest width
            nodes[node_id]["x"] = x
            nodes[node_id]["y"] = y
            y += nodes[node_id]["h"] + gap_y
        x += col_w
        if idx < len(levels) - 1:
            nxt = levels[idx + 1]
            x += max(gap_x, gap_needed.get((level, nxt), 0.0))


def layout(spec: Dict) -> Dict:
    style = spec["style"]
    nodes = spec["nodes"]
    gate_cfg = spec.get("gate", {})
    base_y = float(spec.get("baseline_y", 222))

    # 1. sizing
    for node in nodes.values():
        node["w"], node["h"] = measure_node(node, style, gate_cfg)

    slots = {}
    for node_id, node in nodes.items():
        slot = node.get("slot")
        if slot and slot not in slots:
            slots[slot] = node_id

    # 2. main chain (horizontal baseline)
    if "source" in slots:
        nodes[slots["source"]]["x"] = 34
        nodes[slots["source"]]["y"] = base_y - nodes[slots["source"]]["h"] / 2

    main_x = 158
    if "source" in slots:
        src = nodes[slots["source"]]
        main_x = src["x"] + src["w"] + 12
    if "main" in slots:
        main = nodes[slots["main"]]
        main["x"] = main_x
        main["y"] = base_y - main["h"] / 2

    gate_x = main_x + nodes[slots["main"]]["w"] + 16 if "main" in slots else 296
    if "gate" in slots:
        gate = nodes[slots["gate"]]
        gate["x"] = gate_x
        gate["y"] = base_y - gate["h"] / 2

    center_x = gate_x + (nodes[slots["gate"]]["w"] if "gate" in slots else 18) + 24
    if "center" in slots:
        center = nodes[slots["center"]]
        center["x"] = center_x
        center["y"] = base_y - center["h"] / 2

    # 3. upper/lower branches
    if "center" in slots:
        center = nodes[slots["center"]]
        if "upper" in slots:
            upper = nodes[slots["upper"]]
            upper["x"] = center["x"] + 8
            upper["y"] = center["y"] - 100 - upper["h"]
        if "lower" in slots:
            lower = nodes[slots["lower"]]
            lower["x"] = center["x"] - 2
            lower["y"] = center["y"] + center["h"] + 85

    # 4. right column (horizontal alignment)
    max_right = 0.0
    for slot in ("center", "upper", "lower"):
        if slot in slots:
            x, y, w, h = wrapper_rect(nodes[slots[slot]])
            max_right = max(max_right, x + w)
    right_x = max_right + 38 if max_right else 470

    if "upper_right" in slots:
        node = nodes[slots["upper_right"]]
        node["x"] = right_x
        if "upper" in slots:
            node["y"] = nodes[slots["upper"]]["y"] + nodes[slots["upper"]]["h"] + 2
        else:
            node["y"] = base_y - 170

    if "lower_right" in slots:
        node = nodes[slots["lower_right"]]
        node["x"] = right_x
        if "lower" in slots:
            node["y"] = nodes[slots["lower"]]["y"] - 25
        else:
            node["y"] = base_y + 90

    # layered layout mode: auto-arrange by edge dependency (works for CTE chains, DAGs)
    if str(spec.get("layout", {}).get("mode", "slot")) == "layered":
        _assign_layered(nodes, spec, base_y, style)

    # nodes without a declared slot get a fallback position to avoid crashes on missing config fields
    for node in nodes.values():
        node.setdefault("x", 34)
        node.setdefault("y", base_y - node["h"] / 2)

    # 5. canvas auto-fit: expand and shift down when content is too tall/wide (leave top margin for labels)
    def visual_top(node: Dict) -> float:
        top = wrapper_rect(node)[1]
        if node.get("label"):
            top -= float(style.get("label_size", 17)) + 4
        return top

    # first estimate the downward shift from node tops to keep the topmost nodes/labels off the edge
    min_top = min(visual_top(n) for n in nodes.values())
    dy = 12 - min_top if min_top < 12 else 0
    if dy:
        for node in nodes.values():
            node["y"] += dy

    # once node positions are final, run one label-anti-overlap pass
    label_pos = _resolve_label_positions(nodes, spec, style)

    # labels pushed upward may exceed the canvas top; shift everything down once more and re-resolve
    if label_pos:
        font = float(style.get("font_size", 16))
        max_lh = max((_label_size(e["label"], style)[1]
                      for e in expand_edges(spec) if e.get("label")),
                     default=font + 6)
        min_label_top = min(y for _, y in label_pos.values()) - max_lh / 2
        if min_label_top < 4:
            extra = 4 - min_label_top
            for node in nodes.values():
                node["y"] += extra
            dy += extra
            label_pos = _resolve_label_positions(nodes, spec, style)

    # recompute all element bounds after shifting down (nodes, curve control points, edge labels)
    tops = [visual_top(n) for n in nodes.values()]
    bottoms = [wrapper_rect(n)[1] + wrapper_rect(n)[3] for n in nodes.values()]
    rights = [wrapper_rect(n)[0] + wrapper_rect(n)[2] for n in nodes.values()]
    lefts = [wrapper_rect(n)[0] for n in nodes.values()]

    for p in _curve_points(nodes, spec, style):
        tops.append(p[1])
        bottoms.append(p[1])
        rights.append(p[0])
        lefts.append(p[0])

    for edge in expand_edges(spec):
        bbox = _edge_label_bbox(nodes, edge, style, label_pos)
        if bbox is None:
            continue
        lx, ly, lw, lh = bbox
        tops.append(ly)
        bottoms.append(ly + lh)
        rights.append(lx + lw)
        lefts.append(lx)

    max_bottom = max(bottoms) if bottoms else 0.0
    max_right = max(rights) if rights else 0.0
    min_left = min(lefts) if lefts else 0.0
    dx = 0.0
    if min_left < 0:
        dx = -min_left
        for node in nodes.values():
            node["x"] += dx
        # label/curve bounds shift right with the nodes as a whole; does not affect canvas width, only need right padding
        max_right += dx

    width = max(float(spec["canvas"].get("width", 720)), max_right + 34)
    height = max(float(spec["canvas"].get("height", 452)), max_bottom + 28)

    return {
        "nodes": nodes,
        "width": round(width, 1),
        "height": round(height, 1),
        "dy": dy,
        "label_pos": label_pos,
    }


# ==================== SVG rendering ====================

def _label_lines(text: object) -> List[str]:
    """Split clause labels by ';': 'GROUP BY x; ORDER BY y; LIMIT 1' -> 3 lines.
    width takes the longest line rather than the whole sentence as one super-wide bar; multiple long annotations in the same column stack vertically."""
    parts = [p.strip() for p in str(text).split(";")]
    return [p for p in parts if p] or [str(text)]


def _label_size(text: object, style: Dict) -> Tuple[float, float]:
    font = float(style.get("font_size", 16))
    size = font * 0.85
    lines = _label_lines(text)
    line_h = size * 1.32
    return max(text_width(t, size) for t in lines) + 12, line_h * len(lines) + 8


def _overlap_area(bbox: Tuple[float, float, float, float], rects: Sequence[Tuple[float, float, float, float]]) -> float:
    """Sum of overlap areas between the label box and each entity rectangle."""
    bx, by, bw, bh = bbox
    total = 0.0
    for rx, ry, rw, rh in rects:
        ix = max(0.0, min(bx + bw, rx + rw) - max(bx, rx))
        iy = max(0.0, min(by + bh, ry + rh) - max(by, ry))
        total += ix * iy
    return total


def _edge_key(edge: Dict) -> Tuple:
    return (
        edge.get("from"),
        edge.get("to"),
        edge.get("from_attr"),
        edge.get("to_attr"),
        str(edge.get("label", "")),
    )


def edge_label_anchor(
    nodes: Dict, edge: Dict, style: Dict,
    resolved: Optional[Dict] = None,
) -> Optional[Tuple[float, float]]:
    """Placement anchor of an edge label.

    - positions already resolved by _resolve_label_positions are returned directly
    - curve edges: placed at the Bezier control point (already in a blank area)
    - straight edges spanning multiple layers: snapped to the midline of the first column gap after the source node,
      to avoid labels landing on tables in the middle column
    - other straight edges: offset vertically from the segment midpoint, between the up/down two candidates
      choose the one with smaller overlap area with entity rectangles, to avoid pressing onto tables
    """
    if not edge.get("label"):
        return None
    if resolved and _edge_key(edge) in resolved:
        return resolved[_edge_key(edge)]

    start, control, end = edge_geometry(nodes, edge, style)
    mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2

    # cross-layer straight edges: move the midpoint to the midline of the "source layer -> next layer" gap
    node_a = nodes[edge["from"]]
    node_b = nodes[edge["to"]]
    ra, rb = node_a.get("rank"), node_b.get("rank")
    if not control and ra is not None and rb is not None and rb > ra + 1 and end[0] != start[0]:
        right_a = max(n["x"] + n["w"] for n in nodes.values() if n.get("rank") == ra)
        left_next = min(n["x"] for n in nodes.values() if n.get("rank") == ra + 1)
        gap_cx = (right_a + left_next) / 2
        t = (gap_cx - start[0]) / (end[0] - start[0])
        t = min(max(t, 0.1), 0.9)
        mx = start[0] + t * (end[0] - start[0])
        my = start[1] + t * (end[1] - start[1])

    if control:
        off_x = float(edge.get("label_dx", 0))
        off_y = float(edge.get("label_dy", 0))
        ax, ay = control[0] + off_x, control[1] + off_y
        # the control point of same-column left-side curves hugs the table left edge (only 30~34px outside): labels are right-aligned to
        # the left of the control point to avoid half-width overflow pressing back into the table body
        node_a = nodes[edge["from"]]
        node_b = nodes[edge["to"]]
        ra, rb = node_a.get("rank"), node_b.get("rank")
        if ra is not None and rb is not None and ra == rb and control[0] < start[0]:
            lw, _ = _label_size(edge["label"], style)
            ax = control[0] - lw / 2 - 8 + off_x
        return ax, ay

    dx, dy = end[0] - start[0], end[1] - start[1]
    norm = math.hypot(dx, dy) or 1.0
    px, py = -dy / norm, dx / norm  # unit normal vector

    base_offset = float(edge.get("label_offset", style.get("label_offset", 14)))
    lw, lh = _label_size(edge["label"], style)
    rects = [wrapper_rect(n) for n in nodes.values()]

    best = None
    for sign in (1.0, -1.0):
        cand = (mx + px * base_offset * sign, my + py * base_offset * sign)
        bbox = (cand[0] - lw / 2, cand[1] - lh / 2, lw, lh)
        area = _overlap_area(bbox, rects)
        if best is None or area < best[0]:
            best = (area, cand)
    return best[1]


def _resolve_label_positions(nodes: Dict, spec: Dict, style: Dict) -> Dict:
    """Label anti-overlap: first compute initial anchors from edge geometry, then iteratively push overlapping labels
    apart vertically (only y moves, keeping horizontal position and the owning gap consistent)."""
    items = []  # [key, x, y, w, h]
    for edge in expand_edges(spec):
        if not edge.get("label"):
            continue
        if edge.get("from") not in nodes or edge.get("to") not in nodes:
            continue
        anchor = edge_label_anchor(nodes, edge, style)
        if anchor is None:
            continue
        lw, lh = _label_size(edge["label"], style)
        items.append([_edge_key(edge), anchor[0], anchor[1], lw, lh])

    for _ in range(40):
        moved = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                _, xi, yi, wi, hi = items[i]
                _, xj, yj, wj, hj = items[j]
                overlap_x = (wi + wj) / 2 - abs(xi - xj)
                overlap_y = (hi + hj) / 2 - abs(yi - yj)
                if overlap_x > 0 and overlap_y > 0:
                    shift = overlap_y / 2 + 1
                    if yi <= yj:
                        items[i][2] -= shift
                        items[j][2] += shift
                    else:
                        items[i][2] += shift
                        items[j][2] -= shift
                    moved = True
        if not moved:
            break
    return {it[0]: (it[1], it[2]) for it in items}


def _edge_label_bbox(
    nodes: Dict, edge: Dict, style: Dict,
    resolved: Optional[Dict] = None,
) -> Optional[Tuple[float, float, float, float]]:
    """Approximate bounding box of label text, used for canvas-boundary computation to avoid clipping."""
    anchor = edge_label_anchor(nodes, edge, style, resolved)
    if anchor is None:
        return None
    lw, lh = _label_size(edge["label"], style)
    return anchor[0] - lw / 2, anchor[1] - lh / 2, lw, lh


def _render_edge_label(nodes: Dict, edge: Dict, style: Dict,
                       resolved: Optional[Dict] = None) -> Optional[str]:
    anchor = edge_label_anchor(nodes, edge, style, resolved)
    if anchor is None:
        return None
    font = float(style.get("font_size", 16))
    x, y, w, h = _edge_label_bbox(nodes, edge, style, resolved)
    font_family = style["font_family"]
    size = font * 0.85
    lines = _label_lines(edge["label"])
    line_h = size * 1.32
    # the text block as a whole is arranged vertically around the anchor
    first_base = anchor[1] - h / 2 + 4 + size * 0.88
    texts = "".join(
        f'<text x="{anchor[0]:.1f}" y="{first_base + i * line_h:.1f}" text-anchor="middle" '
        f'font-family="{font_family}" font-size="{size:.1f}" '
        f'fill="#333333">{escape(ln)}</text>'
        for i, ln in enumerate(lines)
    )
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="#ffffff" fill-opacity="0.94" stroke="#d8d8d8" stroke-width="0.8" rx="3"/>'
        + texts
    )


def render_edge_labels(laid: Dict, spec: Dict) -> List[str]:
    """Edge labels are rendered separately and drawn after nodes to avoid being covered by the table white background."""
    nodes = laid["nodes"]
    style = spec["style"]
    resolved = laid.get("label_pos")
    parts = []
    for edge in expand_edges(spec):
        if edge.get("from") not in nodes or edge.get("to") not in nodes:
            continue
        # label_hidden: still participates in layout (occupies width) but the label is not drawn,
        # used for comparison diagrams that "highlight one path and hide the other annotations"
        if edge.get("label_hidden"):
            continue
        html = _render_edge_label(nodes, edge, style, resolved)
        if html:
            parts.append(html)
    return parts


def _arrow_head(
    tip: Tuple[float, float],
    direction: Tuple[float, float],
    style: Dict,
    color: Optional[str] = None,
) -> str:
    """Manually draw an arrow polygon, tip at tip, direction along the path tangent.
    Do not use SVG markers: when multiple edges converge at the same endpoint markers overlap into a reverse visual,
    and some viewers have bugs in orient="auto" support."""
    dx, dy = direction
    norm = math.hypot(dx, dy) or 1.0
    ux, uy = dx / norm, dy / norm
    size = float(style.get("arrow_size", 7))
    back = size * 1.7   # arrow length
    wing = size * 0.62  # half width
    bx, by = tip[0] - ux * back, tip[1] - uy * back
    px, py = -uy, ux
    p1 = (bx + px * wing, by + py * wing)
    p2 = (bx - px * wing, by - py * wing)
    return (
        f'<polygon points="{tip[0]:.1f},{tip[1]:.1f} {p1[0]:.1f},{p1[1]:.1f} '
        f'{p2[0]:.1f},{p2[1]:.1f}" fill="{color or style["stroke"]}"/>'
    )


def render_edges(laid: Dict, spec: Dict) -> List[str]:
    nodes = laid["nodes"]
    style = spec["style"]
    parts = []

    for edge in expand_edges(spec):
        a_id, b_id = edge.get("from"), edge.get("to")
        if a_id not in nodes or b_id not in nodes:
            continue

        start, control, end = edge_geometry(nodes, edge, style)

        if control:
            path = (
                f"M {start[0]:.1f},{start[1]:.1f} "
                f"Q {control[0]:.1f},{control[1]:.1f} {end[0]:.1f},{end[1]:.1f}"
            )
            direction = (end[0] - control[0], end[1] - control[1])
        else:
            path = f"M {start[0]:.1f},{start[1]:.1f} L {end[0]:.1f},{end[1]:.1f}"
            direction = (end[0] - start[0], end[1] - start[1])

        # edge.highlight: when true the whole edge is drawn red and thicker (trace-path highlight)
        e_hl = bool(edge.get("highlight"))
        stroke = style.get("highlight_color", "#e01b1b") if e_hl else style["stroke"]
        stroke_w = (
            float(style.get("highlight_width", 2.6)) if e_hl else style["stroke_width"]
        )
        parts.append(
            f'<path d="{path}" fill="none" stroke="{stroke}" '
            f'stroke-width="{stroke_w}"/>'
        )
        # arrow: false is for symmetric relations (e.g. JOIN key pairs), no arrow is drawn
        if edge.get("arrow", True):
            parts.append(_arrow_head(end, direction, style, stroke))
    return parts


def render_node(node: Dict, style: Dict) -> List[str]:
    """All entities are rendered uniformly as rectangles: header + per-row attributes. Operations/clauses are not drawn as nodes but written on edge labels.

    Highlight support:
      node.highlight: true            - whole table outlined in red (thicker border + red header)
      node.highlight_rows: ["Total"]  - highlight specific attribute rows (light-red background + bold red text)
    """
    x, y, w, h = node["x"], node["y"], node["w"], node["h"]
    header_h = float(style["header_height"])
    row_h = float(style["row_height"])
    font = float(style["font_size"])
    font_family = style["font_family"]

    hl_color = node.get("highlight_color", style.get("highlight_color", "#e01b1b"))
    hl_fill = node.get("highlight_fill", style.get("highlight_fill", "#ffe9e9"))
    node_hl = bool(node.get("highlight"))
    stroke = hl_color if node_hl else style["stroke"]
    stroke_w = float(style.get("highlight_width", 2.6)) if node_hl else style["stroke_width"]

    header_fill = node.get("header_fill")
    if header_fill is None:
        header_fill = hl_color if node_hl else style.get("header_fill", "#111111")
    header_color = node.get("header_color", style.get("header_color", "#ffffff"))
    row_color = node.get("row_color", style.get("row_color", "#111111"))

    hl_rows = {str(r).strip().lower() for r in node.get("highlight_rows", [])}
    parts = []

    parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="#ffffff" stroke="{stroke}" stroke-width="{stroke_w}"/>'
    )
    parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{header_h:.1f}" '
        f'fill="{header_fill}" stroke="{stroke}" stroke-width="{stroke_w}"/>'
    )

    header_y = y + header_h / 2 + font * 0.35
    parts.append(
        f'<text x="{x + w / 2:.1f}" y="{header_y:.1f}" text-anchor="middle" '
        f'font-family="{font_family}" font-size="{font}" font-weight="bold" '
        f'fill="{header_color}">{escape(node.get("header", ""))}</text>'
    )

    for i, row in enumerate(node.get("rows", [])):
        top = y + header_h + i * row_h
        is_hl = str(row).strip().lower() in hl_rows
        if is_hl:
            parts.append(
                f'<rect x="{x:.1f}" y="{top:.1f}" width="{w:.1f}" height="{row_h:.1f}" '
                f'fill="{hl_fill}" stroke="none"/>'
            )
        if i > 0:
            parts.append(
                f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x + w:.1f}" y2="{top:.1f}" '
                f'stroke="{stroke}" stroke-width="1.1"/>'
            )
        text_y = top + row_h / 2 + font * 0.35
        weight = ' font-weight="bold"' if is_hl else ""
        parts.append(
            f'<text x="{x + w / 2:.1f}" y="{text_y:.1f}" text-anchor="middle" '
            f'font-family="{font_family}" font-size="{font}" '
            f'fill="{hl_color if is_hl else row_color}"{weight}>'
            f'{escape(row)}</text>'
        )
    return parts


def render_labels(nodes: Dict, style: Dict) -> List[str]:
    parts = []
    for node in nodes.values():
        label = node.get("label")
        if not label:
            continue
        x, y, w, h = wrapper_rect(node)
        parts.append(
            f'<text x="{x + w / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" '
            f'font-family="{style["font_family"]}" font-size="{style["label_size"]}" '
            f'font-weight="bold" fill="{style["label_color"]}">{escape(label)}</text>'
        )
    return parts


def render_svg(spec: Dict) -> str:
    laid = layout(spec)
    style = spec["style"]
    width, height = laid["width"], laid["height"]
    background = spec["canvas"].get("background", "#ffffff")

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{background}"/>',
    ]

    parts.extend(render_edges(laid, spec))
    for node in laid["nodes"].values():
        parts.extend(render_node(node, style))
    parts.extend(render_labels(laid["nodes"], style))
    # edge labels are drawn last, on top of tables, to avoid being occluded by the white table background
    parts.extend(render_edge_labels(laid, spec))
    parts.append("</svg>")
    return "\n".join(parts)


def render_diagram(
    overrides: Optional[Dict] = None,
    out_path: Optional[str] = None,
    table_overrides: Optional[Sequence[str]] = None,
) -> str:
    """Generate SVG; overrides only need to provide the fields to change."""
    spec = build_spec(overrides, table_overrides)
    svg = render_svg(spec)
    if out_path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(svg, encoding="utf-8")
    return svg


# ==================== Command-line entry point ====================
def main() -> None:
    parser = argparse.ArgumentParser(description="SQL intent visualisation metaphor-diagram renderer")
    parser.add_argument("--config", help="JSON config (recursively merged with defaults)", default=str(Path(__file__).parent / "default.json"))
    parser.add_argument("--out", default=str(Path(__file__).parent / "output" / "diagram.svg"),
                        help="output SVG path")
    parser.add_argument("--table", action="append", default=[],
                        help="dynamically modify a table, format 'node=header|row1,row2', repeatable")
    parser.add_argument("--write-default", help="write the default config to the specified JSON and exit")
    args = parser.parse_args()

    if args.write_default:
        path = Path(args.write_default)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(DEFAULT_SPEC, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print(f"[OK] default config written to: {path}")
        return

    overrides = None
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            overrides = json.load(f)

    svg = render_diagram(overrides, args.out, args.table)
    print(f"[OK] SVG generated: {args.out} ({len(svg)} bytes)")


if __name__ == "__main__":
    main()
