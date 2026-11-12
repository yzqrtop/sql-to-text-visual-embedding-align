#!/usr/bin/env python3
"""
Paper figure generator (generates all at once, output to tool/output/)

  1. fig_pipeline.svg        SQL -> AST -> graph-config JSON -> metaphor diagram conversion description
  2. fig_layout_lr           layout demo: left->right layering (default direction=lr)
  3. fig_layout_tb           layout demo: top->bottom layering (direction=tb)
  4. fig_layout_dag          layout demo: branch-merge DAG (multiple sources into a CTE then to result set)
  5. fig_highlight_row       highlight demo: attribute-row highlight (node.highlight_rows)
  6. fig_highlight_node      highlight demo: whole-table highlight (node.highlight)
  7. fig_highlight_path      highlight demo: cross-table trace-path highlight (edge.highlight + row highlight)

Except fig_pipeline, each example outputs two files:
    fig_<name>.svg      - metaphor diagram only
    fig_<name>_sql.svg  - SQL statement (top) + metaphor diagram (bottom)

Usage:
    python tool/paper_figures.py
    python tool/paper_figures.py --out tool/output
"""

import argparse
import copy
import re
from pathlib import Path
from typing import Dict, List, Sequence

from sql_metaphor_diagram import escape, render_diagram, text_width

FONT = "Arial, Helvetica, 'Microsoft YaHei', sans-serif"
MONO = "'Courier New', Courier, monospace"


# ==================== Basic primitives ====================

def box(x, y, w, h, fill="#ffffff", stroke="#111111", sw=1.4, rx=0):
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def text(x, y, s, size=14, anchor="middle", weight="", fill="#111111", family=FONT):
    w = f' font-weight="{weight}"' if weight else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="{family}" font-size="{size}" fill="{fill}"{w}>'
        f'{escape(s)}</text>'
    )


def arrow_h(x1, y, x2, color="#111111", sw=1.8):
    """Horizontal arrow (left->right)."""
    return (
        f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2 - 9:.1f}" y2="{y:.1f}" '
        f'stroke="{color}" stroke-width="{sw}"/>'
        f'<polygon points="{x2:.1f},{y:.1f} {x2 - 9:.1f},{y - 4.5:.1f} '
        f'{x2 - 9:.1f},{y + 4.5:.1f}" fill="{color}"/>'
    )


def elbow(x1, y1, x2, y2, color="#111111", sw=1.3):
    """Right-angle polyline: vertical then horizontal then vertical (tree connection)."""
    mid = (y1 + y2) / 2
    return (
        f'<path d="M {x1:.1f},{y1:.1f} L {x1:.1f},{mid:.1f} '
        f'L {x2:.1f},{mid:.1f} L {x2:.1f},{y2:.1f}" fill="none" '
        f'stroke="{color}" stroke-width="{sw}"/>'
    )


# ==================== 1. Conversion pipeline description ====================

# the "real rendered result" to embed at the end of the pipeline (corresponds to panel 1's SQL)
PIPELINE_MINI_CFG = {
    "layout": {"mode": "layered", "gap_x": 26, "gap_y": 16},
    "style": {
        "font_size": 11, "header_height": 17, "row_height": 18,
        "min_width": 64, "text_padding": 7, "label_size": 11, "label_offset": 9,
    },
    "nodes": {
        "artists": {"kind": "table", "header": "artists a",
                    "rows": ["ArtistId", "Name"]},
        "result": {"kind": "table", "header": "OUTPUT", "rows": ["Name"],
                   "highlight": True, "header_fill": "#cccccc",
                   "header_color": "#666666"},
    },
    "edges": [
        {"from": "artists", "to": "result", "kind": "line",
         "from_attr": "Name", "to_attr": "Name"},
        {"from": "artists", "to": "result", "kind": "line",
         "label": "WHERE ArtistId = 1"},
    ],
}

SQL_LINES = [
    "SELECT a.Name",
    "FROM artists a",
    "WHERE a.ArtistId = 1;",
]

JSON_LINES = [
    '"nodes": {',
    '  "artists": {',
    '    "header": "artists a",',
    '    "rows": ["ArtistId", "Name"]},',
    '  "result": {',
    '    "header": "OUTPUT",',
    '    "rows": ["Name"] } },',
    '"edges": [',
    '  {"from": "artists", "to": "result",',
    '   "from_attr": "Name",',
    '   "to_attr": "Name"},',
    '  {"from": "artists", "to": "result",',
    '   "label": "WHERE ArtistId = 1"} ]',
]


def _embed_svg(svg: str, x: float, y: float):
    """Embed the full SVG produced by render_diagram as a nested <svg>, return (fragment, w, h)."""
    m = re.search(r'<svg[^>]*width="([\d.]+)"[^>]*height="([\d.]+)"[^>]*>', svg)
    w, h = float(m.group(1)), float(m.group(2))
    inner = svg[m.end():svg.rindex("</svg>")]
    return (
        f'<svg x="{x:.1f}" y="{y:.1f}" width="{w:.0f}" height="{h:.0f}" '
        f'viewBox="0 0 {w:.0f} {h:.0f}">{inner}</svg>'
    ), w, h


def compose_with_sql(sql_lines: Sequence[str], svg: str) -> str:
    """Compose an SQL code block (top) + metaphor diagram (bottom) into one SVG (with-statement figure)."""
    size = 14.0
    line_h = 23.0
    pad_x, pad_y = 16.0, 15.0
    margin = 24.0

    _, w, h = _embed_svg(svg, 0, 0)
    code_w = max(text_width(t, size) for t in sql_lines) + 2 * pad_x
    code_h = line_h * len(sql_lines) + 2 * pad_y
    inner_w = w
    code_left = margin + max(0.0, (inner_w - code_w) / 2)
    width = w + 2 * margin
    sql_top = margin + 24.0
    dia_top = sql_top + code_h + 26.0
    height = dia_top + h + margin

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
        f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" '
        f'fill="#ffffff"/>',
        text(code_left, margin + 12, "SQL", size=15, weight="bold",
             anchor="start", family=MONO),
        box(code_left, sql_top, code_w, code_h, fill="#f7f7f7",
            stroke="#999999", sw=1.2, rx=4),
    ]
    for i, line in enumerate(sql_lines):
        parts.append(
            text(code_left + pad_x, sql_top + pad_y + size * 0.92 + i * line_h,
                 line, size=size, anchor="start", family=MONO)
        )
    embed, _, _ = _embed_svg(svg, (width - w) / 2, dia_top)
    parts.append(embed)
    parts.append("</svg>")
    return "\n".join(parts)


def fig_pipeline(out_dir: Path) -> Path:
    mini_svg = render_diagram(PIPELINE_MINI_CFG)
    embed, mini_w, mini_h = _embed_svg(mini_svg, 0, 0)  # get dimensions first

    # four panels: SQL / AST / JSON / metaphor diagram
    pw = [310, 330, 372, max(330, mini_w + 40)]
    gap = 74                      # inter-panel arrow zone width
    x0, top = 30, 66
    ph = max(300, mini_h + 96)
    xs = [x0]
    for i in range(3):
        xs.append(xs[-1] + pw[i] + gap)
    width = xs[-1] + pw[3] + 30
    height = top + ph + 56

    titles = ["(1) SQL statement", "(2) Abstract Syntax Tree (AST)", "(3) Graph config JSON", "(4) Metaphor diagram"]
    steps = ["Lexing + parsing", "Rule mapping", "Layered layout + rendering"]
    substeps = ["tokenize * recursive descent (R1-R2)",
                "entities R3-R10 * edges R11-R21",
                "sql_metaphor_diagram.py（R22–R26）"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
        f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" fill="#ffffff"/>',
        text(width / 2, 34, "SQL -> AST -> graph config -> metaphor diagram conversion pipeline",
             size=19, weight="bold"),
    ]

    for i in range(4):
        parts.append(box(xs[i], top, pw[i], ph, fill="#fafafa", stroke="#999999",
                         sw=1.2, rx=8))
        parts.append(text(xs[i] + pw[i] / 2, top + 28, titles[i], size=16,
                          weight="bold"))

    for i in range(3):
        ax1, ax2 = xs[i] + pw[i] + 10, xs[i + 1] - 10
        ay = top + ph / 2 - 16
        parts.append(arrow_h(ax1, ay, ax2, color="#555555", sw=2.0))
        parts.append(text((ax1 + ax2) / 2, ay - 34, steps[i], size=13,
                          weight="bold", fill="#333333"))
        parts.append(text((ax1 + ax2) / 2, ay + 26, substeps[i], size=10.5,
                          fill="#666666"))

    # --- panel 1: original SQL ---
    bx, by = xs[0] + 18, top + 62
    bw = pw[0] - 36
    parts.append(box(bx, by, bw, 24 * len(SQL_LINES) + 24, fill="#ffffff",
                     stroke="#111111", sw=1.4, rx=4))
    for i, line in enumerate(SQL_LINES):
        parts.append(text(bx + 14, by + 34 + i * 24, line, size=13.5,
                          anchor="start", family=MONO))
    parts.append(text(xs[0] + pw[0] / 2, by + 24 * len(SQL_LINES) + 52,
                      "input: a syntactically valid SQL query text", size=12, fill="#666666"))

    # --- panel 2: AST tree ---
    px = xs[1]
    cx = px + pw[1] / 2
    ty = top + 66
    parts.append(box(cx - 52, ty, 104, 26, rx=4))
    parts.append(text(cx, ty + 17.5, "OUTPUT", size=12.5, family=MONO))
    kids = [("OUTPUT_list", px + 20), ("from", px + 118), ("where", px + 216)]
    ky = ty + 74
    for label, kx in kids:
        parts.append(elbow(cx, ty + 26, kx + 47, ky))
        parts.append(box(kx, ky, 94, 26, rx=4))
        parts.append(text(kx + 47, ky + 17.5, label, size=12, family=MONO))
    parts.append(text(px + 67, ky + 52, "Name", size=11.5, family=MONO,
                      fill="#333333"))
    parts.append(text(px + 165, ky + 52, "artists a", size=11.5, family=MONO,
                      fill="#333333"))
    wy = ky + 74
    wx = px + 216 + 47
    parts.append(elbow(wx, ky + 26, wx, wy))
    parts.append(box(wx - 40, wy, 80, 24, rx=4))
    parts.append(text(wx, wy + 16.5, "=", size=12.5, family=MONO))
    parts.append(elbow(wx, wy + 24, wx - 52, wy + 62))
    parts.append(elbow(wx, wy + 24, wx + 52, wy + 62))
    parts.append(box(wx - 112, wy + 62, 120, 24, rx=4))
    parts.append(text(wx - 52, wy + 78.5, "col a.ArtistId", size=11,
                      family=MONO))
    parts.append(box(wx + 12, wy + 62, 80, 24, rx=4))
    parts.append(text(wx + 52, wy + 78.5, "literal 1", size=11, family=MONO))

    # --- panel 3: JSON config ---
    bx, by = xs[2] + 16, top + 58
    bw = pw[2] - 32
    parts.append(box(bx, by, bw, 16.5 * len(JSON_LINES) + 18, fill="#ffffff",
                     stroke="#111111", sw=1.4, rx=4))
    for i, line in enumerate(JSON_LINES):
        parts.append(text(bx + 10, by + 24 + i * 16.5, line, size=10.5,
                          anchor="start", family=MONO))
    parts.append(text(xs[2] + pw[2] / 2, by + 16.5 * len(JSON_LINES) + 44,
                      "nodes = relational entities; edges = clauses and operations", size=12,
                      fill="#666666"))

    # --- panel 4: real rendered metaphor diagram ---
    ex = xs[3] + (pw[3] - mini_w) / 2
    ey = top + 62
    parts.append(_embed_svg(mini_svg, ex, ey)[0])
    parts.append(text(xs[3] + pw[3] / 2, ey + mini_h + 28,
                      "output: metaphor diagram (rectangles = entities, arrows = clauses/operations)", size=12,
                      fill="#666666"))

    parts.append("</svg>")
    out = out_dir / "fig_pipeline.svg"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


# ==================== 2/3. Example figures (layout + highlight) ====================
# each example corresponds to one SQL, used to generate the *_sql.svg (SQL + diagram) version

SQL_LR = [
    "OUTPUT a.Name, s.Total",
    "FROM artists a",
    "JOIN ArtistSales s ON a.ArtistId = s.ArtistId",
    "GROUP BY a.ArtistId",
    "ORDER BY SUM(s.Total) DESC",
    "LIMIT 1;",
]

SQL_DAG = [
    "WITH CustOrders AS (",
    "  OUTPUT o.CustomerId, c.Name, o.Total",
    "  FROM orders o",
    "  JOIN customers c ON o.CustomerId = c.CustomerId",
    ")",
    "OUTPUT Name, Total",
    "FROM CustOrders;",
]

SQL_ROW = [
    "OUTPUT SUM(Total) AS Total",
    "FROM invoices i;",
]

SQL_NODE = [
    "WITH BigOrders AS (",
    "  OUTPUT CustomerId, Total",
    "  FROM orders o",
    "  WHERE Total > 100",
    ")",
    "OUTPUT CustomerId, Total",
    "FROM BigOrders;",
]

_LR_NODES = {
    "artists": {"kind": "table", "header": "artists a",
                "rows": ["ArtistId", "Name"]},
    "sales": {"kind": "table", "header": "ArtistSales",
              "rows": ["ArtistId", "Total"]},
    "result": {"kind": "table", "header": "OUTPUT", "rows": ["Name", "Total"],
               "header_fill": "#cccccc", "header_color": "#666666"},
}
_LR_EDGES = [
    {"from": "artists", "to": "sales", "kind": "curve", "no_rank": True,
     "arrow": False, "from_side": "right", "to_side": "right",
     "from_attr": "ArtistId", "to_attr": "ArtistId", "bend": [30, 0]},
    {"from": "artists", "to": "result", "kind": "line",
     "from_attr": "Name", "to_attr": "Name"},
    {"from": "sales", "to": "result", "kind": "line",
     "from_attr": "Total", "to_attr": "Total",
     "label": "GROUP BY ArtistId; ORDER BY SUM(Total) DESC; LIMIT 1"},
]


def emit_figure(stem: str, config: Dict, sql_lines: Sequence[str],
                out_dir: Path) -> List[Path]:
    """Output two files from the same config: fig_<stem>.svg (diagram only) / fig_<stem>_sql.svg (SQL + diagram)."""
    svg = render_diagram(copy.deepcopy(config))
    outs = []
    out_plain = out_dir / f"{stem}.svg"
    out_plain.write_text(svg, encoding="utf-8")
    outs.append(out_plain)
    if sql_lines:
        out_sql = out_dir / f"{stem}_sql.svg"
        out_sql.write_text(compose_with_sql(sql_lines, svg), encoding="utf-8")
        outs.append(out_sql)
    return outs


def fig_layout_lr(out_dir: Path) -> List[Path]:
    return emit_figure(
        "fig_layout_lr",
        {"layout": {"mode": "layered", "gap_x": 46, "gap_y": 26},
         "nodes": _LR_NODES, "edges": _LR_EDGES},
        SQL_LR, out_dir,
    )


def fig_layout_tb(out_dir: Path) -> List[Path]:
    # in tb mode same-row connections are top->top; bend's vertical component is negative = arches upward, avoiding the header
    tb_edges = [dict(e) for e in _LR_EDGES]
    tb_edges[0]["bend"] = [0, -30]
    return emit_figure(
        "fig_layout_tb",
        {"layout": {"mode": "layered", "direction": "tb",
                    "gap_x": 46, "gap_y": 30},
         "nodes": _LR_NODES, "edges": tb_edges},
        SQL_LR, out_dir,
    )


def fig_layout_dag(out_dir: Path) -> List[Path]:
    """Branch merge: two base tables flow into a CTE, then to the result set."""
    return emit_figure("fig_layout_dag", {
        "layout": {"mode": "layered", "gap_x": 46, "gap_y": 30},
        "nodes": {
            "orders": {"kind": "table", "header": "orders o",
                       "rows": ["OrderId", "CustomerId", "Total"]},
            "customers": {"kind": "table", "header": "customers c",
                          "rows": ["CustomerId", "Name"]},
            "L1": {"kind": "table", "header": "CustOrders",
                   "rows": ["CustomerId", "Name", "Total"]},
            "result": {"kind": "table", "header": "OUTPUT",
                       "rows": ["Name", "Total"],
                       "header_fill": "#cccccc", "header_color": "#666666"},
        },
        "edges": [
            {"from": "orders", "to": "customers", "kind": "curve",
             "no_rank": True, "arrow": False, "from_side": "right",
             "to_side": "right", "from_attr": "CustomerId",
             "to_attr": "CustomerId", "bend": [34, 0]},
            {"from": "orders", "to": "L1", "kind": "line",
             "label": "FROM orders o"},
            {"from": "customers", "to": "L1", "kind": "line",
             "label": "JOIN customers c ON CustomerId"},
            {"from": "L1", "to": "result", "kind": "line",
             "from_attr": "Name", "to_attr": "Name"},
            {"from": "L1", "to": "result", "kind": "line",
             "from_attr": "Total", "to_attr": "Total"},
        ],
    }, SQL_DAG, out_dir)


# ==================== 3. Highlight demos ====================

def fig_highlight_row(out_dir: Path) -> List[Path]:
    return emit_figure("fig_highlight_row", {
        "layout": {"mode": "layered", "gap_x": 46, "gap_y": 26},
        "nodes": {
            "invoices": {"kind": "table", "header": "invoices i",
                         "rows": ["InvoiceId", "CustomerId", "Total"],
                         "highlight_rows": ["Total"]},
            "result": {"kind": "table", "header": "OUTPUT", "rows": ["Total"],
                       "header_fill": "#cccccc", "header_color": "#666666"},
        },
        "edges": [
            {"from": "invoices", "to": "result", "kind": "line",
             "from_attr": "Total", "to_attr": "Total",
             "label": "SUM(Total) AS Total"},
        ],
    }, SQL_ROW, out_dir)


def fig_highlight_node(out_dir: Path) -> List[Path]:
    return emit_figure("fig_highlight_node", {
        "layout": {"mode": "layered", "gap_x": 46, "gap_y": 26},
        "nodes": {
            "orders": {"kind": "table", "header": "orders o",
                       "rows": ["OrderId", "CustomerId", "Total"]},
            "L1": {"kind": "table", "header": "BigOrders",
                   "rows": ["CustomerId", "Total"], "highlight": True},
            "result": {"kind": "table", "header": "OUTPUT",
                       "rows": ["CustomerId", "Total"],
                       "header_fill": "#cccccc", "header_color": "#666666"},
        },
        "edges": [
            {"from": "orders", "to": "L1", "kind": "line",
             "label": "WHERE Total > 100"},
            {"from": "L1", "to": "result", "kind": "line",
             "from_attr": "CustomerId", "to_attr": "CustomerId"},
            {"from": "L1", "to": "result", "kind": "line",
             "from_attr": "Total", "to_attr": "Total"},
        ],
    }, SQL_NODE, out_dir)


def fig_highlight_path(out_dir: Path) -> List[Path]:
    """Trace-path highlight: result.Name <- L1.Name <- customers.Name."""
    return emit_figure("fig_highlight_path", {
        "layout": {"mode": "layered", "gap_x": 46, "gap_y": 30},
        "nodes": {
            "orders": {"kind": "table", "header": "orders o",
                       "rows": ["OrderId", "CustomerId", "Total"]},
            "customers": {"kind": "table", "header": "customers c",
                          "rows": ["CustomerId", "Name"],
                          "highlight_rows": ["Name"]},
            "L1": {"kind": "table", "header": "CustOrders",
                   "rows": ["CustomerId", "Name", "Total"],
                   "highlight_rows": ["Name"]},
            "result": {"kind": "table", "header": "OUTPUT",
                       "rows": ["Name", "Total"], "highlight_rows": ["Name"],
                       "header_fill": "#cccccc", "header_color": "#666666"},
        },
        "edges": [
            {"from": "orders", "to": "L1", "kind": "line",
             "label": "FROM orders o", "label_hidden": True},
            {"from": "customers", "to": "L1", "kind": "line",
             "label": "JOIN customers c ON CustomerId", "label_hidden": True},
            {"from": "customers", "to": "L1", "kind": "line",
             "from_attr": "Name", "to_attr": "Name", "highlight": True,
             "label": "OUTPUT c.Name"},
            {"from": "L1", "to": "result", "kind": "line",
             "from_attr": "Name", "to_attr": "Name", "highlight": True,
             "label": "OUTPUT Name"},
            {"from": "L1", "to": "result", "kind": "line",
             "from_attr": "Total", "to_attr": "Total", "label_hidden": True},
        ],
    }, SQL_DAG, out_dir)


# ==================== 4. Multiple layouts / highlights of the same SQL ====================

# one SQL covering CTE + multi-table JOIN + GROUP BY + ORDER BY + LIMIT,
# both the layout and highlight figures below use it as the sole input.
SQL_ONE = [
    "WITH TopSales AS (",
    "  OUTPUT ar.Name, i.ArtistId, SUM(i.Total) AS Total",
    "  FROM invoice_items ii",
    "  JOIN invoices i ON ii.InvoiceId = i.InvoiceId",
    "  JOIN artists ar ON i.ArtistId = ar.ArtistId",
    "  WHERE ii.TrackId = 1",
    "  GROUP BY i.ArtistId, ar.Name",
    ")",
    "OUTPUT Name, Total",
    "FROM TopSales",
    "ORDER BY Total DESC",
    "LIMIT 1;",
]


def _one_sql_config(direction: str = "lr", key_arcs: bool = True) -> Dict:
    """Graph config for SQL_ONE. When key_arcs=False the key-pair arcs are omitted and replaced with plain JOIN annotations (branch-merge form)."""
    nodes = {
        "invoice_items": {"kind": "table", "header": "invoice_items ii",
                          "rows": ["InvoiceId", "TrackId"]},
        "invoices": {"kind": "table", "header": "invoices i",
                     "rows": ["InvoiceId", "ArtistId", "Total"]},
        "artists": {"kind": "table", "header": "artists ar",
                    "rows": ["ArtistId", "Name"]},
        "L1": {"kind": "table", "header": "TopSales",
               "rows": ["Name", "ArtistId", "Total"]},
        "result": {"kind": "table", "header": "OUTPUT",
                   "rows": ["Name", "Total"], "header_fill": "#cccccc",
                   "header_color": "#666666"},
    }
    edges = [
        {"from": "invoice_items", "to": "L1", "kind": "line",
         "label": "FROM invoice_items ii; WHERE TrackId = 1"},
        {"from": "invoices", "to": "L1", "kind": "line",
         "label": "JOIN invoices i ON InvoiceId; GROUP BY ArtistId"},
        {"from": "artists", "to": "L1", "kind": "line",
         "label": "JOIN artists ar ON ArtistId"},
        {"from": "artists", "to": "L1", "kind": "line",
         "from_attr": "Name", "to_attr": "Name"},
        {"from": "L1", "to": "result", "kind": "line",
         "from_attr": "Name", "to_attr": "Name"},
        {"from": "L1", "to": "result", "kind": "line",
         "from_attr": "Total", "to_attr": "Total",
         "label": "ORDER BY Total DESC; LIMIT 1"},
    ]
    if key_arcs:
        bend = [0, 40] if direction == "tb" else [34, 0]
        edges.insert(0, {"from": "invoice_items", "to": "invoices",
                         "kind": "curve", "no_rank": True, "arrow": False,
                         "from_attr": "InvoiceId", "to_attr": "InvoiceId",
                         "bend": bend})
        edges.insert(1, {"from": "invoices", "to": "artists",
                         "kind": "curve", "no_rank": True, "arrow": False,
                         "from_attr": "ArtistId", "to_attr": "ArtistId",
                         "bend": bend})

    layout = {"mode": "layered", "gap_x": 46, "gap_y": 26}
    if direction == "tb":
        layout.update({"direction": "tb", "gap_y": 30})
    return {"layout": layout, "nodes": nodes, "edges": edges}


def compose_document(sql_lines: Sequence[str], panels: Sequence,
                     note: str = "") -> str:
    """Compose multiple diagrams side by side into one SVG: optional SQL code block (top) + panel titles + diagrams

    panels: [(title, svg string)], title may be empty
    """
    margin, gap, cap_h = 28.0, 34.0, 26.0
    size, line_h, pad_x, pad_y = 13.5, 22.0, 15.0, 14.0

    prepared = []
    for cap, svg in panels:
        _, w, h = _embed_svg(svg, 0, 0)
        prepared.append((cap, svg, w, h))

    panels_w = sum(p[2] for p in prepared) + gap * (len(prepared) - 1)
    code_w = code_h = 0.0
    if sql_lines:
        code_w = max(text_width(t, size) for t in sql_lines) + 2 * pad_x
        code_h = line_h * len(sql_lines) + 2 * pad_y
    content_w = max(panels_w, code_w)

    y = margin
    head = []
    if sql_lines:
        code_x = margin + (content_w - code_w) / 2
        head.append(text(code_x, y + 12, "SQL", size=15, weight="bold",
                         anchor="start", family=MONO))
        head.append(box(code_x, y + 20, code_w, code_h, fill="#f7f7f7",
                        stroke="#999999", sw=1.2, rx=4))
        base = y + 20 + pad_y + size * 0.92
        for i, ln in enumerate(sql_lines):
            head.append(text(code_x + pad_x, base + i * line_h, ln,
                             size=size, anchor="start", family=MONO))
        y += 20 + code_h + 32

    panel_h = max(p[3] for p in prepared) + cap_h
    width = content_w + 2 * margin
    height = y + panel_h + (30 if note else 0) + margin

    body = []
    x = margin + (content_w - panels_w) / 2
    for cap, svg, w, h in prepared:
        if cap:
            body.append(text(x + w / 2, y + 16, cap, size=14.5, weight="bold"))
        frag, _, _ = _embed_svg(svg, x, y + cap_h)
        body.append(frag)
        x += w + gap
    if note:
        body.append(text(width / 2, y + panel_h + 22, note, size=12,
                         fill="#666666"))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
        f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" '
        f'fill="#ffffff"/>',
    ]
    parts.extend(head)
    parts.extend(body)
    parts.append("</svg>")
    return "\n".join(parts)


def _emit_panels(stem: str, panels: Sequence, sql_lines: Sequence[str],
                 note: str, out_dir: Path) -> List[Path]:
    outs = []
    plain = out_dir / f"{stem}.svg"
    plain.write_text(compose_document([], panels, note), encoding="utf-8")
    outs.append(plain)
    out_sql = out_dir / f"{stem}_sql.svg"
    out_sql.write_text(compose_document(sql_lines, panels, note),
                       encoding="utf-8")
    outs.append(out_sql)
    return outs


def fig_all_layouts(out_dir: Path) -> List[Path]:
    """Three layouts of the same SQL, side by side in one diagram."""
    panels = [
        ("(a) horizontal layering direction = lr",
         render_diagram(_one_sql_config("lr", key_arcs=True))),
        ("(b) vertical layering direction = tb",
         render_diagram(_one_sql_config("tb", key_arcs=True))),
        ("(c) simplified layout, key-pair arcs omitted",
         render_diagram(_one_sql_config("lr", key_arcs=False))),
    ]
    return _emit_panels("fig_all_layouts", panels, SQL_ONE,
                        "Three layouts of the same SQL", out_dir)


def fig_all_highlights(out_dir: Path) -> List[Path]:
    """Three highlight styles of the same SQL, side by side in one diagram."""
    row_cfg = _one_sql_config()
    row_cfg["nodes"]["invoices"]["highlight_rows"] = ["Total"]
    node_cfg = _one_sql_config()
    node_cfg["nodes"]["L1"]["highlight"] = True
    path_cfg = _one_sql_config()
    for nid in ("artists", "L1", "result"):
        path_cfg["nodes"][nid]["highlight_rows"] = ["Name"]
    for e in path_cfg["edges"]:
        if e.get("from_attr") == "Name":
            # annotations on the path are kept; the rest only occupy space and are not drawn
            e["highlight"] = True
            e["label"] = ("OUTPUT ar.Name" if e.get("from") == "artists"
                          else "OUTPUT Name")
        elif e.get("label"):
            e["label_hidden"] = True

    panels = [
        ("(a) attribute-row highlight highlight_rows", render_diagram(row_cfg)),
        ("(b) whole-table highlight highlight", render_diagram(node_cfg)),
        ("(c) trace-path highlight edge.highlight", render_diagram(path_cfg)),
    ]
    return _emit_panels("fig_all_highlights", panels, SQL_ONE,
                        "Three highlight styles of the same SQL", out_dir)


# ==================== 5. BigOrders example (SQL for fig_highlight_node) ============

def _node_config(direction: str = "lr", folded: bool = False) -> Dict:
    """Graph config for SQL_NODE (BigOrders CTE); when folded=True the CTE is folded and orders connect directly to the result set."""
    result_node = {"kind": "table", "header": "OUTPUT",
                   "rows": ["CustomerId", "Total"],
                   "header_fill": "#cccccc", "header_color": "#666666"}
    orders_node = {"kind": "table", "header": "orders o",
                   "rows": ["OrderId", "CustomerId", "Total"]}
    if folded:
        return {
            "layout": {"mode": "layered", "gap_x": 46, "gap_y": 26},
            "nodes": {"orders": orders_node, "result": result_node},
            "edges": [
                {"from": "orders", "to": "result", "kind": "line",
                 "from_attr": "CustomerId", "to_attr": "CustomerId"},
                {"from": "orders", "to": "result", "kind": "line",
                 "from_attr": "Total", "to_attr": "Total",
                 "label": "WHERE Total > 100"},
            ],
        }
    layout = {"mode": "layered", "gap_x": 46, "gap_y": 26}
    if direction == "tb":
        layout = {"mode": "layered", "direction": "tb",
                  "gap_x": 46, "gap_y": 30}
    return {
        "layout": layout,
        "nodes": {
            "orders": orders_node,
            "L1": {"kind": "table", "header": "BigOrders",
                   "rows": ["CustomerId", "Total"]},
            "result": result_node,
        },
        "edges": [
            {"from": "orders", "to": "L1", "kind": "line",
             "from_attr": "CustomerId", "to_attr": "CustomerId"},
            {"from": "orders", "to": "L1", "kind": "line",
             "from_attr": "Total", "to_attr": "Total"},
            {"from": "orders", "to": "L1", "kind": "line",
             "label": "WHERE Total > 100"},
            {"from": "L1", "to": "result", "kind": "line",
             "from_attr": "CustomerId", "to_attr": "CustomerId"},
            {"from": "L1", "to": "result", "kind": "line",
             "from_attr": "Total", "to_attr": "Total"},
        ],
    }


def fig_all_layouts_node(out_dir: Path) -> List[Path]:
    """Three layouts of the BigOrders example: horizontal / vertical / folded CTE."""
    panels = [
        ("(a) horizontal layering direction = lr",
         render_diagram(_node_config("lr"))),
        ("(b) vertical layering direction = tb",
         render_diagram(_node_config("tb"))),
        ("(c) folded intermediate-result CTE, not expanded",
         render_diagram(_node_config("lr", folded=True))),
    ]
    return _emit_panels("fig_all_layouts_node", panels, SQL_NODE,
                        "Three layouts of the same SQL", out_dir)


def fig_all_highlights_node(out_dir: Path) -> List[Path]:
    """Three highlights of the BigOrders example: attribute row / whole table / trace path."""
    row_cfg = _node_config()
    row_cfg["nodes"]["orders"]["highlight_rows"] = ["Total"]

    node_cfg = _node_config()
    node_cfg["nodes"]["L1"]["highlight"] = True

    path_cfg = _node_config()
    for nid in ("orders", "L1", "result"):
        path_cfg["nodes"][nid]["highlight_rows"] = ["CustomerId"]
    for e in path_cfg["edges"]:
        if e.get("from_attr") == "CustomerId":
            # annotations on the path are kept; the rest only occupy space and are not drawn
            e["highlight"] = True
            # e["label"] = ("OUTPUT o.CustomerId" if e.get("from") == "orders"
            #               else "OUTPUT CustomerId")
        elif e.get("label"):
            e["label_hidden"] = True

    panels = [
        ("(a) attribute-row highlight highlight_rows", render_diagram(row_cfg)),
        ("(b) whole-table highlight highlight", render_diagram(node_cfg)),
        ("(c) trace-path highlight edge.highlight", render_diagram(path_cfg)),
    ]
    return _emit_panels("fig_all_highlights_node", panels, SQL_NODE,
                        "Three highlight styles of the same SQL", out_dir)


# ==================== Entry point ====================

def main() -> None:
    parser = argparse.ArgumentParser(description="Paper figure generator")
    parser.add_argument("--out", default=str(Path(__file__).parent / "output"))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    figs = [fig_pipeline(out_dir)]
    figs.extend(fig_layout_lr(out_dir))
    figs.extend(fig_layout_tb(out_dir))
    figs.extend(fig_layout_dag(out_dir))
    figs.extend(fig_highlight_row(out_dir))
    figs.extend(fig_highlight_node(out_dir))
    figs.extend(fig_highlight_path(out_dir))
    figs.extend(fig_all_layouts(out_dir))
    figs.extend(fig_all_highlights(out_dir))
    figs.extend(fig_all_layouts_node(out_dir))
    figs.extend(fig_all_highlights_node(out_dir))
    for f in figs:
        print(f"[OK] {f}")


if __name__ == "__main__":
    main()
