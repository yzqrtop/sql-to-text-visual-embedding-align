"""Inference entry point (paper Section: Element Scoring at Inference).

Runs one forward pass of the alignment scorer over every graph element of a
generated SQL and writes the score-annotated JSON graph:

    S_u = 1 - P_align(u);  elements with S_u >= tau are flagged for highlight.

Example:
    python scripts/score_sql.py --sql-file tool/examples/artist_pipeline.sql \
        --question "How many albums does each American artist have?" \
        --out results/scored_graph.json

The scored JSON can be rendered directly with Module 3:
    python tool/sql_metaphor_diagram.py --config results/scored_graph.json --out diagram.svg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tool"))

from sql_to_ast_config import build_config  # noqa: E402
from src.data.graph_features import (  # noqa: E402
    RELATION_TYPES,
    apply_scores_to_graph,
    build_graph_tensors,
)
from src import config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sql-file", default="tool/examples/easy_example.sql",
                        help="path to the generated SQL file")
    parser.add_argument("--sql", default=None,
                        help="inline SQL text (overrides --sql-file when given)")
    parser.add_argument("--question", required=True,
                        help="natural-language question q")
    parser.add_argument("--checkpoint", default=None,
                        help="optional trained checkpoint (.pt) for the scorer")
    parser.add_argument("--text-model", default=None,
                        help="override the HuggingFace text model name")
    parser.add_argument("--out", default="results/scored_graph.json",
                        help="output path for the score-annotated JSON graph")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sql = args.sql if args.sql is not None else Path(args.sql_file).read_text(encoding="utf-8")

    # Module 1: SQL -> JSON graph.
    _, json_graph = build_config(sql)

    # Module 2: one forward pass -> element-level semantic-gap scores.
    import torch  # deferred: keeps --help fast

    from src.models.aligner import AlignmentScorer

    node_feats, edge_index, edge_type, element_ids = build_graph_tensors(json_graph)
    model = AlignmentScorer(num_relations=len(RELATION_TYPES), text_model_name=args.text_model)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False))
    model.eval()
    scores = model.predict_scores(args.question, node_feats, edge_index, edge_type)
    json_graph = apply_scores_to_graph(json_graph, scores, element_ids)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(json_graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    highlighted = [eid for eid, s in zip(element_ids, scores.tolist())
                   if s >= config.HIGHLIGHT_THRESHOLD]
    print("Scored %d elements -> %s" % (len(element_ids), out_path))
    print("High semantic gap (S_u >= %.2f): %s"
          % (config.HIGHLIGHT_THRESHOLD, highlighted or "none"))
    print("Render with: python tool/sql_metaphor_diagram.py --config %s --out diagram.svg" % out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
