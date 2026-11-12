"""Training entry point (paper Sections: Triplet Construction and Joint
Optimization Objective).

Reads (q, s_gt, s_bug) triples from JSON files, builds the gold/bug JSON graphs
with Module 1, aligns them via Graph Edit Distance to obtain dense element
labels, and optimizes L_BCE + lambda * L_contrast.

Triple file format (one JSON object per file, see data/triples/):
    {
      "question":   "natural-language question q",
      "sql_gt":     "gold SQL s_gt",
      "sql_bug":    "buggy SQL s_bug collected from real LLM outputs",
      "error_type": "optional error category tag"
    }

Example:
    python scripts/train_aligner.py --triples-dir data/triples \
        --epochs 10 --out checkpoints/aligner.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tool"))

from sql_to_ast_config import build_config  # noqa: E402
from src import config  # noqa: E402
from src.data.graph_alignment import align_graphs, label_elements  # noqa: E402
from src.data.graph_features import RELATION_TYPES  # noqa: E402

Sample = Tuple[str, Dict, Dict[str, int]]


def load_triples(triples_dir: Path) -> List[Dict]:
    """Load every triple JSON file under ``triples_dir``."""
    triples = []
    for path in sorted(triples_dir.glob("*.json")):
        triple = json.loads(path.read_text(encoding="utf-8"))
        for key in ("question", "sql_gt", "sql_bug"):
            if key not in triple:
                raise ValueError("%s is missing required key %r" % (path, key))
        triples.append(triple)
    if not triples:
        raise ValueError("no triple JSON files found in %s" % triples_dir)
    return triples


def _to_element_labels(labels: Dict[str, int]) -> Dict[str, int]:
    """Map GED label ids (``e{i}`` for edges) to graph-feature ids
    (``__edge_{i}``) so they match ``build_graph_tensors`` element ids."""
    converted = {}
    for eid, y in labels.items():
        if eid.startswith("e") and eid[1:].isdigit():
            converted["__edge_%s" % eid[1:]] = y
        else:
            converted[eid] = y
    return converted


def prepare_samples(triples: List[Dict]) -> List[Sample]:
    """Build (question, json_graph, element_labels) samples from triples.

    Gold graphs contribute all-positive samples; bug graphs contribute dense
    labels where inserted/substituted elements (GED edit path) are negatives.
    """
    samples: List[Sample] = []
    for triple in triples:
        _, j_gt = build_config(triple["sql_gt"])
        _, j_bug = build_config(triple["sql_bug"])
        _, _, edit_path = align_graphs(j_gt, j_bug, timeout=config.GED_TIMEOUT_SECONDS)
        gt_labels, bug_labels = label_elements(j_gt, j_bug, edit_path)
        samples.append((triple["question"], j_gt, gt_labels))
        samples.append((triple["question"], j_bug, _to_element_labels(bug_labels)))
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--triples-dir", default="data/triples",
                        help="directory containing triple JSON files")
    parser.add_argument("--epochs", type=int, default=config.TRAIN_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--text-model", default=None,
                        help="override the HuggingFace text model name")
    parser.add_argument("--out", default="checkpoints/aligner.pt",
                        help="checkpoint output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    triples = load_triples(Path(args.triples_dir))
    samples = prepare_samples(triples)
    print("Prepared %d samples from %d triples." % (len(samples), len(triples)))

    from src.models.aligner import AlignmentScorer
    from src.training.trainer import AlignmentTrainer

    model = AlignmentScorer(num_relations=len(RELATION_TYPES), text_model_name=args.text_model)
    trainer = AlignmentTrainer(model, lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        loss = trainer.train_epoch(samples)
        print("epoch %d/%d  loss=%.4f" % (epoch, args.epochs, loss))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(str(out_path))
    print("Checkpoint saved to %s" % out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
