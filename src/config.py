"""Central configuration for the semantic gap modeling module.

All hyper-parameters referenced by the paper (alpha = 0.5, tau = 0.6,
Qwen2.5-Coder-7B, hidden dim d) are defined here so the rest of the
package can import them from a single source of truth.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = os.environ.get("SQLVIS_DEVICE", "cuda" if __import__("torch").cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Text encoder (Module 2 - text branch)
# Paper: Qwen2.5-Coder-7B
# ---------------------------------------------------------------------------
TEXT_MODEL_NAME = os.environ.get("SQLVIS_TEXT_MODEL", "Qwen/Qwen2.5-Coder-7B")
TEXT_MAX_LEN = 512
HIDDEN_DIM = 768  # d in the paper (text encoder output dim)

# ---------------------------------------------------------------------------
# Graph encoder (Module 2 - graph branch)
# Paper: Relational Graph Attention Network (R-GAT)
# ---------------------------------------------------------------------------
GRAPH_HIDDEN_DIM = HIDDEN_DIM
GRAPH_NUM_HEADS = 4
GRAPH_NUM_LAYERS = 2
GRAPH_DROPOUT = 0.1

# Node / edge feature dimensions used by the graph encoder.
NODE_FEATURE_DIM = 128
EDGE_FEATURE_DIM = 128

# ---------------------------------------------------------------------------
# Cross-modal attention + scoring head
# ---------------------------------------------------------------------------
ATTENTION_NUM_HEADS = 4
MLP_HIDDEN_DIM = 256

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
TRAIN_EPOCHS = 10
GRAD_CLIP = 1.0

# Loss weights: L_train = L_BCE + lambda * L_contrast
LOSS_LAMBDA = 1.0
# Triplet contrastive margin alpha in the paper
CONTRASTIVE_MARGIN = 0.5

# Inference: highlight threshold tau (paper: 0.6)
HIGHLIGHT_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# Graph alignment (GED) for constructing (q, s_gt, s_bug) supervision
# ---------------------------------------------------------------------------
GED_TIMEOUT_SECONDS = 5.0

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

for _d in (DATA_DIR, CHECKPOINTS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
