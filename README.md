# 🎨 SQL Relational Data-Flow Diagram Framework

A cross-modal alignment framework that turns SQL queries into **low-cognitive-load
relational data-flow diagrams** and highlights the graph elements that may
deviate from the user's natural-language intent.

✨ **Key features**

- 🧱 **Deterministic SQL → JSON graph** — rule-based `sqlglot` parsing, no black-box structure prediction.
- 🧠 **Element-level semantic-gap scoring** — an LLM (Qwen2.5-Coder-7B) + Relational GAT with cross-modal attention.
- 🎯 **Real-error-driven supervision** — training triples `(q, s_gt, s_bug)` built from genuine LLM mistakes, aligned via Graph Edit Distance.
- 🖼️ **Deterministic SVG rendering** — longest-path layering, attribute-level anchoring, and pop-out highlighting of suspicious elements.

---


## 🚀 Quickstart

### 1️⃣ Installation

Python **3.10+** is recommended.

```bash
pip install -r requirements.txt
```

> 💡 On first use, Module 2 downloads `Qwen/Qwen2.5-Coder-7B` from HuggingFace
> (~15 GB). Set the environment variable `SQLVIS_TEXT_MODEL` to any other
> HuggingFace causal LM to use a smaller stand-in, and `SQLVIS_DEVICE` to
> `cpu` / `cuda` to control the device. Modules 1 and 3 need no model downloads.

### 2️⃣ Build the JSON graph (Module 1)

```bash
# From a SQL file -> JSON config
python tool/sql_to_ast_config.py --sql-file tool/examples/artist_pipeline.sql --out config.json

# Inline SQL (pass an empty --sql-file so --sql takes effect)
python tool/sql_to_ast_config.py --sql-file= --sql "SELECT * FROM t WHERE a = 1" --out config.json

# Inspect the intermediate AST only (printed to stdout)
python tool/sql_to_ast_config.py --sql-file tool/examples/easy_example.sql --ast

# Custom canvas size (default 1200x620)
python tool/sql_to_ast_config.py --sql-file tool/examples/cte_pipeline.sql --canvas 1600x800 --out config.json
```

📌 The default output path is `default.json`; pass `--out=` to print the JSON
to stdout instead. (The `--opt=` empty-value form works in both bash and
PowerShell.)

### 3️⃣ Score graph elements (Module 2)

```python
import sys
sys.path.insert(0, "tool")  # make Module 1 importable

from sql_to_ast_config import build_config
from src.aligner import AlignmentScorer
from src.graph_features import build_graph_tensors, apply_scores_to_graph

sql = "SELECT name FROM artists WHERE country = 'USA'"
question = "List the names of American artists."

# Module 1: SQL -> JSON graph
ast, json_graph = build_config(sql)

# Module 2: JSON graph -> element-level semantic-gap scores
node_feats, edge_index, edge_type, element_ids = build_graph_tensors(json_graph)
model = AlignmentScorer(num_relations=12)          # loads Qwen2.5-Coder-7B
model.eval()
scores = model.predict_scores(question, node_feats, edge_index, edge_type)

# Write S_u back onto the graph: elements with S_u >= tau get highlight flags
json_graph = apply_scores_to_graph(json_graph, scores, element_ids)
```

🏋️ **Training** uses `src/trainer.py`: each sample is a
`(question, json_graph, element_labels)` triple; labels come from GED alignment
(`src/graph_alignment.py`) of gold vs. buggy SQL graphs. The objective is
`L_BCE + lambda * L_contrast`, and checkpoints are saved via
`AlignmentTrainer.save(path)`.

### 4️⃣ Render the SVG (Module 3)

```bash
python tool/sql_metaphor_diagram.py --config config.json --out diagram.svg
```

### 5️⃣ Regenerate all paper figures

```bash
python tool/paper_figures.py --out tool/output
```

### 🔁 End-to-end pipeline

```bash
python tool/sql_to_ast_config.py --sql-file tool/examples/artist_pipeline.sql --out config.json
python tool/sql_metaphor_diagram.py --config config.json --out tool/output/diagram.svg
```

---

## 🧩 Module Details

### 🧱 Module 1 — AST-JSON builder (`tool/sql_to_ast_config.py`)

Parses SQL with `sqlglot` (a pure-stdlib recursive-descent parser is the
fallback), then a `ConfigBuilder` instantiates data-source entities, builds
per-query-block scopes (alias / self-join handling), and emits directed edges
with predicate labels. Correlated single-table scalar subqueries are folded
into annotated edges to avoid fragmented graphs.

SQL is mapped onto three visual primitives:

| Graph element | SQL component               | Visual representation        |
|---------------|-----------------------------|------------------------------|
| node          | base table, CTE, subquery   | rectangle                    |
| edge          | SELECT, WHERE, JOIN, ...    | arrow (data flow)            |
| edge label    | predicates, computations    | text annotation on the arrow |

The output JSON carries `nodes` (dict), `edges` (list), `layout`, and `canvas`.
After Module 2 scoring, each element additionally holds:

```json
{ "score": 0.73, "highlight": true }
```

📖 See `tool/sql_ast_json_rules.md` for the complete rule catalogue (R0–R26).

### 🧠 Module 2 — Semantic-gap scorer (`src/`)

For every graph element `u`, the model predicts
`P_align(u) = sigmoid(MLP(C_u))`:

- **Text encoder** — `Qwen2.5-Coder-7B` produces token hidden states `H_q`.
- **Graph encoder** — a Relational GAT produces one hidden vector `H_u` per element.
- **Cross-modal attention** — `C_u' = MultiHeadAttention(H_q, H_u, H_u)` with the
  text as Query and graph elements as Key/Value, then `C_u = MeanPooling(C_u')`.
- **Scoring head** — an MLP maps the fused representation to a probability.

**Supervision.** Real LLM errors on Spider 2.0 (DISTINCT misuse, window-frame
errors, UNION confusion, predicate drift, wrong aggregation keys) provide
`s_bug`. Because bug graphs drift structurally, GED aligns `J_gt` with `J_bug`;
inserted/substituted elements become negatives (`y=0`), everything else positives.

**Training objective** (element-level BCE + triplet contrastive):

```
L_train    = L_BCE + lambda * L_contrast
L_contrast = max(0, alpha - sim(C_q, C_u_gt+) + sim(C_q, C_u_bug-))
```

**Inference.** `S_u = 1 - P_align(u)`; elements with `S_u >= tau` are flagged as
high semantic gap and trigger pop-out highlighting in Module 3.

⚙️ **Key hyper-parameters** (`src/config.py`):

| Symbol | Meaning                        | Default |
|--------|--------------------------------|---------|
| `alpha`| contrastive margin             | `0.5`   |
| `tau`  | highlight threshold            | `0.6`   |
| `lambda`| contrastive loss weight       | `1.0`   |
| `d`    | hidden dimension               | `768`   |
| —      | text model                     | `Qwen/Qwen2.5-Coder-7B` |
| —      | learning rate / epochs         | `1e-4` / `10` |

### 🖼️ Module 3 — SVG renderer (`tool/sql_metaphor_diagram.py`)

Turns a (score-annotated) JSON graph into an SVG diagram:

- 📐 **Longest-path layering** — left-to-right columns by longest dependency path;
  same-layer nodes stack vertically.
- ⚓ **Attribute-level anchoring** — edges with `from_attr` / `to_attr` attach to
  the exact attribute row on the rectangle border (cubic Bezier curves), not the
  geometric centre; equi-join key bindings use arc links without arrowheads.
- 🧲 **Label anti-collision** — labels are bounding boxes pushed apart
  iteratively; column gaps auto-widen to fit the longest label.
- 🔦 **Visual highlighting** — high-gap nodes get thickened borders and dark-red
  headers; attribute rows get a light-red background (`#ffe9e9`) and bold red
  text; lineage tracing highlights upstream source attributes along the data
  flow; aligned elements stay black-on-white for noise reduction.

---

## 📊 Examples

Six sample SQL queries live in `tool/examples/`:

| File                   | Features exercised                              |
|------------------------|-------------------------------------------------|
| `easy_example.sql`     | simple SELECT + WHERE                           |
| `artist_pipeline.sql`  | CTEs, multi-JOIN, GROUP BY, correlated subquery |
| `cte_pipeline.sql`     | window functions, CTE pipeline                  |
| `patent_citations.sql` | NOT IN subqueries, self-join                    |
| `race_laps.sql`        | aggregation, ORDER BY                           |
| `spatial_events.sql`   | BigQuery-style functions, backtick identifiers  |

`spatial_events.sql` exercises the pure-stdlib fallback parser (sqlglot's
default dialect does not cover all BigQuery syntax).

---

## 🗂️ Data & Outputs

- 📚 `data/spider/` — Spider questions/schemas used to build supervision triples.
- 🖼️ `tool/output/` — generated SVG diagrams and paper figures.
- 💾 `checkpoints/` — trained Module 2 weights (created on demand, git-ignored).
