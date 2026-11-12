# SQL -> AST -> Metaphor-Diagram JSON Configuration: Rule Catalogue

Implementation: `tool/sql_to_ast_config.py` (sqlglot parser with a
pure-stdlib fallback parser; no dependency on `sqlparse`).

Output structure: isomorphic to `tool/config_artist_pipeline.json` and
directly renderable by `tool/sql_metaphor_diagram.py`.

----------------------------------------------------------------------
## 0. Global conventions (R0)

**R0-1** Semantic division of the graph:
- Nodes (`dict`, keyed by `node_id`) = relational entities, always `kind="table"` (rectangle). Covers base tables / CTEs / derived tables / scalar subqueries / result sets.
- Edges (`list[dict]`) = clauses and operations, carried on `label`. Covers `FROM` / `JOIN ... ON` / `WHERE` / `GROUP BY` / `HAVING` / `ORDER BY` / `LIMIT`. When conditions allow, edges use `from_attr` / `to_attr` to anchor to attribute rows (attribute-level wiring).
- Arrow direction = data-flow direction: input entity -> output entity.

**R0-2** No entity (base table, temp table, derived table, virtual table, result set) may be drawn as a circle/diamond/dashed box. Everything is a rectangle with `kind:"table"` (consistent with `sql_metaphor_diagram.py`'s `wrapper_rect` convention).

**R0-3** `header` = the entity's spelling in the original SQL (table name + alias, e.g. `"invoices i"`). Never fabricate aliases or rewrite them.

**R0-4** `rows` = the attribute columns referenced on that entity, in order of first appearance in the SQL, de-duplicated.

**R0-5** Duplicate edges with no attributes and no label are forbidden; `add_edge` de-duplicates by the key `(from, to, from_attr, to_attr, label)`.

----------------------------------------------------------------------
## 1. Lexing (R1)

**R1-1** `tokenize(sql)` produces `Token(kind, text, start, end)` and preserves character offsets so that `text_from(start)` can slice the original SQL (labels must stay faithful to the source, never rewritten).

**R1-2** Supported token kinds: `kw` (keyword, see `KEYWORDS`) / `ident` / `qident` (`"x"`, `` `x` ``, `[x]`) / `str` (`'...'`) / `num` / `op`.

**R1-3** Comments: `--` line comments and `/* ... */` block comments are discarded (not part of the AST, not in labels).

**R1-4** Operators use longest-match: `TWO_CHAR_OPS` take precedence over single-character operators (`<=`, `>=`, `<>`, `!=`, `||`, `::`, `=>`, ...).

**R1-5** Illegal characters -> `ParseError`; on parse failure the CLI exits with "SQL parse failed: ..." (non-zero exit code).

----------------------------------------------------------------------
## 2. Syntax tree (R2)

**R2-1** Statement: `{"ctes": [ {name, columns, query} ], "body": <select|setop>}`. `WITH a AS (...), b AS (...)` populate `ctes` in order; supports `RECURSIVE` and `WITH x(c1,c2)`.

**R2-2** `select` node fields:
```
{type:"select", distinct, select:[{expr, alias, text}], from:[item],
 where, group_by:[expr], having, order_by:[{expr, desc, text}], limit}
```

**R2-3** `from` item fields:
- base table: `{kind:"table", name, alias}`
- subquery:  `{kind:"subquery", query, alias}`
- join info: `join_type` (`"JOIN"`/`"INNER JOIN"`/`"LEFT JOIN"`/`", "` ...), `on`, `using`.

**R2-4** Expression node types:
`column(qualifier,name)` / `literal` / `star` / `function(name,args,distinct,window)` / `binary(op,left,right)` / `unary` / `cast(expr,typename)` / `case` / `between` / `in(items)` / `is` / `exists` / `subquery(query)` / `paren(expr)`.

**R2-5** All expressions carry a `text` field = the original SQL slice. `IN` subquery items additionally record the parenthesised original text for whole-segment label replacement (see R21-3).

**R2-6** Set operations `UNION` / `INTERSECT` / `EXCEPT` -> `{"type":"setop","op","left","right"}`. Visualisation builds entities from the left branch (first `SELECT`) only; the right branch does not get its own nodes.

----------------------------------------------------------------------
## 3. Entity generation (R3 - R8)

**R3-1** `create_entity(node_id, header, role)`: `node = {"kind":"table", "header":..., "rows":[]}`. The result set additionally gets `header_fill="#cccccc"`, `header_color="#666666"`. `role` in `{table, cte, derived, sub, result}` (internal only, not written to JSON).

**R4-1** CTE -> node id `"L1"`, `"L2"`, ... (in `WITH` order). `header` = CTE name (original case preserved); `rows` = the CTE's `SELECT` output column names. If `WITH` carries a column list (`WITH x(a,b) AS ...`), `rows` is overridden by that list.

**R4-2** Node creation happens when the CTE's query body is processed (not at the start of `build`), so the node order is "base tables -> CTE chain -> result set", matching the in-column order of the layered layout.

**R4-3** Nested `WITH` (a `WITH` inside a subquery) -> node ids `"<consumer>_c1"`, `"<consumer>_c2"`, ...

**R5-1** Referencing a CTE does not create a new node: if a name in `FROM` matches `cte_ids`, reuse `L1..Ln`.

**R6-1** The same base table gets a single node; `node_id` = the last segment of the table name (dataset prefix stripped).

**R6-2** Alias rule: the header reads `"table_name alias"`; the alias is always taken from the original SQL spelling.

**R6-3** When the same table appears again:
- (a) no alias, or alias identical to the first occurrence -> reuse the same node.
- (b) first occurrence had no alias, this one has an alias -> still reuse the same node; header becomes `"table_name alias"`.
- (c) both occurrences have non-empty, different aliases (self-join) -> split into node `"table_name_alias"`.

**R7-1** Derived table in `FROM` `(SELECT ...) x` -> node `"d1"`, `"d2"`, ..., `header "(subquery)"`. Its inner query is processed recursively (inner base tables still create entities).

**R8-1** Result set has fixed `node_id "result"`, `header "SELECT"`, `rows` = the outermost `SELECT`'s output column names, and `highlight:true` (whole-table highlight, marking the final result).

**R8-2** Output column name = the `SELECT` item's alias; without an alias: column -> column name, `*` -> `"*"`, anything else -> original SQL text.

----------------------------------------------------------------------
## 4. Scope and attribute registration (R9 - R10)

**R9-1** `build_scope(body)` creates for each query body:
`{by_name: {qualified/table/alias -> entity}, default, sources:[entities], items:[from items]}`.

**R9-2** `resolve_column(col, scope)`:
- with qualifier: `scope.by_name` -> `global_index` -> strip prefix and look up `global_index` -> `None`.
- without qualifier: `scope.default` (the first data source; the only source in a single-source scope).

**R10-1** `register_all(exprs, scope)` uses a two-phase registration to avoid routing columns to the wrong table in multi-source scopes:
- Phase 1: register only qualified columns (e.g. `ii.InvoiceId` -> `invoice_items`).
- Phase 2: register unqualified columns:
  - single-source scope -> assign directly to that source.
  - multi-source scope -> assign to a source only when exactly one existing source already contains that column name in its `rows`; if multiple sources contain it -> assign to `default`; if no source contains it -> do not register (prefer omission over error).

**R10-2** Only `role="table"` (base tables) participate in `rows` registration; `rows` of CTEs/derived tables/subqueries/result sets are determined by their own `SELECT` output.

**R10-3** Registration order: `ON` -> `SELECT` -> `WHERE` -> `GROUP BY` -> `HAVING` -> `ORDER BY` (within a query body: join keys first, then projection, then filters; column order matches reading order).

----------------------------------------------------------------------
## 5. Edges: joins and clauses (R11 - R16)

**R11-1** JOIN key chain (key visualisation, no arrow) is enabled only when ALL hold:
- (a) `FROM` has at least 2 items; (b) all items are base-table entities (`role="table"`); (c) every subsequent item's `ON` is a "simple two-column equality" `a.x = b.y`; (d) this query body is not the outermost result set (`consumer_id != "result"`).

**R11-2** Output edge:
```
{"from": previous entity, "to": current entity, "kind":"curve", "no_rank":true,
 "arrow":false, "from_side":"right","to_side":"right",
 "from_attr": key_col, "to_attr": key_col, "bend":[34,0]}
```
`bend=[34,0]` means exit on the right, arch outward to the right, then reconnect on the right, avoiding overlapping the table body.

**R11-3** `no_rank:true`: this edge does not participate in layering (R23), otherwise tables on the same chain would be squeezed into multiple columns.

**R12-1** `FROM` / `JOIN` label edges (used at the result-set layer and for non-pure-base-table chains):
- If the first source is not connected to the consumer by any edge, add `{"label":"FROM <alias>"}`.
- For each subsequent source: `{"label":"JOIN <alias>[ ON <key>| USING (a, b)]", "_auto_join":true}`.
- The `ON` spelling is compressed by `simplify_on`:
  - same-name key `a.x = b.x` -> write only the column name `x`.
  - `ON` composed by `AND` -> take only the first equality key (the rest are carried by `WHERE` or clause labels).
  - otherwise -> original SQL (single-line).

**R12-1a** JOIN de-duplication (`_is_upstream`): if a joined table is reachable back to the first source along existing edges (data already flows into the consumer via the first source, e.g. `TopAvg JOIN invoices` while an `invoices -> ArtistSales` projection edge already exists), do not draw a separate line (the bottom curve would look like it originates from the last row of the first source, creating redundancy). Instead prepend the `"JOIN ..."` annotation to the head of the first-source -> consumer edge label (`"JOIN i ON InvoiceId; SUM(...) AS AvgAmt"`). If the first source has no labelled edge, add a `"FROM <first>; JOIN ..."` edge. Non-upstream joins (e.g. result's `FROM TopAvg JOIN BotAvg`) are unaffected and drawn normally.

**R12-2** `USING (key)` handling:
- Register the join-key columns into every base-table entity in this query body (add rows if missing, see R10-1).
- If there is a previous source, add an arrowless key arc (same shape as R11-2):
  ```
  {"from": previous, "to": current, "kind":"curve","no_rank":true,"arrow":false,
   "from_side":"right","to_side":"right","from_attr":key,"to_attr":key,"bend":[34,0]}
  ```
- The `JOIN` label still keeps the original `"USING (key)"` text.

**R12-3** `_auto_join` geometry is decided after `compute_ranks` (once ranks are known):
- `rank(to) - rank(from) >= 2` -> `kind:"curve"`, `from_side`/`to_side:"bottom"`, `bend:[0, 90 + 50*k]` where `k` = index of this under-wrapping edge from the same `from` node (90, 140, 190, ...).
- otherwise -> `kind:"line"`.

**R13-1** Clause edges: group `GROUP BY` / `HAVING` / `ORDER BY` / `LIMIT` / `WHERE` into one edge, label joined by `"; "`, attached to the first data source:
`{"from": first_source, "to": consumer, "kind":"line", "label":"A; B; C"}`.

**R13-2** When there is a `GROUP BY`: computed expressions in `SELECT` (non-column, non-`*`, non-literal) are merged into the head of the label (`"expr AS alias, expr AS alias; GROUP BY ..."`), and the per-column projection edges of R14/R15 are suppressed (otherwise the same semantics would be drawn twice).

**R13-3** `ORDER BY` items that contain subqueries do not go into the clause label (handled separately by R19).

**R13-4** When `WHERE` contains a passthrough scalar subquery (R17) or a foldable correlated subquery (R18-2), `WHERE` does not enter the clause label; it is carried by the fold edge of R20 / R18-2 to avoid the same `WHERE` appearing twice.

**R14-1** Attribute-level projection edges (plain columns in `SELECT`):
```
{"from": column's entity, "to": consumer, "kind":"line",
 "from_attr": column_name, "to_attr": output_column_name}
```
Pre-check: the column must actually exist in the source entity's `rows` (`has_attr`), otherwise no edge is drawn. Also record `lineage[(consumer, output_col)] = {(entity, col)}`.

**R15-1** Computed-expression edges (`SUM`/`COUNT`/`CASE`/`CAST`/functions/arithmetic):
- expression references only 1 entity -> `{"from": that entity, "label": original SELECT item text}`.
- expression spans multiple entities (e.g. `ABS(A.x - B.y)`) -> no expression edge; the expression is only written into the consumer's `rows`; inter-entity connection is carried by `FROM`/`JOIN` edges.
- literal output -> no edge.

**R16-1** `SELECT *`:
- sole output item -> draw `{"attr_link": true}` edges for every source (the renderer wires same-named attributes pairwise).
- coexists with other output items (`SELECT *, expr ...`) -> no `*` edge; the wildcard semantics are expressed by the `"*"` in `rows` together with the other output edges (avoids attribute-less duplicate edges).

----------------------------------------------------------------------
## 6. Scalar subqueries (R17 - R20)

**R17-1** Passthrough (foldable) test `_is_passthrough(sub)`, ALL must hold:
- (a) subquery body is `select`; (b) no `GROUP BY` / `HAVING` / `ORDER BY` / `LIMIT`; (c) `FROM` has exactly 1 item; (d) the `SELECT` list is all simple column references.

**R17-2** Fold handling (no node created): the subquery in `WHERE` does not generate a `WHERE` clause label; instead R20 draws a single fold edge at the end. Columns inside the subquery are still registered to its data source per R10.

**R18-1** Correlated (outer-reference) test: a qualifier inside the subquery does not belong to the local scope and resolves to an outer entity -> correlated reference `corr=[(outer_entity, col)]`.

**R18-2** Correlated folding (`try_fold_correlated`, takes precedence over materialisation), ALL hold:
- (a) subquery body is `select`; (b) `FROM` has exactly 1 item and it is a base table (not a derived table); (c) there is a correlated reference from R18-1.

Handling (no node created):
- The inner base table is reused/created as a normal entity node (e.g. `artists`); inner columns referenced by the subquery are registered into that table's `rows` per R10 (e.g. `artists` gets `Name`, `ArtistId`).
- Correlation key-pair edge: if a `WHERE` conjunct contains `inner.x = Outer.y`, draw
  ```
  {"from": inner_table, "to": outer_entity, "kind":"curve", "no_rank":true,
   "arrow":false, "from_side":"right","to_side":"right",
   "from_attr":x, "to_attr":y, "bend":[30,0]}
  ```
  De-duplicate by `(from,to,from_attr,to_attr)` ignoring label: when a same-name attribute wire already exists (e.g. projection edge `artists.ArtistId -> L1.ArtistId`), reuse it and append the correlation condition `"WHERE <original>"` to that wire's label (correlated-join visualisation). Multiple correlated subqueries hitting the same wire are annotated only once; no parallel edges.
- Without such equality, fall back to `{"from": outer_entity, "to": inner_table, "kind":"line", "label":"WHERE <original subquery WHERE text>"}`.
- Consumer edge: `{"from": inner_table, "to": consumer, "kind":"line", "label": <enclosing clause label>}`. The correlation condition is already on the pair edge, so it is not repeated here; `"; WHERE <correlation condition original>"` is appended only when the pair edge cannot carry the annotation. When the subquery output is a single simple column, add `from_attr=<that column>`; in `SELECT` position, replace the subquery text in the consumer's `rows` with that column name and record `lineage`.

Example: `(SELECT Name FROM artists WHERE ArtistId = ArtistSales.ArtistId)`
-> `artists.rows += Name`; wire `artists.ArtistId -> ArtistSales.ArtistId` labelled `"WHERE ArtistId = ArtistSales.ArtistId"`; edge `artists -> TopArtist` labelled `"ORDER BY ... ASC"` (anchored to the `Name` row).

**R18-3** Materialisation (fallback `make_sub_entity` when R18-2 does not apply): `node_id "sub_1"`, `"sub_2"`, ... (in order of appearance); `header "(SELECT output_col[, ...])"`; `rows = ["FROM table_name", ...]`. Correlation-reference edge: `{"from": outer_entity, "to": sub_N, "kind":"line", "label":"WHERE <subquery WHERE text with local qualifiers stripped>"}`. When the subquery `WHERE` is composed by `AND`, take only the conjunct that contains the correlated reference (R21-2).

**R18-4** Non-correlated (subquery has no outer reference):
`{"from": default source entity, "to": sub_N, "label": <compressed predicate of the enclosing clause>}`. In `WHERE` the label looks like `"WHERE patent_id IN (subquery)"` (only the conjunct containing the subquery is kept; the subquery body is replaced by `(subquery)`).

**R18-5** `EXISTS (SELECT ...)` follows the same path as scalar subqueries:
- First test for outer correlation per R18-1 (e.g. `WHERE z.id = x.id`); if correlated and single base table, also fold via R18-2.
- Otherwise materialise as `sub_N` (`header "(SELECT 1)"`, `rows ["FROM t3"]`).
- The clause label uses the `"EXISTS (subquery)"` placeholder; the subquery body is not expanded.
- Column-reference collection does not descend into the `EXISTS` subquery (avoids polluting the outer base tables' `rows`).

**R19-1** Scalar subqueries in `SELECT` / `ORDER BY` / `HAVING` also first try R18-2 folding, then R18-3 materialisation; but the return-edge label is: `ORDER BY` -> `"ORDER BY ... ASC|DESC"`; `HAVING` -> `"HAVING ..."`; `SELECT` -> `"SELECT ..."`.

**R20-1** `sub_N -> consumer` return edge (subquery result consumed):
```
{"from": sub_N, "to": consumer, "kind":"curve", "no_rank":true,
 "from_side":"right","to_side":"right","bend":[30,0]}
```
With correlated references, `label` = enclosing clause label (e.g. `"ORDER BY ... ASC"`); omitted when non-correlated.

**R20-2** `WHERE` fold edge (R17-2) is drawn after projection edges and JOIN edges (so the visual order is expression -> JOIN -> WHERE):
`{"from": subquery's data-source entity, "to": consumer, "kind":"line", "label":"WHERE <compressed predicate>"}`.

**R20-3** A scalar subquery must not be drawn as an "independent data-flow node chain" (no extra intermediate nodes beyond the source -> sub -> consumer triple).

----------------------------------------------------------------------
## 7. Label normalisation (R21)

**R21-1** Every `label` is single-lined at `add_edge` time: consecutive whitespace/newlines collapse to one space (`oneline`).

**R21-2** Predicate compression `short_where(predicate, sub, replacement, scopes)`:
- Among the conjuncts split by top-level `AND`, take only the one that contains the subquery.
- Strip qualifiers known within the scope (e.g. `ArtistSales.`, `c.`, `p.`), preserving semantics.

**R21-3** Subquery shorthand: the whole original text of the compressed subquery is replaced by:
- the data source display name (folded, e.g. `TopArtist`), or
- `"(subquery)"` (materialised).
- If the replacement point is immediately next to `IN` (`"NOT IN "` / `"IN "`), an extra layer of parentheses is added automatically.
- Other subqueries in the same predicate are not expanded (avoids overly long labels).

**R21-4** Labels come only from the original SQL and the compression above; no semantic rewriting (do not abbreviate `SUM(Total)`, etc.).

----------------------------------------------------------------------
## 8. Finalisation (R22 - R24)

**R22-1** Highlight tracing (`apply_highlights`):
- Start = `(entity, column)` referenced by the result set's `SELECT` expressions.
- Layered BFS: `lineage[(consumer, col)]` -> its source `(entity, column)`, until convergence.
- A hit on a same-named row in an entity's `rows` -> write into `node.highlight_rows` (only write rows that actually exist).
- Result set always has `highlight:true`.

**R22-2** Highlighting is "result explainability" tracing; it is forbidden to highlight column names that do not exist in `rows`.

**R23-1** Layered `rank` (layered layout):
- Ignore edges with `no_rank:true` (key chains, subquery return edges do not participate in layering).
- `rank` takes the "longest path": `rank[b] = 1 + max(rank[a])`, iterated to a fixpoint (DAG topology).

**R24-1** Output key order: `layout` -> `baseline_y` -> `canvas` -> `nodes` -> `edges`.

**R24-2** `layout` is fixed: `{"mode":"layered", "gap_x":46, "gap_y":26, "margin_x":34}`. Column gaps are auto-widened by the renderer based on the actual width of the longest label in that gap (an explicit `layout.max_label_gap` may cap it; default is uncapped).

**R24-3** `baseline_y = 320`; `canvas` defaults to `{"width":1200,"height":620,"background":"#ffffff"}` (the renderer auto-expands width to `max_right + 34`).

**R24-4** Edge-field whitelist (only output what is used):
`from` / `to` / `kind` / `no_rank` / `arrow` / `from_side` / `to_side` / `from_attr` / `to_attr` / `attr_link` / `bend` / `label` / `_auto_join` (internal temporary, removed before output).

----------------------------------------------------------------------
## 9. Command line (R25)

**R25-1** `python tool/sql_to_ast_config.py --sql-file tool/examples/artist_pipeline.sql --out tool/config_artist_pipeline.json`

**R25-2** `python tool/sql_to_ast_config.py --sql "SELECT * FROM t WHERE a = 1"` (prints to stdout when `--out` is omitted).

**R25-3** `--ast` prints only the AST (for debugging, no configuration is generated).

**R25-4** `--canvas 1200x620` specifies the canvas.

**R25-5** `--indent N` specifies JSON indentation (default 2).

**R25-6** Rendering: `python tool/sql_metaphor_diagram.py --config <generated json> --out <svg>`.

----------------------------------------------------------------------
## 10. Known boundaries and limitations (R26)

**R26-1** In multi-source scopes, unqualified columns with "no basis for attribution" are not registered (R10-1), so such columns may not appear in base tables' `rows` (prefer omission over error; no erroneous wires).

**R26-2** Set operations (`UNION`, etc.) build entities/edges from the left branch only; right-branch information does not enter the graph (for full representation, split branches into multiple CTEs first).

**R26-3** Window functions `OVER(...)` are kept as a whole string in the expression's original text; `PARTITION`/`ORDER` are not decomposed.

**R26-4** No database metadata/schema is used: column attribution is determined only by qualifiers and "whether the column is already in `rows`".

**R26-5** When a query body's `FROM` list has more than 2 base tables and all joins are simple equalities, the JOIN key chain only draws adjacent "previous -> current" edges; no ring/cross key lines.

**R26-6** The parser covers common OLAP syntax (`WITH`/CTE, `JOIN`, subqueries, `CASE`, `CAST`, window functions, set operations, `LIMIT`/`OFFSET`); unknown dialect extensions may trigger `ParseError`.
