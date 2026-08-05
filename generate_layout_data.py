#!/usr/bin/env python3
"""
生成语义-布局配对训练数据 + Encoder B微调三元组

论文描述：
- MLP训练数据：1,200个查询-指纹对（含语义保持变异）
- Encoder B微调数据：(q, s_bug, s_gt) 三元组
"""

import sys
import json
import re
import random
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.encoder import DualEncoder

# 16维语义指纹名称
FP_NAMES = [
    "table_count", "nest_depth", "join_graph_density", "cross_join_flag",
    "filter_intensity", "having_presence", "group_by_complexity", "agg_func_complexity",
    "distinct_flag", "set_op_flag", "window_func_flag",
    "output_columns_ratio", "orderby_strength", "limit_presence", "result_cardinality",
    "visual_emphasis"
]


def extract_fingerprint(sql: str) -> list:
    """从SQL AST提取16维语义指纹目标 p* ∈ [0,1]^16"""
    sql_upper = sql.upper()
    fp = [0.0] * 16

    tables = set()
    from_match = re.search(r'FROM\s+([^\s;(),]+)', sql_upper)
    if from_match:
        tables.add(from_match.group(1))
    for m in re.finditer(r'JOIN\s+([^\s;(),]+)', sql_upper):
        tables.add(m.group(1))
    num_tables = max(1, len(tables))
    fp[0] = min(1.0, num_tables / 5.0)

    nest_count = len(re.findall(r'\(\s*SELECT', sql_upper))
    fp[1] = min(1.0, nest_count / 3.0)

    join_count = len(re.findall(r'\bJOIN\b', sql_upper))
    fp[2] = min(1.0, join_count / 4.0)

    fp[3] = 1.0 if "CROSS JOIN" in sql_upper else 0.0

    where_match = re.search(r'WHERE\s+(.+?)(GROUP|HAVING|ORDER|LIMIT|UNION|$)', sql_upper, re.S)
    if where_match:
        cond_text = where_match.group(1)
        num_conds = len(re.split(r'\s+AND\s+|\s+OR\s+', cond_text))
        fp[4] = min(1.0, num_conds / 5.0)
    elif "WHERE" in sql_upper:
        fp[4] = 0.2

    fp[5] = 1.0 if "HAVING" in sql_upper else 0.0

    gb_match = re.search(r'GROUP\s+BY\s+(.+?)(HAVING|ORDER|LIMIT|UNION|$)', sql_upper, re.S)
    if gb_match:
        gb_cols = gb_match.group(1).strip()
        num_gb = len([c for c in re.split(r',', gb_cols) if c.strip()])
        fp[6] = min(1.0, num_gb / 3.0)

    agg_count = 0
    for agg in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]:
        agg_count += sql_upper.count(agg)
    fp[7] = min(1.0, agg_count / 5.0)

    fp[8] = 1.0 if "DISTINCT" in sql_upper else 0.0
    fp[9] = 1.0 if any(op in sql_upper for op in ["UNION", "INTERSECT", "EXCEPT"]) else 0.0
    fp[10] = 1.0 if "OVER(" in sql_upper.replace(" ", "") or "OVER (" in sql_upper else 0.0

    select_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql_upper, re.S)
    if select_match:
        select_cols = select_match.group(1).strip()
        cols = [c for c in re.split(r',', select_cols) if c.strip()]
        fp[11] = min(1.0, len(cols) / 5.0)

    if "ORDER BY" in sql_upper:
        ob_match = re.search(r'ORDER\s+BY\s+(.+?)(LIMIT|UNION|$)', sql_upper, re.S)
        if ob_match:
            ob_cols = ob_match.group(1).strip()
            num_ob = len([c for c in re.split(r',', ob_cols) if c.strip()])
            fp[12] = min(1.0, num_ob / 3.0)
        else:
            fp[12] = 0.3

    fp[13] = 1.0 if "LIMIT" in sql_upper else 0.0

    if fp[7] > 0 and fp[6] > 0:
        fp[14] = min(1.0, fp[6] * 0.6 + fp[7] * 0.3)
    elif fp[6] > 0:
        fp[14] = fp[6] * 0.7
    elif fp[7] > 0:
        fp[14] = 0.3
    else:
        fp[14] = 0.5

    complexity = sum(fp[:7]) / 7.0
    fp[15] = min(1.0, 0.3 + complexity * 0.7)

    return fp


def apply_semantic_mutation(sql: str) -> str:
    """语义保持变异（不改变查询语义）：交换JOIN顺序、改变谓词常量等"""
    mutations = []

    # 变异1: 交换 JOIN 顺序 (A JOIN B → B JOIN A)
    join_pattern = re.compile(r'(\w+)\s+(JOIN|INNER JOIN|LEFT JOIN)\s+(\w+)\s+ON\s+(\w+\.\w+)\s*=\s*(\w+\.\w+)', re.I)
    m = join_pattern.search(sql)
    if m:
        t1, join_kw, t2, c1, c2 = m.groups()
        mutated = sql[:m.start()] + f"{t2} {join_kw} {t1} ON {c2} = {c1}" + sql[m.end():]
        mutations.append(mutated)

    # 变异2: AND条件顺序交换 (a AND b → b AND a)
    and_pattern = re.compile(r'(\S+)\s+AND\s+(\S+)', re.I)
    m = and_pattern.search(sql)
    if m:
        mutated = sql[:m.start()] + f"{m.group(2)} AND {m.group(1)}" + sql[m.end():]
        mutations.append(mutated)

    # 变异3: 列顺序交换 (SELECT a, b → SELECT b, a)
    sel_pattern = re.compile(r'SELECT\s+(\w+),\s*(\w+)', re.I)
    m = sel_pattern.search(sql)
    if m:
        mutated = sql[:m.start()] + f"SELECT {m.group(2)}, {m.group(1)}" + sql[m.end():]
        mutations.append(mutated)

    # 变异4: 添加冗余括号
    if 'WHERE' in sql.upper() and '(' not in sql:
        where_match = re.search(r'WHERE\s+(.+?)(GROUP|ORDER|LIMIT|UNION|$)', sql, re.I | re.S)
        if where_match:
            cond = where_match.group(1).strip()
            mutated = sql[:where_match.start(1)] + f"({cond})" + sql[where_match.end(1):]
            mutations.append(mutated)

    if mutations:
        return random.choice(mutations)
    return sql  # 无变异则返回原SQL


def create_bias_sql(sql: str) -> str:
    """生成语义偏差SQL（用于Encoder B对比训练的s_bug）"""
    sql_lower = sql.lower()
    candidates = []

    # AND → OR (改变逻辑)
    if ' and ' in sql_lower:
        candidates.append(re.sub(r'\band\b', 'OR', sql, count=1, flags=re.I))

    # OR → AND
    if ' or ' in sql_lower:
        candidates.append(re.sub(r'\bor\b', 'AND', sql, count=1, flags=re.I))

    # 聚合函数替换
    for orig, repl in [(r'\bcount\s*\(', 'SUM('), (r'\bsum\s*\(', 'COUNT('),
                        (r'\bavg\s*\(', 'MAX('), (r'\bmax\s*\(', 'AVG('),
                        (r'\bmin\s*\(', 'AVG(')]:
        if re.search(orig, sql_lower):
            candidates.append(re.sub(orig, repl, sql, count=1, flags=re.I))
            break

    # 比较运算符翻转
    for pat, repl in [(r'\s=\s', ' != '), (r'\s>\s', ' < '), (r'\s<\s', ' > ')]:
        if re.search(pat, sql):
            candidates.append(re.sub(pat, repl, sql, count=1))
            break

    # JOIN类型替换
    sql_upper = sql.upper()
    for orig, repl in [('INNER JOIN', 'LEFT JOIN'), ('LEFT JOIN', 'INNER JOIN'),
                        ('JOIN', 'LEFT JOIN')]:
        if orig in sql_upper:
            candidates.append(re.sub(re.escape(orig), repl, sql, count=1, flags=re.I))
            break

    if candidates:
        return random.choice(candidates)
    return sql + " WHERE 1 = 0"  # 退化：添加恒假条件


def recommend_template_by_rules(sql: str) -> str:
    """规则推荐模板类型"""
    sql_lower = sql.lower()
    if re.search(r'\(\s*select', sql_lower):
        return "subquery"
    if 'join' in sql_lower:
        return "with_join"
    if any(kw in sql_lower for kw in ['count(', 'sum(', 'avg(', 'max(', 'min(']) or 'group by' in sql_lower:
        return "with_group"
    if 'where' in sql_lower or 'having' in sql_lower:
        return "with_filter"
    return "basic"


def main():
    print("=" * 60)
    print("生成训练数据 (16维语义指纹 + Encoder B三元组)")
    print("=" * 60)

    spider_dir = project_root / "data" / "spider"
    output_mlp = project_root / "data" / "processed" / "layout_train_data.json"
    output_encb = project_root / "data" / "processed" / "encoder_b_train_data.json"
    output_mlp.parent.mkdir(parents=True, exist_ok=True)

    train_path = spider_dir / "train.json"
    if not train_path.exists():
        print("[FAIL] Spider训练数据不存在: " + str(train_path))
        return

    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    print("[OK] 加载Spider训练数据: " + str(len(train_data)) + " 条")

    print("[INIT] 初始化编码器 (Encoder A, 冻结)...")
    encoder = DualEncoder()

    # 采样
    max_samples = 1200
    random.seed(42)
    if len(train_data) > max_samples:
        train_data = random.sample(train_data, max_samples)
    print("采样: " + str(len(train_data)) + " 条")

    # ========== 1. MLP训练数据 ==========
    print("\n[1/2] 生成MLP训练数据 (SQL→16维指纹)...")
    mlp_data = []
    for i, item in enumerate(train_data):
        if (i + 1) % 200 == 0:
            print(f"  处理 {i + 1}/{len(train_data)}...")

        sql = item.get('query', '')
        question = item.get('question', '')
        if not sql or not question:
            continue

        try:
            # Encoder A: 仅SQL → e_s^A (768维)
            e_s_A = encoder.encode_sql_structural(sql)
            embedding = e_s_A.squeeze(0).cpu().numpy()  # [768]

            # 提取16维指纹目标
            target = extract_fingerprint(sql)

            mlp_data.append({
                "embedding": embedding.tolist(),
                "target_params": target,
                "sql": sql[:200],
                "question": question[:200],
                "fp_names": FP_NAMES,
            })

            # 语义保持变异样本（额外50%数据）
            mutated_sql = apply_semantic_mutation(sql)
            if mutated_sql != sql:
                e_mut_A = encoder.encode_sql_structural(mutated_sql)
                embedding_mut = e_mut_A.squeeze(0).cpu().numpy()
                target_mut = extract_fingerprint(mutated_sql)  # 语义保持→指纹应相同
                mlp_data.append({
                    "embedding": embedding_mut.tolist(),
                    "target_params": target_mut,
                    "sql": mutated_sql[:200],
                    "question": question[:200],
                    "fp_names": FP_NAMES,
                })

        except Exception as e:
            if i < 5:
                print(f"  [ERROR] {str(e)[:80]}")

    print(f"\n[OK] MLP训练数据: {len(mlp_data)} 条")
    with open(output_mlp, 'w', encoding='utf-8') as f:
        json.dump(mlp_data, f, ensure_ascii=False)
    print(f"  保存: {output_mlp}")

    # ========== 2. Encoder B三元组 ==========
    print("\n[2/2] 生成Encoder B微调三元组 (q, s_bug, s_gt)...")
    encb_data = []
    for i, item in enumerate(train_data):
        if (i + 1) % 200 == 0:
            print(f"  处理 {i + 1}/{len(train_data)}...")

        sql = item.get('query', '')
        question = item.get('question', '')
        if not sql or not question:
            continue

        s_bug = create_bias_sql(sql)
        if s_bug == sql:
            continue  # 无法生成偏差则跳过

        encb_data.append({
            "question": question[:300],
            "sql_gt": sql[:300],
            "sql_bug": s_bug[:300],
        })

    print(f"\n[OK] Encoder B三元组: {len(encb_data)} 条")
    with open(output_encb, 'w', encoding='utf-8') as f:
        json.dump(encb_data, f, ensure_ascii=False)
    print(f"  保存: {output_encb}")

    # 统计
    if mlp_data:
        targets = np.array([d["target_params"] for d in mlp_data])
        print("\n16维语义指纹统计:")
        for j, name in enumerate(FP_NAMES):
            print(f"  D{j:02d} {name:<25} 均值={targets[:, j].mean():.3f}")


if __name__ == "__main__":
    main()
