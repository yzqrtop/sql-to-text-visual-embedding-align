#!/usr/bin/env python3
"""
Spider数据集处理脚本
"""

import sys
import json
import random
import re
from pathlib import Path
from collections import defaultdict


def classify_query_type(sql):
    sql_lower = sql.lower()
    if re.search(r'\(\s*select', sql_lower):
        return "subquery"
    if re.search(r'\bjoin\b', sql_lower):
        return "join"
    if re.search(r'\b(count|sum|avg|max|min|group by|having)\b', sql_lower):
        return "aggregate"
    if re.search(r'\b(order by|limit|distinct)\b', sql_lower):
        return "simple"
    return "complex"


def calculate_complexity_score(sql):
    score = 1.0
    sql_upper = sql.upper()
    if "JOIN" in sql_upper:
        score += sql_upper.count("JOIN") * 0.5
    if "WHERE" in sql_upper or "HAVING" in sql_upper:
        score += 0.5
    if "GROUP BY" in sql_upper:
        score += 0.5
    if re.search(r'\(\s*SELECT', sql_upper, re.IGNORECASE):
        score += 1.0
    if any(agg in sql_upper for agg in ["COUNT", "SUM", "AVG", "MAX", "MIN"]):
        score += 0.3
    return round(score, 2)


def create_bias_sample(original_sql):
    sql = original_sql
    sql_lower = sql.lower()
    if ' and ' in sql_lower:
        pattern = re.compile(r'\band\b', re.IGNORECASE)
        match = pattern.search(sql)
        if match:
            pos = match.start()
            sql = sql[:pos] + ' OR ' + sql[pos + len('AND'):]
            return sql
    agg_patterns = [
        (r'\bcount\s*\(', 'SUM('),
        (r'\bsum\s*\(', 'COUNT('),
        (r'\bavg\s*\(', 'MAX('),
        (r'\bmax\s*\(', 'MIN('),
        (r'\bmin\s*\(', 'AVG(')
    ]
    for pattern, replacement in agg_patterns:
        if re.search(pattern, sql_lower):
            sql = re.sub(pattern, replacement, sql, count=1, flags=re.IGNORECASE)
            return sql
    comparison_patterns = [
        (r'\s=\s', ' != '),
        (r'\s!=\s', ' = '),
        (r'\s>\s', ' < '),
        (r'\s<\s', ' > ')
    ]
    for pattern, replacement in comparison_patterns:
        if re.search(pattern, sql):
            sql = re.sub(pattern, replacement, sql, count=1)
            return sql
    return sql


def main():
    spider_dir = Path(__file__).parent / "data" / "spider"
    output_file = Path(__file__).parent / "data" / "processed" / "test.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("加载Spider数据集...")
    
    with open(spider_dir / "train.json", 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    print("[OK] 加载训练数据: " + str(len(train_data)) + " 条")

    with open(spider_dir / "dev.json", 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    print("[OK] 加载开发数据: " + str(len(dev_data)) + " 条")

    with open(spider_dir / "tables.json", 'r', encoding='utf-8') as f:
        tables_data = json.load(f)
    print("[OK] 加载表结构数据: " + str(len(tables_data)) + " 个数据库")

    all_data = train_data + dev_data
    print("\n总数据量: " + str(len(all_data)) + " 条")

    print("\n开始分层采样...")
    type_groups = defaultdict(list)
    for item in all_data:
        sql_str = item.get('query', '')
        if not sql_str or not isinstance(sql_str, str):
            sql_str = item.get('sql', '')
            if isinstance(sql_str, dict):
                sql_str = ''
        query_type = classify_query_type(sql_str)
        type_groups[query_type].append(item)

    sampled_data = []
    query_types = ['simple', 'aggregate', 'join', 'subquery', 'complex']
    samples_per_type = 100

    for query_type in query_types:
        available = type_groups.get(query_type, [])
        if available:
            sample_size = min(samples_per_type, len(available))
            sampled = random.sample(available, sample_size)
            sampled_data.extend(sampled)
            print("  " + query_type + ": " + str(sample_size) + " 条")
        else:
            print("  " + query_type + ": 0 条")

    print("采样完成: " + str(len(sampled_data)) + " 条")

    bias_count = int(len(sampled_data) * 0.4)
    bias_indices = random.sample(range(len(sampled_data)), bias_count)

    processed_samples = []
    for idx, item in enumerate(sampled_data):
        schema = {}
        for table_info in tables_data:
            if table_info.get('db_id') == item['db_id']:
                schema = table_info
                break

        sql_str = item.get('query', '')
        if not sql_str or not isinstance(sql_str, str):
            sql_str = item.get('sql', '')
            if isinstance(sql_str, dict):
                sql_str = ''

        complexity = calculate_complexity_score(sql_str)
        query_type = classify_query_type(sql_str)

        if idx in bias_indices:
            modified_sql = create_bias_sample(sql_str)
            label = 1
        else:
            modified_sql = ""
            label = 0

        processed_samples.append({
            "question": item['question'],
            "sql_original": sql_str,
            "sql_modified": modified_sql,
            "schema": schema,
            "label": label,
            "complexity_score": complexity,
            "query_type": query_type,
            "db_id": item['db_id']
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_samples, f, ensure_ascii=False, indent=2)

    print("\n[OK] 数据处理完成!")
    print("  总样本数: " + str(len(processed_samples)))
    print("  输出文件: " + str(output_file))


if __name__ == "__main__":
    main()