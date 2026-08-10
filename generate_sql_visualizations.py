#!/usr/bin/env python3
"""
用真实管线生成SQL→可视化原语SVG图（通过真实双编码器+MLP+Renderer生成）

输出：每个SQL生成一张原语组成的可视化图（容器/漏斗/分支/堆叠柱/边界框等）
"""

import sys
import json
import re
import random
from pathlib import Path
from xml.etree import ElementTree as ET

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.layout_generator import LayoutGenerator

# SQL示例集（论文中典型的5种错误类型 + 基础查询）
SQL_SAMPLES = [
    # ===== 基础查询（无过滤/聚合）=====
    {
        "name": "01_simple_select",
        "question": "List all student names and their ages.",
        "sql": "SELECT name, age FROM students",
        "description": "Simple SELECT — Table container only (无过滤无聚合)"
    },
    {
        "name": "02_simple_distinct",
        "question": "Show all distinct departments.",
        "sql": "SELECT DISTINCT dept FROM employees",
        "description": "SELECT DISTINCT — Container with output label"
    },
    # ===== WHERE过滤查询 =====
    {
        "name": "03_single_filter",
        "question": "List students who are older than 18.",
        "sql": "SELECT name, age FROM students WHERE age > 18",
        "description": "Single WHERE — Container + Funnel (single predicate)"
    },
    {
        "name": "04_multiple_filters",
        "question": "Find male students from Beijing under 25.",
        "sql": "SELECT name FROM students WHERE gender='M' AND city='Beijing' AND age < 25",
        "description": "Multiple WHERE — Funnel with 3 conditions (高filter_intensity)"
    },
    {
        "name": "05_like_filter",
        "question": "Find products with name containing 'Apple'.",
        "sql": "SELECT product_name, price FROM products WHERE product_name LIKE '%Apple%'",
        "description": "LIKE filter — Funnel (模糊匹配)"
    },
    # ===== 聚合查询 =====
    {
        "name": "06_single_aggregation",
        "question": "How many students are there?",
        "sql": "SELECT COUNT(*) AS total FROM students",
        "description": "COUNT(*) — Stacked bar (单聚合)"
    },
    {
        "name": "07_multiple_aggregations",
        "question": "What's the average age and total salary of employees?",
        "sql": "SELECT AVG(age), SUM(salary) FROM employees",
        "description": "AVG + SUM — Stack with 2 aggregations"
    },
    # ===== GROUP BY分组聚合 =====
    {
        "name": "08_group_by_count",
        "question": "Count students per department.",
        "sql": "SELECT dept, COUNT(*) FROM students GROUP BY dept",
        "description": "GROUP BY COUNT — Stack bars per group (分组柱状图)"
    },
    {
        "name": "09_group_by_having",
        "question": "Show departments with more than 10 students.",
        "sql": "SELECT dept, COUNT(*) FROM students GROUP BY dept HAVING COUNT(*) > 10",
        "description": "GROUP BY + HAVING — Stack + Funnel after aggregation (Scope边界)"
    },
    {
        "name": "10_group_avg_sum",
        "question": "Average score and total students per class.",
        "sql": "SELECT class_id, AVG(score), COUNT(*) FROM students GROUP BY class_id",
        "description": "AVG + COUNT by class — Multi-column stacked group"
    },
    # ===== JOIN查询 =====
    {
        "name": "11_inner_join",
        "question": "List students and their enrolled courses.",
        "sql": "SELECT s.name, c.title FROM students s INNER JOIN courses c ON s.id = c.student_id",
        "description": "INNER JOIN — Two containers + Branch connection"
    },
    {
        "name": "12_left_join",
        "question": "All departments and their employees (include empty depts).",
        "sql": "SELECT d.name, e.name FROM departments d LEFT JOIN employees e ON d.id = e.dept_id",
        "description": "LEFT JOIN — Asymmetric branch (保留左容器所有行)"
    },
    {
        "name": "13_join_where",
        "question": "Engineering employees earning over 10000.",
        "sql": "SELECT e.name, d.name FROM employees e JOIN departments d ON e.dept_id = d.id WHERE d.name='Engineering' AND e.salary > 10000",
        "description": "JOIN + WHERE — Two containers + Branch + Funnel after"
    },
    {
        "name": "14_triple_join",
        "question": "Students, their courses, and professors.",
        "sql": "SELECT s.name, c.title, p.name FROM students s JOIN enrollments e ON s.id = e.sid JOIN courses c ON e.cid = c.id JOIN professors p ON c.pid = p.id",
        "description": "3-way JOIN — Multiple connected containers (高join_density)"
    },
    # ===== ORDER BY / LIMIT =====
    {
        "name": "15_order_limit",
        "question": "Top 5 highest paid employees.",
        "sql": "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 5",
        "description": "ORDER BY + LIMIT — Sorted result + capped output"
    },
    # ===== 嵌套子查询 =====
    {
        "name": "16_subquery_where_in",
        "question": "Students enrolled in Computer Science courses.",
        "sql": "SELECT name FROM students WHERE id IN (SELECT student_id FROM enrollments WHERE course_dept = 'CS')",
        "description": "Subquery IN (WHERE) — Boundary box wraps inner subquery scope"
    },
    {
        "name": "17_subquery_comparison",
        "question": "Employees earning above average.",
        "sql": "SELECT name, salary FROM employees WHERE salary > (SELECT AVG(salary) FROM employees)",
        "description": "Subquery scalar comparison — Boundary with threshold"
    },
    {
        "name": "18_correlated_subquery",
        "question": "Employees earning more than their dept average.",
        "sql": "SELECT e.name FROM employees e WHERE e.salary > (SELECT AVG(salary) FROM employees WHERE dept_id = e.dept_id)",
        "description": "Correlated subquery — Outer-Inner scope with linkage + Boundary"
    },
    {
        "name": "19_derived_table_from",
        "question": "Top earners from the aggregated result.",
        "sql": "SELECT dept, avg_sal FROM (SELECT dept, AVG(salary) AS avg_sal FROM employees GROUP BY dept) t WHERE avg_sal > 8000",
        "description": "Subquery in FROM — Boundary wraps aggregated inner result"
    },
    {
        "name": "20_nested_nested_subquery",
        "question": "Students in courses taught by professors who have >5 students.",
        "sql": "SELECT s.name FROM students s WHERE s.id IN (SELECT sid FROM enrollments WHERE cid IN (SELECT id FROM courses WHERE pid IN (SELECT pid FROM professors GROUP BY pid HAVING COUNT(*) > 5)))",
        "description": "3-level Nested — Multiple nested boundaries (高nest_depth)"
    },
    # ===== 复合结构（多原语组合）=====
    {
        "name": "21_full_join_group_having",
        "question": "Depts with avg salary > 7000 for employees > 30 years old.",
        "sql": "SELECT d.name, AVG(e.salary) FROM departments d JOIN employees e ON d.id = e.dept_id WHERE e.age > 30 GROUP BY d.name HAVING AVG(e.salary) > 7000",
        "description": "FULL PIPELINE — JOIN + WHERE + GROUP BY + HAVING (所有原语组合)"
    },
    {
        "name": "22_window_function",
        "question": "Ranked salaries within each department.",
        "sql": "SELECT name, dept, RANK() OVER (PARTITION BY dept ORDER BY salary DESC) AS rnk FROM employees",
        "description": "Window Function — Partition-aware stack + ordered result"
    },
    {
        "name": "23_union_set_operation",
        "question": "All customers and suppliers from New York.",
        "sql": "SELECT name, 'Customer' FROM customers WHERE city='NY' UNION SELECT name, 'Supplier' FROM suppliers WHERE city='NY'",
        "description": "UNION — Two stacked result sets with set operation"
    },
    {
        "name": "24_complex_composite",
        "question": "Most popular course per major for CS students over 20 with 3+ courses.",
        "sql": "SELECT m.major, c.title, COUNT(*) FROM majors m JOIN students s ON m.id = s.mid JOIN enrollments e ON s.id = e.sid JOIN courses c ON e.cid = c.id WHERE m.name='CS' AND s.age > 20 GROUP BY m.major, c.title HAVING COUNT(*) >= 3 ORDER BY COUNT(*) DESC",
        "description": "COMPOSITE — 4-table JOIN + WHERE + GROUP BY + HAVING + ORDER BY"
    },
]


def slugify(name):
    return re.sub(r'[^\w\-]', '_', name)


def main():
    output_dir = project_root / "results" / "sql_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("用真实管线生成 SQL→可视化原语 SVG")
    print("=" * 60)

    print("[INIT] 加载 LayoutGenerator...")
    try:
        lg = LayoutGenerator()
        print("[OK] LayoutGenerator 加载成功")
    except Exception as e:
        print(f"[FAIL] 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    success = 0
    failures = []

    for idx, sample in enumerate(SQL_SAMPLES):
        name = sample["name"]
        sql = sample["sql"]
        question = sample["question"]
        desc = sample["description"]

        print(f"\n[{idx+1:02d}/{len(SQL_SAMPLES)}] {name}")
        print(f"  SQL: {sql[:80]}{'...' if len(sql)>80 else ''}")
        print(f"  Desc: {desc}")

        try:
            # 真实管线：双编码器 → MLP指纹预测 → β调制 → 几何解码 → 锚点约束 → SVG渲染
            svg_str = lg.generate_visualization(sql, question)

            if svg_str is None:
                print(f"  [FAIL] 返回 None")
                failures.append(name)
                continue

            # 保存SVG
            svg_path = output_dir / f"{slugify(name)}.svg"
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(svg_str)

            # 解析SVG获取canvas尺寸
            try:
                root = ET.fromstring(svg_str)
                w = root.get("width", "?")
                h = root.get("height", "?")
                print(f"  [OK] → {svg_path.name} ({w}×{h})")
            except:
                print(f"  [OK] → {svg_path.name}")

            success += 1

        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback
            traceback.print_exc()
            failures.append(name)

    print("\n" + "=" * 60)
    print(f"完成: {success}/{len(SQL_SAMPLES)} 成功")
    if failures:
        print(f"失败: {len(failures)} 个: {', '.join(failures)}")
    print(f"输出目录: {output_dir}")

    # 生成索引HTML
    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>SQL → Visualization Primitives Gallery</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: "Segoe UI", "Microsoft YaHei", Arial; background: #f0f2f5; padding: 20px; }
h1 { text-align: center; color: #1a1a2e; margin-bottom: 8px; font-size: 24px; }
.subtitle { text-align: center; color: #666; margin-bottom: 30px; font-size: 13px; }
.gallery { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; max-width: 1600px; margin: 0 auto; }
.card { background: #fff; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); overflow: hidden; }
.card-head { padding: 14px 18px; border-bottom: 1px solid #eee; }
.card-num { display: inline-flex; width: 26px; height: 26px; background: #3498DB; color: #fff;
            border-radius: 50%; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; margin-right: 8px; }
.card-title { font-size: 14px; font-weight: 600; color: #333; }
.card-question { font-size: 11px; color: #888; margin-top: 4px; }
.card-body { background: #fafbfc; padding: 14px; text-align: center; }
.card-body svg { max-width: 100%; height: auto; border-radius: 6px; background: #fff; }
.card-sql { padding: 10px 18px; background: #1a1a2e; color: #dcdcdc; font-family: Consolas, monospace; font-size: 11px; border-top: 1px solid #333; white-space: pre-wrap; word-break: break-all; }
.card-desc { padding: 8px 18px; font-size: 11px; color: #666; border-top: 1px solid #eee; }
.card-desc b { color: #3498DB; }
</style>
</head>
<body>
<h1>SQL → Visualization Primitives Gallery (真实管线生成)</h1>
<p class="subtitle">Dual-Encoder → 16-Dim Fingerprint → β Entropy Modulation → Anchor Constraints → Symbolic SVG Rendering</p>
<div class="gallery">
"""

    for idx, sample in enumerate(SQL_SAMPLES):
        name = sample["name"]
        fname = f"{slugify(name)}.svg"
        html += f"""
<div class="card">
  <div class="card-head">
    <span class="card-num">{idx+1}</span>
    <span class="card-title">{name[3:]}</span>
    <div class="card-question">Q: {sample['question']}</div>
  </div>
  <div class="card-body">
    <object data="{fname}" type="image/svg+xml" style="max-width:100%; max-height:350px;"></object>
  </div>
  <pre class="card-sql">{sample['sql']}</pre>
  <div class="card-desc">
    <b>结构:</b> {sample['description']}
  </div>
</div>
"""

    html += "</div></body></html>"
    with open(output_dir / "index.html", 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n索引页: {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
