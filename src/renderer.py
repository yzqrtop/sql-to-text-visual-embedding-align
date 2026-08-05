"""
确定性符号渲染模块 (Methodology §6)

将布局参数转换为SVG图形，实现：
- Containment公式: A_i = p_area_i · log(1 + |T_i|) + c
- Pathfinding公式: ω_ij = p_thick_ij · log(1 + |T_i ⋈ T_j|)
- Flow公式: α_filter = p_opac · (1 - γ^ξ), ξ=#ops + #nested conds
- 双路径编码 Dual-Path Encoding:
    低 β (低熵，注意力集中) → Active De-emphasis (灰度 + 低透明度0.2)
    高 β (高熵，注意力分散，歧义) → Visual Pop-out (高饱和警告色 + 粗实线边框)
"""

import math
import numpy as np
from typing import Dict, Any, Optional
from xml.etree.ElementTree import Element, SubElement, tostring
from .config import (
    RENDERER_W, RENDERER_H,
    BETA_THRESHOLD, HIGH_BETA_SATURATION, LOW_BETA_OPACITY, WARNING_COLOR
)


class Renderer:
    """SVG渲染器

    根据布局参数生成SVG图形，包含容器、漏斗、分支、堆叠等视觉原语。
    渲染过程完全确定性，确保可复现性。
    """

    def __init__(self, width: int = RENDERER_W, height: int = RENDERER_H):
        self.width = width
        self.height = height
        self._beta_cache: Dict[str, float] = {}  # 原语 -> β值缓存

    # ================================================================
    # 双路径编码辅助方法 (Methodology §6)
    # ================================================================

    def _set_beta_context(self, params: Dict[str, Dict[str, Any]]):
        """缓存beta调制系数，从params._meta中提取"""
        meta = params.get("_meta", {})
        beta = meta.get("beta", {})
        # 4维beta分组 → 原语的默认beta
        mapping = {
            "container": beta.get("entity", 0.3),
            "container_left": beta.get("entity", 0.3),
            "container_right": beta.get("entity", 0.3),
            "boundary": beta.get("entity", 0.3),
            "funnel": beta.get("cond", 0.3),
            "branch": beta.get("join", 0.3),
            "stack": beta.get("res", 0.3),
            "result_container": beta.get("res", 0.3),
        }
        self._beta_cache = mapping

    def _dual_path_style(self, prim_key: str, default_fill: str,
                          default_stroke: str, default_opacity: float,
                          default_stroke_w: float = 2.0) -> Dict[str, Any]:
        """
        双路径编码样式 (Methodology §6)

        - 低 β (< threshold): Active De-emphasis → 灰度化, 透明度 = LOW_BETA_OPACITY
        - 高 β (>= threshold): Visual Pop-out → 高饱和警告色, 粗实线边框 *HIGH_BETA_SATURATION

        Args:
            prim_key: 原语类型 (container, funnel, ...)
            default_fill: 默认填充色
            default_stroke: 默认描边色
            default_opacity: 默认透明度
            default_stroke_w: 默认描边粗细

        Returns:
            dict: {fill, stroke, opacity, stroke_width, filter, extra_stroke}
        """
        beta = self._beta_cache.get(prim_key, 0.3)
        style = {
            "fill": default_fill,
            "stroke": default_stroke,
            "opacity": default_opacity,
            "stroke_width": default_stroke_w,
            "filter": None,
            "extra_stroke": None,
            "beta": beta,
            "path": "balanced"
        }

        if beta < BETA_THRESHOLD:
            # --- 低熵路径：Active De-emphasis 去强调
            style["opacity"] = max(LOW_BETA_OPACITY, default_opacity * 0.55)
            style["filter"] = "grayscale(80%)"
            style["stroke_width"] = max(0.5, default_stroke_w * 0.7)
            style["path"] = "low_beta"
        else:
            # --- 高熵路径：Visual Pop-out 歧义警告
            gain = HIGH_BETA_SATURATION
            style["fill"] = WARNING_COLOR
            style["stroke"] = "#922B21"
            style["stroke_width"] = default_stroke_w * gain
            style["extra_stroke"] = WARNING_COLOR
            style["path"] = "high_beta"

        return style

    def _apply_style_attrs(self, elem: Element, style: Dict[str, Any]):
        """将样式应用到SVG元素"""
        if style.get("fill") is not None:
            elem.set("fill", style["fill"])
        if style.get("stroke") is not None:
            elem.set("stroke", style["stroke"])
        if style.get("stroke_width") is not None:
            elem.set("stroke-width", str(style["stroke_width"]))
        op = style.get("opacity")
        if op is not None and op < 1.0:
            elem.set("opacity", f"{op:.3f}")
        if style.get("filter") and "filter" not in elem.attrib:
            # SVG filter: 灰度化
            pass  # 用 opacity + 冷色模拟以简化

    def analyze_sql(self, sql: str) -> Dict[str, bool]:
        """
        分析SQL结构，确定需要哪些原语

        Args:
            sql: SQL语句

        Returns:
            原语需求字典
        """
        if not sql:
            return {}

        sql_upper = sql.upper()

        return {
            "container": "FROM" in sql_upper,
            "funnel": "WHERE" in sql_upper or "HAVING" in sql_upper,
            "branch": "JOIN" in sql_upper,
            "stack": "GROUP BY" in sql_upper or any(
                func in sql_upper for func in ["COUNT", "SUM", "AVG", "MAX", "MIN"]
            ),
            "boundary": "(" in sql and "SELECT" in sql_upper[sql_upper.find("("):]
        }

    def _remove_subqueries(self, sql: str) -> str:
        """
        移除SQL语句中的子查询，用(Subquery)替换
        """
        result = []
        depth = 0
        in_subquery = False
        skip_next_right_paren = False
        
        i = 0
        while i < len(sql):
            if sql[i] == '(':
                depth += 1
                if depth == 1 and i + 1 < len(sql):
                    j = i + 1
                    while j < len(sql) and sql[j].isspace():
                        j += 1
                    if sql[j:j+6].upper() == 'SELECT':
                        in_subquery = True
                        skip_next_right_paren = True
                if not in_subquery:
                    result.append(sql[i])
                i += 1
            elif sql[i] == ')':
                depth -= 1
                if depth == 0 and in_subquery:
                    result.append('(Subquery)')
                    in_subquery = False
                    skip_next_right_paren = False
                elif not skip_next_right_paren:
                    result.append(sql[i])
                i += 1
            elif in_subquery:
                i += 1
            else:
                result.append(sql[i])
                i += 1
        
        return ''.join(result)

    def extract_sql_attributes(self, sql: str) -> Dict[str, list]:
        """
        提取SQL语句中的属性信息

        Args:
            sql: SQL语句

        Returns:
            属性信息字典，包含select_cols, where_attrs, join_attrs, group_by_cols
        """
        import re

        result = {
            "select_cols": [],
            "where_attrs": [],
            "join_attrs": [],
            "group_by_cols": [],
            "table_names": []
        }

        if not sql:
            return result

        sql = sql.strip()

        sql_main = sql
        sql_main = self._remove_subqueries(sql_main)

        select_match = re.search(r'\bSELECT\s+(.*?)\s+\bFROM\b', sql_main, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_content = select_match.group(1).strip()
            select_content = re.sub(r'/\*.*?\*/', '', select_content)
            select_content = re.sub(r'--.*?$', '', select_content, flags=re.MULTILINE)
            
            parts = re.split(r',\s*(?![^()]*\))', select_content)
            for part in parts:
                part = part.strip()
                if part:
                    agg_match = re.match(r'(COUNT|SUM|AVG|MAX|MIN)\s*\(\s*(?:DISTINCT\s+)?([a-zA-Z_][a-zA-Z0-9_]*|\*)\s*\)', part, re.IGNORECASE)
                    if agg_match:
                        result["select_cols"].append(f"{agg_match.group(1)}({agg_match.group(2)})")
                    elif part.upper() == '*':
                        result["select_cols"].append("*")
                    else:
                        col_name = re.sub(r'\bAS\s+\w+\b', '', part, flags=re.IGNORECASE).strip()
                        col_name = col_name.split('.')[-1]
                        result["select_cols"].append(col_name)

        from_pattern = re.compile(r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+AS\s+[a-zA-Z_][a-zA-Z0-9_]*)?', re.IGNORECASE)
        from_match = from_pattern.search(sql_main)
        if from_match:
            result["table_names"].append(from_match.group(1))

        join_pattern = re.compile(r'\b(LEFT|RIGHT|INNER|OUTER|FULL)?\s*JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+AS\s+[a-zA-Z_][a-zA-Z0-9_]*)?\s+(?:ON\s+(.+?))?(?=\s+\b(LEFT|RIGHT|INNER|OUTER|FULL)?\s*JOIN|\s+\bWHERE|\s+\bGROUP|\s+\bHAVING|\s+\bORDER|\s+\bLIMIT|$)', re.IGNORECASE | re.DOTALL)
        for join_match in join_pattern.finditer(sql_main):
            table_name = join_match.group(2)
            if table_name:
                result["table_names"].append(table_name)
            on_condition = join_match.group(3)
            if on_condition:
                attrs = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)', on_condition)
                for alias1, col1, alias2, col2 in attrs:
                    result["join_attrs"].append(f"{col1} = {col2}")
                
                if not attrs:
                    attrs = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*)', on_condition)
                    for attr1, attr2 in attrs:
                        result["join_attrs"].append(f"{attr1} = {attr2}")

        where_pattern = re.compile(r'\bWHERE\s+(.+?)(?=\s+\bGROUP|\s+\bHAVING|\s+\bORDER|\s+\bLIMIT|\s+\bINTERSECT|\s+\bUNION|\s+\bEXCEPT|$)', re.IGNORECASE | re.DOTALL)
        where_match = where_pattern.search(sql_main)
        if where_match:
            where_content = where_match.group(1).strip()
            where_content = re.sub(r'/\*.*?\*/', '', where_content)
            where_content = re.sub(r'--.*?$', '', where_content, flags=re.MULTILINE)
            
            conditions = re.split(r'\bAND\b|\bOR\b', where_content, flags=re.IGNORECASE)
            for cond in conditions:
                cond = cond.strip()
                if cond:
                    cond_clean = cond.replace('"', "'").strip()
                    result["where_attrs"].append(cond_clean)

        having_pattern = re.compile(r'\bHAVING\s+(.+?)(?=\s+\bORDER|\s+\bLIMIT|$)', re.IGNORECASE | re.DOTALL)
        having_match = having_pattern.search(sql_main)
        if having_match:
            having_content = having_match.group(1).strip()
            having_content = re.sub(r'/\*.*?\*/', '', having_content)
            having_content = re.sub(r'--.*?$', '', having_content, flags=re.MULTILINE)
            
            attr_matches = re.findall(r'(COUNT|SUM|AVG|MAX|MIN)\s*\(\s*(?:DISTINCT\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\)', having_content, re.IGNORECASE)
            for agg_func, attr in attr_matches:
                result["where_attrs"].append(f"{agg_func}({attr})")

        group_by_pattern = re.compile(r'\bGROUP\s+BY\s+(.+?)(?=\s+\bHAVING|\s+\bORDER|\s+\bLIMIT|$)', re.IGNORECASE | re.DOTALL)
        group_by_match = group_by_pattern.search(sql_main)
        if group_by_match:
            group_by_content = group_by_match.group(1).strip()
            group_by_content = re.sub(r'/\*.*?\*/', '', group_by_content)
            group_by_content = re.sub(r'--.*?$', '', group_by_content, flags=re.MULTILINE)
            
            cols = re.split(r',\s*(?![^()]*\))', group_by_content)
            for col in cols:
                col = col.strip()
                if col:
                    col_name = col.split('.')[-1]
                    result["group_by_cols"].append(col_name)

        return result

    def render(self, params: Dict[str, Dict[str, float]], sql: str = "", question: str = "") -> str:
        """
        确定性符号渲染（流水线布局）

        根据布局参数按顺序绘制各视觉原语，输出SVG图像。
        将原语按SQL执行流程分层排列（源表→过滤→聚合→结果），并绘制连接线。
        渲染过程完全确定性，确保可复现性。

        实现 §6 双路径编码 Dual-Path Encoding：
            低 β (< BETA_THRESHOLD): Active De-emphasis → 灰度 + 低透明度
            高 β (≥ BETA_THRESHOLD): Visual Pop-out → 警告红 + 粗边框

        Args:
            params: 布局参数字典（已修正，含_meta信息）
            sql: SQL语句（用于添加解释和属性提取）
            question: 自然语言问题

        Returns:
            SVG字符串
        """
        svg = Element("svg")
        svg.set("xmlns", "http://www.w3.org/2000/svg")
        svg.set("width", str(self.width))
        svg.set("height", str(self.height))
        svg.set("viewBox", f"0 0 {self.width} {self.height}")

        # 定义SVG滤镜：高β Visual Pop-out 用的阴影滤镜
        defs = SubElement(svg, "defs")
        # 警告阴影滤镜 — 高β模块的视觉跳变效果
        shadow_filter = SubElement(defs, "filter")
        shadow_filter.set("id", "highBetaShadow")
        shadow_filter.set("x", "-30%")
        shadow_filter.set("y", "-30%")
        shadow_filter.set("width", "160%")
        shadow_filter.set("height", "160%")
        # 外发光
        fe_glow = SubElement(shadow_filter, "feGaussianBlur")
        fe_glow.set("stdDeviation", "4")
        fe_glow.set("result", "coloredBlur")
        fe_merge = SubElement(shadow_filter, "feMerge")
        fe_merge_node1 = SubElement(fe_merge, "feMergeNode")
        fe_merge_node1.set("in", "coloredBlur")
        fe_merge_node2 = SubElement(fe_merge, "feMergeNode")
        fe_merge_node2.set("in", "SourceGraphic")
        # 灰度滤镜 — 低β去强调
        gray_filter = SubElement(defs, "filter")
        gray_filter.set("id", "lowBetaGrayscale")
        fe_gray = SubElement(gray_filter, "feColorMatrix")
        fe_gray.set("type", "matrix")
        fe_gray.set("values", "0.3 0.3 0.3 0 0  0.3 0.3 0.3 0 0  0.3 0.3 0.3 0 0  0 0 0 1 0")

        # 准备双路径编码的beta上下文
        self._set_beta_context(params)

        attributes = self.extract_sql_attributes(sql)

        params, last_y = self._apply_pipeline_layout(params, attributes)

        if "boundary" in params:
            self._draw_primitive(svg, "boundary", params["boundary"], attributes)
            self._draw_subquery_flow_arrow(svg, params)

        self._draw_pipeline_connections(svg, params)

        drawing_order = ["container", "container_left", "container_right", 
                         "funnel", "branch", "stack", "result_container", 
                         "arrow", "arrow_0", "arrow_1", "arrow_2"]

        for prim_key in drawing_order:
            if prim_key in params:
                prim_params = params[prim_key]
                self._draw_primitive(svg, prim_key, prim_params, attributes)

        self._add_legend_right(svg, params, sql)

        # 绘制NL高注意力词标注（在各原语旁标注对应的NL关键词）
        self._draw_attn_word_annotations(svg, params)

        sql_bottom = self._add_sql_info(svg, sql, question, last_y)

        final_height = max(sql_bottom + 20, 500)
        
        svg.set("height", str(final_height))
        svg.set("viewBox", f"0 0 {self.width} {final_height}")

        svg_str = tostring(svg, encoding="unicode")

        return svg_str
    
    def _draw_subquery_flow_arrow(self, parent: Element, params: Dict[str, Dict[str, float]]):
        """绘制子查询流向箭头 - 从boundary指向result_container"""
        if "boundary" not in params:
            return
        
        boundary = params["boundary"]
        boundary_bottom_x = boundary.get("x", 10) + boundary.get("width", 600) / 2
        boundary_bottom_y = boundary.get("y", 10) + boundary.get("height", 400)
        
        if "result_container" in params:
            result = params["result_container"]
            target_x = result.get("x", 0) + result.get("width", 150) / 2
            target_y = result.get("y", 0)
        elif "container" in params:
            container = params["container"]
            target_x = container.get("x", 0) + container.get("width", 150) / 2
            target_y = container.get("y", 0)
        else:
            return
        
        path_d = f"M {boundary_bottom_x} {boundary_bottom_y} C {boundary_bottom_x} {boundary_bottom_y + 30}, {target_x} {target_y - 30}, {target_x} {target_y - 5}"
        
        flow_path = SubElement(parent, "path")
        flow_path.set("d", path_d)
        flow_path.set("stroke", "#FF9800")
        flow_path.set("stroke-width", "2")
        flow_path.set("stroke-dasharray", "6,3")
        flow_path.set("fill", "none")
        flow_path.set("opacity", "0.8")
        
        arrow_size = 8
        arrow_angle = math.radians(30)
        
        dx = target_x - boundary_bottom_x
        dy = (target_y - 5) - boundary_bottom_y
        angle = math.atan2(dy, dx) if dx != 0 else math.pi / 2
        
        arrow_x1 = target_x - arrow_size * math.cos(angle + arrow_angle)
        arrow_y1 = (target_y - 5) - arrow_size * math.sin(angle + arrow_angle)
        arrow_x2 = target_x - arrow_size * math.cos(angle - arrow_angle)
        arrow_y2 = (target_y - 5) - arrow_size * math.sin(angle - arrow_angle)
        
        arrow_line1 = SubElement(parent, "line")
        arrow_line1.set("x1", str(target_x))
        arrow_line1.set("y1", str(target_y - 5))
        arrow_line1.set("x2", str(arrow_x1))
        arrow_line1.set("y2", str(arrow_y1))
        arrow_line1.set("stroke", "#FF9800")
        arrow_line1.set("stroke-width", "2")
        
        arrow_line2 = SubElement(parent, "line")
        arrow_line2.set("x1", str(target_x))
        arrow_line2.set("y1", str(target_y - 5))
        arrow_line2.set("x2", str(arrow_x2))
        arrow_line2.set("y2", str(arrow_y2))
        arrow_line2.set("stroke", "#FF9800")
        arrow_line2.set("stroke-width", "2")
        
        mid_x = (boundary_bottom_x + target_x) / 2 + 15
        mid_y = (boundary_bottom_y + target_y) / 2
        flow_label = SubElement(parent, "text")
        flow_label.set("x", str(mid_x))
        flow_label.set("y", str(mid_y))
        flow_label.set("fill", "#E65100")
        flow_label.set("font-size", "9")
        flow_label.set("font-style", "italic")
        flow_label.text = "subquery result"

    def _draw_attn_word_annotations(self, svg: Element, params: Dict[str, Dict[str, float]]):
        """
        在各视觉原语旁标注NL中注意力最高的词

        映射关系 (Methodology §4):
        - entity → container / container_left / container_right / boundary
        - join → branch
        - cond → funnel
        - res → stack / result_container
        """
        meta = params.get("_meta", {})
        attn_words = meta.get("attn_words", {})
        if not attn_words:
            return

        # 组 → 原语键映射
        group_to_prims = {
            "entity": ["container", "container_left", "container_right", "boundary"],
            "join": ["branch"],
            "cond": ["funnel"],
            "res": ["stack", "result_container"],
        }

        # 组 → 标注颜色
        group_colors = {
            "entity": "#2980B9",
            "join": "#8E44AD",
            "cond": "#E67E22",
            "res": "#27AE60",
        }

        for g_name, prim_keys in group_to_prims.items():
            words = attn_words.get(g_name, [])
            if not words:
                continue

            # 找到存在的原语
            target_prim = None
            for pk in prim_keys:
                if pk in params:
                    target_prim = params[pk]
                    break
            if not target_prim:
                continue

            # 获取原语位置
            px = target_prim.get("x", 100)
            py = target_prim.get("y", 100)
            pw = target_prim.get("width", target_prim.get("top_width", 150))
            ph = target_prim.get("height", target_prim.get("max_height", target_prim.get("length", 80)))

            color = group_colors.get(g_name, "#555")

            # 在原语右侧绘制注意力词标签
            label_x = px + pw + 12
            label_y_start = py + 5

            # 背景框
            max_word_len = max(len(w[0]) for w in words) if words else 5
            bg_w = max(70, max_word_len * 7 + 35)
            bg_h = len(words) * 16 + 22

            # 确保不超出画布右边界
            if label_x + bg_w > self.width - 10:
                label_x = px - bg_w - 12

            bg = SubElement(svg, "rect")
            bg.set("x", str(label_x))
            bg.set("y", str(label_y_start))
            bg.set("width", str(bg_w))
            bg.set("height", str(bg_h))
            bg.set("fill", color)
            bg.set("fill-opacity", "0.08")
            bg.set("stroke", color)
            bg.set("stroke-width", "1")
            bg.set("stroke-dasharray", "3,2")
            bg.set("rx", "4")
            bg.set("ry", "4")

            # 组名标题
            title = SubElement(svg, "text")
            title.set("x", str(label_x + 6))
            title.set("y", str(label_y_start + 13))
            title.set("fill", color)
            title.set("font-size", "9")
            title.set("font-weight", "bold")
            g_labels = {"entity": "Entity", "join": "Join", "cond": "Cond", "res": "Result"}
            title.text = f"NL attn: {g_labels.get(g_name, g_name)}"

            # 连接线（从标签框指向原语）
            line = SubElement(svg, "line")
            line.set("x1", str(label_x))
            line.set("y1", str(label_y_start + bg_h / 2))
            line.set("x2", str(px + pw))
            line.set("y2", str(py + ph / 2))
            line.set("stroke", color)
            line.set("stroke-width", "1")
            line.set("stroke-dasharray", "2,2")
            line.set("opacity", "0.5")

            # 每个词
            for i, (word, weight) in enumerate(words):
                wy = label_y_start + 28 + i * 16
                # 词文本
                wt = SubElement(svg, "text")
                wt.set("x", str(label_x + 8))
                wt.set("y", str(wy))
                wt.set("fill", color)
                wt.set("font-size", "10")
                wt.set("font-weight", "bold" if weight > 0.1 else "normal")
                wt.text = word

                # 权重条
                bar_w = min(30, max(5, weight * 200))
                bar = SubElement(svg, "rect")
                bar.set("x", str(label_x + bg_w - bar_w - 8))
                bar.set("y", str(wy - 8))
                bar.set("width", str(bar_w))
                bar.set("height", "8")
                bar.set("fill", color)
                bar.set("fill-opacity", f"{min(1.0, weight * 3):.2f}")
                bar.set("rx", "2")

    def _apply_pipeline_layout(self, params: Dict[str, Dict[str, float]], attributes: Dict[str, list] = None) -> tuple:
        """
        应用流水线布局：将原语按SQL执行流程分层排列

        层级顺序：
        - 第1层：源表容器 (container) - 顶部
        - 第2层：漏斗/过滤 (funnel) - 中间偏上
        - 第3层：分支/JOIN (branch) - 中间
        - 第4层：堆叠/聚合 (stack) - 中间偏下
        - 第5层：结果容器 (result_container) - 底部

        Args:
            params: 原始布局参数字典
            attributes: SQL属性信息字典

        Returns:
            应用流水线布局后的参数字典
        """
        if attributes is None:
            attributes = {}

        has_join = "branch" in params or ("join_attrs" in attributes and attributes["join_attrs"])
        table_names = attributes.get("table_names", [])

        if has_join and len(table_names) >= 2:
            header_height = 28
            row_height = 22
            
            select_cols = attributes.get("select_cols", [])
            num_display = min(len(select_cols), 5)
            
            if num_display > 0:
                max_col_len = max(len(col) for col in select_cols[:num_display])
                min_width = 80
                max_width = 180
                container_width = min(max(min_width + max_col_len * 7, min_width), max_width)
                container_height = header_height + 12 + num_display * row_height
                container_height = max(container_height, 70)
                container_height = min(container_height, 120)
            else:
                container_width = 100
                container_height = 70
            
            gap = 80
            total_width = container_width * 2 + gap
            
            left_x = self.width / 2 - total_width / 2
            right_x = left_x + container_width + gap
            
            params["container_left"] = {
                "x": left_x,
                "y": 40,
                "width": container_width,
                "height": container_height,
                "table_names": [table_names[0]]
            }
            
            params["container_right"] = {
                "x": right_x,
                "y": 40,
                "width": container_width,
                "height": container_height,
                "table_names": [table_names[1]]
            }
            
            if "container" in params:
                del params["container"]

        vertical_gap = 15
        
        current_y = 40
        
        has_two_containers = "container_left" in params and "container_right" in params
        
        if "container" in params:
            select_cols = attributes.get("select_cols", [])
            num_display = min(len(select_cols), 5)
            header_height = 28
            
            if num_display > 0:
                max_col_len = max(len(col) for col in select_cols[:num_display])
                min_width = 60
                max_width = 130
                container_w = min(max(min_width + max_col_len * 6, min_width), max_width)
                
                table_names = attributes.get("table_names", params.get("container", {}).get("table_names", []))
                if table_names:
                    max_name_len = max(len(name) for name in table_names)
                    name_width = max_name_len * 7 + 24
                    container_w = max(container_w, name_width)
                
                row_height = 22
                container_h = header_height + 12 + num_display * row_height
                container_h = max(container_h, 70)
                container_h = min(container_h, 100)
            else:
                container_w = 100
                container_h = 70
            
            params["container"]["width"] = container_w
            params["container"]["height"] = container_h
            params["container"]["x"] = self.width / 2 - container_w / 2
            params["container"]["y"] = current_y
            if "table_names" in attributes and attributes["table_names"]:
                params["container"]["table_names"] = attributes["table_names"]
            current_y += container_h + vertical_gap
        
        if has_two_containers:
            left_h = params["container_left"].get("height", 120)
            right_h = params["container_right"].get("height", 120)
            max_h = max(left_h, right_h)
            current_y += max_h + vertical_gap
        
        if "branch" in params and has_two_containers:
            params["branch"]["x"] = self.width / 2
            params["branch"]["y"] = current_y
            params["branch"]["num_branches"] = 2
            params["branch"]["draw_lines"] = False
            if "join_attrs" in attributes:
                params["branch"]["join_attrs"] = attributes["join_attrs"]
            current_y += 50 + vertical_gap
        
        if "funnel" in params and has_two_containers:
            container_w = max(params.get("container_left", {}).get("width", 108), 
                             params.get("container_right", {}).get("width", 108))
            container_h = max(params.get("container_left", {}).get("height", 70), 
                             params.get("container_right", {}).get("height", 70))
            
            top_width = max(80, container_w + 15)
            bottom_width = max(25, int(top_width * 0.3))
            funnel_height = max(70, min(int(container_h * 1.2), 85))
            
            params["funnel"]["top_width"] = top_width
            params["funnel"]["bottom_width"] = bottom_width
            params["funnel"]["height"] = funnel_height
            params["funnel"]["x"] = self.width / 2 - top_width / 2
            params["funnel"]["y"] = current_y
            if "where_attrs" in attributes:
                params["funnel"]["where_attrs"] = attributes["where_attrs"]
            current_y += funnel_height + vertical_gap
        
        if "funnel" in params and not has_two_containers:
            container_w = params.get("container", {}).get("width", 108)
            if not container_w:
                container_w = max(params.get("container_left", {}).get("width", 108), 
                                 params.get("container_right", {}).get("width", 108))
            
            container_h = params.get("container", {}).get("height", 70)
            if not container_h:
                container_h = max(params.get("container_left", {}).get("height", 70), 
                                 params.get("container_right", {}).get("height", 70))
            
            top_width = max(80, container_w + 15)
            bottom_width = max(25, int(top_width * 0.3))
            funnel_height = max(70, min(int(container_h * 1.2), 85))
            
            params["funnel"]["top_width"] = top_width
            params["funnel"]["bottom_width"] = bottom_width
            params["funnel"]["height"] = funnel_height
            params["funnel"]["x"] = self.width / 2 - top_width / 2
            params["funnel"]["y"] = current_y
            if "where_attrs" in attributes:
                params["funnel"]["where_attrs"] = attributes["where_attrs"]
            current_y += funnel_height + vertical_gap
        
        if "branch" in params and not has_two_containers:
            params["branch"]["x"] = self.width / 2
            params["branch"]["y"] = current_y
            table_names = attributes.get("table_names", [])
            params["branch"]["num_branches"] = len(table_names) if len(table_names) >= 2 else 2
            if "join_attrs" in attributes:
                params["branch"]["join_attrs"] = attributes["join_attrs"]
            current_y += 50 + vertical_gap
        
        if "stack" in params:
            num_bars = params["stack"].get("num_bars", 5)
            bar_width = params["stack"].get("bar_width", 30)
            spacing = params["stack"].get("spacing", 15)
            total_width = num_bars * (bar_width + spacing) - spacing
            params["stack"]["x"] = self.width / 2 - total_width / 2
            params["stack"]["y"] = current_y
            if "group_by_cols" in attributes:
                params["stack"]["group_by_cols"] = attributes["group_by_cols"]
            current_y += 80 + vertical_gap

        if "result_container" in params:
            source_w = params.get("container", {}).get("width", 108)
            if not source_w:
                source_w = max(params.get("container_left", {}).get("width", 108), 
                               params.get("container_right", {}).get("width", 108))
            result_w = source_w
            result_h = params.get("result_container", {}).get("height", 70)
            params["result_container"]["width"] = result_w
            params["result_container"]["height"] = result_h
            params["result_container"]["x"] = self.width / 2 - result_w / 2
            params["result_container"]["y"] = current_y
            if "select_cols" in attributes:
                params["result_container"]["select_cols"] = attributes["select_cols"]
            current_y += result_h + vertical_gap

        has_processing = any(p in params for p in ["funnel", "branch", "stack", "container_left", "container_right"])
        if has_processing and "result_container" not in params:
            source_w = params.get("container", {}).get("width", 108)
            if not source_w:
                source_w = max(params.get("container_left", {}).get("width", 108), 
                               params.get("container_right", {}).get("width", 108))
            result_params = {
                "x": self.width / 2 - source_w / 2,
                "y": current_y,
                "width": source_w,
                "height": 70,
                "opacity": 0.9
            }
            if "select_cols" in attributes:
                result_params["select_cols"] = attributes["select_cols"]
            params["result_container"] = result_params
            current_y += 70 + vertical_gap

        if "boundary" in params:
            min_x = min(params.get("container_left", {}).get("x", 50), 
                       params.get("container", {}).get("x", 50),
                       params.get("container_right", {}).get("x", 50),
                       params.get("funnel", {}).get("x", 50),
                       params.get("branch", {}).get("x", 300) - 50)
            
            max_x = max(params.get("container_left", {}).get("x", 0) + params.get("container_left", {}).get("width", 180),
                       params.get("container", {}).get("x", 0) + params.get("container", {}).get("width", 180),
                       params.get("container_right", {}).get("x", 0) + params.get("container_right", {}).get("width", 180),
                       params.get("funnel", {}).get("x", 0) + params.get("funnel", {}).get("top_width", 120),
                       params.get("branch", {}).get("x", 0) + 50)
            
            min_y = min(params.get("container_left", {}).get("y", 60),
                       params.get("container", {}).get("y", 60),
                       params.get("container_right", {}).get("y", 60),
                       params.get("funnel", {}).get("y", 230),
                       params.get("branch", {}).get("y", 270))
            
            max_y = max(params.get("result_container", {}).get("y", 380) + params.get("result_container", {}).get("height", 120),
                       params.get("stack", {}).get("y", 420) + params.get("stack", {}).get("max_height", 100),
                       params.get("funnel", {}).get("y", 230) + params.get("funnel", {}).get("height", 100))
            
            max_y = min(max_y, self.height - 80)
            
            params["boundary"]["x"] = min_x - 20
            params["boundary"]["y"] = min_y - 20
            params["boundary"]["width"] = max_x - min_x + 40
            params["boundary"]["height"] = max_y - min_y + 40

        last_y = 0
        for prim_key in ["result_container", "stack", "funnel", "branch", "container", "container_left", "container_right"]:
            if prim_key in params:
                y = params[prim_key].get("y", 0)
                height = params[prim_key].get("height", 0)
                if prim_key == "branch":
                    height = 50
                if prim_key == "funnel":
                    height = params[prim_key].get("height", 100)
                last_y = max(last_y, y + height)
                break
        
        return params, last_y

    def _draw_pipeline_connections(self, parent: Element, params: Dict[str, Dict[str, float]]):
        """
        绘制流水线连接线：在相邻层级的原语之间绘制带箭头的连接线

        Args:
            parent: 父SVG元素
            params: 布局参数字典
        """
        has_join = "branch" in params and ("container_left" in params or "container_right" in params)

        if has_join:
            if "container_left" in params and "branch" in params:
                left_container = params["container_left"]
                branch = params["branch"]
                source_x = left_container.get("x", 0) + left_container.get("width", 160) / 2
                source_y = left_container.get("y", 0) + left_container.get("height", 120)
                target_x = branch.get("x", 300)
                target_y = branch.get("y", 270) - 5
                self._draw_flow_arrow(parent, source_x, source_y, target_x, target_y)

            if "container_right" in params and "branch" in params:
                right_container = params["container_right"]
                branch = params["branch"]
                source_x = right_container.get("x", 0) + right_container.get("width", 160) / 2
                source_y = right_container.get("y", 0) + right_container.get("height", 120)
                target_x = branch.get("x", 300)
                target_y = branch.get("y", 270) - 5
                self._draw_flow_arrow(parent, source_x, source_y, target_x, target_y)

            if "branch" in params and "funnel" in params:
                branch = params["branch"]
                funnel = params["funnel"]
                source_x = branch.get("x", 300)
                source_y = branch.get("y", 270) + 30
                target_x = funnel.get("x", 0) + funnel.get("top_width", 120) / 2
                target_y = funnel.get("y", 0) - 5
                self._draw_flow_arrow(parent, source_x, source_y, target_x, target_y)

            if "branch" in params and "result_container" in params and "funnel" not in params:
                branch = params["branch"]
                result = params["result_container"]
                source_x = branch.get("x", 300)
                source_y = branch.get("y", 270) + 30
                target_x = result.get("x", 0) + result.get("width", 150) / 2
                target_y = result.get("y", 0) - 5
                self._draw_flow_arrow(parent, source_x, source_y, target_x, target_y)

            if "funnel" in params and "result_container" in params:
                funnel = params["funnel"]
                result = params["result_container"]
                source_x = funnel.get("x", 0) + funnel.get("top_width", 120) / 2
                source_y = funnel.get("y", 0) + funnel.get("height", 100) + 5
                target_x = result.get("x", 0) + result.get("width", 150) / 2
                target_y = result.get("y", 0) - 5
                self._draw_flow_arrow(parent, source_x, source_y, target_x, target_y)

            return

        layer_order = ["container", "funnel", "branch", "stack", "result_container"]
        visible_layers = [l for l in layer_order if l in params]

        for i in range(len(visible_layers) - 1):
            source_layer = visible_layers[i]
            target_layer = visible_layers[i + 1]

            source_params = params[source_layer]
            target_params = params[target_layer]

            if source_layer == "container":
                source_x = source_params.get("x", 0) + source_params.get("width", 200) / 2
                source_y = source_params.get("y", 0) + source_params.get("height", 150)
            elif source_layer == "funnel":
                source_x = source_params.get("x", 0) + source_params.get("top_width", 120) / 2
                source_y = source_params.get("y", 0) + source_params.get("height", 100) + 5
            elif source_layer == "branch":
                source_x = source_params.get("x", 300)
                source_y = source_params.get("y", 270) + 30
            elif source_layer == "stack":
                source_x = source_params.get("x", 100) + (source_params.get("num_bars", 5) * 
                          (source_params.get("bar_width", 30) + source_params.get("spacing", 15)) - 
                          source_params.get("spacing", 15)) / 2
                source_y = source_params.get("y", 200) + source_params.get("max_height", 100) + 5
            else:
                continue

            if target_layer == "funnel":
                target_x = target_params.get("x", 0) + target_params.get("top_width", 120) / 2
                target_y = target_params.get("y", 0) - 5
            elif target_layer == "branch":
                target_x = target_params.get("x", 300)
                target_y = target_params.get("y", 270) - 5
            elif target_layer == "stack":
                target_x = target_params.get("x", 100) + (target_params.get("num_bars", 5) * 
                          (target_params.get("bar_width", 30) + target_params.get("spacing", 15)) - 
                          target_params.get("spacing", 15)) / 2
                target_y = target_params.get("y", 200) - 5
            elif target_layer == "result_container":
                target_x = target_params.get("x", 0) + target_params.get("width", 150) / 2
                target_y = target_params.get("y", 0) - 5
            else:
                continue

            self._draw_flow_arrow(parent, source_x, source_y, target_x, target_y)

    def _draw_flow_arrow(self, parent: Element, x1: float, y1: float, x2: float, y2: float):
        """
        绘制带箭头的数据流连接线

        Args:
            parent: 父SVG元素
            x1, y1: 起点坐标
            x2, y2: 终点坐标
        """
        offset = 10
        
        start_y = y1 + offset
        end_y = y2 - offset
        
        if abs(x1 - x2) < 5:
            path_d = f"M {x1} {start_y} L {x2} {end_y}"
        else:
            path_d = f"M {x1} {start_y} C {x1} {start_y + (end_y - start_y) / 2}, {x2} {start_y + (end_y - start_y) / 2}, {x2} {end_y}"
        
        path = SubElement(parent, "path")
        path.set("d", path_d)
        path.set("stroke", "#5DADE2")
        path.set("stroke-width", "3")
        path.set("stroke-linecap", "round")
        path.set("stroke-linejoin", "round")
        path.set("fill", "none")
        path.set("opacity", "0.8")

        arrow_angle = math.radians(30)
        arrow_len = 12

        if y2 > y1:
            arrow_x1 = x2 - arrow_len * math.cos(math.radians(270) + arrow_angle)
            arrow_y1 = y2 - arrow_len * math.sin(math.radians(270) + arrow_angle)
            arrow_x2 = x2 - arrow_len * math.cos(math.radians(270) - arrow_angle)
            arrow_y2 = y2 - arrow_len * math.sin(math.radians(270) - arrow_angle)
        else:
            angle = math.atan2(y2 - y1, x2 - x1)
            arrow_x1 = x2 - arrow_len * math.cos(angle + arrow_angle)
            arrow_y1 = y2 - arrow_len * math.sin(angle + arrow_angle)
            arrow_x2 = x2 - arrow_len * math.cos(angle - arrow_angle)
            arrow_y2 = y2 - arrow_len * math.sin(angle - arrow_angle)

        arrow_line1 = SubElement(parent, "line")
        arrow_line1.set("x1", str(x2))
        arrow_line1.set("y1", str(y2))
        arrow_line1.set("x2", str(arrow_x1))
        arrow_line1.set("y2", str(arrow_y1))
        arrow_line1.set("stroke", "#3498DB")
        arrow_line1.set("stroke-width", "3")

        arrow_line2 = SubElement(parent, "line")
        arrow_line2.set("x1", str(x2))
        arrow_line2.set("y1", str(y2))
        arrow_line2.set("x2", str(arrow_x2))
        arrow_line2.set("y2", str(arrow_y2))
        arrow_line2.set("stroke", "#3498DB")
        arrow_line2.set("stroke-width", "3")

    def _draw_primitive(self, parent: Element, prim_key: str, params: Dict[str, float], attributes: Dict[str, list] = None):
        """
        绘制单个视觉原语

        Args:
            parent: 父SVG元素
            prim_key: 原语类型
            params: 原语参数
            attributes: SQL属性信息字典
        """
        if attributes is None:
            attributes = {}
            
        if prim_key.startswith("arrow"):
            self._draw_arrow(parent, params)
        elif prim_key == "container" or prim_key == "container_left" or prim_key == "container_right":
            self._draw_container(parent, params, attributes)
        elif prim_key == "result_container":
            self._draw_result_container(parent, params, attributes)
        elif prim_key == "funnel":
            self._draw_funnel(parent, params, attributes)
        elif prim_key == "branch":
            self._draw_branch(parent, params, attributes)
        elif prim_key == "stack":
            self._draw_stack(parent, params, attributes)
        elif prim_key == "boundary":
            self._draw_boundary(parent, params)

    def _draw_container(self, parent: Element, params: Dict[str, float], attributes: Dict[str, list] = None):
        """绘制容器（数据表）"""
        if attributes is None:
            attributes = {}
            
        x = params.get("x", 50)
        y = params.get("y", 50)
        w = params.get("width", 200)
        h = params.get("height", 150)
        opacity = params.get("opacity", 0.8)

        header_height = 28

        select_cols = attributes.get("select_cols", [])
        num_display = min(len(select_cols), 5)
        
        if num_display > 0:
            max_col_len = max(len(col) for col in select_cols[:num_display])
            min_width = 60
            max_width = 150
            w = min(max(min_width + max_col_len * 6, min_width), max_width)
            
            row_height = 22
            h = header_height + 12 + num_display * row_height
            h = max(h, 70)
            h = min(h, 120)
        else:
            w = 100
            h = 70

        params["width"] = w
        params["height"] = h

        rect = SubElement(parent, "rect")
        rect.set("x", str(x))
        rect.set("y", str(y))
        rect.set("width", str(w))
        rect.set("height", str(h))
        rect.set("fill", "#4A90D9")
        rect.set("fill-opacity", str(opacity))
        rect.set("stroke", "#2E5C8A")
        rect.set("stroke-width", "2")
        rect.set("rx", "4")
        rect.set("ry", "4")

        header_height = 28
        header = SubElement(parent, "rect")
        header.set("x", str(x))
        header.set("y", str(y))
        header.set("width", str(w))
        header.set("height", str(header_height))
        header.set("fill", "#2E5C8A")

        label = SubElement(parent, "text")
        label.set("x", str(x + w / 2))
        label.set("y", str(y + header_height / 2 + 6))
        label.set("text-anchor", "middle")
        label.set("fill", "white")
        label.set("font-size", "13")
        label.set("font-weight", "bold")
        
        table_names = params.get("table_names", attributes.get("table_names", []))
        if table_names:
            label.text = table_names[0]
        else:
            label.text = "Table"

        row_height = 22
        
        for i in range(num_display):
            row_y = y + header_height + 10 + i * row_height
            row = SubElement(parent, "line")
            row.set("x1", str(x + 8))
            row.set("y1", str(row_y))
            row.set("x2", str(x + w - 8))
            row.set("y2", str(row_y))
            row.set("stroke", "#2E5C8A")
            row.set("stroke-opacity", "0.3")
            
            col_text = SubElement(parent, "text")
            col_text.set("x", str(x + w / 2))
            col_text.set("y", str(row_y + 4))
            col_text.set("text-anchor", "middle")
            col_text.set("fill", "#2E5C8A")
            col_text.set("font-size", "13")
            col_text.text = select_cols[i]

    def _draw_result_container(self, parent: Element, params: Dict[str, float], attributes: Dict[str, list] = None):
        """绘制结果容器（查询结果）"""
        if attributes is None:
            attributes = {}
            
        x = params.get("x", 50)
        y = params.get("y", 50)
        w = params.get("width", 150)
        h = params.get("height", 80)
        opacity = params.get("opacity", 0.9)

        header_height = 24

        select_cols = params.get("select_cols", attributes.get("select_cols", []))
        num_display = min(len(select_cols), 5)
        
        if num_display > 0:
            max_col_len = max(len(col) for col in select_cols[:num_display])
            min_width = 60
            max_width = 150
            w = min(max(min_width + max_col_len * 6, min_width), max_width)
            
            row_height = 22
            h = header_height + 12 + num_display * row_height
            h = max(h, 70)
            h = min(h, 120)
        else:
            w = 100
            h = 70

        params["width"] = w
        params["height"] = h

        rect = SubElement(parent, "rect")
        rect.set("x", str(x))
        rect.set("y", str(y))
        rect.set("width", str(w))
        rect.set("height", str(h))
        rect.set("fill", "#58D68D")
        rect.set("fill-opacity", str(opacity))
        rect.set("stroke", "#1E8449")
        rect.set("stroke-width", "2")
        rect.set("rx", "6")
        rect.set("ry", "6")

        header_height = 24
        header = SubElement(parent, "rect")
        header.set("x", str(x))
        header.set("y", str(y))
        header.set("width", str(w))
        header.set("height", str(header_height))
        header.set("fill", "#1E8449")

        label = SubElement(parent, "text")
        label.set("x", str(x + w / 2))
        label.set("y", str(y + header_height / 2 + 5))
        label.set("text-anchor", "middle")
        label.set("fill", "white")
        label.set("font-size", "13")
        label.set("font-weight", "bold")
        label.text = "Result"

        row_height = 22
        
        for i in range(num_display):
            row_y = y + header_height + 10 + i * row_height
            row = SubElement(parent, "line")
            row.set("x1", str(x + 8))
            row.set("y1", str(row_y))
            row.set("x2", str(x + w - 8))
            row.set("y2", str(row_y))
            row.set("stroke", "#1E8449")
            row.set("stroke-opacity", "0.3")
            
            col_text = SubElement(parent, "text")
            col_text.set("x", str(x + w / 2))
            col_text.set("y", str(row_y + 4))
            col_text.set("text-anchor", "middle")
            col_text.set("fill", "#1E8449")
            col_text.set("font-size", "13")
            col_text.text = select_cols[i]

    def _draw_funnel(self, parent: Element, params: Dict[str, float], attributes: Dict[str, list] = None):
        """
        绘制漏斗（过滤）— 实现双路径编码 (Methodology §6)

        高β (β_cond ≥ BETA_THRESHOLD): Visual Pop-out 突出弹窗渲染
          → 不透明度1.0、高饱和度橙/警示红、粗实线边框 + SVG阴影滤镜
          → 歧义区域形成视觉跳变，被人眼前置注意力优先捕捉

        低β (β_cond < BETA_THRESHOLD): Active De-emphasis 去强调
          → 灰度化、低透明度0.2
        """
        if attributes is None:
            attributes = {}

        x = params.get("x", 150)
        y = params.get("y", 50)
        top_width = params.get("top_width", 120)
        bottom_width = params.get("bottom_width", 40)
        height = params.get("height", 100)

        # 获取该原语的β值（从_beta_cache中读取，由_set_beta_context设置）
        beta_cond = self._beta_cache.get("funnel", 0.3)
        is_high_beta = beta_cond >= BETA_THRESHOLD

        # 根据双路径编码选择样式
        if is_high_beta:
            # === 高β: Visual Pop-out 突出弹窗渲染 ===
            fill_color = WARNING_COLOR          # #E74C3C 警示红
            fill_opacity = 1.0                   # 不透明度1.0
            stroke_color = "#922B21"             # 深红粗实线边框
            stroke_width = 3.5                   # 粗边框
            label_color = "white"                # 白色标签
            attrs_color = "#FFF3E0"              # 浅黄属性文字
            arrow_color = "#922B21"              # 深红箭头
            arrow_width = 3
            filter_id = "highBetaShadow"         # SVG阴影滤镜
        else:
            # === 低β: Active De-emphasis 去强调 ===
            fill_color = "#FFB347"
            fill_opacity = LOW_BETA_OPACITY      # 0.2 低透明度
            stroke_color = "#E08A00"
            stroke_width = 1.5
            label_color = "#8B5A00"
            attrs_color = "#8B5A00"
            arrow_color = "#E08A00"
            arrow_width = 2
            filter_id = "lowBetaGrayscale"       # 灰度滤镜

        points = [
            (x, y),
            (x + top_width, y),
            (x + bottom_width + (top_width - bottom_width) / 2, y + height),
            (x + (top_width - bottom_width) / 2, y + height)
        ]

        point_str = " ".join([f"{p[0]},{p[1]}" for p in points])
        polygon = SubElement(parent, "polygon")
        polygon.set("points", point_str)
        polygon.set("fill", fill_color)
        polygon.set("fill-opacity", str(fill_opacity))
        polygon.set("stroke", stroke_color)
        polygon.set("stroke-width", str(stroke_width))

        # 高β时添加SVG阴影滤镜 + 脉冲动画效果
        if is_high_beta:
            polygon.set("filter", f"url(#{filter_id})")
            # 添加脉冲动画 — 视觉跳变效果
            animate = SubElement(polygon, "animate")
            animate.set("attributeName", "stroke-width")
            animate.set("values", f"{stroke_width};{stroke_width * 1.5};{stroke_width}")
            animate.set("dur", "2s")
            animate.set("repeatCount", "indefinite")

        label = SubElement(parent, "text")
        label.set("x", str(x + top_width / 2))
        label.set("y", str(y + height / 2 - 5))
        label.set("text-anchor", "middle")
        label.set("fill", label_color)
        label.set("font-size", "13")
        label.set("font-weight", "bold")
        label.text = "Filter"

        # 高β时添加β值标注
        if is_high_beta:
            beta_label = SubElement(parent, "text")
            beta_label.set("x", str(x + top_width + 10))
            beta_label.set("y", str(y + 12))
            beta_label.set("fill", "#E74C3C")
            beta_label.set("font-size", "10")
            beta_label.set("font-weight", "bold")
            beta_label.set("font-family", "monospace")
            beta_label.text = f"β={beta_cond:.3f}"

        where_attrs = params.get("where_attrs", attributes.get("where_attrs", []))
        if where_attrs:
            max_lines = 2
            max_chars_per_line = 30

            simple_conds = []
            complex_conds = []

            for cond in where_attrs:
                if '(Subquery)' in cond or len(cond) > max_chars_per_line:
                    complex_conds.append(cond)
                else:
                    simple_conds.append(cond)

            display_lines = []

            if simple_conds:
                current_line = []
                current_len = 0
                for cond in simple_conds:
                    cond_len = len(cond)
                    if current_line and current_len + cond_len + 2 > max_chars_per_line:
                        display_lines.append(", ".join(current_line))
                        current_line = []
                        current_len = 0
                    current_line.append(cond)
                    current_len += cond_len + (2 if current_line else 0)
                if current_line:
                    display_lines.append(", ".join(current_line))

            if complex_conds and len(display_lines) < max_lines:
                complex_display = complex_conds[0]
                if len(complex_display) > max_chars_per_line:
                    complex_display = complex_display[:max_chars_per_line - 3] + "..."
                display_lines.append(complex_display)

            num_remaining = len(complex_conds) - 1
            if num_remaining > 0:
                if display_lines:
                    display_lines[-1] += f" +{num_remaining}"
                else:
                    display_lines.append(f"+{num_remaining}")

            for i, line_text in enumerate(display_lines):
                attrs_text = SubElement(parent, "text")
                attrs_text.set("x", str(x + top_width / 2))
                attrs_text.set("y", str(y + height / 2 + 12 + i * 14))
                attrs_text.set("text-anchor", "middle")
                attrs_text.set("fill", attrs_color)
                attrs_text.set("font-size", "11")
                attrs_text.text = line_text

        arrow = SubElement(parent, "path")
        arrow_x = x + top_width / 2
        arrow_y = y + height + 8
        arrow_d = f"M {arrow_x} {arrow_y - 5} L {arrow_x} {arrow_y + 5} M {arrow_x - 4} {arrow_y + 2} L {arrow_x} {arrow_y + 5} L {arrow_x + 4} {arrow_y + 2}"
        arrow.set("d", arrow_d)
        arrow.set("stroke", arrow_color)
        arrow.set("stroke-width", str(arrow_width))
        arrow.set("fill", "none")

    def _draw_branch(self, parent: Element, params: Dict[str, float], attributes: Dict[str, list] = None):
        """绘制分支（JOIN汇合）"""
        if attributes is None:
            attributes = {}
            
        x = params.get("x", 300)
        y = params.get("y", 100)
        num_branches = params.get("num_branches", 2)
        spread = params.get("spread", 60)
        length = params.get("length", 80)
        angle = params.get("angle", 90)
        draw_lines = params.get("draw_lines", True)

        if draw_lines:
            start_angle = angle - spread / 2
            angle_step = spread / (num_branches - 1) if num_branches > 1 else 0

            for i in range(num_branches):
                branch_angle = math.radians(start_angle + i * angle_step)
                end_x = x + length * math.cos(branch_angle)
                end_y = y + length * math.sin(branch_angle)

                line = SubElement(parent, "line")
                line.set("x1", str(end_x))
                line.set("y1", str(end_y))
                line.set("x2", str(x))
                line.set("y2", str(y))
                line.set("stroke", "#9B59B6")
                line.set("stroke-width", "3")
                line.set("stroke-linecap", "round")

                arrow_len = 10
                arrow_angle_val = math.radians(30)
                
                arrow1_x = x - arrow_len * math.cos(branch_angle + arrow_angle_val)
                arrow1_y = y - arrow_len * math.sin(branch_angle + arrow_angle_val)
                arrow2_x = x - arrow_len * math.cos(branch_angle - arrow_angle_val)
                arrow2_y = y - arrow_len * math.sin(branch_angle - arrow_angle_val)

                arrow_line1 = SubElement(parent, "line")
                arrow_line1.set("x1", str(x))
                arrow_line1.set("y1", str(y))
                arrow_line1.set("x2", str(arrow1_x))
                arrow_line1.set("y2", str(arrow1_y))
                arrow_line1.set("stroke", "#9B59B6")
                arrow_line1.set("stroke-width", "2")

                arrow_line2 = SubElement(parent, "line")
                arrow_line2.set("x1", str(x))
                arrow_line2.set("y1", str(y))
                arrow_line2.set("x2", str(arrow2_x))
                arrow_line2.set("y2", str(arrow2_y))
                arrow_line2.set("stroke", "#9B59B6")
                arrow_line2.set("stroke-width", "2")

        circle = SubElement(parent, "circle")
        circle.set("cx", str(x))
        circle.set("cy", str(y))
        circle.set("r", "8")
        circle.set("fill", "#9B59B6")
        circle.set("stroke", "#7D3C98")
        circle.set("stroke-width", "2")

        label = SubElement(parent, "text")
        label.set("x", str(x))
        label.set("y", str(y - 15))
        label.set("text-anchor", "middle")
        label.set("fill", "#7D3C98")
        label.set("font-size", "13")
        label.set("font-weight", "bold")
        label.text = "JOIN"
        
        join_attrs = params.get("join_attrs", attributes.get("join_attrs", []))
        if join_attrs:
            max_lines = 2
            max_chars_per_line = 30
            
            display_lines = []
            remaining = join_attrs.copy()
            
            while remaining and len(display_lines) < max_lines:
                current_line = []
                current_len = 0
                
                while remaining:
                    cond = remaining[0]
                    cond_len = len(cond)
                    
                    if current_line and current_len + cond_len + 2 > max_chars_per_line:
                        break
                    
                    current_line.append(cond)
                    current_len += cond_len + (2 if current_line else 0)
                    remaining.pop(0)
                
                display_lines.append(", ".join(current_line))
            
            if remaining:
                display_lines[-1] += f" +{len(remaining)}"
            
            for i, line_text in enumerate(display_lines):
                attrs_text = SubElement(parent, "text")
                attrs_text.set("x", str(x))
                attrs_text.set("y", str(y + 22 + i * 14))
                attrs_text.set("text-anchor", "middle")
                attrs_text.set("fill", "#7D3C98")
                attrs_text.set("font-size", "11")
                attrs_text.text = line_text

    def _draw_stack(self, parent: Element, params: Dict[str, float], attributes: Dict[str, list] = None):
        """绘制堆叠柱（GROUP BY）"""
        if attributes is None:
            attributes = {}
            
        x = params.get("x", 100)
        y = params.get("y", 200)
        num_bars = params.get("num_bars", 5)
        bar_width = params.get("bar_width", 30)
        max_height = params.get("max_height", 100)
        spacing = params.get("spacing", 15)

        group_by_cols = params.get("group_by_cols", attributes.get("group_by_cols", []))
        colors = ["#1ABC9C", "#2ECC71", "#3498DB", "#9B59B6", "#E67E22"]

        for i in range(num_bars):
            bar_x = x + i * (bar_width + spacing)
            bar_height = max_height * (0.3 + 0.7 * (1 - i / num_bars))

            bar = SubElement(parent, "rect")
            bar.set("x", str(bar_x))
            bar.set("y", str(y + max_height - bar_height))
            bar.set("width", str(bar_width))
            bar.set("height", str(bar_height))
            bar.set("fill", colors[i % len(colors)])
            bar.set("fill-opacity", "0.85")
            bar.set("stroke", "#2C3E50")
            bar.set("stroke-width", "1")

            label = SubElement(parent, "text")
            label.set("x", str(bar_x + bar_width / 2))
            label.set("y", str(y + max_height + 15))
            label.set("text-anchor", "middle")
            label.set("fill", "#2C3E50")
            label.set("font-size", "13")
            if i < len(group_by_cols):
                label.text = group_by_cols[i]
            else:
                label.text = f"组{i + 1}"

        base_line = SubElement(parent, "line")
        base_line.set("x1", str(x - 5))
        base_line.set("y1", str(y + max_height))
        base_line.set("x2", str(x + num_bars * (bar_width + spacing) - spacing + 5))
        base_line.set("y2", str(y + max_height))
        base_line.set("stroke", "#2C3E50")
        base_line.set("stroke-width", "2")

        title = SubElement(parent, "text")
        title.set("x", str(x + (num_bars * (bar_width + spacing) - spacing) / 2))
        title.set("y", str(y - 10))
        title.set("text-anchor", "middle")
        title.set("fill", "#2C3E50")
        title.set("font-size", "11")
        title.set("font-weight", "bold")
        title.text = "分组聚合"

    def _draw_boundary(self, parent: Element, params: Dict[str, float]):
        """绘制上下文边界框（子查询作用域）- 改进样式"""
        x = params.get("x", 10)
        y = params.get("y", 10)
        width = params.get("width", 600)
        height = params.get("height", 400)
        dashed = params.get("dashed", True)

        rect = SubElement(parent, "rect")
        rect.set("x", str(x))
        rect.set("y", str(y))
        rect.set("width", str(width))
        rect.set("height", str(height))
        rect.set("fill", "#FFF3E0")
        rect.set("fill-opacity", "0.3")
        rect.set("stroke", "#FF9800")
        rect.set("stroke-width", "2")
        if dashed:
            rect.set("stroke-dasharray", "8,4")
        rect.set("rx", "8")
        rect.set("ry", "8")

        corner_size = 15
        corners = [
            (x, y, x + corner_size, y, x, y + corner_size),
            (x + width, y, x + width - corner_size, y, x + width, y + corner_size),
            (x, y + height, x + corner_size, y + height, x, y + height - corner_size),
            (x + width, y + height, x + width - corner_size, y + height, x + width, y + height - corner_size),
        ]
        
        for cx, cy, x1, y1, x2, y2 in corners:
            line1 = SubElement(parent, "line")
            line1.set("x1", str(x1))
            line1.set("y1", str(y1))
            line1.set("x2", str(cx))
            line1.set("y2", str(cy))
            line1.set("stroke", "#FF9800")
            line1.set("stroke-width", "3")
            
            line2 = SubElement(parent, "line")
            line2.set("x1", str(x2))
            line2.set("y1", str(y2))
            line2.set("x2", str(cx))
            line2.set("y2", str(cy))
            line2.set("stroke", "#FF9800")
            line2.set("stroke-width", "3")

        label_bg = SubElement(parent, "rect")
        label_bg.set("x", str(x + 5))
        label_bg.set("y", str(y + 5))
        label_bg.set("width", "140")
        label_bg.set("height", "22")
        label_bg.set("fill", "#FF9800")
        label_bg.set("rx", "4")
        
        label = SubElement(parent, "text")
        label.set("x", str(x + 75))
        label.set("y", str(y + 20))
        label.set("text-anchor", "middle")
        label.set("fill", "white")
        label.set("font-size", "11")
        label.set("font-weight", "bold")
        label.text = "Subquery Scope"

    def _draw_arrow(self, parent: Element, params: Dict[str, float]):
        """绘制箭头"""
        x1 = params.get("x1", 0)
        y1 = params.get("y1", 0)
        x2 = params.get("x2", 100)
        y2 = params.get("y2", 0)
        color = params.get("color", "#3498DB")

        line = SubElement(parent, "line")
        line.set("x1", str(x1))
        line.set("y1", str(y1))
        line.set("x2", str(x2))
        line.set("y2", str(y2))
        line.set("stroke", color)
        line.set("stroke-width", "2")
        line.set("stroke-linecap", "round")

        angle = math.atan2(y2 - y1, x2 - x1)
        arrow_len = 10
        arrow_angle = math.radians(30)

        arrow1_x = x2 - arrow_len * math.cos(angle + arrow_angle)
        arrow1_y = y2 - arrow_len * math.sin(angle + arrow_angle)
        arrow2_x = x2 - arrow_len * math.cos(angle - arrow_angle)
        arrow2_y = y2 - arrow_len * math.sin(angle - arrow_angle)

        arrow_line1 = SubElement(parent, "line")
        arrow_line1.set("x1", str(x2))
        arrow_line1.set("y1", str(y2))
        arrow_line1.set("x2", str(arrow1_x))
        arrow_line1.set("y2", str(arrow1_y))
        arrow_line1.set("stroke", color)
        arrow_line1.set("stroke-width", "2")

        arrow_line2 = SubElement(parent, "line")
        arrow_line2.set("x1", str(x2))
        arrow_line2.set("y1", str(y2))
        arrow_line2.set("x2", str(arrow2_x))
        arrow_line2.set("y2", str(arrow2_y))
        arrow_line2.set("stroke", color)
        arrow_line2.set("stroke-width", "2")

    def _add_legend_right(self, parent: Element, params: Dict[str, Dict[str, float]], sql: str = ""):
        """在右侧添加原语解释图例"""
        legend_x = self.width - 180
        legend_y = 30

        prim_info = {
            "container": {"label": "Container", "desc": "Table/Result", "color": "#4A90D9"},
            "funnel": {"label": "Funnel", "desc": "WHERE/HAVING", "color": "#FFB347"},
            "branch": {"label": "Branch", "desc": "JOIN Merge", "color": "#9B59B6"},
            "stack": {"label": "Stack", "desc": "GROUP BY", "color": "#1ABC9C"},
            "boundary": {"label": "Boundary", "desc": "Subquery Scope", "color": "#7F8C8D"}
        }

        row_height = 35
        current_y = legend_y

        for prim_key, info in prim_info.items():
            if prim_key in params:
                box = SubElement(parent, "rect")
                box.set("x", str(legend_x))
                box.set("y", str(current_y))
                box.set("width", "12")
                box.set("height", "12")
                box.set("fill", info["color"])
                box.set("fill-opacity", "0.8")

                label = SubElement(parent, "text")
                label.set("x", str(legend_x + 20))
                label.set("y", str(current_y + 10))
                label.set("fill", "#2C3E50")
                label.set("font-size", "13")
                label.set("font-weight", "bold")
                label.text = info["label"]

                desc = SubElement(parent, "text")
                desc.set("x", str(legend_x + 60))
                desc.set("y", str(current_y + 10))
                desc.set("fill", "#7F8C8D")
                desc.set("font-size", "12")
                desc.text = info["desc"]

                current_y += row_height

    def _add_sql_info(self, parent: Element, sql: str = "", question: str = "", last_y: float = 0) -> float:
        """在底部添加SQL信息"""
        line_height = 12
        sql_lines = []
        if sql:
            line_width = 90
            current_line = ""
            for word in sql.split():
                if len(current_line) + len(word) + 1 <= line_width:
                    current_line += " " + word if current_line else word
                else:
                    if current_line:
                        sql_lines.append(current_line)
                    current_line = word
            if current_line:
                sql_lines.append(current_line)
        
        sql_height = len(sql_lines) * line_height if sql_lines else 0
        question_height = 16 if question else 0
        total_height = sql_height + question_height + 10
        
        info_y = last_y + 10

        if question:
            q_label = SubElement(parent, "text")
            q_label.set("x", "20")
            q_label.set("y", str(info_y))
            q_label.set("fill", "#2C3E50")
            q_label.set("font-size", "12")
            q_label.set("font-weight", "bold")
            q_label.text = "Query: "

            q_text = SubElement(parent, "text")
            q_text.set("x", "70")
            q_text.set("y", str(info_y))
            q_text.set("fill", "#34495E")
            q_text.set("font-size", "11")
            q_text.text = question[:50] + "..." if len(question) > 50 else question

        if sql:
            s_label = SubElement(parent, "text")
            s_label.set("x", "20")
            s_label.set("y", str(info_y + 14))
            s_label.set("fill", "#2C3E50")
            s_label.set("font-size", "11")
            s_label.set("font-weight", "bold")
            s_label.text = "SQL: "

            line_width = 90
            font_size = 10
            line_height = 12
            
            lines = []
            current_line = ""
            for word in sql.split():
                if len(current_line) + len(word) + 1 <= line_width:
                    current_line += " " + word if current_line else word
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            
            for i, line in enumerate(lines):
                line_text = SubElement(parent, "text")
                line_text.set("x", "60")
                line_text.set("y", str(info_y + 18 + (i + 1) * line_height))
                line_text.set("fill", "#7F8C8D")
                line_text.set("font-size", str(font_size))
                line_text.text = line
        
        return info_y + total_height
