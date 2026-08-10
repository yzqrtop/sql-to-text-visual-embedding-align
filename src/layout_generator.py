"""
布局参数回归模块
实现 MetaphorSearch 核心方法论：

§3 语义指纹: M: ℝ¹⁵³⁶ → ℝ¹⁶，三层MLP回归器
    16维 = L1表结构(D0-D3) + L2操作语义(D4-D10) + L3输出聚焦(D11-D15)
    训练损失: L_base = MSE(p_base, p*) + λ‖Θ‖²

§4 熵调制: β = Φ(H_s, H_q) = [β_ent, β_join, β_cond, β_res]

§5 参数合成: p_final = p_base ⊙ (1 + 0.3 · β_expand)

§6 锚点约束：将P_pred投影到有效约束空间
"""

import re
import math
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List, Optional
from .config import (
    MLP_INPUT_DIM, MLP_HIDDEN_DIM, MLP_OUTPUT_DIM,
    DEVICE, MLP_CHECKPOINT, MLP_CHECKPOINT_STR,
    BETA_GAIN, ATTN_ENTROPY_GROUPS,
    CONTAINMENT_LOG_BASE, CONTAINMENT_OFFSET,
    PATHFINDING_THICK_GAIN, FLOW_OPACITY_DECAY,
    ENCODER_DIM
)
from .encoder import DualEncoder


# 16维语义指纹定义（Methodology §3）
FINGERPRINT_DIM_NAMES = [
    # L1 表与结构 (D0-D3)
    "table_count",         # D0: 表数量
    "nest_depth",          # D1: 嵌套查询深度
    "join_graph_density",  # D2: JOIN图密度
    "cross_join_flag",     # D3: 笛卡尔积标记 (0/1)
    # L2 操作语义 (D4-D10)
    "filter_intensity",    # D4: WHERE谓词数
    "having_presence",     # D5: HAVING存在 (0/1)
    "group_by_complexity", # D6: GROUP BY复杂度
    "agg_func_complexity", # D7: 聚合函数复杂度
    "distinct_flag",       # D8: DISTINCT标记 (0/1)
    "set_op_flag",         # D9: 集合操作标记 (0/1)
    "window_func_flag",    # D10: 窗口函数标记 (0/1)
    # L3 输出聚焦 (D11-D15)
    "output_columns_ratio",# D11: 输出列比例
    "orderby_strength",    # D12: ORDER BY强度
    "limit_presence",      # D13: LIMIT存在 (0/1)
    "result_cardinality",  # D14: 结果基数估计
    "visual_emphasis",     # D15: 视觉强调 (MLP学习到的)
]


class LayoutMLP(nn.Module):
    """轻量级三层MLP回归器 (Methodology §3)

    输入：拼接后的语义嵌入 E = [E_q; E_s] ∈ ℝ¹⁵³⁶
    输出：16维语义指纹 p_base ∈ ℝ¹⁶

    训练目标：L = ‖MLP(E) - p*‖² + λ‖Θ‖²  (MSE + L2正则)
    """

    def __init__(self, input_dim=MLP_INPUT_DIM, hidden_dim=MLP_HIDDEN_DIM,
                 output_dim=MLP_OUTPUT_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播输出p_base ∈ [0,1]^16"""
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        x = torch.sigmoid(self.fc3(x))  # 所有维度归一化到[0,1]
        return x


class LayoutGenerator:
    """布局生成器 - 核心方法论实现

    在线推理管线 (给定问题q和SQL s)：
    1. 语义编码: e_q,e_s,H_q,H_s ← Encoder(q,s)
    2. 指纹回归: p_base = MLP([e_q; e_s])  ∈ ℝ¹⁶
    3. 熵调制:   β = Φ(H_s, H_q)  ∈ ℝ⁴
    4. 参数合成: p_final = p_base ⊙ (1 + BETA_GAIN·β_expand)
    5. 约束投影: params = AnchorConstraints(p_final)
    6. 几何解码: prims = DecodeGeometry(params, sql)
    """

    def __init__(self, model_path=None):
        self.device = DEVICE
        self.encoder = self._init_encoder()
        self.mlp = self._init_mlp(model_path)
        # 4维β到16维p的广播映射
        self._beta_to_dim_map = self._build_beta_expand_map()
        # 渲染器
        from .renderer import Renderer
        self.renderer = Renderer()

    def _init_encoder(self) -> DualEncoder:
        return DualEncoder()

    def _init_mlp(self, model_path) -> LayoutMLP:
        mlp = LayoutMLP().to(self.device)
        # 优先使用字符串路径（避免Windows中文路径问题）
        import os
        checkpoint_path = model_path if model_path else MLP_CHECKPOINT_STR
        if isinstance(checkpoint_path, str) and os.path.exists(checkpoint_path):
            try:
                mlp.load_state_dict(torch.load(checkpoint_path, map_location=self.device),
                                    strict=False)
            except Exception as e:
                print(f"[WARN] 加载MLP权重失败（将使用默认权重）: {e}")
        mlp.eval()
        return mlp

    def _build_beta_expand_map(self) -> List[str]:
        """
        构建 4维β → 16维p的广播映射 (Methodology §5)

        β_ent  → Entity相关维 (D0表数, D1嵌套, D11输出比例)
        β_join → Join相关维   (D2密度, D3笛卡尔)
        β_cond → Cond相关维   (D4过滤, D5HAVING, D13LIMIT)
        β_res  → Res相关维    (D6分组, D7聚合, D8DISTINCT, D9SET, D10窗口,
                               D12ORDER, D14基数, D15视觉)
        """
        return [
            "entity",  # D0 table_count
            "entity",  # D1 nest_depth
            "join",    # D2 join_graph_density
            "join",    # D3 cross_join_flag
            "cond",    # D4 filter_intensity
            "cond",    # D5 having_presence
            "res",     # D6 group_by_complexity
            "res",     # D7 agg_func_complexity
            "res",     # D8 distinct_flag
            "res",     # D9 set_op_flag
            "res",     # D10 window_func_flag
            "entity",  # D11 output_columns_ratio
            "res",     # D12 orderby_strength
            "cond",    # D13 limit_presence
            "res",     # D14 result_cardinality
            "res",     # D15 visual_emphasis
        ]

    def generate_layout(self, sql: str, question: str) -> Dict[str, Dict[str, float]]:
        """主入口 - 在线推理完整管线 (双编码器架构)"""
        # §2. 双编码器前向传播
        # Encoder A (冻结): SQL → e_s^A (768维) → MLP
        # Encoder B (微调后冻结): SQL+NL → H_s^B, H_q^B → 交叉注意力
        enc = self.encoder.encode_pair(question, sql)
        e_s_A = enc["e_s_A"]          # [1, 768] — Encoder A 输出
        H_q, H_s = enc["H_q"], enc["H_s"]  # Encoder B 输出
        tokens_q, tokens_s = enc["tokens_q"], enc["tokens_s"]

        # §3. 指纹回归 p_base = MLP(e_s^A)  — 仅用Encoder A的SQL嵌入
        with torch.no_grad():
            p_base = self.mlp(e_s_A).squeeze(0).cpu().numpy()  # [16]

        # §4. 交叉注意力 + 熵调制 β = Φ(H_s^B, H_q^B)
        A = self.encoder.compute_cross_attention(H_s, H_q)
        beta = self.encoder.compute_entropy_modulation(A, tokens_s, len(tokens_q))

        # 提取各SQL功能组的高注意力NL词（供renderer标注）
        attn_words = self._extract_attn_words(A, tokens_s, tokens_q)

        # §5. 参数合成 p_final = p_base ⊙ (1 + β_GAIN · β_expand)
        p_final = self._synthesize_params(p_base, beta)

        # §6. 几何解码 + 锚点约束
        params = self._decode_geometry(p_final, sql)
        params = self._apply_anchor_constraints(params)

        # 附加信息（供renderer双路径编码 + NL注意力标注使用）
        params["_meta"] = {
            "p_base": p_base.tolist(),
            "beta": beta,
            "p_final": p_final.tolist(),
            "fingerprint_names": FINGERPRINT_DIM_NAMES,
            "attn_words": attn_words,  # {"entity": [("word", 0.85), ...], "join": [...], ...}
            "question": question,
        }
        return params

    def generate_visualization(self, sql: str, question: str) -> str:
        """
        完整管线：SQL + NL → SVG字符串 (真实端到端)
        步骤：generate_layout → renderer.render
        """
        params = self.generate_layout(sql, question)
        svg_str = self.renderer.render(params, sql=sql, question=question)
        return svg_str

    def _extract_attn_words(self, A: torch.Tensor, tokens_s: List[str],
                             tokens_q: List[str], top_k: int = 3) -> Dict[str, list]:
        """
        提取每个SQL功能组中注意力最高的NL词

        对每个SQL group (Entity/Join/Cond/Res)：
        - 取该组内所有SQL token对NL的注意力分布
        - 找出注意力权重最高的top_k个NL token
        - 清理token (去除特殊符号)

        Returns:
            {group_name: [(word, weight), ...]}
        """
        groups = self.encoder._assign_sql_groups(tokens_s)
        result = {}
        L_q = A.size(1)

        for g_name, g_idx in groups.items():
            if not g_idx:
                continue
            # 该组内所有SQL token对NL的注意力求平均
            A_g = A[g_idx, :].mean(dim=0)  # [L_q]
            # 取top_k
            top_vals, top_ids = torch.topk(A_g, min(top_k * 3, L_q))

            words = []
            seen = set()
            for v, idx in zip(top_vals.tolist(), top_ids.tolist()):
                if idx >= len(tokens_q):
                    continue
                raw = tokens_q[idx]
                # 清理XLM-RoBERTa token前缀
                word = raw.replace("Ġ", "").replace("▁", "").strip()
                if not word or len(word) < 2:
                    continue
                if word in seen:
                    continue
                # 过滤特殊token
                if word in ("<s>", "</s>", "<pad>", "<unk>", "<mask>"):
                    continue
                seen.add(word)
                words.append((word, round(v, 4)))
                if len(words) >= top_k:
                    break

            result[g_name] = words

        return result

    def _synthesize_params(self, p_base: np.ndarray,
                            beta: Dict[str, float]) -> np.ndarray:
        """
        参数合成 (Methodology §5)
        p_final = p_base ⊙ (1 + 0.3 · β_expand)
        """
        p_final = p_base.copy()
        for i, beta_group in enumerate(self._beta_to_dim_map):
            g = beta.get(beta_group, 0.0)
            p_final[i] = p_base[i] * (1.0 + BETA_GAIN * g)
        # 仍截断到[0,1]以防数值异常
        return np.clip(p_final, 0.0, 1.0)

    # ========================================================================
    # §6 几何解码：将16维指纹 → 具体原语的布局参数
    # ========================================================================

    def _decode_geometry(self, p_final: np.ndarray,
                          sql: str) -> Dict[str, Dict[str, float]]:
        """
        将16维语义指纹解码为各视觉原语的几何参数
        结合Containment/Pathfinding/Flow公式 (Methodology §6)
        """
        fp = {name: float(p_final[i]) for i, name in enumerate(FINGERPRINT_DIM_NAMES)}

        # 先根据SQL结构确定有哪些激活的原语
        active_prims = self._analyze_sql_structure(sql)
        structural = self._extract_structural(sql)

        final_params = {}

        # --- Container (表容器)：Containment 公式
        if "container" in active_prims:
            num_tables = structural.get("num_tables", 1)
            # A_i = p_area_i * log(1 + |T_i|) + c
            area_scale = fp["table_count"] * math.log(1 + num_tables) + 1.0
            width = 100 + fp["visual_emphasis"] * 120 * area_scale
            height = 60 + fp["table_count"] * 80 + CONTAINMENT_OFFSET

            x = 200 + (1 - fp["table_count"]) * 60
            y = 60 + fp["nest_depth"] * 40

            if structural["has_multi_tables"]:
                # 双表拆分
                half_w = width / 2
                final_params["container_left"] = {
                    "x": x - half_w - 30, "y": y,
                    "width": half_w, "height": height,
                    "opacity": 0.7 + fp["visual_emphasis"] * 0.25,
                    "num_rows": max(3, int(fp["result_cardinality"] * 10))
                }
                final_params["container_right"] = {
                    "x": x + 30, "y": y,
                    "width": half_w, "height": height,
                    "opacity": 0.7 + fp["visual_emphasis"] * 0.25,
                    "num_rows": max(3, int(fp["result_cardinality"] * 10))
                }
            else:
                final_params["container"] = {
                    "x": x, "y": y,
                    "width": width, "height": height,
                    "opacity": 0.7 + fp["visual_emphasis"] * 0.25,
                    "num_rows": max(5, int(fp["result_cardinality"] * 15))
                }

        # --- Funnel (过滤漏斗)：Flow公式 α = p_opac * (1 - 0.7^ξ)
        if "funnel" in active_prims:
            xi = structural.get("num_conditions", 1) + fp["nest_depth"]
            # α_filter = p_opac * (1 - γ^ξ)
            opac = fp["filter_intensity"] * (1.0 - FLOW_OPACITY_DECAY ** xi)
            opac = max(0.4, min(1.0, opac + 0.3))

            top_w = 100 + fp["filter_intensity"] * 80
            bot_w = top_w * max(0.2, 1.0 - fp["filter_intensity"] * 0.8)
            h = 60 + fp["filter_intensity"] * 60

            final_params["funnel"] = {
                "x": 200 + (1 - fp["filter_intensity"]) * 40,
                "y": 200,  # 锚点约束会重新调整
                "top_width": top_w,
                "bottom_width": bot_w,
                "height": h,
                "angle": 0,
                "opacity": opac,
                "flow_strength": xi,
                "beta_cond": fp.get("_beta_cond", fp["filter_intensity"])
            }

        # --- Branch (分支/JOIN)：Pathfinding 公式
        if "branch" in active_prims:
            join_tuples = structural.get("num_joins", 1)
            # ω_ij = p_thick_ij * log(1 + |T_i ⋈ T_j|)
            thickness = fp["join_graph_density"] * math.log(1 + join_tuples) * PATHFINDING_THICK_GAIN
            thickness = max(1.5, thickness + 1.0)

            num_branches = min(4, max(2, join_tuples + 1))

            final_params["branch"] = {
                "x": 350, "y": 220,
                "num_branches": num_branches,
                "spread": 60 + fp["join_graph_density"] * 100,
                "length": 60 + fp["join_graph_density"] * 60,
                "angle": 270,
                "line_thickness": thickness,
                "beta_join": fp["join_graph_density"]
            }

        # --- Stack (堆叠柱/GROUP BY聚合)
        if "stack" in active_prims:
            num_bars = min(8, max(3, int(fp["group_by_complexity"] * 6) + 3))
            max_h = 60 + fp["agg_func_complexity"] * 100
            bar_w = 20 + fp["visual_emphasis"] * 30

            final_params["stack"] = {
                "x": 200 + fp["group_by_complexity"] * 60,
                "y": 300,
                "num_bars": num_bars,
                "bar_width": bar_w,
                "max_height": max_h,
                "agg_intensity": fp["agg_func_complexity"],
                "distinct_hint": fp["distinct_flag"]
            }

        # --- Boundary (子查询边界框)
        if "boundary" in active_prims:
            depth = structural.get("nest_depth", 1)
            w = 500 + fp["nest_depth"] * 100
            h = 300 + fp["nest_depth"] * 80
            final_params["boundary"] = {
                "x": 30, "y": 40,
                "width": w, "height": h,
                "dashed": True,
                "nest_depth": depth,
                "beta_entity": fp["nest_depth"]
            }

        # --- Result Container (结果容器)
        if structural["has_result"]:
            result_w = 120 + fp["output_columns_ratio"] * 80
            result_h = 50 + fp["result_cardinality"] * 60
            final_params["result_container"] = {
                "x": 300,
                "y": 450,
                "width": result_w,
                "height": result_h,
                "opacity": 0.85 + fp["visual_emphasis"] * 0.15,
                "output_ratio": fp["output_columns_ratio"],
                "has_orderby": fp["orderby_strength"] > 0.3,
                "has_limit": fp["limit_presence"] > 0.5
            }
        return final_params

    def _extract_structural(self, sql: str) -> Dict:
        """从SQL文本中提取结构信息（用于几何公式）"""
        sql_upper = sql.upper()
        info = {}

        # 表数量
        from_match = re.search(r'FROM\s+([^\s;]+)', sql_upper)
        tables = set()
        if from_match:
            tables.add(re.split(r'\s+AS\s+|\s+', from_match.group(1))[0])
        for m in re.finditer(r'JOIN\s+([^\s]+)', sql_upper):
            tables.add(m.group(1))
        info["num_tables"] = max(1, len(tables))
        info["has_multi_tables"] = len(tables) >= 2 or "JOIN" in sql_upper

        # 嵌套深度
        info["nest_depth"] = len(re.findall(r'\(\s*SELECT', sql_upper)) or 0

        # JOIN数量
        info["num_joins"] = len(re.findall(r'\bJOIN\b', sql_upper))

        # 条件数
        where_match = re.search(r'WHERE\s+(.+?)(GROUP|HAVING|ORDER|LIMIT|$)', sql_upper, re.S)
        if where_match:
            cond_text = where_match.group(1)
            info["num_conditions"] = len(re.split(r'\s+AND\s+|\s+OR\s+', cond_text))
        else:
            info["num_conditions"] = 1 if "WHERE" in sql_upper else 0

        # 是否有结果展示
        info["has_result"] = "SELECT" in sql_upper
        return info

    def _analyze_sql_structure(self, sql: str) -> List[str]:
        """分析SQL结构，确定激活的原语"""
        primitives = ["container"]
        sql_upper = sql.upper()

        if "WHERE" in sql_upper or "HAVING" in sql_upper:
            primitives.append("funnel")
        if "JOIN" in sql_upper:
            primitives.append("branch")
        if "GROUP BY" in sql_upper or any(
            agg in sql_upper for agg in [
                "COUNT(", "SUM(", "AVG(", "MAX(", "MIN("
            ]
        ):
            primitives.append("stack")
        if re.search(r'\(\s*SELECT', sql_upper):
            primitives.append("boundary")
        return primitives

    # ========================================================================
    # 锚点约束
    # ========================================================================

    def _apply_anchor_constraints(self, params: Dict[str, Dict[str, float]]
                                   ) -> Dict[str, Dict[str, float]]:
        """
        锚点约束投影 (Methodology §6)
        - 漏斗附着在容器出口
        - 分支起点与漏斗出口对齐
        - 堆叠柱在容器下方合理位置
        - 边界框包含所有嵌套原语
        """
        corrected = {k: v.copy() for k, v in params.items() if k != "_meta"}
        meta = params.get("_meta")

        # ---- 垂直位置（流水线拓扑：容器 → 漏斗 → 分支 → 堆叠柱 → 结果）
        vertical_gap = 25
        current_y = 60

        container_key = None
        if "container" in corrected:
            container_key = "container"
        elif "container_left" in corrected:
            container_key = "container_left"

        if container_key:
            c = corrected[container_key]
            c["y"] = current_y
            current_y = c["y"] + c["height"] + vertical_gap

            if "container_right" in corrected and container_key == "container_left":
                corrected["container_right"]["y"] = c["y"]

        if "funnel" in corrected:
            f = corrected["funnel"]
            f["y"] = current_y
            current_y = f["y"] + f["height"] + vertical_gap
            # 水平居中于上方容器
            if container_key:
                c = corrected[container_key]
                c_x = c["x"]
                c_w = c["width"]
                if "container_right" in corrected:
                    cr = corrected["container_right"]
                    c_x = (c["x"] + cr["x"] + cr["width"]) / 2
                    c_w = (cr["x"] + cr["width"]) - c["x"]
                f["x"] = c_x + (c_w - f["top_width"]) / 2

        if "branch" in corrected:
            b = corrected["branch"]
            b["y"] = current_y
            current_y = b["y"] + b["length"] + vertical_gap
            if "funnel" in corrected:
                f = corrected["funnel"]
                b["x"] = f["x"] + f["bottom_width"] / 2
            elif container_key:
                c = corrected[container_key]
                b["x"] = c["x"] + c["width"] / 2

        if "stack" in corrected:
            s = corrected["stack"]
            s["y"] = current_y
            current_y = s["y"] + s["max_height"] + vertical_gap
            # 对齐容器中心
            if container_key:
                c = corrected[container_key]
                cx = c["x"] + c["width"] / 2
                s["x"] = cx - (s["num_bars"] * s["bar_width"]) / 2

        if "result_container" in corrected:
            r = corrected["result_container"]
            r["y"] = current_y
            # 水平居中
            if "stack" in corrected:
                s = corrected["stack"]
                total = s["num_bars"] * s["bar_width"]
                r["x"] = s["x"] + total / 2 - r["width"] / 2
            elif "funnel" in corrected:
                f = corrected["funnel"]
                r["x"] = f["x"] + f["bottom_width"] / 2 - r["width"] / 2
            elif container_key:
                c = corrected[container_key]
                r["x"] = c["x"] + c["width"] / 2 - r["width"] / 2

        # ---- Boundary包住所有
        if "boundary" in corrected:
            all_x = []
            all_y = []
            all_w = []
            all_h = []
            for k, v in corrected.items():
                if k in ("boundary", "_meta"):
                    continue
                if isinstance(v, dict) and "x" in v and "y" in v:
                    all_x.append(v["x"])
                    all_y.append(v["y"])
                    all_w.append(v.get("width", v.get("top_width", 100)))
                    all_h.append(v.get("height", v.get("max_height", v.get("length", 80))))
            if all_x:
                min_x = min(all_x) - 20
                min_y = min(all_y) - 20
                max_x = max(x + w for x, w in zip(all_x, all_w)) + 20
                max_y = max(y + h for y, h in zip(all_y, all_h)) + 20
                corrected["boundary"]["x"] = min_x
                corrected["boundary"]["y"] = min_y
                corrected["boundary"]["width"] = max_x - min_x
                corrected["boundary"]["height"] = max_y - min_y

        # 恢复_meta
        if meta is not None:
            corrected["_meta"] = meta
        return corrected
