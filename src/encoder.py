"""
双编码器语义编码模块 (Dual-Encoder Architecture)

论文设计：
- Encoder A（结构编码器）：XLM-RoBERTa-base，全层冻结，仅输入SQL → e_s^A ∈ ℝ⁷⁶⁸
- Encoder B（调制编码器）：XLM-RoBERTa-base，前10层冻结、最后2层可训练，输入SQL+NL → H_s^B, H_q^B

实现：
- e_s = E_CLS^A(s)                全局嵌入 (Methodology §2, Training §2)
- H_q = E_token^B(q), H_s = E_token^B(s)  token级隐藏状态 (Methodology §2)
- A = softmax(H_s^B (H_q^B)^T / √768)     交叉注意力 (Methodology §4)
- β = [β_ent, β_join, β_cond, β_res]      按SQL功能分组的归一化熵 (Methodology §4)
"""

import os
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Tuple, List, Dict, Optional
from .config import (
    ENCODER_MODEL_NAME, ENCODER_MODEL_DIR, DEVICE,
    ENCODER_DIM, ATTN_ENTROPY_GROUPS,
    ENCODER_B_FREEZE_LAYERS, ENCODER_B_TRAIN_LAYERS,
    ENCODER_B_CHECKPOINT
)


class StructuralEncoder(nn.Module):
    """
    Encoder A — 结构编码器 (论文 Training §2)

    XLM-RoBERTa-base, 所有层完全冻结, 仅接收SQL查询 s
    输出: 全局 [CLS] 嵌入 e_s^A ∈ ℝ⁷⁶⁸
    """

    def __init__(self, model_name=ENCODER_MODEL_NAME, model_dir=ENCODER_MODEL_DIR):
        super().__init__()
        if model_dir.exists() and (model_dir / "config.json").exists():
            print(f"[Encoder A] 从本地加载: {model_dir}")
            load_path = str(model_dir)
        else:
            print(f"[Encoder A] 从HuggingFace下载: {model_name}")
            load_path = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model = AutoModel.from_pretrained(load_path, output_hidden_states=True)
        self.model.eval()
        # 完全冻结
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def encode_sql(self, sql: str) -> torch.Tensor:
        """编码SQL → e_s^A ∈ [1, 768]"""
        inputs = self.tokenizer(sql, return_tensors="pt", truncation=True, max_length=512).to(next(self.model.parameters()).device)
        outputs = self.model(**inputs)
        e_cls = outputs.last_hidden_state[:, 0, :]  # [1, 768]
        return e_cls

    @torch.no_grad()
    def encode_sql_tokens(self, sql: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """编码SQL → (e_cls, H_tokens, tokens) — 兼容旧接口"""
        inputs = self.tokenizer(sql, return_tensors="pt", truncation=True, max_length=512).to(next(self.model.parameters()).device)
        outputs = self.model(**inputs)
        H = outputs.last_hidden_state  # [1, L, D]
        e_cls = H[:, 0, :]  # [1, D]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        return e_cls, H, tokens


class ModulationEncoder(nn.Module):
    """
    Encoder B — 调制编码器 (论文 Training §2)

    XLM-RoBERTa-base, 前10层冻结, 最后2层可训练
    接收SQL查询 s 和自然语言问题 q
    输出: token级隐藏状态 H_s^B, H_q^B
    """

    def __init__(self, model_name=ENCODER_MODEL_NAME, model_dir=ENCODER_MODEL_DIR):
        super().__init__()
        if model_dir.exists() and (model_dir / "config.json").exists():
            print(f"[Encoder B] 从本地加载: {model_dir}")
            load_path = str(model_dir)
        else:
            print(f"[Encoder B] 从HuggingFace下载: {model_name}")
            load_path = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model = AutoModel.from_pretrained(load_path, output_hidden_states=True)

        # 冻结前 N 层, 最后 2 层可训练
        total_layers = len(self.model.encoder.layer)
        freeze_count = total_layers - ENCODER_B_TRAIN_LAYERS  # 12 - 2 = 10
        for i, layer in enumerate(self.model.encoder.layer):
            if i < freeze_count:
                for param in layer.parameters():
                    param.requires_grad_(False)
            else:
                print(f"  [Encoder B] 层 {i} 可训练")

        # embeddings 也要冻结
        for param in self.model.embeddings.parameters():
            param.requires_grad_(False)

    def forward(self, text: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """编码文本 → (H_tokens, tokens) — 带梯度用于Encoder B训练"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
        outputs = self.model(**inputs)
        H = outputs.last_hidden_state  # [1, L, D]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        return H, tokens, inputs["input_ids"]

    @torch.no_grad()
    def encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """推理时编码（无梯度）"""
        H, tokens, _ = self.forward(text)
        return H, tokens

    def load_finetuned(self, checkpoint_path: str = ENCODER_B_CHECKPOINT):
        """加载微调后的权重"""
        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.model.device)
            self.load_state_dict(state, strict=False)
            print(f"[Encoder B] 加载微调权重: {checkpoint_path}")
        else:
            print(f"[Encoder B] 未找到微调权重, 使用预训练权重: {checkpoint_path}")


class DualEncoder:
    """
    双编码器封装 (论文 Inference §1)

    推理时:
    - Encoder A (冻结) → e_s^A → MLP → p_base
    - Encoder B (微调后冻结) → H_s^B, H_q^B → 交叉注意力 → β
    """

    # SQL功能分组关键字
    SQL_ENTITY_KEYWORDS = [
        "from", "select", "as", "table", "column", "group_concat",
        "count", "sum", "avg", "max", "min", "distinct"
    ]
    SQL_JOIN_KEYWORDS = [
        "join", "inner", "left", "right", "outer", "full", "cross",
        "natural", "on", "using"
    ]
    SQL_COND_KEYWORDS = [
        "where", "having", "and", "or", "not", "in", "exists",
        "between", "like", "is", "null", ">", "<", "=", "!=", ">=", "<="
    ]
    SQL_RES_KEYWORDS = [
        "order", "by", "limit", "asc", "desc", "offset", "union",
        "intersect", "except"
    ]

    def __init__(self, load_finetuned_b: bool = True):
        self.device = DEVICE
        self.encoder_a = StructuralEncoder()
        self.encoder_a.model = self.encoder_a.model.to(self.device)
        self.encoder_b = ModulationEncoder()
        self.encoder_b.model = self.encoder_b.model.to(self.device)
        # 加载Encoder B微调权重 (如果存在)
        if load_finetuned_b:
            self.encoder_b.load_finetuned()

    # ==================== Encoder A 接口 ====================

    def encode_sql_structural(self, sql: str) -> torch.Tensor:
        """Encoder A: SQL → e_s^A ∈ [1, 768]"""
        return self.encoder_a.encode_sql(sql)

    # ==================== Encoder B 接口 ====================

    def encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Encoder B: 编码文本 → (e_cls, H_tokens, tokens)

        兼容旧接口, 返回 (e_cls, H_tokens, tokens)
        e_cls = H_tokens[:, 0, :] 用于兼容
        """
        H, tokens = self.encoder_b.encode(text)
        e_cls = H[:, 0, :]  # [1, D]
        return e_cls, H, tokens

    def encode_pair(self, question: str, sql: str) -> Dict[str, object]:
        """双编码器联合编码: Encoder A → e_s^A, Encoder B → H_s^B, H_q^B"""
        # Encoder A: SQL → e_s^A (768维, 用于MLP)
        e_s_A = self.encode_sql_structural(sql)

        # Encoder B: SQL + NL → H_s^B, H_q^B (用于交叉注意力)
        e_q_B, H_q_B, tokens_q = self.encode(question)
        e_s_B, H_s_B, tokens_s = self.encode(sql)

        return {
            "e_s_A": e_s_A,         # [1, 768] — MLP输入 (Encoder A)
            "e_q": e_q_B,           # [1, 768] — 兼容
            "e_s": e_s_B,           # [1, 768] — 兼容
            "H_q": H_q_B,           # [1, L_q, 768] — Encoder B
            "H_s": H_s_B,           # [1, L_s, 768] — Encoder B
            "tokens_q": tokens_q,
            "tokens_s": tokens_s,
        }

    # ==================== 交叉注意力 + 熵调制 ====================

    def compute_cross_attention(self, H_s: torch.Tensor, H_q: torch.Tensor) -> torch.Tensor:
        """A = softmax(H_s^B (H_q^B)^T / √d) ∈ [L_s, L_q]"""
        d = H_s.size(-1)
        H_s_sq = H_s.squeeze(0)
        H_q_sq = H_q.squeeze(0)
        raw_scores = torch.matmul(H_s_sq, H_q_sq.T) / math.sqrt(d)
        A = F.softmax(raw_scores, dim=-1)
        return A

    def _assign_sql_groups(self, tokens_s: List[str]) -> Dict[str, List[int]]:
        """将SQL tokens按功能分组 (Entity / Join / Cond / Res)"""
        groups = {g: [] for g in ATTN_ENTROPY_GROUPS}
        for i, tok in enumerate(tokens_s):
            t = tok.lower().strip("Ġ")
            if not t:
                continue
            found = False
            if t in self.SQL_ENTITY_KEYWORDS:
                groups["entity"].append(i); found = True
            elif t in self.SQL_JOIN_KEYWORDS:
                groups["join"].append(i); found = True
            elif t in self.SQL_COND_KEYWORDS:
                groups["cond"].append(i); found = True
            elif t in self.SQL_RES_KEYWORDS:
                groups["res"].append(i); found = True
            if not found and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', t):
                groups["entity"].append(i)
        for g in groups:
            if len(groups[g]) == 0:
                groups[g].append(0)
        return groups

    def compute_entropy_modulation(self, A: torch.Tensor, tokens_s: List[str],
                                    L_q: int) -> Dict[str, float]:
        """β_g = 1 - (1/log L_q) Σ A_ij log A_ij"""
        groups = self._assign_sql_groups(tokens_s)
        L_q = max(L_q, 2)
        norm_factor = math.log(L_q)
        beta = {}
        eps = 1e-9
        for g_name, g_idx in groups.items():
            A_g = A[g_idx, :]
            A_g_clamped = torch.clamp(A_g, min=eps, max=1.0)
            H_g = -(A_g_clamped * torch.log2(A_g_clamped)).sum()
            H_g_norm = H_g.item() / (len(g_idx) * norm_factor) if norm_factor > 0 else 0.0
            beta[g_name] = 1.0 - max(0.0, min(1.0, 1.0 - H_g_norm))
        return beta

    def get_concatenated_embedding(self, e_q: torch.Tensor, e_s: torch.Tensor) -> torch.Tensor:
        """兼容旧接口: [e_q; e_s] — 但新架构MLP只用e_s^A (768维)"""
        return torch.cat([e_q, e_s], dim=-1)
