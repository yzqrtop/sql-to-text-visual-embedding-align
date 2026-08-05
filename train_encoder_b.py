#!/usr/bin/env python3
"""
Encoder B 辅助分布对齐微调脚本 (论文 Training §4)

目标：让Encoder B学会区分"对齐良好"和"错位"的SQL-NL对

损失函数：
  L_align    = KL(T ∥ A_gt)                              — 正确SQL注意力接近目标
  L_contrast = max(0, KL(T∥A_gt) - KL(T∥A_bug) + 0.5)    — 错误SQL注意力更分散
  L_aux      = 1.0 · L_align + 0.5 · L_contrast

梯度传播：仅更新 Encoder B 的最后2层
"""

import sys
import json
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.encoder import ModulationEncoder
from src.config import (
    ENCODER_B_LR, ENCODER_B_EPOCHS, ENCODER_B_BATCH_SIZE,
    ENCODER_B_L_ALIGN_WEIGHT, ENCODER_B_L_CONTRAST_WEIGHT,
    ENCODER_B_CONTRAST_MARGIN, ENCODER_B_CHECKPOINT,
    ENCODER_DIM, DEVICE
)


def build_target_distribution(sql_tokens: list) -> torch.Tensor:
    """
    根据AST句法角色构造目标注意力分布 T (论文 Training §4)

    规则：
    - 表名/列名 = 1.0 (实体词应获得高注意力)
    - SQL关键字 = 0.3 (功能词注意力较低)
    - 其他 = 0.5

    Returns:
        T: [L_s] 每个SQL token的目标权重 (归一化后为分布)
    """
    # SQL关键字集合
    sql_keywords = {
        "select", "from", "where", "join", "inner", "left", "right", "outer",
        "full", "cross", "on", "as", "group", "by", "order", "having",
        "and", "or", "not", "in", "exists", "between", "like", "is", "null",
        "limit", "asc", "desc", "union", "intersect", "except", "distinct",
        "count", "sum", "avg", "max", "min", "over", "case", "when", "then",
        "else", "end", "(", ")", ",", ";", "*", "<", ">", "=", "!=", ">=", "<=",
        "<s>", "</s>", "<pad>", "<unk>", "<mask>"
    }

    weights = []
    for tok in sql_tokens:
        t = tok.lower().strip("Ġ▁")
        if t in sql_keywords:
            weights.append(0.3)
        elif re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', t):
            # 表名/列名 → 高权重
            weights.append(1.0)
        else:
            # 数字/字符串常量等
            weights.append(0.5)

    w = torch.tensor(weights, dtype=torch.float32)
    # 归一化为分布
    T = F.softmax(w, dim=0)
    return T


def compute_attention_to_question(A: torch.Tensor, L_q: int) -> torch.Tensor:
    """
    将 [L_s, L_q] 注意力矩阵聚合为 SQL token 级别的分布

    a_i = (1/L_q) Σ_j A[i,j]  → [L_s]
    """
    return A.mean(dim=1)  # [L_s]


def compute_kl_divergence(P: torch.Tensor, Q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """KL(P ∥ Q) = Σ P * log(P/Q)"""
    P_clamped = torch.clamp(P, min=eps)
    Q_clamped = torch.clamp(Q, min=eps)
    return F.kl_div(Q_clamped.log(), P_clamped, reduction='sum')


def main():
    print("=" * 60)
    print("Encoder B 辅助分布对齐微调")
    print("L_aux = 1.0·L_align + 0.5·L_contrast")
    print("=" * 60)

    # 加载三元组数据
    data_path = project_root / "data" / "processed" / "encoder_b_train_data.json"
    if not data_path.exists():
        print(f"[FAIL] 训练数据不存在: {data_path}")
        print("请先运行 generate_layout_data.py")
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"[OK] 加载三元组数据: {len(data)} 条")

    # CPU模式下减少数据量和轮次
    if not torch.cuda.is_available():
        print("[WARN] CPU模式, 减少数据量(200条)和轮次(3轮)以加速训练")
        data = data[:200]
        epochs = 3
        batch_size = 4
    else:
        epochs = ENCODER_B_EPOCHS
        batch_size = ENCODER_B_BATCH_SIZE
    print(f"  使用数据: {len(data)} 条, 轮数: {epochs}, batch: {batch_size}")

    # 初始化 Encoder B
    print("\n[INIT] 初始化 Encoder B (前10层冻结, 最后2层可训练)...")
    encoder_b = ModulationEncoder()
    encoder_b.model = encoder_b.model.to(DEVICE)
    encoder_b.model.train()

    # 只优化可训练参数
    trainable_params = [p for p in encoder_b.parameters() if p.requires_grad]
    print(f"  可训练参数: {sum(p.numel() for p in trainable_params):,}")
    print(f"  总参数:     {sum(p.numel() for p in encoder_b.parameters()):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=ENCODER_B_LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 训练
    print(f"\n  学习率: {ENCODER_B_LR}")
    print(f"  轮数: {epochs}")
    print(f"  Batch: {batch_size}")
    print(f"  L_align权重: {ENCODER_B_L_ALIGN_WEIGHT}")
    print(f"  L_contrast权重: {ENCODER_B_L_CONTRAST_WEIGHT}")
    print(f"  对比间隔: {ENCODER_B_CONTRAST_MARGIN}")

    print(f"\nEpoch  L_align   L_contrast  L_aux     Train Loss")
    print("-" * 55)

    best_loss = float('inf')

    for epoch in range(epochs):
        encoder_b.model.train()
        total_align = 0.0
        total_contrast = 0.0
        total_aux = 0.0
        total_count = 0

        # Shuffle
        indices = np.random.permutation(len(data))

        for batch_start in range(0, len(indices), batch_size):
            batch_idx = indices[batch_start:batch_start + batch_size]
            batch_loss = 0.0

            optimizer.zero_grad()

            for idx in batch_idx:
                item = data[idx]
                question = item["question"]
                sql_gt = item["sql_gt"]
                sql_bug = item["sql_bug"]

                try:
                    # Encoder B 前向传播 (带梯度)
                    H_q, tokens_q, _ = encoder_b.forward(question)
                    H_s_gt, tokens_s_gt, _ = encoder_b.forward(sql_gt)
                    H_s_bug, tokens_s_bug, _ = encoder_b.forward(sql_bug)

                    # 交叉注意力
                    d = H_q.size(-1)
                    H_q_sq = H_q.squeeze(0)     # [L_q, D]
                    H_gt_sq = H_s_gt.squeeze(0)  # [L_s_gt, D]
                    H_bug_sq = H_s_bug.squeeze(0)  # [L_s_bug, D]

                    A_gt = F.softmax(
                        torch.matmul(H_gt_sq, H_q_sq.T) / math.sqrt(d), dim=-1
                    )  # [L_s_gt, L_q]
                    A_bug = F.softmax(
                        torch.matmul(H_bug_sq, H_q_sq.T) / math.sqrt(d), dim=-1
                    )  # [L_s_bug, L_q]

                    # 构造目标分布 T (基于正确SQL的句法角色)
                    T = build_target_distribution(tokens_s_gt).to(DEVICE)  # [L_s_gt]

                    # 聚合注意力到SQL token级别
                    a_gt = compute_attention_to_question(A_gt, H_q_sq.size(0))   # [L_s_gt]
                    a_bug = compute_attention_to_question(A_bug, H_q_sq.size(0)) # [L_s_bug]

                    # 对齐长度 (取较短的)
                    min_len = min(len(T), a_gt.size(0), a_bug.size(0))
                    T = T[:min_len]
                    a_gt = a_gt[:min_len]
                    a_bug = a_bug[:min_len]

                    # 重新归一化
                    T = T / (T.sum() + 1e-9)
                    a_gt = a_gt / (a_gt.sum() + 1e-9)
                    a_bug = a_bug / (a_bug.sum() + 1e-9)

                    # L_align = KL(T ∥ A_gt)
                    l_align = compute_kl_divergence(T, a_gt)

                    # L_contrast = max(0, KL(T∥A_gt) - KL(T∥A_bug) + margin)
                    kl_gt = compute_kl_divergence(T, a_gt)
                    kl_bug = compute_kl_divergence(T, a_bug)
                    l_contrast = F.relu(kl_gt - kl_bug + ENCODER_B_CONTRAST_MARGIN)

                    # L_aux = 1.0 · L_align + 0.5 · L_contrast
                    l_aux = ENCODER_B_L_ALIGN_WEIGHT * l_align + ENCODER_B_L_CONTRAST_WEIGHT * l_contrast

                    batch_loss += l_aux
                    total_align += l_align.item()
                    total_contrast += l_contrast.item()
                    total_aux += l_aux.item()
                    total_count += 1

                except Exception as e:
                    continue

            if batch_loss > 0:
                batch_loss = batch_loss / len(batch_idx)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

        scheduler.step()

        avg_align = total_align / max(1, total_count)
        avg_contrast = total_contrast / max(1, total_count)
        avg_aux = total_aux / max(1, total_count)

        print(f"{epoch+1:<6} {avg_align:<9.4f} {avg_contrast:<11.4f} {avg_aux:<9.4f} {avg_aux:<9.4f}")

        if avg_aux < best_loss:
            best_loss = avg_aux
            # 只保存可训练层的权重
            trainable_state = {}
            for name, param in encoder_b.named_parameters():
                if param.requires_grad:
                    trainable_state[name] = param.data.clone()
            torch.save(trainable_state, ENCODER_B_CHECKPOINT)

    print("-" * 55)
    print(f"\n训练完成!")
    print(f"  最佳 L_aux: {best_loss:.6f}")
    print(f"  权重保存: {ENCODER_B_CHECKPOINT}")

    # 验证：对比微调前后的注意力分布
    print("\n[验证] 微调后注意力分布检查...")
    encoder_b.model.eval()
    test_item = data[0]
    with torch.no_grad():
        H_q, tokens_q, _ = encoder_b.forward(test_item["question"])
        H_s_gt, tokens_s_gt, _ = encoder_b.forward(test_item["sql_gt"])
        H_s_bug, tokens_s_bug, _ = encoder_b.forward(test_item["sql_bug"])

        d = H_q.size(-1)
        A_gt = F.softmax(torch.matmul(H_s_gt.squeeze(0), H_q.squeeze(0).T) / math.sqrt(d), dim=-1)
        A_bug = F.softmax(torch.matmul(H_s_bug.squeeze(0), H_q.squeeze(0).T) / math.sqrt(d), dim=-1)

        T = build_target_distribution(tokens_s_gt).to(DEVICE)
        a_gt = A_gt.mean(dim=1)[:len(T)]
        a_bug = A_bug.mean(dim=1)[:len(T)]
        T = T / (T.sum() + 1e-9)
        a_gt = a_gt / (a_gt.sum() + 1e-9)
        a_bug = a_bug / (a_bug.sum() + 1e-9)

        kl_gt = compute_kl_divergence(T, a_gt).item()
        kl_bug = compute_kl_divergence(T, a_bug).item()
        print(f"  KL(T ∥ A_gt)  = {kl_gt:.4f}  (正确SQL, 应更小)")
        print(f"  KL(T ∥ A_bug) = {kl_bug:.4f}  (错误SQL, 应更大)")
        print(f"  差值 = {kl_bug - kl_gt:.4f}  (正差值表示对比成功)")


if __name__ == "__main__":
    main()
