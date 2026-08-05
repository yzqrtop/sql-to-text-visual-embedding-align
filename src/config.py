"""
MetaphorSearch 配置模块
包含模型、训练和渲染的所有配置参数

根据方法论核心公式定义：
- 16维语义指纹 (L1表结构/L2操作语义/L3输出聚焦)
- 4维熵调制 β = [β_ent, β_join, β_cond, β_res]
"""

import torch
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODER_MODEL_NAME = "xlm-roberta-base"
ENCODER_MODEL_DIR = Path("models/AI-ModelScope/xlm-roberta-base")
ENCODER_MAX_LEN = 512
ENCODER_DIM = 768

# === 双编码器配置 (论文 Training §2) ===
# Encoder A: 全层冻结, 仅SQL → e_s^A (768维)
# Encoder B: 前10层冻结, 最后2层可训练, SQL+NL → H_s^B, H_q^B
ENCODER_B_FREEZE_LAYERS = 10   # 冻结前10层
ENCODER_B_TRAIN_LAYERS = 2     # 可训练最后2层
ENCODER_B_CHECKPOINT = "./checkpoints/encoder_b_finetuned.pth"
ENCODER_B_LR = 2e-5            # Encoder B微调学习率
ENCODER_B_EPOCHS = 10          # Encoder B微调轮数
ENCODER_B_BATCH_SIZE = 8       # Encoder B batch (显存限制)
ENCODER_B_L_ALIGN_WEIGHT = 1.0 # L_align 权重
ENCODER_B_L_CONTRAST_WEIGHT = 0.5  # L_contrast 权重
ENCODER_B_CONTRAST_MARGIN = 0.5     # 对比损失间隔

# === 语义指纹配置 (Methodology §3) ===
# MLP输入: e_s^A (768维, 来自Encoder A)
MLP_INPUT_DIM = 768
MLP_HIDDEN_DIM = 512
MLP_OUTPUT_DIM = 16
MLP_CHECKPOINT = Path("checkpoints/layout_mlp.pth")
MLP_CHECKPOINT_STR = "./checkpoints/layout_mlp.pth"

# === 熵调制配置 (Methodology §4) ===
# SQL功能分组: Entity(表/列) / Join / Condition(WHERE/HAVING/ON) / Result(SELECT/ORDER)
ATTN_ENTROPY_GROUPS = ["entity", "join", "cond", "res"]
BETA_GAIN = 0.3  # p_final = p_base * (1 + BETA_GAIN * β_expand)

# === 双路径编码配置 (Methodology §6) ===
BETA_THRESHOLD = 0.5  # 高/低熵分界
HIGH_BETA_SATURATION = 1.3  # 高β视觉凸出
LOW_BETA_OPACITY = 0.2  # 低β主动去强调
WARNING_COLOR = "#E74C3C"  # 高熵警告色

# === 渲染几何公式配置 (Methodology §6) ===
CONTAINMENT_LOG_BASE = 1.0  # A_i = p_area_i * log(1 + |T_i|) + c
CONTAINMENT_OFFSET = 50     # 偏移常数 c

PATHFINDING_THICK_GAIN = 2.0  # ω_ij = p_thick_ij * log(1 + |T_i ⋈ T_j|)

FLOW_OPACITY_DECAY = 0.7  # α = p_opac * (1 - 0.7^ξ)

# === 训练配置 ===
TRAIN_EPOCHS = 50
TRAIN_BATCH_SIZE = 32
TRAIN_LR = 1e-4
TRAIN_WEIGHT_DECAY = 1e-5
TRAIN_L2_LAMBDA = 1e-5  # L_base = MSE + λ‖Θ‖²

RENDERER_W = 800
RENDERER_H = 800

ANCHOR_EPS = 1e-6
ANCHOR_MAX_ITER = 100

DATA_DIR = Path("data")
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
CHECKPOINTS_DIR = Path("checkpoints")
SVGS_DIR = Path("results/svgs")
