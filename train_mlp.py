#!/usr/bin/env python3
"""
MLP回归器训练脚本 - 16维语义指纹
从 src.layout_generator 导入 LayoutMLP，使用 mini-batch 策略
训练损失: L = MSE(p_base, p*) + λ‖Θ‖²
"""

import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.layout_generator import LayoutMLP, FINGERPRINT_DIM_NAMES
from src.config import MLP_INPUT_DIM, MLP_HIDDEN_DIM, MLP_OUTPUT_DIM, TRAIN_L2_LAMBDA

checkpoint_dir = project_root / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
# 使用相对路径避免Windows下中文路径的torch.save Unicode问题
checkpoint_path = "./checkpoints/layout_mlp.pth"
print(f"Checkpoint path: {checkpoint_path}")

EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 20  # 早停patience (论文: ~120-150轮收敛)


def main():
    data_path = project_root / "data" / "processed" / "layout_train_data.json"

    if not data_path.exists():
        print("[FAIL] 训练数据文件不存在: " + str(data_path))
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("[OK] 加载训练数据: " + str(len(data)) + " 条")

    features = np.array([item['embedding'] for item in data], dtype=np.float32)
    targets = np.array([item['target_params'] for item in data], dtype=np.float32)

    print("特征维度: " + str(features.shape))
    print("目标维度: " + str(targets.shape))

    print("\n16维语义指纹统计:")
    for i, name in enumerate(FINGERPRINT_DIM_NAMES):
        print(f"  D{i:02d} {name:<25} 均值={targets[:, i].mean():.3f}  范围=[{targets[:, i].min():.2f}, {targets[:, i].max():.2f}]")

    indices = np.random.permutation(len(features))
    split_idx = int(len(features) * 0.85)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    train_features = torch.tensor(features[train_idx])
    train_targets = torch.tensor(targets[train_idx])
    test_features = torch.tensor(features[test_idx])
    test_targets = torch.tensor(targets[test_idx])

    print("\n训练集: " + str(len(train_features)) + " 条")
    print("测试集: " + str(len(test_features)) + " 条")

    print("\n" + "=" * 60)
    print(f"MLP回归器训练 ({MLP_OUTPUT_DIM}维语义指纹)")
    print(f"输入: e_s^A (Encoder A, {MLP_INPUT_DIM}维)")
    print("=" * 60)

    device = torch.device("cpu")
    model = LayoutMLP(
        input_dim=MLP_INPUT_DIM,
        hidden_dim=MLP_HIDDEN_DIM,
        output_dim=MLP_OUTPUT_DIM
    ).to(device)
    criterion = nn.MSELoss()
    # 论文: AdamW + 余弦退火
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=TRAIN_L2_LAMBDA)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    print(f"  设备: cpu")
    print(f"  输入维度: {MLP_INPUT_DIM} (e_s^A)")
    print(f"  输出维度: {MLP_OUTPUT_DIM}")
    print(f"  优化器: AdamW (lr={LEARNING_RATE}, wd={TRAIN_L2_LAMBDA})")
    print(f"  调度器: CosineAnnealing (T_max={EPOCHS})")
    print(f"  Batch大小: {BATCH_SIZE}")
    print(f"  早停patience: {EARLY_STOP_PATIENCE}")

    print("\nEpoch  Train Loss  Test Loss  Test MAE")
    print("-" * 50)

    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(EPOCHS):
        model.train()

        permutation = torch.randperm(train_features.size()[0])
        epoch_loss = 0.0
        batch_count = 0

        for i in range(0, train_features.size()[0], BATCH_SIZE):
            indices = permutation[i:i + BATCH_SIZE]
            batch_x = train_features[indices].to(device)
            batch_y = train_targets[indices].to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)
            batch_count += batch_x.size(0)

        epoch_loss /= batch_count
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(test_features.to(device))
                test_loss = criterion(test_outputs, test_targets.to(device))
                test_mae = torch.mean(torch.abs(test_outputs - test_targets.to(device)))

            print(f"{epoch + 1:<6} {epoch_loss:<10.4f} {test_loss.item():<10.4f} {test_mae.item():<9.4f}")

            if test_loss.item() < best_loss:
                best_loss = test_loss.item()
                torch.save(model.state_dict(), str(checkpoint_path))
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print(f"\n早停: {EARLY_STOP_PATIENCE}轮无改善, 停止训练")
                    break

    print("-" * 50)

    print("\n训练完成!")
    print(f"  最佳测试损失: {best_loss:.6f}")
    print(f"  模型已保存: {checkpoint_path}")

    # 最终评估
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_features.to(device))
        test_loss = criterion(test_outputs, test_targets.to(device))
        test_mae = torch.mean(torch.abs(test_outputs - test_targets.to(device)))

    print(f"\n最终评估结果:")
    print(f"  MSE损失: {test_loss.item():.6f}")
    print(f"  MAE: {test_mae.item():.4f}")

    print("\n各维度预测 vs 目标对比:")
    pred_mean = test_outputs.mean(dim=0).cpu().numpy()
    target_mean = test_targets.mean(dim=0).cpu().numpy()

    for i, name in enumerate(FINGERPRINT_DIM_NAMES):
        diff = abs(pred_mean[i] - target_mean[i])
        print(f"  D{i:02d} {name:<25} 预测={pred_mean[i]:.3f}  目标={target_mean[i]:.3f}  偏差={diff:.3f}")


if __name__ == "__main__":
    main()
