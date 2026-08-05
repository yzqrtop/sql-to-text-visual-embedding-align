#!/usr/bin/env python3
"""
完整流程执行脚本
直接通过import调用各模块
"""

import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_step(step_func, description):
    print("\n" + "=" * 60)
    print("步骤: " + description)
    print("=" * 60)
    
    try:
        step_func()
        print("\n[OK] 步骤完成")
        return True
    except Exception as e:
        print("\n执行失败: " + str(e))
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("MetaphorSearch - 完整流程执行")
    print("=" * 60)
    print()

    print("加载模块...")
    
    from src.download_spider import main as download_main
    from process_spider_data import main as process_main
    from generate_layout_data import main as generate_main
    from train_encoder_b import main as train_encb_main
    from train_mlp import main as train_main
    from experiments.run_evaluation import main as evaluate_main

    print("[OK] 模块加载完成")
    print()

    steps = [
        {
            "func": download_main,
            "desc": "1. 下载Spider数据集",
            "required": True
        },
        {
            "func": process_main,
            "desc": "2. 数据预处理 (分层采样+偏差标注)",
            "required": True
        },
        {
            "func": generate_main,
            "desc": "3. 生成训练数据 (16维指纹 + Encoder B三元组)",
            "required": True
        },
        {
            "func": train_encb_main,
            "desc": "4. Encoder B辅助分布对齐微调 (KL+对比损失)",
            "required": True
        },
        {
            "func": train_main,
            "desc": "5. 训练MLP回归器 (AdamW+余弦退火+早停)",
            "required": True
        },
        {
            "func": evaluate_main,
            "desc": "6. 实验评估 (布局预测+模板分布+消融)",
            "required": True
        }
    ]

    print("流程步骤:")
    for step in steps:
        status = "[REQ]" if step["required"] else "[OPT]"
        print("  " + status + " " + step["desc"])
    print()

    success_count = 0
    for step in steps:
        if run_step(step["func"], step["desc"]):
            success_count += 1
        else:
            if step["required"]:
                print("[FAIL] 关键步骤失败，终止流程")
                print("\n" + "=" * 60)
                print("流程终止: " + str(success_count) + "/" + str(len(steps)) + " 步骤完成")
                print("=" * 60)
                return
            else:
                print("[OPT] 非关键步骤跳过")

    print("\n" + "=" * 60)
    print("流程完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
