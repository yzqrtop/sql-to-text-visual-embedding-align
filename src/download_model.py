#!/usr/bin/env python3
"""
从ModelScope下载XLM-RoBERTa模型到本地目录
"""

import os
from pathlib import Path
from modelscope import snapshot_download

MODEL_ID = "AI-ModelScope/xlm-roberta-base"
MODEL_DIR = Path(__file__).parent.parent / "models" / "xlm-roberta-base"

print(f"开始从ModelScope下载模型...")
print(f"模型ID: {MODEL_ID}")
print(f"目标目录: {MODEL_DIR}")

os.makedirs(MODEL_DIR.parent, exist_ok=True)

snapshot_download(
    MODEL_ID,
    cache_dir=str(MODEL_DIR.parent),
    revision="master"
)

print(f"\n[完成] 模型已下载到 {MODEL_DIR}")