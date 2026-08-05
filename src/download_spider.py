#!/usr/bin/env python3
"""
从ModelScope下载Spider数据集
仓库: ModelM/spider
"""

import json
import requests
from pathlib import Path

def download_file(repo_id, filename, output_path):
    if output_path.exists():
        print(f"  跳过: {filename} (已存在)")
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    
    url = f"https://www.modelscope.cn/api/v1/datasets/{repo_id}/repo?Revision=master&FilePath={filename}&View=False"
    print(f"下载 {filename}...")
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        data = response.json()
        print(f"  成功: {len(data)} 条")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  已保存: {output_path}")
        return data
    except Exception as e:
        print(f"  失败: {e}")
        return None

def main():
    output_dir = Path(__file__).parent.parent / "data" / "spider"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("从ModelScope下载Spider数据集")
    print("仓库: ModelM/spider")
    print("=" * 60)
    
    repo_id = "ModelM/spider"
    
    # 下载 train_spider.json
    train_spider = download_file(repo_id, "train_spider.json", output_dir / "train_spider.json")
    
    # 下载 train_others.json
    train_others = download_file(repo_id, "train_others.json", output_dir / "train_others.json")
    
    # 下载 dev.json
    dev_data = download_file(repo_id, "dev.json", output_dir / "dev.json")
    
    # 下载 tables.json
    tables_data = download_file(repo_id, "tables.json", output_dir / "tables.json")
    
    # 合并训练集
    if train_spider is not None or train_others is not None:
        train_data = []
        if train_spider:
            train_data.extend(train_spider)
        if train_others:
            train_data.extend(train_others)
        
        processed_train = []
        for item in train_data:
            if isinstance(item, dict):
                processed_train.append({
                    "db_id": str(item.get('db_id', '')),
                    "query": str(item.get('query', '')),
                    "question": str(item.get('question', '')),
                    "sql": str(item.get('query', ''))
                })
        
        with open(output_dir / "train.json", 'w', encoding='utf-8') as f:
            json.dump(processed_train, f, ensure_ascii=False, indent=2)
            
        print(f"\n合并训练集: {len(processed_train)} 条")
    
    # 处理验证集
    if dev_data is not None:
        processed_dev = []
        for item in dev_data:
            if isinstance(item, dict):
                processed_dev.append({
                    "db_id": str(item.get('db_id', '')),
                    "query": str(item.get('query', '')),
                    "question": str(item.get('question', '')),
                    "sql": str(item.get('query', ''))
                })
        
        with open(output_dir / "dev.json", 'w', encoding='utf-8') as f:
            json.dump(processed_dev, f, ensure_ascii=False, indent=2)
        print(f"验证集: {len(processed_dev)} 条")
    
    print("\n" + "=" * 60)
    print("下载完成!")

if __name__ == "__main__":
    main()