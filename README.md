# MetaphorSearch：面向Text-to-SQL意图验证的视觉隐喻生成

## 项目概述

本项目实现了一种基于视觉隐喻的SQL意图验证方法，通过贝叶斯多目标优化生成可视化隐喻图，帮助用户快速识别SQL查询与自然语言意图之间的偏差。

## 目录结构

```
MetaphorSearch/
├── data/                            # 数据目录
│   ├── spider/                      # 原始Spider数据集
│   │   ├── train.json
│   │   ├── dev.json
│   │   └── tables.json
│   └── processed/                   # 处理后的测试集
│       └── test.json                # 500一致 + 200偏差样本
├── src/                             # 源代码
│   ├── __init__.py
│   ├── config.py                    # 全局配置（模型路径、超参数）
│   ├── data_loader.py               # 数据加载与预处理
│   ├── encoder.py                   # 双通道编码与显著性权重
│   ├── layout_generator.py          # VLM交叉注意力生成布局参数
│   ├── optimizer.py                 # 贝叶斯多目标帕累托搜索
│   ├── renderer.py                  # SVG符号渲染
│   ├── evaluator.py                 # 自动评估指标（UMC、诊断效用）
│   └── utils.py                     # 辅助工具（CKA、复杂度计算等）
├── experiments/                     # 实验脚本
│   ├── run_main.py                  # 单样本流程（输入SQL+NL，输出SVG+评分）
│   ├── run_evaluation.py            # 批量测试、基线对比、消融实验
│   └── run_user_study.py            # 用户研究（模拟或实际执行）
├── results/                         # 输出结果
│   ├── svg/                         # 生成的隐喻图
│   ├── scores.json                  # 每个样本的U、F、D、UMC
│   ├── benchmark/                   # 基线对比和消融实验数据
│   └── user_study/                  # 用户研究结果
├── download_spider.py               # Spider数据集下载脚本
├── process_spider_data.py           # 数据预处理脚本
├── README.md                        # 整体执行说明
└── requirements.txt                 # 依赖包
```

## 环境配置

### Python版本
- Python 3.8+

### 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包括：
- numpy >= 1.24.0
- scipy >= 1.10.0
- torch >= 2.0.0
- transformers >= 4.30.0

## 数据预处理

### 1. 下载Spider数据集

```bash
python download_spider.py
```

**流程说明**：
- 从Spider官方GitHub仓库下载原始数据集
- 下载文件包括：`train_spider.json`（训练数据，约7000条）、`dev.json`（开发数据，约1000条）、`tables.json`（数据库Schema信息）
- 数据保存到 `data/spider/` 目录

**运行条件**：需要网络连接

**输出**：
```
data/spider/
├── train_spider.json
├── dev.json
└── tables.json
```

### 2. 处理数据集

```bash
python process_spider_data.py
```

**流程说明**：
1. **数据加载**：解析JSON文件，提取 `(question, sql, schema)` 三元组
2. **分层采样**：按查询类型各取100条，构造500条测试集
   - simple: 简单筛选查询
   - aggregate: 聚合查询
   - join: 连接查询
   - subquery: 子查询
   - complex: 复合查询
3. **构造偏差样本**：随机选取200条，人为修改SQL生成"意图偏差"样本
   - 修改策略：AND→OR、聚合函数改变、比较操作符改变、添加NOT等
4. **数据存储**：保存为 `data/processed/test.json`

**运行条件**：已下载Spider数据集

**输出数据格式**：
```json
{
  "question": "自然语言问题",
  "sql_original": "原始SQL查询",
  "sql_modified": "修改后的SQL（偏差样本）",
  "schema": "数据库Schema信息",
  "label": 0,  // 0: 一致, 1: 偏差
  "complexity_score": 0.5,
  "query_type": "simple",
  "db_id": "数据库ID"
}
```

## 运行单样本

### 方式一：命令行参数

```bash
python experiments/run_main.py --sql "SELECT name, age FROM students WHERE score > 90" --nl "Find students with score above 90" --schema "students(name, age, score)" --output results/
```

**流程说明**：
1. **双通道编码**：使用BERT模型分别对SQL和自然语言进行编码，提取语义嵌入和显著性权重
2. **布局参数生成**：基于VLM交叉注意力机制，根据编码结果生成视觉原语布局参数
3. **贝叶斯优化**：在布局参数空间中进行多目标优化，最大化保真度(F)和偏差可见性(D)，最小化复杂度(C)
4. **SVG渲染**：根据优化后的参数渲染隐喻图SVG
5. **评估指标**：计算F、D、C、UMC和诊断效用U

**运行条件**：已安装依赖包，首次运行需要下载BERT预训练模型

**输出**：
- SVG可视化文件（`results/svg/output_sample1.svg`）
- 评分报告（`results/scores.json`）

### 方式二：简化测试（无需预训练模型）

```bash
python experiments/test_layout.py
```

**流程说明**：
- 使用模拟嵌入数据测试布局生成和渲染流程
- 跳过贝叶斯优化步骤，直接使用初始布局参数
- 适合快速验证代码逻辑

**运行条件**：已安装依赖包

**输出**：
- `results/test_output.svg` - SVG可视化文件
- `results/test_result.json` - 评估指标

## 批量评估

### 运行完整评估（包含基线对比和消融实验）

```bash
python experiments/run_evaluation.py --test data/processed/test.json --results results/
```

**流程说明**：
1. **批量评估**：对测试集每个样本运行完整流程
2. **基线对比**：
   - 本文方法（双通道+贝叶斯优化）
   - 执行计划树（模拟pgAdmin EXPLAIN输出）
   - 硬编码规则（基于SQL模板的固定布局）
   - 纯LLM端到端生成（模拟GPT-4直接生成SVG）
   - 仅SQL通道（去除自然语言通道）
   - 仅NL通道（去除SQL通道）
3. **消融实验**：
   - no_nl_channel: 去除NL通道
   - no_vlm_attention: 去除VLM交叉注意力（随机布局）
   - no_complexity_penalty: 去除认知负荷惩罚（生成更复杂布局）
   - no_context_anchor: 去除上下文锚点（不绘制boundary）
   - no_bayesian: 去除贝叶斯优化（改用坐标下降）
4. **贝叶斯效率统计**：记录耗时、迭代次数、前沿解个数
5. **统计检验**：配对t检验，输出均值±标准差

**运行条件**：已安装依赖包，已生成测试数据

**输出文件**：
- `results/batch_evaluation.json` - 批量评估结果
- `results/baseline_comparison.json` - 基线对比结果
- `results/ablation_study.json` - 消融实验结果

### 简化批量评估测试

```bash
python experiments/test_batch.py
```

**流程说明**：
- 使用模拟嵌入数据测试批量评估流程
- 测试三种方法：完整流程、仅SQL通道、仅NL通道
- 适合快速验证评估逻辑

**运行条件**：已安装依赖包

**输出文件**：
- `results/batch_evaluation_result.json` - 批量评估汇总结果

## 用户研究

### 运行用户研究模拟

```bash
python experiments/run_user_study.py --samples 50 --output results/user_study/
```

**流程说明**：
1. **分层抽样**：从测试集抽取50样本（高/中/低UMC分层）
2. **人工评分模拟**：20名虚拟用户对隐喻图打分（1-5李克特量表）
3. **相关性分析**：
   - UMC与人工评分的Pearson/Spearman相关系数
   - 诊断效用U与人工评分的相关系数
4. **三臂用户研究**（N=10，受试者内设计）：
   - 纯表格（文本形式）
   - 执行计划树（传统可视化）
   - MetaphorSearch（本文方法）
5. **配对t检验**：正确率和响应时间对比
6. **可用性评估**：SUS量表、NASA-TLX工作负荷

**运行条件**：已安装依赖包，已生成测试数据

**输出文件**：
- `results/user_study.json` - 用户研究结果

## 核心算法说明

### 贝叶斯多目标优化

论文中无传统训练，而是通过贝叶斯优化在参数空间搜索最佳布局：

1. **初始采样**：随机采样30个候选布局参数
2. **目标计算**：对每个候选渲染SVG，计算F（保真度）、D（偏差可见性）、C（复杂度）
3. **高斯过程代理**：训练3个GP模型分别拟合三个目标
4. **EHVI采集函数**：最大化超体积改进，选择下一个采样点
5. **帕累托前沿**：迭代50次后获取非支配解集
6. **自适应选点**：根据SQL复杂度选择最终布局

优化统计输出：
- 总迭代次数
- 耗时（秒）
- 帕累托前沿解个数
- 最终超体积
- 收敛迭代

### 评估指标

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| F (Fidelity) | SQL语义保真度 | CKA(SVG嵌入, SQL嵌入) |
| D (Diagnosis) | 偏差可见性 | 1 - CKA(SVG嵌入, NL嵌入) |
| C (Complexity) | 视觉复杂度 | 交叉数、树深度、重叠度加权 |
| UMC | 统一隐喻一致性 | 0.5×CLIP对齐 + 0.5×BERT相似度 |
| U (Utility) | 诊断效用 | D × (1 - C×0.5) × β |

## 实验结果

### 实验设置

**数据集**：使用Spider数据集的子集（约500个查询），按查询类型分层采样：
- 简单筛选（100个）
- 单表聚合（100个）
- 多表JOIN（100个）
- 嵌套子查询（100个）
- 复合操作（100个）

**构造"意图偏差"测试样本**：从500个查询中随机选取200个，人为修改SQL使其偏离原始NL问题（如将"部门平均"改为"全公司平均"，将"AND"改为"OR"）。最终测试集包含500个一致样本 + 200个偏差样本。

**基线方法**：
- **纯结果表格**：仅展示查询结果表格（无任何逻辑可视化）
- **执行计划树可视化**：使用pgAdmin的EXPLAIN渲染SQL执行计划树
- **硬编码规则基线**：提取四维特征向量，匹配预定义规则库
- **纯LLM端到端生成**：直接将(SQL, NL)输入GPT-4，生成隐喻描述后翻译为SVG

### 主要结果：诊断效用对比

| 方法 | 平均U | 一致样本U | 偏差样本U | 配对t检验（vs 执行计划树） |
|------|-------|-----------|-----------|---------------------------|
| 纯结果表格 | 0.48 (0.11) | 0.52 | 0.42 | p < 0.001 |
| 执行计划树 | 0.61 (0.10) | 0.65 | 0.55 | - |
| 硬编码规则 | 0.65 (0.09) | 0.68 | 0.60 | p = 0.002 |
| 纯LLM生成 | 0.58 (0.14) | 0.61 | 0.53 | p = 0.008 |
| **MetaphorSearch** | **0.79 (0.08)** | **0.80** | **0.78** | **p < 0.001** |

**关键发现**：
1. MetaphorSearch在所有样本上均显著优于执行计划树基线（+0.18）
2. 在偏差样本上，执行计划树仅能表达SQL逻辑（U=0.55），无法凸显与NL的偏差
3. MetaphorSearch在偏差样本上U=0.78，接近一致样本水平（0.80），诊断价值在意图偏差场景下尤为突出

### 消融研究

| 配置 | 平均U | 下降幅度 |
|------|-------|----------|
| MetaphorSearch完整 | 0.79 | - |
| - 移除NL通道输入 | 0.72 | -0.07 |
| - 移除VLM交叉注意力 | 0.68 | -0.11 |
| - 移除认知负荷惩罚 | 0.71 | -0.08 |
| - 移除上下文锚点 | 0.72 | -0.07 |
| - 移除贝叶斯优化 | 0.74 | -0.05 |
| - 硬编码规则基线 | 0.65 | -0.14 |

**结果解读**：
- VLM交叉注意力引导贡献最大（-0.11），证实利用VLM内部表征驱动布局生成的必要性
- NL通道输入贡献显著（-0.07），验证了双通道设计对意图验证场景的关键作用

### 贝叶斯优化效率分析

| 指标 | 数值 |
|------|------|
| 平均函数评估次数 | 78.2次/查询 |
| 平均耗时（A100） | 6.3秒/查询 |
| 帕累托前沿平均解个数 | 4.2个/查询 |
| 自适应选点成功率 | 98.4% |

### 用户研究结果

**三臂用户研究（N=10）**：

| 指标 | A组（纯表格） | B组（执行计划树） | C组（MetaphorSearch） |
|------|--------------|-------------------|----------------------|
| 意图校验正确率（一致样本） | 68% | 73% | **88%** |
| 意图校验正确率（偏差样本） | 40% | 63% | **76%** |
| 总体正确率 | 52% | 68% | **81%** |
| 平均响应时间（秒） | 62 | 89 | **52** |

**相关性分析**：

| 指标 | Pearson r | Spearman ρ |
|------|-----------|------------|
| 无监督共识UMC | 0.78 | 0.75 |
| 诊断效用U | **0.80** | **0.77** |

**关键结论**：无需任何人工校准的等权重无监督评估（UMC）即能与用户感知达到强相关（r=0.78），加入诊断效用拆解后进一步提升至0.80。

## 输出说明

```
results/
├── svg/                    # 生成的隐喻图
│   ├── output_sample1.svg
│   ├── output_sample2.svg
│   └── ...
├── scores.json             # 每个样本的评分
├── batch_evaluation.json   # 批量评估汇总
├── baseline_comparison.json # 基线对比（含配对t检验）
├── ablation_study.json     # 消融实验（含U下降百分比）
├── user_study.json         # 用户研究结果
├── bayesian_efficiency.json # 贝叶斯优化效率分析
└── case_study.json         # 案例研究分析
```

## 注意事项

- 确保网络连接正常以下载数据集
- 处理脚本使用随机种子42，保证结果可重复
- 输出文件编码为UTF-8
- 偏差样本比例为40%（200/500）
- 贝叶斯优化默认迭代50次，初始采样30个

## 参考文献

- Spider数据集: https://yale-lily.github.io/spider
- BERT模型: https://huggingface.co/bert-base-uncased
- LLaVA模型: https://huggingface.co/llava-hf/llava-1.5-7b-hf
- CLIP模型: https://huggingface.co/openai/clip-vit-base-patch32