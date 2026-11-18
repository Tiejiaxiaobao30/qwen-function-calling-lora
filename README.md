# Qwen2.5 工具调用微调与评测

基于 [ArunarkSingh/qwen-function-calling-lora](https://github.com/ArunarkSingh/qwen-function-calling-lora) 的个人实验仓库。保留上游 QLoRA SFT、生成与评分流程及用户源码注释，扩展教师改写对照（Exp1）与 SFT→DPO（Exp2）。这些是复现和扩展实验，不宣称自研算法或官方 BFCL 成绩。

## 实际结果

同为项目 AST 严格匹配、greedy、500 条 Test，但来源和训练条件有差异，不能将本表解释为严格受控的逐阶段提升。

| 实验 | 训练数据/起点 | JSON 有效 | 函数名 | 严格匹配 | 失败 |
|---|---|---:|---:|---:|---:|
| Base（Exp0 报告记录） | 未微调 | 76.6% | 44.6% | 31.0% | — |
| Exp0 SFT | 原始 5000 条，3 epochs | 99.2% | 98.8% | 77.4%（387/500） | 113 |
| Exp1 Control | 脱敏 5000 + 重复 490 | 99.4% | 98.8% | 77.2%（386/500） | 114 |
| Exp1 Rewrite | 脱敏 5000 + 教师改写 490 | 99.6% | 99.2% | 76.8%（384/500） | 116 |
| Exp2 DPO | 从 Exp0 adapter 继续，451 对偏好 | 99.2% | 98.8% | 78.0%（390/500） | 110 |

Exp1 Rewrite 对 Control 的严格匹配低 0.4 个百分点，本轮没有整体收益。Exp2 Dev（原 500 val）为 JSON 99.8%、函数名 99.4%、严格匹配 79.2%（396/500）；这是生成质量验证，区别于 DPO 的 50 对偏好验证集。

Exp2 对历史 Exp0 的表面差值为 +0.6 个百分点（3 题），但没有本轮同条件 Exp0 逐题预测，不能声称已验证 DPO 净收益、显著提升或泛化收益。归档 `exp0_reference` 是 **499 条的上游报告**，不是本人的 500 条 Exp0 基线，已排除。Exp2 Test 预测的 gold、数据指纹与本地原始 500 条 Test 已核对一致；Exp1 使用脱敏输入，val 零基 128 / test 零基 126 与原始输入不同。

## 目录与发布范围

```text
qwen-function-calling-lora/
├── config.py             # 原始注释文件，本轮字节不变
├── prepare_data.py       # 原始注释文件，本轮字节不变
├── train.py              # 原始注释文件，本轮字节不变（保留此前 seed 修复）
├── scoring.py            # 原始注释文件，本轮字节不变
├── evaluate.py           # 原始注释文件，本轮字节不变
├── error_analysis.py     # 原始注释文件，本轮字节不变
├── requirements.txt      # 原有依赖声明
├── exp1/
│   ├── augment_data.py / augmentation_utils.py / __init__.py
│   ├── README.md
│   ├── reports/          # 聚合统计与历史核验报告，公开
│   └── local/            # 生成进度、两组数据/最终模型/checkpoint，仅本机
├── exp2/
│   ├── generate_candidates.py / generate_candidates_batch.py
│   ├── build_preferences.py / train_dpo.py / compare_exp2.py
│   ├── utils.py / __init__.py
│   ├── README.md
│   ├── reports/          # 最终实验报告、统计、审查与验证边界，公开
│   └── local/archives/   # 完整云端归档、原始源码、模型与备份，仅本机
├── README.md
└── .gitignore            # 默认排除，逐文件公开白名单
```

Exp0 数据、最终 adapter 和 checkpoint 已整体移到本机相邻的 `E:/qwen-function-calling-lora-artifacts/exp0/`。整理前文档、旧 dpo 源码、publish 脚本、旧包和逐文件哈希在 `E:/qwen-function-calling-lora-artifacts/before-final-20261118/`。移动而非删除；原路径映射、完整性核验见 [整理与发布审查](exp2/reports/organization-review.md)。

本分支公开文件树只含源码、运行说明和无样本的聚合结论。原始/脱敏 xLAM、教师响应、偏好对、预测/gold、错误样例、模型权重、checkpoint、二进制与压缩包均不提交。历史 Git 提交仍包含过去发布的产物；本次从新分支索引移除这些路径，没有改写历史，也不代表旧历史已清除。合并前 main 保持旧版本。

## 运行入口

从仓库根目录执行；Exp1/Exp2 用 `python -m ...`。包内源码仍复用根目录配置和评分，不能直接双击子目录脚本。

```powershell
cd E:\qwen-function-calling-lora
# 读取已保存 Exp0 产物；不要重新 prepare_data 覆盖它们。
$env:OUTPUT_ROOT = 'E:/qwen-function-calling-lora-artifacts/exp0'
# 仅示例：以下评测会加载 GPU，并可能覆盖已有报告，先选新输出或备份。
# python evaluate.py
# python error_analysis.py
```

新 Exp0 训练必须把 `OUTPUT_ROOT` 设为全新、仓库外路径，再依次运行 `prepare_data.py`、`train.py`、`evaluate.py`、`error_analysis.py`。Exp1 运行方式见 [Exp1](exp1/README.md)，Exp2 见 [Exp2](exp2/README.md)。本轮只做离线核验，没有重训或 GPU 重评。原 requirements 是既有声明；Exp2 实际归档环境与本地轻量检查环境的区别见实验报告。

## 评分、来源与边界

`scoring.score` 解析 JSON 后按函数名与 arguments 的集合匹配，忽略调用顺序，也会折叠重复调用；JSON validity 包含原解析器的代码块/片段提取行为，不等价于完整 JSON Schema 合法率。无 LLM judge，无真实工具执行验证。`diagnose` 是启发式错误分类。

数据源为 [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)，来自 [APIGen](https://arxiv.org/abs/2406.18518)。本项目按用户要求不再分发数据及其衍生样本；脱敏不等于获得再分发授权。来源访问和使用条件由使用者自行确认。本仓库继承上游来源与署名，未发现独立的上游 LICENSE 文件，本次不补授代码或数据许可。个人贡献包括源码阅读注释、Exp1/Exp2 实验扩展、恢复检查、离线核验和报告整理；不把上游实现宣称原创。

详细结果：[Exp1 报告](exp1/reports/exp1-results.md)、[Exp2 最终报告](exp2/reports/experiment-report.md)、[Exp2 机器可读证据](exp2/reports/results-summary.json)。
