# Qwen2.5 工具调用微调与评测

基于 **Qwen2.5-1.5B-Instruct + QLoRA** 的个人工具调用实验项目，围绕结构化 JSON 调用完成数据处理、监督微调、Base / Tuned 对照评测与错误分析。Exp0 在 500 条测试样本上，严格 Exact Match 从 **31.0% 提升至 77.4%**；Exp1 教师改写对照未观察到整体 Exact Match 收益。

项目基于开源训练流程开展复现，保留中文源码阅读注释、实际运行产物与来源记录，并在本仓库持续维护。实现基础与当前工作范围见文末；归档校验详情见 [实验记录](docs/run-20261115.md)。

## 项目结构

```text
qwen-function-calling-lora/
├── README.md                         # 项目说明、实验结果与运行入口
├── config.py                         # 模型、数据量、路径和训练参数
├── prepare_data.py                   # 数据处理与划分（中文注释）
├── train.py                          # QLoRA 训练、恢复与保存（中文注释）
├── scoring.py                        # 生成、JSON 解析与评分（中文注释）
├── evaluate.py                       # Base / Tuned 对照评测（中文注释）
├── error_analysis.py                 # 错误分类与宽松匹配（中文注释）
├── publish.py                        # 上游 Hugging Face 发布辅助脚本
├── augment_data.py                   # Exp1：一次尝试，按成功N条生成Rewrite及Control
├── augmentation_utils.py             # Exp1：样本转换、提示词与基本校验
├── requirements.txt                  # 项目依赖声明
│
├── outputs/                          # 历史成果与当前Exp1（旧中间产物已移出归档）
│   ├── data/                         # 原始划分，Arrow文件仅保存在本地
│   │   ├── meta.json                 # 数据来源、seed、数量等
│   │   ├── train/                    # 5,000 条训练数据
│   │   ├── val/                      # 500 条验证数据
│   │   └── test/                     # 500 条测试数据
│   ├── data-redacted/                # GitHub公开副本：疑似凭证已替换
│   │   ├── train/                    # 5,000 条，其中8条脱敏
│   │   ├── val/                      # 500 条，其中1条脱敏
│   │   └── test/                     # 500 条，其中1条脱敏
│   ├── checkpoints/
│   │   ├── checkpoint-600/           # 验证损失最好的 checkpoint
│   │   ├── checkpoint-900/           # 后期训练快照
│   │   └── checkpoint-939/           # 训练结束快照（3 epochs）
│   │       ├── adapter_model.safetensors
│   │       ├── adapter_config.json
│   │       ├── trainer_state.json    # 步数、最佳 checkpoint 与日志
│   │       ├── optimizer.pt          # 恢复训练所需优化器状态
│   │       ├── scheduler.pt
│   │       ├── rng_state.pth
│   │       └── ...                   # tokenizer、训练参数等
│   ├── adapter-final/                # 按最佳验证损失选择的最终 adapter
│   │   ├── adapter_model.safetensors # 约 70.5 MiB，不含完整底座
│   │   ├── adapter_config.json
│   │   ├── tokenizer.json
│   │   └── ...
│   ├── exp1-rewrite-v2/              # 已完成的本地Rewrite数据与生成记录
│   │   └── data/                     # 候选/成功清单、配置与导出的数据
│   ├── eval_results.json             # 本项目实验的 Base / Tuned 对照结果
│   └── error_analysis.json           # 本项目实验的错误分类与部分样例
│
├── docs/
│   ├── run-20261115.md               # 来源、校验依据与验证边界
│   ├── backup-manifest.json          # 产物体积、SHA-256 与存储方式
│   ├── DATA_ATTRIBUTION.md           # 数据来源、署名与条件入口
│   ├── data-redaction.json           # 脱敏行号与替换规则（不含原值）
│   ├── cloud-run-source/             # 备份中的原始云端源码与依赖声明
│   └── upstream-README.md            # 上游原始说明与原始结果
├── eval_results.json                 # 上游报告，非本次结果，原样保留
├── error_analysis.json               # 上游错误分析，非本次结果，原样保留
├── .gitattributes                    # 二进制文件按原始字节保存
└── .gitignore                        # 排除缓存、临时文件与本机配置
```

`data` 保留以前实验的原始划分，默认评测仍读取它；`data-redacted` 是同一划分的脱敏副本，今天的增强读取它，不能混作完全相同的历史评测输入。`adapter-final`、`checkpoints` 和两个评测JSON保留以前成果。

Exp1 已完成：500 个候选中 accepted=490；`outputs/exp1-control-flex/data` 仅重复这 490 个成功 seed 的原始样本，Rewrite 增加 490 条教师改写。两组均为 5490/500/500。云端完整归档位于本地 `exp1-cloud-results/`；结果、核验边界及命令见 [Exp1 实验记录](docs/exp1-results.md)，旧生成过程见本地 `docs/exp1-teacher-augmentation.md`（尚未提交）。

## 实验结果

来源：[outputs/eval_results.json](outputs/eval_results.json)。测试集 **500 条**，greedy decoding，采用项目实现的 BFCL-style AST match；这不是官方 BFCL 榜单评测。

| 指标 | Base | Tuned |
|---|---:|---:|
| JSON 格式有效率 | 76.6% | 99.2% |
| 函数名准确率 | 44.6% | 98.8% |
| 严格 Exact Match | 31.0% | 77.4% |

严格 Exact Match 提高 **46.4 个百分点**。错误分析统计 113 条失败：参数值错误 61 条、参数键错误 46 条、JSON 无效 4 条、函数名错误 2 条；宽松匹配率为 77.8%。详见 [outputs/error_analysis.json](outputs/error_analysis.json)。

**不要混用根目录报告。** 根目录两份 JSON 和 [上游 README](docs/upstream-README.md) 属于初始仓库内容，其中 78.8% 等数字不代表本次复现结果。

## Exp1：教师改写与重复采样对照

固定 5000 条训练源，seed=42 抽取 500 个候选，实际接受 490 个。Control 追加成功 seed 的 490 条原样本，Rewrite 追加对应的 490 条教师改写；成功 seed 清单完全相同，两组 train=5490、val/test=500，val/test 已逐行核对一致。规则校验通过不代表人工语义验收。

| 模型 / 实验 | JSON 有效 | 函数名准确 | 严格 Exact Match | 失败数 |
|---|---:|---:|---:|---:|
| Base（报告记录值） | 76.6% | 44.6% | 31.0% | — |
| Exp0 Tuned（历史参考） | 99.2% | 98.8% | 77.4% | 113 |
| Exp1 Control Tuned | 99.4% | 98.8% | 77.2% | 114 |
| Exp1 Rewrite Tuned | 99.6% | 99.2% | 76.8% | 116 |

Rewrite 相比 Control：JSON **+0.2 pp**、函数名 **+0.4 pp**、严格匹配 **−0.4 pp（少正确 2 道）**。参数值错误 62→58，但参数键错误 45→51；主要瓶颈仍在参数生成。增加数据不一定提升性能，本轮没有整体 Exact Match 改善证据。

**比较边界：** Exp1 两组测试集完全相同，但与 Exp0 原始 test 有 1 条输入不同（脱敏），val 也有 1 条不同；不能宣称三组严格同一输入。Exp0/Exp1 训练量为 5000/5490、结束步数为 939/1032，Control−Exp0 不能解释为纯数据量因果。单次运行、500 条测试、差异很小，可能随机波动，未做多随机种子或显著性检验。Exp1 JSON 中 Base 数字与历史相同，本次未验证其是否独立重评。

汇总统计、错误样例观察（不公开原始样本）、数据哈希与产物定位见 [Exp1 实验记录](docs/exp1-results.md) 和 [机器可读汇总](docs/exp1-results-summary.json)。

## 训练配置与模型选择

- 底座：`Qwen/Qwen2.5-1.5B-Instruct`。
- 数据：`Salesforce/xlam-function-calling-60k`，处理后使用 5000 / 500 / 500 划分，seed 42。
- QLoRA：4-bit NF4；LoRA r=16、alpha=32、dropout=0.05；训练 3 epochs。
- `checkpoint-939/trainer_state.json` 记录训练结束于 step 939，最佳验证损失出现在 **step 600**，eval_loss 约 **0.02535**。
- `adapter-final` 的 392 个张量与 `checkpoint-600` 在 BF16 → FP32 转换后逐元素相同。最终文件更大是保存精度不同，不表示选用了最后一步权重。

以上来自备份文件核对；本次整理没有重新训练、重新运行 GPU 评测或验证恢复训练。

## 模型与数据获取

克隆可获取已提交的代码、说明与历史文件；Exp1 完整归档仅保存在本地：

```bash
git clone https://github.com/Tiejiaxiaobao30/qwen-function-calling-lora.git
cd qwen-function-calling-lora
```

历史归档统计约 **441.6 MiB**；此前公开版本用 `data-redacted/` 替代原始 Arrow 数据，其余产物按原文件保存。单文件均小于 100 MiB。这个公开 Fork 不能上传新的 Git LFS 对象，因此采用普通 Git；首次克隆需要下载较多内容。后续多轮实验应采用独立模型存储或发布附件，避免重复累积大文件历史。

原始数据10条样本的工具定义默认值含疑似GitHub凭证，公开副本将其替换为 `[REDACTED_CREDENTIAL]`，保留数量与行顺序。原始Arrow文件只在本地和完整备份中保留，未推送。**Exp0 指标来自原始数据，不是对脱敏副本重跑的结果；Exp1 使用脱敏源。** 从GitHub恢复时，需在本地核对后将 `data-redacted/` 复制为 `data/` 才能用于默认脚本；这不会严格复现 Exp0 原始输入。脱敏行号见 [data-redaction.json](docs/data-redaction.json)。

本次发布仅新增说明与汇总统计；遵循 xLAM gated / research-only 使用边界，不新增公开原始或衍生样本、教师输出、权重及 checkpoint。历史已跟踪文件仍在仓库中，本次忽略规则不会撤销其跟踪，也不代表重新审查或授权再分发；不要将已有文件视为数据许可。

中间 `cache-*.arrow` 仅在本地保留，不纳入提交。完整 Qwen 底座和 Hugging Face 缓存没有打包，需要另行下载。数据来源及署名见 [DATA_ATTRIBUTION.md](docs/DATA_ATTRIBUTION.md)，使用前请阅读并接受来源站点的适用条件。

## 运行与开发

推荐顺序：`config.py → prepare_data.py → train.py → scoring.py → evaluate.py → error_analysis.py`。

```bash
pip install -r requirements.txt
python prepare_data.py
python train.py
python evaluate.py
python error_analysis.py
```

脚本默认使用 `outputs/`，重新执行可能更新已有产物。新实验应先指定独立目录，例如 PowerShell：

```powershell
$env:OUTPUT_ROOT = "outputs-new-run"
```

根目录保留学习注释版；[cloud-run-source](docs/cloud-run-source/) 保存备份中的原始云端代码，便于追溯。依赖声明不是完整环境锁文件，恢复时仍需核验 GPU、驱动和库版本。`publish.py` 为上游发布工具，使用前应检查目标账号、模型卡和指标路径；本次整理没有向 Hugging Face 发布模型。

## 项目基础与工作范围

本项目以开源项目 `qwen-function-calling-lora` 的 `33a10fc` 版本为起点，训练、生成与评分流程复用上游实现。原作者说明及发布链接见 [上游 README](docs/upstream-README.md)，保留原 Git 历史与署名。

| 范围 | 当前内容 |
|---|---|
| 开源实现基础 | 数据准备、QLoRA 训练、生成、AST 风格评分与错误分类脚本 |
| 本仓库的阅读整理 | 中文源码注释、文件结构导航、运行源码与阅读版本对照保存 |
| 本项目的实验产物 | 5000 / 500 / 500 数据划分、训练 checkpoint、最终 adapter、500 条测试的 Base / Tuned 报告与错误统计 |
| 结果追溯与发布整理 | 产物哈希清单、最佳 checkpoint 核对、数据脱敏与原始/公开数据边界说明 |

后续实验与代码改进继续在本仓库维护，新增结果以独立运行记录为依据。当前版本属于基于开源实现的复现实验，已完成上述 Exp1 配对对照；尚未验证自研方法收益或跨数据集泛化收益。
