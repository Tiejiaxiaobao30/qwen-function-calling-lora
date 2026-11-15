# Qwen2.5 工具调用微调：源码注释与实验备份

基于 **Qwen2.5-1.5B-Instruct + QLoRA** 的工具调用学习与复现项目。保留源码阅读时添加的中文注释，归档云端训练产生的数据、checkpoint、最终 LoRA adapter、评测与错误分析。

本次备份在 **2026-11-15** 完成本地核验与整理；该日期不是对云端训练时间的推断。完整压缩包内 69 个文件均与本地解压副本通过 SHA-256 比对。

## 先看文件树

```text
qwen-function-calling-lora/
├── README.md                         # 当前导航与本次结果
├── config.py                         # 模型、数据量、路径和训练参数
├── prepare_data.py                   # 数据处理与划分（中文注释）
├── train.py                          # QLoRA 训练、恢复与保存（中文注释）
├── scoring.py                        # 生成、JSON 解析与评分（中文注释）
├── evaluate.py                       # Base / Tuned 对照评测（中文注释）
├── error_analysis.py                 # 错误分类与宽松匹配（中文注释）
├── publish.py                        # 上游 Hugging Face 发布辅助脚本
├── requirements.txt                  # 项目依赖声明
│
├── outputs/                          # ★ 本次实际运行产物
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
│   ├── eval_results.json             # ★ 本次 Base / Tuned 指标
│   └── error_analysis.json           # ★ 本次错误统计与部分样例
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

## 本次结果

来源：[outputs/eval_results.json](outputs/eval_results.json)。测试集 **500 条**，greedy decoding，采用项目实现的 BFCL-style AST match；这不是官方 BFCL 榜单评测。

| 指标 | Base | Tuned |
|---|---:|---:|
| JSON 格式有效率 | 76.6% | 99.2% |
| 函数名准确率 | 44.6% | 98.8% |
| 严格 Exact Match | 31.0% | 77.4% |

严格 Exact Match 提高 **46.4 个百分点**。错误分析统计 113 条失败：参数值错误 61 条、参数键错误 46 条、JSON 无效 4 条、函数名错误 2 条；宽松匹配率为 77.8%。详见 [outputs/error_analysis.json](outputs/error_analysis.json)。

**不要混用根目录报告。** 根目录两份 JSON 和 [上游 README](docs/upstream-README.md) 属于初始仓库内容，其中 78.8% 等数字不代表本次复现结果。

## 最终权重来自哪里

- 底座：`Qwen/Qwen2.5-1.5B-Instruct`。
- 数据：`Salesforce/xlam-function-calling-60k`，处理后使用 5000 / 500 / 500 划分，seed 42。
- QLoRA：4-bit NF4；LoRA r=16、alpha=32、dropout=0.05；训练 3 epochs。
- `checkpoint-939/trainer_state.json` 记录训练结束于 step 939，最佳验证损失出现在 **step 600**，eval_loss 约 **0.02535**。
- `adapter-final` 的 392 个张量与 `checkpoint-600` 在 BF16 → FP32 转换后逐元素相同。最终文件更大是保存精度不同，不表示选用了最后一步权重。

以上来自备份文件核对；本次整理没有重新训练、重新运行 GPU 评测或验证恢复训练。

## 获取完整文件

直接克隆即可获取完整文件：

```bash
git clone https://github.com/Tiejiaxiaobao30/qwen-function-calling-lora.git
cd qwen-function-calling-lora
```

本地有效运行产物合计约 **441.6 MiB**；公开版本用 `data-redacted/` 替代原始 Arrow 数据，其余产物按原文件保存。单文件均小于 100 MiB。这个公开 Fork 不能上传新的 Git LFS 对象，因此采用普通 Git；首次克隆需要下载较多内容。后续多轮实验应采用独立模型存储或发布附件，避免重复累积大文件历史。

原始数据10条样本的工具定义默认值含疑似GitHub凭证，公开副本将其替换为 `[REDACTED_CREDENTIAL]`，保留数量与行顺序。原始Arrow文件只在本地和完整备份中保留，未推送。**本次指标来自原始数据，不是对脱敏副本重跑的结果。** 从GitHub恢复时，需在本地核对后将 `data-redacted/` 复制为 `data/` 才能用于默认脚本；这不会严格复现原始输入。脱敏行号见 [data-redaction.json](docs/data-redaction.json)。

中间 `cache-*.arrow` 仅在本地保留，不纳入提交。完整 Qwen 底座和 Hugging Face 缓存没有打包，需要另行下载。数据来源及署名见 [DATA_ATTRIBUTION.md](docs/DATA_ATTRIBUTION.md)，使用前请阅读并接受来源站点的适用条件。

## 源码阅读与运行入口

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

## 来源

初始仓库版本为 `33a10fc`，保留其 Git 历史和上游署名。原作者说明及发布链接见 [上游 README](docs/upstream-README.md)；本 Fork 中的中文阅读注释与本次运行产物分别保留，避免把上游实现或结果表述成本人原创。
