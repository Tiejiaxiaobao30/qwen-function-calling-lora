# Exp1：教师改写与配对 Control

保留 `augment_data.py` 与 `augmentation_utils.py` 中用户已有注释，只调整包导入和新任务默认输出。SFT 复用根目录 `train.py`，不再复制一套训练代码。原始云端运行源码保留 `local/cloud-results/*.py`，当前根源码与其已核对一致；该结论不等于逐张量核验 Exp1 模型。

固定脱敏 train/val/test=5000/500/500；尝试 500 个 seed，每个一次，不合格跳过，实际接受 490。Rewrite 加入 490 条改写；Control 加入相同成功 seed 的 490 条原样本，均为 5490/500/500。教师记录 `qwen3.8-flash`、temperature 0.7、max_tokens 1024。基本结构/工具名/参数名和 rewrite 答案保持校验，不含完整类型、语义或真实执行验证。

## 本机归档

| 旧路径 | 新路径 | 用途 |
|---|---|---|
| `exp1-cloud-results/` | `exp1/local/cloud-results/` | 两组云端数据、最终模型、checkpoint 与原始报告 |
| `outputs/exp1-rewrite-v2/` | `exp1/local/generation/exp1-rewrite-v2/` | 本地生成原记录，完整保留 |
| `outputs/exp1-control-flex/` | `exp1/local/generation/exp1-control-flex/` | 配对原样本数据 |
| `outputs/data-redacted/` | `E:/qwen-function-calling-lora-artifacts/exp0/data-redacted/` | 固定脱敏来源 |

归档 manifest 保留历史绝对路径和指纹，不为适配移动而伪造历史身份；已完成任务不在新位置 `--resume`。新生成用全新路径。两组正式模型分别位于 `local/cloud-results/outputs/exp1-control-flex/adapter-final/` 和 `local/cloud-results/outputs/exp1-rewrite-v2/adapter-final/`。

## 新任务命令（本轮未执行教师调用或训练）

先在已授权环境配置 `TEACHER_API_KEY`、`TEACHER_BASE_URL`，不要把值写入源码或 Git。以下从仓库根目录执行，使用本机已有 tokenizer，不会由此自动获得数据访问权。

```powershell
$env:OUTPUT_ROOT = 'E:/qwen-function-calling-lora-artifacts/exp0'
python -m exp1.augment_data --source E:/qwen-function-calling-lora-artifacts/exp0/data-redacted --output exp1/local/new-rewrite/data --control-output exp1/local/new-control/data --tokenizer E:/qwen-function-calling-lora-artifacts/exp0/adapter-final --mode rewrite --limit 500 --seed 42 --model qwen3.8-flash --max-length 1024 --temperature 0.7 --max-tokens 1024
```

默认 limit 是 5 的试跑，正式尝试范围必须显式传 500；实际接受数决定两组规模。仅在同一个新任务、同来源路径和配置下使用 `--resume`。不要对已生成的 Exp1 数据再次运行 `prepare_data.py`。

训练从底座独立开始，不加载 Exp0 adapter。下列仅用于新生成的数据：

```powershell
$env:OUTPUT_ROOT = 'exp1/local/new-control'
python train.py
python evaluate.py
python error_analysis.py
$env:OUTPUT_ROOT = 'exp1/local/new-rewrite'
python train.py
python evaluate.py
python error_analysis.py
```

两组各 3 epochs、有效 batch16、lr2e-4、最大训练长度1024。历史结束 step1032，最佳 Control500 / Rewrite700。当前根 `train.py` 保留此前 seed 修复，本轮未新增行为。评测会写当前 OUTPUT_ROOT 下报告；勿将新任务路径指向归档。

严格匹配 Control77.2%、Rewrite76.8%，未验证整体收益。与 Exp0 的测试脱敏差异及完整统计见 [历史结果](reports/exp1-results.md)、[聚合 JSON](reports/exp1-results-summary.json)。没有保存完整 Exp1 Dev 生成质量结果，留空，不拿训练 eval_loss 代替。
