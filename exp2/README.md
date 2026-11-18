# Exp2：从 Exp0 SFT 继续 DPO

最终结果与核验见 [实验报告](reports/experiment-report.md)。包从原 `dpo/` 迁移到 `exp2/`，保留归档中的历史代码；正式双 SFT 模型加载和根评分复用不变。

## 归档与环境

完整原始备份位于 `exp2/local/archives/exp2-final/`，包含云端源码和 `outputs/exp2_sft_dpo/`。其原始 tar.gz 与 sha256 也在 `local/archives/`。最终模型：`exp2/local/archives/exp2-final/outputs/exp2_sft_dpo/runs/main/adapter-final/`；最佳 checkpoint-50、末步87、sanity-b2、失败日志均保留。历史状态记录云端路径，迁移后不能假设 checkpoint 可以直接恢复；本轮没有恢复演练或 GPU 推理。

归档实际 Python3.12.3、torch2.6.0+cu126、transformers5.12.1、peft0.19.1、trl1.7.1、datasets5.0.1、bitsandbytes0.49.2、accelerate1.14.0。根 requirements 的 datasets4.0.0 与归档不同，不能当精确冻结环境。本地轻量检查不代表这些训练依赖全部安装或 GPU 可用。

## 新运行流程

从仓库根目录执行，设置三个变量；输出必须为新任务目录。本轮未运行以下 GPU 命令。数据来源为历史 Exp0 原始划分，区别于 Exp1 脱敏源。模型与原始数据均须从自己的合法备份准备。

```powershell
$env:OUTPUT_ROOT = 'E:/qwen-function-calling-lora-artifacts/exp0'
$env:SFT_DATA_DIR = 'E:/qwen-function-calling-lora-artifacts/exp0/data'
$env:EXP0_ADAPTER_DIR = 'E:/qwen-function-calling-lora-artifacts/exp0/adapter-final'
$env:DPO_OUTPUT_ROOT = 'exp2/local/new-run'
python -m exp2.generate_candidates --limit 100
python -m exp2.generate_candidates_batch --batch-size 16
python -m exp2.build_preferences
```

批量生成要求先有 pilot；逐题与 batch 的随机数轨迹不同。本次新增完整 metadata 比较后才允许续做/修剪尾行，修复旧归档 batch 仅哈希 pilot metadata 的缺口。来源/权重/参数改变会拒绝续做；不要在归档上修改路径来绕过校验。单写入进程，不并发生成与训练。

偏好构造保留原 score 语义；每源最多一对，chosen 为采样正确，rejected 为合法 JSON 但 score 错误的候选。没有 gold fallback。过滤来源错配、重复/val-test query 重叠和超长，再按规范化 query group 切分约10%偏好验证；这不是原500 val。人工审查与冻结后再训练，禁止重建数据接旧 checkpoint。

实际成功设置是 **2×8**，默认 CLI 仍为4×4；实际4×4曾OOM，必须显式传2，不能把有效batch相同当数值完全相同。

```powershell
python -m exp2.train_dpo --sanity --batch-size 2 --run-name sanity-b2 --stop-after-step-5
python -m exp2.train_dpo --sanity --batch-size 2 --run-name sanity-b2 --resume exp2/local/new-run/runs/sanity-b2/checkpoints/checkpoint-5
python -m exp2.train_dpo --batch-size 2
```

sanity 从5步恢复到10步；正式从 Exp0 开始3 epochs，lr5e-6、beta0.1、sigmoid、max_length2048、BF16/NF4、seed42、仅Policy LoRA可训练，Reference独立且冻结。最优按50对偏好验证集最低eval_loss选择。恢复用同run的完整 checkpoint 和原manifest，不能用 adapter-final；改 batch 等配置必须新任务。底座 revision 未锁定，不保证逐位复现。

冻结模型后先原500 val，再最终500 test：

```powershell
python -m exp2.compare_exp2 --split val
python -m exp2.compare_exp2 --split test
```

默认同输入评测 Exp0/Exp2；支持 `--models exp2` 单独复评与 `--adapter` 显式选定模型。任一 split 目标目录非空即拒绝覆盖，需换全新 `DPO_OUTPUT_ROOT`。例如复评已有最终模型时用新输出根并指定 `--adapter exp2/local/archives/exp2-final/outputs/exp2_sft_dpo/runs/main/adapter-final`。不会 merge Exp0 adapter 后再叠加 Exp2。输出每题 prediction/gold 只留 local，不进入公开 Git。

归档只含 Exp2 的500 val/test预测；所存 compare源码默认两模型，但没有可核对的当时单模型运行命令。因此报告仅认证归档预测与统计，不声称云端按现有入口完整跑了配对基线。
