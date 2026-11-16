# Exp1：教师改写对照结果（2026-11-16）

目的：检验保持训练条数一致时，教师改写能否优于重复采样。底座为 Qwen2.5-1.5B-Instruct，QLoRA r=16、alpha=32、dropout=0.05，3 epochs、有效 batch 16、学习率 2e-4、训练长度 1024、seed 42。复用上游训练和项目 BFCL-style AST 评分，并非官方 BFCL 榜单。

## 数据流与对照

本地 `outputs/data-redacted` 固定 train/val/test=5000/500/500 → 从 train 固定抽取 500 个候选 → 教师改写及规则过滤 → accepted=490 → Control 追加这些成功 seed 的 490 条原样本，Rewrite 追加对应的 490 条改写 → 同 seed 打乱 → 各自训练和评测。教师配置记录为 `qwen3.8-flash`、temperature=0.7、max_tokens=1024；配置名称不等于可独立验证的服务端模型版本。改写保持工具与答案，规则通过不代表人工语义或执行验证。

助手实际读取两组归档 train/val/test：均为 5490/500/500；`accepted_seed_indices.json` 与 Control `seed_indices.json` 整体相等，共 490 个成功 seed，候选清单 500 个。两组 val/test 逐行一致，且均与本地脱敏源一致。Exp0 原始 val 的零基索引 128、test 的零基索引 126 与 Exp1 不同；因此三组不能视为严格同一评测输入。这里只核验当前本地数据，不能凭此证明历史运行中没有更换输入。

哈希按有序行列表、`json.dumps(ensure_ascii=False, sort_keys=True, separators=(',', ':'))` 的 UTF-8 字节计算 SHA-256，不使用可变的 datasets 指纹或 Arrow 容器字节。全部划分的哈希见 [汇总 JSON](exp1-results-summary.json)。

| test | 行内容 SHA-256 |
|---|---|
| Exp0 原始 | `4beaee2f677f796343cdccc6c06abc45abb4f62340a3af699ba2bbcf24fd7697` |
| Exp1 Control / Rewrite / 脱敏源 | `5ebc6b1a01fcd60fb838f2effdaf0142fd0f65e37e2c8d7ffa548f783bbfd3c8` |

## 指标与错误分析

实际读取 Exp0 与两组 Exp1 的 `eval_results.json`、`error_analysis.json`（含 breakdown 和全部 sample_misses），每组测试 500 条、greedy。汇总仅复制聚合指标，不发布 sample_misses。根目录同名 JSON 属于上游报告，不参与比较。

| 实验 | JSON 有效率 | 函数名准确率 | 严格匹配 | 宽松匹配 | 失败 |
|---|---:|---:|---:|---:|---:|
| Base（报告记录） | 76.6% | 44.6% | 31.0% | — | — |
| Exp0 Tuned | 99.2% | 98.8% | 77.4% | 77.8% | 113 |
| Control Tuned | 99.4% | 98.8% | 77.2% | 78.0% | 114 |
| Rewrite Tuned | 99.6% | 99.2% | 76.8% | 77.2% | 116 |

| 错误类别 | Exp0 | Control | Rewrite |
|---|---:|---:|---:|
| wrong_arg_values | 61 | 62 | 58 |
| wrong_arg_keys | 46 | 45 | 51 |
| wrong_function_name | 2 | 3 | 1 |
| invalid_json | 4 | 3 | 2 |
| wrong_call_count | 0 | 0 | 1 |
| other | 0 | 1 | 3 |

三组类别计数分别加总为 113/114/116，与严格匹配的失败数一致。sample_misses 反映：三组都有把工具要求的参数键替换成近义键、遗漏默认值/其他参数，以及数字、枚举或类型不匹配的问题；Rewrite 样例还出现遗漏材料参数，Control 和 Rewrite 均有坐标值不符。保存的 query/pred 已截短，不能仅凭片段判断完整生成是否无效、是否因长度截断；类别沿用原报告，不重新标注。样例是少量展示，不能用来估计错误类别比例或认定同一道题发生修复。

Rewrite−Control 为 JSON +0.2 pp、函数名 +0.4 pp、严格匹配 −0.4 pp，即正确数 386→384。参数值错误减少 4，参数键错误增加 6；这是类别总量变化，不是逐题迁移分析。增加改写数据未提高本轮整体严格匹配，不能宣称方法有效。

## 训练与复现入口

Exp0 结束于 step 939，Exp1 两组结束于 step 1032、epoch 3；归档 trainer_state 分别记录 Control 最佳 checkpoint-500、Rewrite 最佳 checkpoint-700。此处未做最终 adapter 与最佳 checkpoint 的逐张量核验。Exp0 原始/Exp1 脱敏输入、训练条数、训练步数以及本地未提交训练脚本版本均构成比较边界，Control−Exp0 不能解释为纯数据量因果。

以下是读取已准备数据后的训练/评测入口，需要已接受来源访问条件、本地依赖和 GPU。**本次收尾没有执行这些命令。** 云端实际运行源码保存在本地 `exp1-cloud-results/*.py`，当前根目录 `train.py` 与增强脚本含用户未提交修改；本次仅提交文档和汇总，不能把本提交称作完整冻结的训练代码复现包。精确复现还需另行审查并版本化这些代码、环境和教师响应。

```powershell
# 从仓库根目录执行；新目录避免覆盖归档。
pip install -r requirements.txt
# Exp0 从来源准备数据：先选全新目录，再执行 prepare_data.py。
$env:OUTPUT_ROOT = "outputs-new-exp0"
python prepare_data.py
python train.py
python evaluate.py
python error_analysis.py

# Exp1 已生成的每组 data 放入独立 OUTPUT_ROOT/data，勿再运行 prepare_data.py。
# 下列路径仅作新运行示例，需先放好相应的 5490/500/500 数据。
$env:OUTPUT_ROOT = "outputs-new-control"
python train.py
python evaluate.py
python error_analysis.py
# Rewrite 同样操作，改用 outputs-new-rewrite 及其自己的 data。
```

## 证据、发布范围与局限

- Exp0 原始报告：本地 `outputs/{eval_results.json,error_analysis.json}`。
- Exp1 原始报告：本地 `exp1-cloud-results/outputs/{exp1-control-flex,exp1-rewrite-v2}/{eval_results.json,error_analysis.json}`；同组 `data/`、`checkpoints/`、`adapter-final/` 保存完整产物。
- 本次公开证据：[聚合指标、源报告 SHA-256 与数据核验](exp1-results-summary.json)。仅含统计与哈希，无原始/衍生样本、教师响应、权重、访问凭证。
- 遵循本项目 xLAM gated / research-only 使用边界，数据由使用者自行在接受来源条件后取得；脱敏不等于获得再分发许可。本次不新增公开数据或模型文件。历史跟踪文件不因 `.gitignore` 自动移除，历史发布范围不在此次提交中重写。
- 单次训练、500 条测试、差异仅 2 道，可能随机波动；未做多 seed、显著性检验、逐题配对检验、跨数据集或工具执行评测。Exp1 的 Base 字段沿用相同报告值，未验证是否独立重评。本次只做归档核对与文档更新，未调用教师 API、启动训练或 GPU 复评。
