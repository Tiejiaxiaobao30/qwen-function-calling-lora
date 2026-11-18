# Exp0 → Exp1 → Exp2 最终实验报告

核验日期：2026-11-18。依据本机原始报告、数据划分、Exp1云端结果及Exp2完整归档；本轮执行者为助手，只做本地离线核验，未调用教师、未训练、未GPU复评。原始证据与受限内容仅保留本地，公开统计见 [Exp1 JSON](../../exp1/reports/exp1-results-summary.json) 与 [Exp2 JSON](results-summary.json)。

## 1. 问题与对照

任务是根据用户 query 和工具 schema 生成 JSON 调用列表。底座 Qwen2.5-1.5B-Instruct，数据为 Salesforce/APIGen xLAM function-calling，评价为本项目 `scoring.score` 的 JSON解析、函数名集合、调用结构集合匹配，不是官方 BFCL、真实工具执行或人工语义评分。

Exp0：5000/500/500原始划分，QLoRA r16/alpha32/dropout0.05，七类 projection 目标，训练最大长度1024、3epochs、lr2e-4、batch8×累积2。历史结束939步、最佳600；不能把final自动解释为最后一步。根六文件保留用户注释及整理前字节；train.py中seed/data_seed修复是此前已有变更，不倒推历史训练一定包含它。

Exp1：固定脱敏源5000/500/500，从train尝试500个seed，教师改写实际接受490；对照补入这些seed的原样本490，实验组补入对应改写490，两组均5490/500/500。3epochs、有效batch16、lr2e-4、长度1024；结束1032步，最佳Control500、Rewrite700。验证和Test两组逐行一致；教师及规则通过不保证语义正确。此次没有逐张量核对Exp1 final与best。

Exp2从 **Exp0** adapter 继续，不从Exp1继续。它是独立的SFT→DPO实验，表格顺序不代表串行训练链。

## 2. Exp2 数据构造

候选元数据、5000条记录与本地Exp0原始train逐条核对通过：source_id覆盖0–4999、无重复，每源4候选，共20000个候选槽。先逐题pilot，后batch16；temperature0.7、top_p0.9、max_new_tokens200，batch seed为42+窗口起点。batch与逐题、部分窗口恢复不保证随机轨迹一致，底座revision未锁定。

过滤无效JSON/结构，使用既有score选正确/错误；每源取首个正确与首个错误形成至多一对，无gold fallback。原始val/test规范化query作为排除集合；按query group划分偏好train/val，避免同query跨划分。

| audit字段 | 数量 | 含义 |
|---|---:|---|
| sources | 5000 | 原train候选记录 |
| duplicate_or_overlap | 12 | 重复prompt或与原val/test规范化query重叠 |
| unique_candidates | 6255 | 排除上述源后，各源内部去重累计 |
| correct_candidate | 4658 | 原score判正确 |
| wrong_candidate | 1532 | 合法结构且原score判错 |
| invalid_candidate | 65 | JSON或调用结构不合格 |
| canonical_score_disagreement | 2 | 列表规范化与score的集合语义有分歧；仍按score选择 |
| no_sampled_pair | 4487 | 无同源正负组合 |
| 最终偏好对 | 501 | 500个有效query group；一个group可有多个不同prompt |
| 偏好训练 / 验证 | 451 / 50 | 450 / 50个query group |

6255=4658+1532+65；5000−12−4487=501。原score使用set，折叠重复调用，不能把canonical分歧说成已修正；当前实现保留历史标签。此次验证偏好train/val query无交集、与原500val/test query无交集，核验chosen方向及内容指纹。未做逐对人工语义/工具执行验证，不从manual_review文件存在推断人工验收已完成。

## 3. DPO训练与模型

Policy、Reference均从同一Exp0 adapter初始化且独立；Policy只训练LoRA，Reference冻结。4×4正式尝试日志出现CUDA OOM；实际成功配置为microbatch2×累积8，名义有效16对，最大长度2048，3epochs、seed42、lr5e-6、beta0.1、sigmoid loss、BF16/NF4、gradient checkpointing。

sanity-b2有第5步与恢复后第10步checkpoint和日志；正式main完成87/87步、epoch3，训练日志runtime534.5秒、train_loss约0.6844。该runtime是训练日志字段，不是全流程耗时或本机测量。每25步及末步验证保存，偏好验证最低eval_loss=0.6850159168243408，最佳checkpoint-50；final应与best关联，不能标成checkpoint-87。

此次用safetensors在CPU加载：final与checkpoint-50同392个键，best BF16转final FP32后逐元素一致。final SHA256 `0dc2b3c9a32c4a43f970f95d45409494454e2984e5e46c4be3f1502240b6967e`，best SHA256 `b14419c10a355b75fb2d3d689cc04de9f7eab19276840259660f1623c42b1d2e`。文件哈希不同来自序列化/精度差异，不代表不同选模。元数据指向该final，但本轮没有重新GPU推理以证明生成时实际加载行为。

归档实际版本：Python3.12.3，torch2.6.0+cu126，transformers5.12.1，peft0.19.1，trl1.7.1，datasets5.0.1，bitsandbytes0.49.2，accelerate1.14.0。没有据包清单推断GPU型号；日志显示23.76GiB设备及OOM，本轮未连接云端确认硬件。根requirements datasets4.0.0与实际不同。

## 4. Dev / Test与对照

原500 val用于生成质量开发评测；50对偏好val用于训练eval_loss选模，两者不同。归档Exp2 predictions逐条重新调用根score，确认id0–499、数量、每条gold与本地原始数据、metadata数据指纹和汇总完全一致，无重新生成。

| 组别 | Dev JSON/函数名/严格匹配 | Test JSON | Test函数名 | Test严格匹配 |
|---|---|---:|---:|---:|
| Base（Exp0记录） | 未找到可用报告 | 76.6% | 44.6% | 31.0% |
| Exp0 SFT | 未找到可用报告 | 99.2% | 98.8% | 77.4%（387/500） |
| Exp1 Control | 未找到可用报告 | 99.4% | 98.8% | 77.2%（386/500） |
| Exp1 Rewrite | 未找到可用报告 | 99.6% | 99.2% | 76.8%（384/500） |
| Exp2 DPO | 99.8% / 99.4% / 79.2%（396/500） | 99.2% | 98.8% | 78.0%（390/500） |

Exp1 Base字段沿用相同记录，不作为独立复评证明。Exp2归档`exp0_reference`报告的test_n=499、tuned78.8%，对应上游；与本地本人Exp0 500条77.4%不同，排除为配对基线。归档没有Exp0逐题新预测，也没有Dev同时对照结果。故仅能说Exp2 Test观察到78.0%，相对历史Exp0差+0.6pp，不能归因于DPO净收益。Dev高于Test不说明过拟合或显著差异。

Exp1 Rewrite−Control为−0.4pp（2题），没有整体提升证据。尚无多seed、显著性检验、配对迁移检验、跨数据集或真实执行结果。不能根据Test事后改超参；本轮只核验归档，无法独立证明历史Test访问次数或确切评测命令顺序。

## 5. 错误分析

原`diagnose`启发式，分类有顺序和同名多调用局限；以下保留既有含义。Exp2直接复用落盘prediction/gold重算分类，Test与原始error_analysis.json聚合一致。Dev分类为此次离线新增统计。

| 类别 | Exp0 Test | Control Test | Rewrite Test | Exp2 Dev | Exp2 Test |
|---|---:|---:|---:|---:|---:|
| 参数值 | 61 | 62 | 58 | 57 | 63 |
| 参数键 | 46 | 45 | 51 | 44 | 41 |
| 函数名 | 2 | 3 | 1 | 2 | 2 |
| 无效JSON | 4 | 3 | 2 | 1 | 4 |
| 调用数 | 0 | 0 | 1 | 0 | 0 |
| other | 0 | 1 | 3 | 0 | 0 |
| 总失败 | 113 | 114 | 116 | 104 | 110 |

Exp2 Test参数错误104/110，约94.5%；相对历史Exp0参数键少5、参数值多2，是类别总量差异，不是“修复5题”或逐题迁移结论。公开报告不含具体query、schema、prediction或gold。没有Exp2宽松匹配报告，不补造宽松成绩。

## 6. 数据与证据边界

Exp0原始和Exp1脱敏的val零基128、test零基126不同，脱敏替换schema默认值；三组不能宣称完全同输入。本轮Exp2原始train/val/test与本机对应，候选元数据、偏好run.meta内容指纹、评测gold均通过核验。

Exp2的`key`为`json.dumps(ensure_ascii=False,sort_keys=True)`后SHA256（保留默认分隔空格）；Exp1历史报告用紧凑分隔符，两套哈希不能直接比较。Exp2原始Test key为`08e40f635c338c48f1651c5283ac941938a97958b68dc1993635143d5b639509`，原始val为`dd3b2b2cbd83dfebaf2057ab8bafa4a109bf96b72b3adf136523e456d44a7dd2`。完整指纹见汇总JSON。

原始归档源码与本地整理前dpo共同文件字节一致；云端额外batch脚本纳入exp2。归档compare源码默认会生成两模型输出，而现有归档只有Exp2预测，缺失实际单模型命令来源。因此不能把本分支视作逐字节、逐命令、随机轨迹完全冻结的复现包。新batch完整metadata校验、评测防覆盖和包路径适配是本轮维护变更，未倒写历史源码或候选metadata。

原始数据及衍生样例、教师响应、候选、偏好对、预测、checkpoint/模型和全量环境日志留本地，不公开。历史Git内容不重写，本分支仅从索引移除已有产物。保留来源署名，不从历史公开状态推导再分发授权。
