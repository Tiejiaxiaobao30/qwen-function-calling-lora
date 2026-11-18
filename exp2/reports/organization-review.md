# 整理、公开范围与核验记录

2026-11-18，助手直接操作用户Windows目录 `E:/qwen-function-calling-lora`；GitHub插件确认用户fork、main和权限，本地核对起点 `1bb87f8a7348299074c6eefb6f41416b105e180d`。使用安全分支 `organize/exp0-exp1-exp2-20261118`，PR只面向用户fork的main，不向上游作者投稿。

## 保留与移动

整理前1417个非.git文件全部记录路径、字节数及SHA256，移动后逐项核验相同，无永久删除。根六个用户注释文件保持整理前字节；train.py已有的set_seed与seed/data_seed修改保留并纳入版本化，不回退。原requirements保留。根目录仅六源码、requirements、README、.gitignore和exp1/exp2（另有.git）。

| 原位置 | 新位置 |
|---|---|
| `outputs/`（除两Exp1子目录） | `E:/qwen-function-calling-lora-artifacts/exp0/` |
| `outputs/exp1-control-flex`、`outputs/exp1-rewrite-v2` | `exp1/local/generation/`下原名目录 |
| `exp1-cloud-results/` | `exp1/local/cloud-results/` |
| `archives/` | `exp2/local/archives/` |
| `dpo/` | 仓库外`before-final-20261118/dpo/`保留原版；发布适配为`exp2/` |
| `augment_data.py`、`augmentation_utils.py` | `exp1/`；原版另在仓库外root-source备份 |
| `docs/` | 仓库外`before-final-20261118/docs/`；仅安全聚合报告复制到公开reports |
| 根`eval_results.json`、`error_analysis.json` | 仓库外`before-final-20261118/upstream-reports/`（上游，不能当个人结果） |
| 根两个Exp1 tar.gz | 仓库外`before-final-20261118/packages/` |
| `publish.py`、空tests、.vscode、缓存 | 仓库外`before-final-20261118/`下保留 |

仓库外备份完整路径 `E:/qwen-function-calling-lora-artifacts/before-final-20261118/`。`inventory-before.json`、`moves.json`和`preservation-verification.json`用于恢复与查重；root-source保留整理前根源码/README/ignore。重复的完整压缩包和解包结果均保留本地，不再混入源码或公开树。旧manifest中的绝对路径未改；历史运行不能因移动就假定可直接resume。

## 公开与许可边界

`.gitignore`采取默认忽略、逐文件允许；暂存时另外使用精确文件白名单，不使用`git add .`或`git add -A`。从索引移除旧产物/含样例报告等路径，保留本地归档。白名单只包括六根源码、requirements、README、ignore、Exp1/2源码及无样例报告/聚合JSON。

排除原始及脱敏数据、Arrow/缓存、教师响应、候选、偏好对、prediction/gold、sample_misses、模型/adapter/tokenizer产物、训练状态、checkpoint、压缩包、原始日志、环境和凭证文件。公开源码与报告扫描凭证模式及敏感payload键，扫描结果以文件名/计数记录，不输出匹配值。模式扫描不等于识别所有可能秘密，因此仍依赖逐文件范围审查和默认拒绝发布。

历史分支、Git提交和fork关系保留；本次不改写或清除旧远端历史，不把旧数据的历史公开状态解释为许可。根源码和Exp1/2依赖上游实现，保留仓库链接和贡献界限；没有新增许可证授权。用户要求不公开受限数据和衍生样本，因此本分支不再分发这些内容。

## 代码变化

1. Exp1两文件移入包；修改增强脚本导入为`exp1.augmentation_utils`，默认来源随OUTPUT_ROOT，新输出指向`exp1/local/new-*`。注释保留。
2. dpo源码移为exp2包，内部导入同步；默认DPO输出`exp2/local/new-run`，默认SFT数据为原config.DATA_DIR，明确与实际Exp2原始输入一致。
3. 云端batch脚本纳入源码；完整核验当前metadata后才调用原续做校验/尾行修复，避免旧脚本只检查pilot部分内容。
4. compare支持显式`--adapter`和`--models`，原val/test目录非空拒绝覆盖；默认同输入两模型评分不变。
5. README/实验报告重写为真实最终状态，保留Exp1历史聚合报告；Exp2 summary只含统计/哈希，无原始样例。原始云端源码仍封存在local归档。

## 验证范围

已完成原文件移动前后SHA256、根六源码字节、Python语法、包导入、原评分自检、轻量离线回归、文档本地链接、diff空白及公开文件扫描。Exp2真实落盘预测重新评分、gold与原始500val/test核对，候选5000×4及来源核对，偏好组隔离/内容指纹核对，final与best392张量核对。结果详见`results-summary.json`及`local-checks.json`。

本地解释器为 `E:/Anaconda/envs/nlp/python.exe`，本地没有TRL，未安装/升级依赖，不进行真实DPOTrainer初始化、GPU前后向、显存测试或恢复。Trainer部分仅静态语法与隔离替代依赖的逻辑检查，不能当GPU验收。既有datasets/multiprocess退出RLock警告单列记录，不用它否定已通过且exit0的离线核验，也不隐瞒环境问题。原始证据与完整审查记录留本地。
