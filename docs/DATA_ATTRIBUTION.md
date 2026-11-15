# 数据来源与署名

数据来自 Salesforce AI Research 的 **APIGen / xLAM function-calling 60k**：

- 数据集：https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
- 论文：https://arxiv.org/abs/2406.18518
- 项目：https://apigen-pipeline.github.io/
- 数据集卡在 2026-11-15 核对时标注 `CC-BY-4.0`，同时有访问条件确认入口及研究用途说明。
- 许可证：https://creativecommons.org/licenses/by/4.0/

`outputs/data/` 是处理后子集，并非原创数据集。上游脚本清理样本、重排格式、长度筛选后划分 train 5000 / val 500 / test 500，seed 42；详见 `cloud-run-source/prepare_data.py` 与 `outputs/data/meta.json`。错误分析报告含少量源数据衍生样例。这不是官方 BFCL 测试集。

公开的Arrow文件实际位于 `outputs/data-redacted/`：保留数量与顺序，替换10条样本工具schema默认值中的疑似GitHub凭证。原始Arrow仅保留本地。脱敏行号与替换标记见 `data-redaction.json`，这里不保存原值。已发布的评测指标使用原始输入，未对脱敏副本重跑。

保留原始作者署名、数据集与许可证链接，并说明上述处理。使用者须遵守来源站点当前适用条件；本仓库不另行授予原始数据的权利。初始 `.gitignore` 将 gated 数据概括为不可再分发，与当前公开数据卡的 CC-BY-4.0 标注不同；整理时按当前卡片记录来源，不沿用这一未经核实的概括。

```bibtex
@article{liu2024apigen,
  title={APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets},
  author={Liu, Zuxin and Hoang, Thai and Zhang, Jianguo and Zhu, Ming and Lan, Tian and Kokane, Shirley and Tan, Juntao and Yao, Weiran and Liu, Zhiwei and Feng, Yihao and others},
  journal={arXiv preprint arXiv:2406.18518},
  year={2024}
}
```
