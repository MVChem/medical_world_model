# 09-14：将已有结果写入论文 Table 1 / Table 2

按本次要求，两表原来的 Qwen2.5-VL-7B 零样本行均替换为 **Qwen3.5-9B**，并使用该模型的真实评估结果。不是仅替换行名。当前 Table 1 有 11 行、63 个已填写指标；Table 2 有 13 行、38 个已填写指标。

- [Table 1 源码](../27cvpr/tables/table1_future.tex)
- [Table 2 源码](../27cvpr/tables/table2_downstream.tex)
- [两页结果预览](../27cvpr/plans/table1_table2_plan.pdf)
- 数值来源见表格各行的 TeX 注释；聚合 JSON 已在目录整理时移除。
- [从原始结果重建表格的脚本](../27cvpr/plans/populate_results.py)

论文正文和独立预览共享两份 TeX 源码。每行的原始指标文件写在 TeX 注释中；JSON 记录全精度值、列名、实验协议及原文件哈希，不包含患者记录或生成报告。

## Qwen3.5-9B 的对应结果

官方模型目录为 `Qwen/Qwen3.5-9B`，revision 为 `c202236235762e1c871ad0ccb60c8ee5ba337b9a`。

| 评估 | 已写入的值 |
| --- | --- |
| Table 1，同一 297 对／94 位患者 | AP .8555；AUROC .8177；Transition .1360；RadGraph .2044；GREEN .3170；Brier .2022；ECE .2790 |
| Table 2，353 张当前分类 | AUROC .7154；AP .8546 |
| Table 2，507 份当前报告 | RadGraph .2398；CheXbert .3797 |
| Table 2，冻结视觉 slots + 任务头 | Dice pseudo .9231；Dice human .7538；PSNR 31.8731；SSIM .9082 |
| Table 2，打乱冻结视觉 slots | Dice pseudo .9118；Dice human .7566；PSNR 31.8621；SSIM .9079 |

零样本来源为 `code/medworld_baselines/runs/raw_models_20260911/qwen9b/test/`。两行密集任务来源分别为 `frozen_slots_20260913/qwen9b/` 与 `frozen_slots_shuffled_remaining_20260913/qwen9b/`。密集任务只加载冻结的约 456M 视觉塔、提取四个固定深度池化 tokens 并训练任务头，故在表中与 9B 原生零样本行分开。

## 回填口径

Table 1 填入三个公开零样本模型、BioViL-T/CheXWorld 编码器适配、Copy Current、统计 prior 和 0.8B 三种状态条件。两个编码器适配的 GREEN 已完整评完 297/297，分别为 .1537 和 .1263。现有八查询 forecast pilot 取最后语言层，不能称为已实现论文中多层 fusion/vision 4+4 架构；full-token 条件也保留共同预测器，行名据此修正。

Table 2 的所有已填密集任务统一用 **4096 train / 249 validation / 447 test、20 epochs、batch 8、seed 20260913**，人工肺区另用 Montgomery 138 张测试图。这样 DINOv2、CheXWorld、SwinIR、image-only 与 Qwen-9B 视觉条件处于同一训练数据规模。已完成的 18,708 张扩大数据结果保留在运行报告和审计记录，不与 4096 版本混在主表。DINOv2/CheXWorld 的分类实际为独立 13681/160/353 划分；原字段 `macro_auprc` 已确认实现为非插值 AP。

实际 GPU 小时并不相等。SwinIR 4096 训练加评估约 8.313 GPU-h，image-only SR 约 .161 GPU-h；表注和正文已披露。此处不宣称等计算量优势。

以下位置保留 `TBD`：297 对 cohort 的 Direction、prior 的 Transition、CheXagent、MAIRA-2、正式 VQA、MS-CXR grounding、计划中的完整 EPA 六任务及其 matched no-slots 行。旧 82 对方向、派生疾病列表 QA、解剖区域定位、报告辅助分类、×2 SR，以及另一个两轮 VLM 适配 pilot，都不替代这些指标。

## 重建

在项目根目录运行：

```bash
/home/data2/chk/workspace/2026/.venv/bin/python 27cvpr/plans/populate_results.py
make -C 27cvpr
make -C 27cvpr plan-preview
```

填表脚本读取本次固定的已完成结果，校验 cohort 数量、dense epochs 与完整 GREEN 状态。更换任务、训练规模或增补新方法时，应先更新映射和协议，不能把运行中间指标当作最终结果。
