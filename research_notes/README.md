# 研究笔记与实验记录

本目录统一保存研究计划、论文思路、实验复盘、数据审计和讨论决定。

2026-09-09 将原 `plans` 目录更名为 `research_notes`；2026-09-11 已迁移旧文档和样例清单中的路径引用，并移除项目根目录的兼容软链接。研究文档与资源统一使用 `research_notes/` 路径。

## 最近一次讨论

**最新计划：[2026-09-11 Stage 1：4＋4 slots 分配与四任务训练](0911_stage1_slot_allocation_plan.md)。** 用户已确认：分类读 S₁～S₄，疾病识别读 S₁～S₈，分割和超分读 S₅～S₈；当前先写计划供审阅，暂不安排消融。新分组尚未训练。

**最新下游指标：[2026-09-11 上一轮 Stage 1 指标汇总](0911_stage1_downstream_results.md)。** 从已完成的原始 JSON 核对分类、报告生成、分割和 SR 指标，附测试规模、冻结 encoder 对照含义、逐类结果和 CSV。旧 diagnosis 指报告生成，Fig1 Disease recognition 尚未单独实现。

**数据总览：[2026-09-10 数据规模、变化与时间分布](0910_dataset_summary.md)。** 包含实际 6,000 对训练子集、可扩展候选池、Qwen 136,383/136,385 对完成状态、原始标签与语义判定差异，以及可切换队列的汇总页。

**此前纵向实验：[2026-09-10 早间新旧结果比较](0910_overnight_results_comparison.md)。** 夜间任务全部完成；新增 267 对相同影像/报告病例的配对复算和患者级区间。主模型未整体改善，报告模板化仍在。

**此前执行：[2026-09-09 夜间新链接数据训练](0909_overnight_linked_training.md)。** 用户随后授权启动训练，并允许使用所有空闲 GPU（始终排除 GPU 4）；该次运行见该文入口。下方早些时候的“不训练”决定为历史状态。

**入口：[2026-09-09 讨论摘要与后续决定](0909_session_summary.md)。**

- Session ID：`01a083e0-c7cb-7df3-9794-b2eb11160dd1`，已通过当前运行环境核实。
- 主题：8-slot 状态、报告 decoder、Copy Current、纵向配对噪声、MIMIC-CXR＋IV 连接、EHRXDiff 数据规模与后续诊断。
- 2026-09-09 当时决定：先审计与讨论，不启动新训练，不更换 decoder。当前讨论以本页最新计划为准。

| 文档 | 内容 |
|---|---|
| [讨论摘要](0909_session_summary.md) | 精简结论、证据边界、后续顺序和本次 Session ID |
| [实验复盘](0909_experiment_review_and_plan.md) | 完整结果表、训练实现、报告退化、slots/decoder 诊断设计 |
| [配对质量审计](0909_pair_quality_audit.md) | 旧 pilot 的住院边界、标签覆盖、采集变化及事后分层 |
| [连接规模与示例](0909_mimic_linked_cohort_capacity.md) | 全量 CXR＋IV 3.1 可连接规模及真实病例入口 |

以上文档存在详略关系，不是多套独立实验。标为计划、假设或待核验的条目，不代表已经执行或得到证实。

## 历史记录

| 日期 | 文档 |
|---|---|
| 2026-09-08 | [Table 1 实验计划](0908_table_1.md) |
| 2026-09-07 | [论文计划](0907_paper_plan.md)、[方法修订](0907_methods_revision.md) |
| 2026-09-06 | [方法记录](0906_methods.md) |
| 2026-08-29 | [研究思路](0829_ideas.md) |
| 2026-08-28 | [研究思路](0828_ideas.md) |
| 2026-08-14 | [数据附录图设计](0814_mimic_cxr_data_appendix_figure_prompt.md)、[训练验证图设计](0814_mimic_vla_jepa_training_validation_figure_prompt.md) |
| 2026-08-12 | [小规模实验计划](0812_mimic_vla_jepa_small_scale.md) |

历史文档保留当时的设想与口径；当前讨论以最新摘要及对应审计证据为准。配图和示例资源保留在 `assets/`。
