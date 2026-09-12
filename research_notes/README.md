# 研究笔记

[项目入口](../README.md) · [当前与历史实验](../experiments/README.md) · [实时训练状态](../scripts/project_status.py)

## 当前决定与执行

**2026-09-12：4＋4 slots／无 slots 的匹配训练与六模型零样本评测均已完成，冻结 VLM 密集任务队列已启动。** 已完成分数和负结果见下方实验小结；在跑状态以本地报告为准。旧文档中的“尚未训练”“先不做消融”等描述属于当时状态。

| 文档 | 用途 |
|---|---|
| [09-12 实验小结与结果表](0912_recent_experiments_summary.md) | 已完成的匹配比较、六模型零样本分数、负结果与在跑快照；GitHub 可读 |
| [六模型零样本评测](0911_raw_model_baseline_sweep.md) | 未来预测／当前分类、报告、派生 QA 的固定协议 |
| [冻结 VLM 密集任务](0912_frozen_vlm_dense_baselines.md) | 六主干、三分支、20 epochs、分割／×4 SR／解剖定位与独立方向队列 |
| [4＋4 slots 计划](0911_stage1_slot_allocation_plan.md) | 当前任务分配和模型设计 |
| [4＋4 执行记录](0911_stage1_slot44_run.md) | 实际数据接入、派生 QA、训练配置与指标解释 |
| [无 slots 对照](0911_qwen08_noslots_baseline.md) | 与旧 DirectQwen／shuffle 的区别、相同步数协议 |
| [Table 2 方法讨论](0911_table2_top_conference_review.md) | 精简后的候选方法和论文依据 |
| [Table 1 四组八指标](0911_future_state_evaluation_redesign.md) | 已确认并写入论文的指标方案、文献依据和实现缺口 |
| [Table 1 旧指标与来源](0910_table1_metrics_and_data_sources.md) | 早期实验评分解释与 MIMIC 数据来源 |
| [数据总览](0910_dataset_summary.md) | 数据规模、变化与时间分布 |

当前论文 Results 使用简短 setup、Table 1 future prediction、Table 2 downstream tasks。Table 1 已固定为临床状态、病情演变、报告内容、概率可靠性四组，每组两项指标；结果仍待正式实验填写。

原始 VLM 评测已得到其中七项指标，论文主表未回填；Direction 的新增 82 对子集另有输入和参考协议，不能直接当作原 297 对的第八列。

## 历史实验和数据审计

| 文档 | 内容 |
|---|---|
| [Table 1／2 扩展候选](0911_table_expansion_proposal.md) | 保留方案讨论过程；候选方法与计划任务不代表已完成实验 |
| [旧 Stage 1 指标](0911_stage1_downstream_results.md) | 09-10 运行；diagnosis 当时是报告生成 |
| [旧四任务计划](0910_stage1_four_task_plan.md) | 4＋4 分组之前的设计 |
| [纵向实验结果复盘](0910_overnight_results_comparison.md) | 新旧配对实验与误差分析 |
| [夜间训练记录](0909_overnight_linked_training.md) | 09-09 CXR＋IV 纵向实验 |
| [语义评价讨论](0910_richer_semantic_evaluation.md) | 额外语义评价的试验与限制 |
| [09-09 讨论摘要](0909_session_summary.md) | 当次讨论与决定 |
| [实验复盘和后续计划](0909_experiment_review_and_plan.md) | 报告退化、状态与 decoder 诊断 |
| [配对质量审计](0909_pair_quality_audit.md) | 住院边界、标签和采集变化 |
| [连接规模与样例](0909_mimic_linked_cohort_capacity.md) | CXR＋IV 连接容量与样例入口 |

## 早期设计与配图记录

- 09-08：[Table 1 初始计划](0908_table_1.md)。
- 09-07：[论文计划](0907_paper_plan.md)、[方法修订](0907_methods_revision.md)。
- 09-06：[方法记录](0906_methods.md)。
- 08-28／29：[研究思路](0828_ideas.md)、[后续想法](0829_ideas.md)。
- 08-14：[数据附录图](0814_mimic_cxr_data_appendix_figure_prompt.md)、[训练验证图](0814_mimic_vla_jepa_training_validation_figure_prompt.md)。
- 08-12：[小规模 VLA-JEPA 计划](0812_mimic_vla_jepa_small_scale.md)。
- [历史任务原文](archive/task_history.md)：原根目录 TODO 的日期记录。

日期文档保留原位置，避免破坏引用；`assets/` 保存对应资源。新增决定写入带日期的文档，并更新此索引；实时步数不复制到首页。历史文档和导出结果不代表当前协议。
