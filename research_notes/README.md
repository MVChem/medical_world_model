# 研究笔记

[项目入口](../README.md) · [当前与历史实验](../experiments/README.md) · [实时训练状态](../scripts/project_status.py)

## 项目维护与复现

2026-09-13 将原文档目录的四个文件统一迁入本目录，开发说明和实验记录从这里进入。同日停用 `TODO.md`，仅保留历史快照；后续计划和决定继续写入带日期的研究笔记。

| 文档 | 用途 |
|---|---|
| [目录说明](PROJECT_STRUCTURE.md) | 论文、代码、笔记、实验记录与导出产物的实际位置及存放规则 |
| [历史待办（Deprecated）](TODO.md) | 已停用，不再更新或作为任务安排依据 |
| [本地开发与复现](DEVELOPMENT.md) | 环境、协议检查、源码快照与 Git 操作 |
| [09-11 环境快照](runtime_20260911.json) | 当时的 Python 与包版本，非通用安装锁文件 |
| [09-11 项目整理记录](PROJECT_CLEANUP_20260911.md) | 代码整理与验证，以及 09-13 文档迁移记录 |

## 当前决定与执行

**2026-09-15 融合前快照：** 当前两表的新版代码仍独立训练，先归档为旧版本；下一步统一状态编码、checkpoint 和两阶段训练。见[快照说明与后续方向](0915_pre_unification_snapshot.md)。

**2026-09-15 回填：9B 三组预测对照及 CheXagent 的正式结果已写入两表，新增 32 个指标值。** Table 1 现有 70 个、Table 2 有 42 个已填指标；完整多深度模型、Direction、正式 VQA／grounding 等缺项见[本次回填记录](0915_table_results_filled.md)。

**2026-09-13 核对：09-11 的 4＋4 slots／无 slots 匹配训练、六模型零样本评测，以及 09-12 的冻结 VLM 密集任务队列均已完成。** 密集任务于 09-12 12:14 完成 55/55 项；表格来源、实验时间和最终结果见下方溯源记录。09-13 新多层视觉 slots 是独立实验，其状态以对应运行报告为准。旧文档中的“尚未训练”“在跑”等描述属于当时快照。

| 文档 | 用途 |
|---|---|
| [09-15 slots 版本与文本解码](0915_slots_version_and_text_codec.md) | 核对两表旧结果与新版 4＋4 的归属；补充预测 latent 直接生成报告的接口、0.8B／9B 检查及正式质量验收边界 |
| [09-15 Table 1 原生图文预测重做](0915_table1_native_forecast_redesign.md) | 保留当前原图／历史的 Qwen decoder，新增预测 4＋4 slots；匹配 native SFT／slots／shuffled，新旧结果分开记录 |
| [09-15 共享模型短测、baseline 与缓存范围](0915_table2_joint_prototype_and_baselines.md) | 0.8B 四组与 9B 两组真实短测；区分原生 Qwen 微调、JEPA 结构对照、SwinIR 参照及缓存特征训练 |
| [09-15 两表补跑结果回填](0915_table_results_filled.md) | 9B full-token／slots／shuffled 与 CheXagent 的最终结果、完成时间、来源与剩余缺项 |
| [09-15 候选图复盘与后续制作思路](0915_figure_review_and_next_steps.md) | 汇总分享聊天与本轮看图讨论；区分状态证据与未来预测，明确现有试图不足、缺失材料及后续制作边界 |
| [09-14 候选图尝试与论文占位计划](0914_candidate_figure_plan.md) | 临床证据、状态检索、slot 条件扰动、空间输出、纵向预测及真实 attention 候选；全部尝试后筛选，先在论文保留简短图注占位 |
| [Table 1／2 结果来源与实验时间](0913_table1_table2_results_provenance.md) | 聊天中表格的具体路径、09-11／12 实验时间、18 行分割／×4 SR 最终分数，以及与新旧 slots 实验的区别 |
| [09-13 冻结多层视觉 slots 对照](0913_frozen_multidepth_slots.md) | 第 5–8 个 slots 固定，只训练分割／×4 SR decoder；六个视觉塔与匹配 image-only 基线 |
| [8 个多层 slots 方法小图 prompt](0912_multiscale_slots_figure_prompt.md) | 紧凑版：VLM 原生视觉与融合特征经轻量 adapter 对齐为八个 slots；沿用 fig1 字体 |
| [特征上采样与 VLM hidden states 源码核对](0912_feature_upsampling_vlm_code_review.md) | FeatUp／UPLiFT 的实现，以及 LISA、S1-Omni-Image、HealthGPT、PURE 如何连接下游解码器 |
| [09-12 slots 训练研究计划](0912_slots_training_research_plan.md) | 两次聊天总结；联系 FeatUp／UPLiFT，说明 8 slots 联合训练、冻结对照与特征取点 |
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

当前论文 Results 使用简短 setup、Table 1 future prediction、Table 2 downstream tasks。Table 1 已固定为临床状态、病情演变、报告内容、概率可靠性四组，每组两项指标；已完成且符合对应协议的结果已回填。

原始 VLM 评测已得到其中七项指标并写入论文主表；Direction 的新增 82 对子集另有输入和参考协议，不能直接当作原 297 对的第八列。

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
