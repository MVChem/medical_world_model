# 实验索引

新增：[09-15 Table 1 原生 Qwen3.5-9B 重做协议](../research_notes/0915_table1_native_forecast_redesign.md)：decoder 保留当前原图与历史，对比 native SFT／预测 4＋4 slots／shuffled。0.8B 与 9B 执行检查通过，新的正式 9B 队列已启动；[运行入口](../code/medworld_native_forecast/runs/qwen9b_native_20260915/REPORT.md)与[实时状态](../code/medworld_native_forecast/runs/qwen9b_native_20260915/queue_status.json)。旧 Table 1 分数仍属于 legacy prototype。

新增：[09-13 开源对比方法复现](../research_notes/0913_open_comparators.md)：DINOv2／CheXWorld任务头、BioViL-T／CheXWorld未来预测、SwinIR，以及表中其他公开ZS模型。[持久队列与结果](../code/medworld_open_baselines/runs/comparators_20260913/LIVE.md)自动更新；所有空闲GPU可调度，真实GPU小时单独记录。

本轮新增：[09-13 扩大数据与 VLM 联合优化过夜实验](../research_notes/0913_expanded_overnight_joint.md)，最多 5 张 GPU，09-14 08:00 北京时间停止。[合并 Table 1/2](../results/overnight_20260913/REPORT.md) 与各运行调度状态自动更新。

当前状态以原始 `status.json` 和运行报告为准。运行 `python scripts/project_status.py` 查看当前实验，加 `--all` 查看历史实验；[registry.json](registry.json) 只登记入口，不复制不断变化的指标。

**GitHub 结果入口：[09-12 实验小结](../research_notes/0912_recent_experiments_summary.md)**，包含已完成的 4＋4／无 slots 比较、六模型零样本结果和新队列快照。下表的 `runs/` 报告仅在本地存在，不随 Git 上传；协议和日期小结可在仓库中直接阅读。

| 实验 | 代码入口 | 结果与解释 |
|---|---|---|
| 09-13 下午，补齐四个模型的打乱 slots 对照 | [补跑协议与恢复命令](../research_notes/0913_remaining_shuffled_slots.md) | [合并结果表](../code/medworld_dense_baselines/runs/frozen_slots_shuffled_remaining_20260913/REPORT.md)、[新增 8 项状态](../code/medworld_dense_baselines/runs/frozen_slots_shuffled_remaining_20260913/status.json) |
| 09-13 冻结多层视觉 slots，分割／×4 SR 对照 | [固定协议](../research_notes/0913_frozen_multidepth_slots.md) | [自动更新主表](../code/medworld_dense_baselines/runs/frozen_slots_20260913/REPORT.md)、[队列状态](../code/medworld_dense_baselines/runs/frozen_slots_20260913/status.json) |
| 09-12 冻结 VLM 密集基线，55/55 已完成 | [密集任务适配](../code/medworld_dense_baselines/README.md) | [结果与实验时间](../research_notes/0913_table1_table2_results_provenance.md)、[本地完整矩阵](../code/medworld_dense_baselines/runs/dense_20260912/preview/full_tables.md)、[固定协议与限制](../research_notes/0912_frozen_vlm_dense_baselines.md) |
| 09-11 原始模型基线，已完成 | [零样本评测](../code/medworld_baselines/README.md) | [本地 Table 1 / Table 2 报告](../code/medworld_baselines/runs/raw_models_20260911/REPORT.md)、[固定协议与限制](../research_notes/0911_raw_model_baseline_sweep.md) |
| 09-11 4＋4 slots，四任务已完成 | [Stage 1](../code/medworld_stage1/README.md) | [本地运行报告](../code/medworld_stage1/runs/slot44_20260911/REPORT.md)、[数据与设置](../research_notes/0911_stage1_slot44_run.md) |
| 09-11 无 slots，四任务已完成 | 同一 Stage 1 训练循环，`slots=0` | [本地运行报告及比较](../code/medworld_stage1/runs/qwen08_noslots_20260911/REPORT.md)、[对照定义](../research_notes/0911_qwen08_noslots_baseline.md) |
| 09-10 旧 Stage 1 | `legacy` 变体；8 slots，报告生成 | [历史指标](../research_notes/0911_stage1_downstream_results.md) |
| 09-09 纵向配对训练 | [Table 1](../code/medworld_table1/README.md) | [结果复盘](../research_notes/0910_overnight_results_comparison.md) |
| 09-08 未来预测 pilot | Table 1 初始设置 | [历史分析](../results/table1_pilot_20260909.md) |

当前 Ours 与无 slots 对照使用相同患者划分、每步样本、公共参数初始值及优化配置。早期比较固定在 3,176 步，最终比较已采用两组 24,000 步的 final checkpoint。验证集和测试集结果分开记录。

旧 `null`／`shuffle` 是推理时的状态清零／跨患者互换；旧 frozen encoder 仍保留 slots；`DirectQwen` 使用原生 Qwen 视觉前端做未来报告预测。它们与当前四任务无 slots baseline 分别登记，不能相互替代。

## 数据审计和展示

- [数据总览](../research_notes/0910_dataset_summary.md)，导出页位于 `results/data_audit_20260910/`。
- [CXR 与 IV 配对规模](../research_notes/0909_mimic_linked_cohort_capacity.md)，展示产物位于 `results/mimic_linked_examples_20260909/` 和 `results/mimic_cxr_iv_review_20260910/`。
- [语义评价讨论](../research_notes/0910_richer_semantic_evaluation.md)，历史小样本输出位于 `results/semantic_schema_pilot_20260910*/`。

`runs/*/source/`、历史 checkpoint 和原始预测属于实验档案。新配置用新目录，不覆盖旧实验。新增正式实验时更新登记表；短检查放入明确命名的 smoke/check 目录，不当作论文结果。
