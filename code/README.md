# 代码入口

项目总入口见 [根目录 README](../README.md)，实验清单见 [experiments](../experiments/README.md)。

| 目录 | 职责 | 状态 |
|---|---|---|
| [medworld_spatial](medworld_spatial/README.md) | Sparse slots 的多视图特征一致性、VLM 弱语义对齐与 attention 导出 | 独立空间任务试验；8 小时六组对照 |
| [medworld](medworld/README.md) | Table 1／2 融合：4＋4 状态、EMA、双向时间预测、状态独立文本解码 | 当前开发入口；正式融合结果待训练 |
| [medworld_multitask](medworld_multitask/README.md) | 融合前 Table 2：当前状态四任务 | 旧版快照 |
| [medworld_native_forecast](medworld_native_forecast/README.md) | 融合前 Table 1：原生 VLM＋未来状态 | 旧版快照 |
| [medworld_stage1](medworld_stage1/README.md) | 早期 Table 2：分类、疾病识别、分割、SR | 历史训练入口 |
| [medworld_table1](medworld_table1/README.md) | Table 1：未来状态／报告／finding 预测，DirectQwen 对照 | 保留纵向实验 |
| [medworld_baselines](medworld_baselines/README.md) | 六个原始 Qwen／MedGemma 的未来预测、分类、报告和派生 QA | 09-11 零样本评测已完成 |
| [medworld_dense_baselines](medworld_dense_baselines/README.md) | 冻结 VLM 密集任务对照；冻结多层视觉 slots 的分割／×4 SR 对照 | 09-12 队列 55/55 已完成；09-13 独立运行见下方入口 |
| `medworld_common` | 共享 Qwen 加载、LoRA、状态 encoder、报告 decoder、保存与随机种子工具 | 两个实验共用 |
| [mimic_cxr_iv_linked](mimic_cxr_iv_linked/README.md) | CXR＋IV 连接、筛选、审计及本地 VLM 作业 | 数据管线 |
| [MIMIC_example](MIMIC_example/README.md) | 时间配对、示例和数据约束 | 早期数据工具 |
| [mimic_vla_jepa](mimic_vla_jepa/README.md) | 8 月的特征提取和预测器实验 | 历史实现 |

`VLA-JEPA`、`VLA-JEPA-reference`、`Clin-JEPA`、`vjepa2`、`ChestXRayAnatomySegmentation` 是独立第三方仓库，保留其原路径和 Git。自有代码的整理不会把它们并入根仓库。

09-13 冻结多层视觉 slots 实验的提取、训练、队列和报告代码位于 `medworld_dense_baselines/frozen_slots_{extract,train,queue,report}.py`；[固定协议](../research_notes/0913_frozen_multidepth_slots.md)、[运行报告](medworld_dense_baselines/runs/frozen_slots_20260913/REPORT.md) 和 [队列状态](medworld_dense_baselines/runs/frozen_slots_20260913/status.json) 分别记录定义、结果和进度。该轮使用独立的 `runs/frozen_slots_20260913/`，09-12 的结果继续保存在 `runs/dense_20260912/`。

## 运行和开发

当前解释器是 `/home/data2/chk/workspace/2026/.venv/bin/python`。从项目根目录执行命令；融合版本使用 `PYTHONPATH=code python -m medworld.train`，配置与命令见 [MedWorld README](medworld/README.md)。以下为历史代码的兼容关系。

- Stage 1 三种训练共用 `training.py`；`training_variants.py` 管理模型选择与梯度约束。
- `train.py` 根据配置选择变体；`slot44_train.py`、`noslots_train.py` 是旧命令兼容入口。
- 公共 Qwen 实现在 `medworld_common/qwen.py`。Table 1 的 `model.py` 保留未来预测模型，Stage 1 的 `model.py` 仅转发公共组件。
- 运行时第三方 Python 包仍位于 `medworld_table1/vendor/`、`metric_vendor/`，模型权重仍位于原路径；这些不是实验源码依赖。
- 特征、slots、hidden states 和可重算的图像预处理结果应按 batch 即时计算，不落盘缓存；测试时直接从图像前向。已清理历史缓存；遇到旧代码找不到缓存时，重构计算路径，不恢复或重建缓存。本轮未完成这些旧代码的重构。完整约定见[根 README](../README.md#目录与存放规则)和 [AGENTS.md](../AGENTS.md)。
- 原始数据、固定标注、模型权重、checkpoint、日志和最终预测保留，不纳入根 Git；新增大体积产物存放于 `/home/data2/chk/data`，仓库通过 symbolic link 引用。

[开发与检查命令](../research_notes/DEVELOPMENT.md) · [实时状态](../scripts/project_status.py)
