# 代码入口

项目总入口见 [根目录 README](../README.md)，实验清单见 [experiments](../experiments/README.md)。

| 目录 | 职责 | 状态 |
|---|---|---|
| [medworld_stage1](medworld_stage1/README.md) | Table 2：分类、疾病识别、分割、SR | 当前开发主线 |
| [medworld_table1](medworld_table1/README.md) | Table 1：未来状态／报告／finding 预测，DirectQwen 对照 | 保留纵向实验 |
| [medworld_baselines](medworld_baselines/README.md) | 六个原始 Qwen／MedGemma 的未来预测、分类、报告和派生 QA | 09-11 零样本评测已完成 |
| [medworld_dense_baselines](medworld_dense_baselines/README.md) | 冻结 VLM 的分割、×4 SR、解剖定位 heads 与独立方向评测 | 09-12 队列 55/55 已完成 |
| `medworld_common` | 共享 Qwen 加载、LoRA、状态 encoder、报告 decoder、保存与随机种子工具 | 两个实验共用 |
| [mimic_cxr_iv_linked](mimic_cxr_iv_linked/README.md) | CXR＋IV 连接、筛选、审计及本地 VLM 作业 | 数据管线 |
| [MIMIC_example](MIMIC_example/README.md) | 时间配对、示例和数据约束 | 早期数据工具 |
| [mimic_vla_jepa](mimic_vla_jepa/README.md) | 8 月的特征提取和预测器实验 | 历史实现 |

`VLA-JEPA`、`VLA-JEPA-reference`、`Clin-JEPA`、`vjepa2`、`ChestXRayAnatomySegmentation` 是独立第三方仓库，保留其原路径和 Git。自有代码的整理不会把它们并入根仓库。

## 运行和开发

当前解释器是 `/home/data2/chk/workspace/2026/.venv/bin/python`。从项目根目录执行命令；新 Stage 1 训练统一使用 `code/medworld_stage1/launch.py`，具体配置和启动示例见 [Stage 1 README](medworld_stage1/README.md)。

- Stage 1 三种训练共用 `training.py`；`training_variants.py` 管理模型选择与梯度约束。
- `train.py` 根据配置选择变体；`slot44_train.py`、`noslots_train.py` 是旧命令兼容入口。
- 公共 Qwen 实现在 `medworld_common/qwen.py`。Table 1 的 `model.py` 保留未来预测模型，Stage 1 的 `model.py` 仅转发公共组件。
- 运行时第三方 Python 包仍位于 `medworld_table1/vendor/`、`metric_vendor/`，模型权重仍位于原路径；这些不是实验源码依赖。
- 训练缓存、checkpoint、日志和预测实际位于各实验的 `data/`、`weights/`、`runs/` 中，不纳入根 Git。

[开发与检查命令](../research_notes/DEVELOPMENT.md) · [实时状态](../scripts/project_status.py)
