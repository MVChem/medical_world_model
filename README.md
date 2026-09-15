# MedWorld-JEPA

## 融合前旧版本快照：2026-09-15

本次提交保存 Table 1 与 Table 2 尚未融合的实现，标签为
`pre-table1-table2-unification-20260915`。下一版本将统一状态编码器、
checkpoint 和两阶段训练，以同一个模型完成未来预测与当前状态下游评测。
本快照不代表融合版本或完整六任务实验已经完成。
当前代码入口、已知差异和归档范围见
[融合前快照说明](research_notes/0915_pre_unification_snapshot.md)。

医疗多模态状态表示与未来预测研究。主模型使用 Qwen3.5-0.8B，论文以两个问题组织实验：**Table 1：未来预测；Table 2：当前状态的下游任务**。已有四任务训练，并在补充原始 VLM 与冻结主干的下游适配基线。

## 从这里开始

| 内容 | 入口 |
|---|---|
| CVPR 2027 论文与编译方法 | [27cvpr/README.md](27cvpr/README.md)，主文件 [main.tex](27cvpr/main.tex)；暂用 CVPR 2026 官方模板 |
| 各目录放什么、如何对应 | [目录说明](research_notes/PROJECT_STRUCTURE.md) |
| 当前实验与历史对照 | [实验索引](experiments/README.md) |
| 训练、评估和代码模块 | [code/README.md](code/README.md) |
| 最新计划、讨论决定、数据说明 | [research_notes/README.md](research_notes/README.md) |
| 运行环境、检查和本地 Git | [开发说明](research_notes/DEVELOPMENT.md) |
| 历史待办（Deprecated，已停用） | [TODO.md](research_notes/TODO.md)，仅保留快照 |

## 当前实验

**[Table 1／2 结果来源与实验时间](research_notes/0913_table1_table2_results_provenance.md)** 记录 09-11／12 实验的完成时间、原始表格路径和分割／×4 SR 最终分数；此前的结果与失败尝试见 [09-12 实验小结](research_notes/0912_recent_experiments_summary.md)。

- **4＋4 slots／无 slots：已完成匹配训练与最终测试。** slots 的分类指标较高，疾病列表 F1 和分割 Dice 较低，SR 差距很小。[训练与数据](research_notes/0911_stage1_slot44_run.md) · [对照定义](research_notes/0911_qwen08_noslots_baseline.md)。
- **六个原始 Qwen／MedGemma：零样本评测已完成。** 包含未来预测、当前分类／报告和派生 QA；官方 VQA 与原测试集 Direction 仍缺合格数据。[评测协议](research_notes/0911_raw_model_baseline_sweep.md)。
- **冻结 VLM＋下游 heads：09-12 12:14 已完成 55/55 项。** 比较原图、V-JEPA、V-JEPA＋adapter 分支，完成分割、×4 SR、解剖区域定位及独立方向评测。[运行设置与边界](research_notes/0912_frozen_vlm_dense_baselines.md)。

README 不固定记录训练步数。查看实时状态：

```bash
python scripts/project_status.py
```

09-11 四任务中的疾病识别使用本地 Chest ImaGenome 派生问答，分割衡量 CXAS 伪标签一致性，SR 使用合成 ×2 退化；09-12 密集基线另测 ×4 SR 和人工两肺分割。历史报告生成分数与疾病列表 F1 含义不同。论文结果表保留空值，待各任务正式协议与对应结果核对后填写。

## 目录与存放规则

```text
medical_world_model/
├── README.md                项目入口
├── code/                    自有代码、独立第三方仓库及本地运行目录
│   ├── medworld_common/     公共 Qwen 组件与运行工具
│   ├── medworld_stage1/     Table 2：当前状态四任务
│   ├── medworld_table1/     Table 1：未来预测
│   ├── medworld_baselines/ 原始 VLM 零样本评测
│   └── medworld_dense_baselines/ 冻结 VLM 的密集任务适配
├── experiments/             实验索引和机器可读登记表
├── research_notes/          研究记录、历史待办、环境和复现实验说明
├── 27cvpr/                  CVPR 2027 工作稿、补充材料和配图；暂用 2026 模板
├── results/                 数据审计、病例展示等导出产物
├── scripts/                 跨实验状态查询与诊断工具
├── imgs/                    目前仅有空的 ppt/、scripts/、source/ 子目录
└── related_works/           参考文献笔记与本地 PDF
```

训练权重、缓存、日志和预测继续保留在原实验的 `runs/`、`data/`、`weights/` 下，便于已有实验恢复。新实验必须使用新运行目录；运行目录中的 `source/` 是该次实验的冻结源码，开发修改在 `code/` 中进行。CVPR 工作稿的论文配图位于 `27cvpr/ppt/` 和 `27cvpr/imgs/`。

## 版本管理

项目根目录的 Git 管理自有代码、配置、文档、实验登记表和 `27cvpr/` 论文源码。远端仓库为 [MVChem/medical_world_model](https://github.com/MVChem/medical_world_model)。数据、模型、实验产物和第三方仓库不纳入该 Git。

整理前的版本标记为 `before-project-cleanup-20260911`。Git 历史不替代实验的源码快照、原始指标和数据文件。
