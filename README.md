# MedWorld-JEPA

医疗多模态状态表示与未来预测研究。主模型使用 Qwen3.5-0.8B，论文以两个问题组织实验：**Table 1：未来预测；Table 2：当前状态的下游任务**。已有四任务训练，并在补充原始 VLM 与冻结主干的下游适配基线。

## 从这里开始

| 内容 | 入口 |
|---|---|
| 论文与编译方法 | [26iclr/README.md](26iclr/README.md)，主文件 [main.tex](26iclr/main.tex) |
| 当前实验与历史对照 | [实验索引](experiments/README.md) |
| 训练、评估和代码模块 | [code/README.md](code/README.md) |
| 最新计划、讨论决定、数据说明 | [research_notes/README.md](research_notes/README.md) |
| 运行环境、检查和本地 Git | [开发说明](docs/DEVELOPMENT.md) |
| 接下来要做的事 | [TODO.md](docs/TODO.md) |

## 当前实验

**[09-12 实验小结与结果表](research_notes/0912_recent_experiments_summary.md)** 汇总最近的已完成结果、失败尝试和在跑任务，可直接在 GitHub 阅读。

- **4＋4 slots／无 slots：已完成匹配训练与最终测试。** slots 的分类指标较高，疾病列表 F1 和分割 Dice 较低，SR 差距很小。[训练与数据](research_notes/0911_stage1_slot44_run.md) · [对照定义](research_notes/0911_qwen08_noslots_baseline.md)。
- **六个原始 Qwen／MedGemma：零样本评测已完成。** 包含未来预测、当前分类／报告和派生 QA；官方 VQA 与原测试集 Direction 仍缺合格数据。[评测协议](research_notes/0911_raw_model_baseline_sweep.md)。
- **冻结 VLM＋下游 heads：09-12 已启动。** 比较原图、V-JEPA、V-JEPA＋adapter 分支，补充分割、×4 SR、解剖区域定位及独立方向评测。[运行设置与边界](research_notes/0912_frozen_vlm_dense_baselines.md)。

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
├── docs/                    当前 TODO、环境和复现实验说明
├── research_notes/          按日期保留的计划、讨论和实验解释
├── 26iclr/                  论文、论文配图；独立 Git 仓库
├── results/                 数据审计、病例展示等导出产物
└── related_works/           参考文献笔记与本地 PDF
```

训练权重、缓存、日志和预测继续保留在原实验的 `runs/`、`data/`、`weights/` 下，便于已有实验恢复。新实验必须使用新运行目录；运行目录中的 `source/` 是该次实验的冻结源码，开发修改在 `code/` 中进行。论文配图以 `26iclr/ppt/` 和 `26iclr/imgs/` 为准。

## 版本管理

项目根目录的 Git 管理自有代码、配置、文档和实验登记表。远端仓库为 [MVChem/medical_world_model](https://github.com/MVChem/medical_world_model)。数据、模型、实验产物和第三方仓库不纳入该 Git。`26iclr/` 保留原有独立 Git，其已有稿件修改单独管理。

整理前的版本标记为 `before-project-cleanup-20260911`。Git 历史不替代实验的源码快照、原始指标和数据文件。
