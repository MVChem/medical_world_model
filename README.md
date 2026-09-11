# MedWorld-JEPA

医疗多模态状态表示与未来预测研究。当前使用 Qwen3.5-0.8B，论文以两个问题组织实验：**Table 1：未来预测；Table 2：当前状态的四个下游任务**。

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

2026-09-11 的 Stage 1 主实验采用 **4＋4 slots**：分类读前 4 个，疾病识别读全部 8 个，分割和超分读后 4 个。无 slots 对照保留相同视觉前端、Qwen 骨干和任务 heads，直接读取完整图文 tokens；两边匹配训练样本顺序和 optimizer 更新数，上限各 24,000 步。

- [Ours：4＋4 slots，运行报告](code/medworld_stage1/runs/slot44_20260911/REPORT.md)
- [Qwen0.8B：无 slots，运行报告与同一步数比较](code/medworld_stage1/runs/qwen08_noslots_20260911/REPORT.md)
- [本轮数据与实现说明](research_notes/0911_stage1_slot44_run.md) · [无 slots 对照定义](research_notes/0911_qwen08_noslots_baseline.md)

README 不固定记录训练步数。查看实时状态：

```bash
python scripts/project_status.py
```

本轮疾病识别使用本地 Chest ImaGenome 派生问答；分割衡量 CXAS 伪标签一致性；SR 使用合成 ×2 退化。历史报告生成分数和这轮疾病列表 F1 含义不同。论文结果表保留空值，待正式协议和相同步数结果确认后填写。

## 目录与存放规则

```text
medical_world_model/
├── README.md                项目入口
├── code/                    自有代码、独立第三方仓库及本地运行目录
│   ├── medworld_common/     公共 Qwen 组件与运行工具
│   ├── medworld_stage1/     Table 2：当前状态四任务
│   └── medworld_table1/     Table 1：未来预测
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
