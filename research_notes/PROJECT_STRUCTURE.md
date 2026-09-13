# 项目目录说明

2026-09-13 按本地实际目录核对。[项目首页](../README.md) · [实验索引](../experiments/README.md) · [研究笔记](README.md)

当前分层基本合理，适合继续研究。主要需要说明的是：`experiments/` 只登记实验入口，原始运行结果在 `code/*/runs/`；根目录 `results/` 主要保存展示和审计导出。代码目录还包含共享运行资产和独立第三方仓库，搬动前需要核对引用关系。

## 顶层目录放什么

| 目录 | 当前实际内容 | 新内容如何存放 |
|---|---|---|
| `27cvpr/` | 当前 CVPR 论文、附录、表格、配图制作文件和编译产物 | 论文正文、正式表格和论文用图从这里维护 |
| `code/` | 自有训练／评估／数据代码、5 个第三方仓库，以及本地缓存、权重、运行档案 | 开发源码改对应模块，正式实验使用独立运行目录 |
| `experiments/` | `README.md` 人工索引、`registry.json` 机器可读登记表 | 新正式实验登记代码、run、状态文件与报告入口 |
| `research_notes/` | 带日期的想法、协议、执行记录、结果复盘、论文修改与作图说明；另有已停用的 TODO 历史快照和开发说明 | 讨论决定、实验解释写日期文档，再更新索引 |
| `results/` | 数据审计页、病例展示、结果矩阵预览、诊断导出；还有 PPT、图片和打包文件 | 供阅读、展示或检查的导出按主题与日期建目录 |
| `scripts/` | 跨实验状态查询、特征诊断工具 | 项目级工具放这里，具体训练入口保留在代码模块 |
| `related_works/` | 31 篇本地论文 PDF 和旧翻译任务说明 `CN_related_works.md` | 参考文献 PDF 放这里；目前还没有统一文献索引 |
| `imgs/` | 只有空的 `ppt/`、`scripts/`、`source/` 子目录 | 当前论文配图统一从 `27cvpr/` 进入；本目录可在后续清理时移除 |

## 论文与配图

| 路径 | 内容 |
|---|---|
| `27cvpr/main.tex`、`sections/` | 主稿入口、摘要、引言、方法、实验、结论等分节源码 |
| `27cvpr/supplementary.tex` | 附录入口，当前载入 `sections/b_protocol_details.tex` |
| `27cvpr/tables/` | Table 1 未来预测和 Table 2 当前下游任务的正式 TeX 表源 |
| `27cvpr/main.bib` | 论文参考文献条目 |
| `27cvpr/plans/` | 实验计划说明、共用正式表源的两页预览及构建文件 |
| `27cvpr/imgs/` | 正文引用的图片与部分旧图版本 |
| `27cvpr/ppt/source/` | 原始图像、参考图和作图素材 |
| `27cvpr/ppt/prompt/` | 配图提示词和配套说明 |
| `27cvpr/ppt/scripts/` | 按图及版本分目录的构建、修改、导出脚本 |
| `27cvpr/ppt/ppt/` | 可编辑 PPTX，以及 PDF、SVG、PNG 导出和历史版本 |
| `27cvpr/preview/` | 主稿、附录的逐页图片预览 |

当前正文 Fig. 1 为 `imgs/fig1.pdf`，对应 `ppt/ppt/fig1_v7.pptx`；Fig. 2 为 `imgs/fig2_v2.pdf`，对应 `ppt/ppt/fig2_v2.pptx`。附录直接引用 `ppt/ppt/appendix_fig_v3.pdf`。相对路径均以 `27cvpr/` 为起点。以后更新图时，同步可编辑文件、导出和正文引用。

## 代码模块

| `code/` 下的目录 | 主要职责 |
|---|---|
| `medworld_stage1/` | 当前状态四任务：分类、疾病列表识别、分割、SR；4＋4 slots、无 slots 和历史变体共用训练主循环 |
| `medworld_table1/` | 未来状态、报告、finding 预测及 DirectQwen 对照；保留早期 pilot 和纵向配对实验 |
| `medworld_baselines/` | 原始 Qwen／MedGemma 零样本评测，覆盖未来预测和部分当前任务 |
| `medworld_dense_baselines/` | 冻结 VLM 下游 heads；包含 09-12 密集基线与 09-13 冻结多层视觉 slots 对照 |
| `medworld_common/` | 公共 Qwen、LoRA、状态 encoder、报告 decoder 与运行工具 |
| `mimic_cxr_iv_linked/` | MIMIC-CXR＋IV 连接、筛选、质控、审计和 VLM 初筛管线 |
| `MIMIC_example/` | 早期时间配对、数据示例和数据约束工具 |
| `mimic_vla_jepa/` | 8 月的特征／状态提取和 predictor 实验 |
| `VLA-JEPA/`、`VLA-JEPA-reference/`、`Clin-JEPA/`、`vjepa2/`、`ChestXRayAnatomySegmentation/` | 5 个保留各自 Git 的第三方仓库；两个 VLA-JEPA 目录是独立检出，不能仅凭名称判定重复 |

模块内的 `configs/` 保存配置，`data/` 保存本地数据入口或派生缓存，`weights/` 保存模型权重，`vendor/`、`metric_vendor/` 保存本地依赖，`runs/` 保存运行档案；这些子目录并非每个模块都有。

共享资源目前有跨模块引用：Stage 1 使用 `medworld_table1/vendor/` 和其中的模型权重；dense 代码使用 Stage 1 的派生数据和原始基线某次运行的模型清单。它们是当前依赖，目录日期较早并不代表可以删除。`code/pyproject.toml` 的包名和打包范围仍沿用早期项目，当前训练主要通过各模块脚本运行。

## 一次实验的记录分别在哪里

| 要找的内容 | 入口 |
|---|---|
| 做过哪些正式实验、属于当前还是历史对照 | `experiments/README.md`、`experiments/registry.json` |
| 为什么做、固定什么协议、结果如何解释 | `research_notes/日期_主题.md` |
| 实际执行的源码、配置、checkpoint、状态与逐样本预测 | 对应 `code/模块/runs/运行名/`；文件布局按实验而异 |
| 实验冻结源码 | 运行目录内的 `source/`，按原运行保留 |
| 汇总报告与原始指标 | 登记表指向的 `REPORT.md`、`status.json`、`evaluation/` 等 |
| 可读表格、病例图、网页与数据审计 | `results/主题_日期/`，部分实验的预览在自身 `runs/` 内 |
| 最终写进论文的数值 | `27cvpr/tables/`，需先核对协议、划分、checkpoint 和指标来源 |

例如 09-13 多层视觉 slots 对照：协议在 [0913_frozen_multidepth_slots.md](0913_frozen_multidepth_slots.md)，代码在 `code/medworld_dense_baselines/frozen_slots_*.py`，运行目录为 `code/medworld_dense_baselines/runs/frozen_slots_20260913/`，由实验登记表关联。运行 `python scripts/project_status.py` 可读取已登记实验状态。

`research_notes/assets/` 是笔记的配套案例与指标快照；`research_notes/archive/` 保存历史需求记录。它们与正式运行档案用途不同。Git 当前忽略数据、权重、`runs/`、`results/`、`assets/` 和本地文献 PDF，仓库中的索引不等于已备份这些本地文件。按当前存放约定，`results/` 内的生成脚本也随整个目录保留在本地，不纳入 Git。

## 值得逐步改善的地方

1. 在登记表中补充关键输入、协议说明和共享资源依赖，方便判断某个旧目录是否仍被使用。
2. 论文图目录保留清晰的当前版本入口；`ppt/ppt/` 的名称和历史版本混放可等下一次整理时统一。
3. 为 `related_works/` 增加文献索引，记录论文主题、与本项目的关系及对应笔记。当前 `CN_related_works.md` 只是旧任务说明。

上述是后续整理建议；本轮保留实际目录与实验路径，更新说明和论文资源引用。
