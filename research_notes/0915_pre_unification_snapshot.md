# 2026-09-15：Table 1／Table 2 融合前旧版本快照

## 快照定位

本提交按用户要求归档融合前的本地工作，推送到私有仓库
[MVChem/medical_world_model](https://github.com/MVChem/medical_world_model)。
对应标签：`pre-table1-table2-unification-20260915`。

这是旧版实现的保存点。融合版本尚未实现，本次不修改模型、训练目标、
正在运行的实验或论文已填分数。

## 当前实现分别在哪里

| 内容 | 代码入口 | 当前边界 |
| --- | --- | --- |
| 新 Table 1：未来状态／报告预测 | `code/medworld_native_forecast/` | 独立的分类／报告 Stage 1，随后预测未来 4＋4 slots；原生 decoder 同时读取当前图文和预测槽 |
| 新 Table 2：当前状态四任务 | `code/medworld_multitask/` | 分类、报告、分割、×4 SR 轮转训练，共享本分支 checkpoint；尚未接入未来预测 |
| 旧 Table 1 与已填预测行 | `code/medworld_table1/`、`code/medworld_common/qwen.py` | 已填 9B slots 行使用最后语言层八查询，不能当作新版多深度结果 |
| 旧当前状态任务与对照 | `code/medworld_stage1/`、`code/medworld_dense_baselines/`、`code/medworld_joint/` | 历史训练、冻结视觉槽对照及在线适配原型，协议各自保留 |
| 开放模型对照与调度工具 | `code/medworld_open_baselines/`、`scripts/` | 本次补齐尚未进入 Git 的自有源码及协议 |
| 论文与本地研究材料 | `27cvpr/`、`research_notes/`、`results/` | 保存当前稿、图表、导出和研究记录；参考文献目录留在本地 |

两条新版路径都使用 `[B,8,1024]` 状态，但分别实现和训练。Table 1 的
encoder／decoder LoRA 独立，Table 2 的报告编码／解码共享语言 LoRA；
输入处理、任务头和 checkpoint 结构也未统一。形状相同不等于状态空间兼容。
目前没有一个完成两阶段训练并通过两张表完整评测的统一 checkpoint。

## 下一版本：统一模型与两阶段训练

1. 统一 4 个融合深度槽＋4 个原生视觉深度槽的状态编码实现、输入协议和 checkpoint 格式。
2. Stage 1 使用同一个模型学习分类、报告、分割、SR 等当前任务；不同任务可轮转采样。正式 VQA 和 grounding 仍需接入相应数据与评测。
3. Stage 2 直接继承这一 Stage 1 checkpoint，训练当前 encoder、未来预测器和未来报告／finding 读出；从同一 Stage 1 encoder 建立冻结 target 副本。
4. 决定并固定 Stage 2 是否混入当前任务 replay；保留 Stage 1 checkpoint，比较未来预测训练对下游能力的影响。
5. 用同一个 Stage 2 checkpoint 完成 Table 1 的未来预测评测和 Table 2 的当前状态评测；按匹配协议另列必要消融。
6. 统一前明确正式报告 decoder 是否继续读取当前原始图文，并相应定义 slots 的贡献及对照，避免仅凭状态形状宣称模型已融合。

Table 1／Table 2 是评测划分。当前独立训练来自两条开发路径尚未整合，
不是论文方法要求建立两套互不关联的状态空间。

## 本次归档范围

纳入自有源码、配置、实验登记、论文 TeX／PDF、PPT、配图与预览、
研究笔记及附件、`results/` 中的研究导出和构建脚本。
旧讨论 ZIP 内含日志，因此另存去掉日志目录／文件的
`results/medworld_jepa_discussion_20260909_snapshot_no_logs.zip`；原 ZIP 留在本地。

以下继续留在本地，不属于 Git 快照：

- 整个 `related_works/` 目录，包括参考文献 PDF 和文献笔记；
- checkpoint、预训练权重、优化器状态；
- 日志、调度运行档案、锁文件、进程文件和编译临时文件；
- `runs/`、`data/`、特征／像素数组缓存（`.npy`、`.npz`）和早期自动生成的 output 目录；
- 下载的 vendor／第三方独立 Git 仓库、环境、依赖及其机器绝对路径软链接。

这些本地资产体积约百 GB。Git 快照保留开发与研究材料，不是可直接恢复
训练的全磁盘备份；恢复训练仍需本地运行目录、对应权重和数据。
研究附件中包含病例材料，沿用各材料的访问与使用条件，仓库保持私有。
