# 2026-09-11 项目整理记录

入口：[项目 README](../README.md) · [开发说明](DEVELOPMENT.md)。

## 改动

- 根目录建立 Git，整理前提交为 `e97013f`，标签为 `before-project-cleanup-20260911`。按用户后续授权连接 `MVChem/medical_world_model`。
- 根 README 统一论文、代码、笔记、实验和开发入口。当前待办独立成文档，09-13 再迁入 [research_notes/TODO.md](TODO.md)；旧 TODO 原文保存在 `research_notes/archive/task_history.md`。
- Stage 1 三个 trainer 合并为 `training.py`，旧命令保留兼容入口。模型差异及梯度约束放在 `training_variants.py`，当前实验统一从 `launch.py` 启动。
- 公共 Qwen 组件与运行工具移到 `medworld_common/`，Table 1 和 Stage 1 共用。新运行的源码快照包含公共模块。
- 增加实验登记表和 `scripts/project_status.py`。修正文档中“尚未训练”、过期目录、报告生成／疾病识别混用等入口说明。
- 数据缓存、权重、运行目录和第三方仓库保持原位置。正在运行的两组实验继续执行原源码快照。

## 验证

Stage 1 的 5 项训练协议测试通过；Table 1 CPU 测试 18 项通过，另在 GPU 上执行的 2 项模型约束测试通过。

整理后的 4＋4 与无 slots 模型各完成 8 步真实数据训练和四任务评估。两种模型的 step 0 参数和初始验证 loss 均与整理前完全一致；24,000 步对应的 48,000 个 microbatches 核验一致。无 slots 模型的梯度、padding 隔离、SR 的 HR 输入隔离和 checkpoint 重载检查通过。

恢复检查从 4＋4 模型第 4 步的 checkpoint 继续到第 8 步，恢复后的日志只包含第 5～8 步；最终更新数和各任务样本数与连续训练一致。

独立进程中的 BF16 更新不是逐位确定的。相同 GPU 上，旧／新无 slots trainer 运行 8 步后，参数最大绝对差为 `3.85e-5`，更新数和各任务样本数相同；这项检查不声称逐位相同的训练轨迹。

本机详细检查记录位于 `code/medworld_stage1/runs/refactor_check_20260911/`。该目录为实现验证，不是新的论文实验。当前实验的源码文件在整理前后逐文件校验；其配置和模型结构未因整理改变。

## 2026-09-13 文档迁移

按用户要求，将原 `docs` 目录中的 `DEVELOPMENT.md`、`TODO.md`、本页和 `runtime_20260911.json` 全部迁入 `research_notes`，移除原目录。开发说明与环境快照原文保留，当前待办补记 09-12 队列完成；根 README、代码入口、研究笔记和历史任务入口中的相关链接同步更新。统一入口见 [研究笔记索引](README.md)。

同日按用户最新决定，将 [TODO.md](TODO.md) 标记为 **Deprecated／已停用**，保留停用时的内容，不再更新或作为后续待办依据；入口同步标为历史记录。`results/` 及其中的生成脚本继续保留在本地，不纳入 Git。
