# Qwen3.5-0.8B 无 slots 四任务对照

2026-09-11，按“去掉 slots，重新训练，训练 steps 一致”的要求补充。运行目录为 [qwen08_noslots_20260911](../code/medworld_stage1/runs/qwen08_noslots_20260911/REPORT.md)，协调器 PID 2541127，训练 PID 2541403，GPU 1。Ours 的 4＋4 训练继续使用 GPU 0。

## 和已有代码的关系

- [旧 Stage 1 评估](../code/medworld_stage1/evaluation.py) 已实现 `null` 和 `shuffled`。`null` 在推理时把状态清零；`shuffled` 用 `state.roll(1, 0)` 跨患者替换整组状态，单样本时另取患者。它没有打乱同一患者内部的 slot 顺序，也没有重新训练无 slots 模型。旧的冻结 encoder 对照仍保留 slots。
- [DirectQwen](../code/medworld_table1/direct.py) 已实现原生 Qwen3.5-0.8B 的无 slots 未来报告／finding 预测，使用 Qwen 原生视觉前端。它属于 Table 1 的早期试验，不包含当前四任务训练；旧直接模型采用相同训练时长预算，也不能直接当作本轮相同更新步数的对照。
- 本次新增的是 Table 2 四任务的无状态 slots 模型，代码为 `code/medworld_stage1/noslots_*.py`，执行的是运行目录内冻结的 `source/` 副本。

## 对照定义

保留与 Ours 相同的冻结 V-JEPA 视觉前端、Qwen3.5-0.8B encoder、独立 Qwen decoder、LoRA 和四任务 heads。去掉 encoder 末端的 8 个可学习状态 slots，heads 直接读取 Qwen 输出的全部有效图像／报告 tokens。分类、分割和 SR 中的任务查询仍保留，两边相同；它们不属于被移除的共享状态 slots。变长输入使用 padding mask。

这是一项控制视觉前端和任务 heads 的 slots 对照，应标为 “Qwen3.5-0.8B, full-token readout”。它和更早采用原生 Qwen 视觉编码器的 `DirectQwen` 实验不是同一种配置。

SR encoder 只接收低分辨率图像特征和同样的报告，HR 只用于监督。与 Ours 共用的可训练参数精确复制自 Ours 的 **step 0** checkpoint；不使用任何训练后的 Ours 权重。只移除 `encoder.slots` 的 8,192 个参数，剩余可训练参数为 9,247,173。

## 公平性和评估

两边相同的项目包括：预训练权重、公共参数初始值、患者划分、缓存、报告预算、问答来源和固定词表、标签屏蔽、样本顺序、任务轮转、batch、梯度累积、优化器、学习率、warmup、损失权重和评估实现。

已逐一核验全部 24,000 次 optimizer 更新对应的 48,000 个 microbatches，其样本索引与顺序一致。每任务 6,000 次更新；含累积的样本呈现次数分别为分类 96,000、识别 48,000、分割 96,000、SR 96,000。

Baseline 跟随 Ours 实际完成的 optimizer 更新数，上限 24,000；晚启动不会触发相同墙钟截止时间而少训。完整 token 上下文的 FLOPs 和运行时间可能不同，本对照匹配步数和训练样本量。

在第 **3,176** 步保存并自动评估 checkpoint，与 Ours 已保存的同一步早期验证比较：分类 160 张、分割 249 张、SR 318 张、疾病识别同一组 128 题。最终比较使用相同步数的 final checkpoint 和相同测试集。比较程序核验预测文件中的样本、问题、答案顺序一致后才输出差值。不同步数和不同划分不混在一起。

## 启动核验

真实数据上的四任务梯度、padding 隔离、SR 的 HR 输入隔离、checkpoint 文件重载和公共参数初始化检查均通过，记录见 [contracts.json](../code/medworld_stage1/runs/qwen08_noslots_20260911/checks/contracts.json)。独立的 8 步 smoke 训练与四任务评估完成；正式训练重新从共同的 step 0 初始化，未继承 smoke 权重。

[完整公平性检查](../code/medworld_stage1/runs/qwen08_noslots_20260911/checks/fairness_audit.json) · [正式训练状态](../code/medworld_stage1/runs/qwen08_noslots_20260911/joint/status.json) · [配置](../code/medworld_stage1/runs/qwen08_noslots_20260911/config.json) · [自动更新的比较报告](../code/medworld_stage1/runs/qwen08_noslots_20260911/REPORT.md)

与 Ours 相同，本轮使用当前报告辅助读出、Chest ImaGenome 本地派生阳性问答、CXAS 分割伪标签和合成 ×2 SR；不声称复现官方 VQA benchmark。论文指标继续留空，等待相同步数评估。
