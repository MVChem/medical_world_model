# 2026-09-15：MedWorld 融合实现与验证

## 本次实现

开发目录为 [code/medworld](../code/medworld/README.md)。融合前快照仍为
`pre-table1-table2-unification-20260915`，对应 `fc4be80e2a6abd47829835b968ddd14a1daba982`。
本次没有将旧表格成绩改写成融合模型成绩，也没有启动正式预算训练。

融合版采用一个 online 观察编码器，Stage 1 训练当前四任务，Stage 2 继续训练时间预测并
replay 当前任务。状态为 `[B,8,1024]`；Qwen3.5-0.8B 的语言层取 6／12／18／24，
视觉层取 3／6／9／12。target 从完成 Stage 1 的 online encoder 初始化，所有编码器
可训练参数通过 EMA 更新，冻结基座共享。文本 decoder 有独立 LoRA，读取预测 slots
与固定提示，其报告 CE 可回传到 World Model 和 online encoder。

正／负时间条件使用实际小时差，回溯样本交换两个观察。当前初版输入仍需影像；
时间任务可读取该观察的报告，不读取 EHR。纯报告输入、VQA、grounding 和临床报告
指标适配仍待实现。细节与命令见开发目录 README。

## 跨表患者隔离

合并原数据后发现，Table 2 测试池有 53 位患者出现在 Table 1 训练池，1 位出现在
验证池。融合版保留测试集，剔除冲突的训练／验证行，不移动到其他 split：

| 时间配对 | 原始 | 保留 | 排除患者 |
|---|---:|---:|---:|
| train | 16,000 | 15,877 | 53 |
| validate | 230 | 227 | 1 |
| test | 297 | 297 | 0 |

四项当前任务的样本数不变；启用双向后时间样本数翻倍。所有剩余训练、验证、测试池
通过跨任务患者隔离审计。数据协议指纹：
`35668c6534a2f8627e85e46e7ee220838a172a32c9977fed94fa6a06ec1a45a1`。

## 已完成验证

### 单元测试

11 项测试通过，覆盖：

- EMA 数值公式、可训练参数独立存储、冻结基座共享、target 无梯度和 eval 状态。
- 梯度累积与当前任务 replay 后，EMA 仍只按 optimizer 更新计数。
- 中断／恢复与连续训练的 CPU 参数完全一致，包括 EMA、优化器、随机数和采样位置。
- 正负实际时间、双向端点交换、推理时目标端不可访问、跨任务 holdout 优先级。
- 报告 CE 对状态的梯度、EOS／padding、逐样本结束、缓存位置、CPU 状态保存恢复。
- 分割读出对后四槽有梯度，对前四槽无梯度。

### 真实 Qwen3.5-0.8B 端到端测试

运行目录：
[qwen035_08b_smoke_cpu_v2_20260915](../code/medworld/runs/qwen035_08b_smoke_cpu_v2_20260915/)。
使用本地真实 Qwen 和 V-JEPA 权重、真实数据，设备为 **CPU**，BF16 基座／FP32 adapter，
8 个 CPU 线程。Stage 1 共 4 次更新，四任务各一次；Stage 2 共 4 次更新，第 4 次
附加当前分类 replay。batch size／梯度累积均为 1，报告监督预算 64 tokens，
观察文本预算 96 tokens，生成预算 24 tokens。

所有训练及验证 loss 均有限。关键验收结果：

| 检查 | 结果 |
|---|---|
| 仅报告 CE → online slot queries 梯度范数 | 0.63248 |
| 仅报告 CE → World Model 输出层梯度范数 | 1.52560 |
| 仅报告 CE → 文本投影梯度范数 | 2.58289 |
| target 状态 requires_grad | false |
| online／target 共享冻结 Parameter 数 | 473 |
| EMA 独立可训练参数张量数 | 96 |
| Stage 2 optimizer 更新／EMA 更新 | 4／4 |
| online 与 EMA slot queries 差异范数 | 0.01923 |
| CPU slots 保存重载后的解码 | 完全一致 |
| 重新构造模型并恢复 checkpoint 后的解码 | 完全一致 |
| checkpoint 重载前后预测状态最大误差 | 0 |

详见 [smoke_audit.json](../code/medworld/runs/qwen035_08b_smoke_cpu_v2_20260915/smoke_audit.json)。
独立 `infer` CLI 只读取 checkpoint 与 `predicted_state.pt`，复现了同样的两份输出。
`evaluate --task all --split validate --limit 2` 也已完成：分类、报告、分割、SR、
双向时间预测均从同一个 Stage 2 checkpoint 运行，结果见
[evaluation_validate_smoke](../code/medworld/runs/qwen035_08b_smoke_cpu_v2_20260915/evaluation_validate_smoke/)。

该 checkpoint SHA256 为
`46247345bb2eadac97c54f43738a8c7d469e0a7d15ea8b8a16faf6a9fc64c580`。
checkpoint、数据、生成记录和运行日志都位于被 Git 忽略的本地运行目录。

### 结果能说明什么

本轮证明两阶段流程、梯度方向、EMA、同一 checkpoint 多任务使用和独立状态解码可运行。
8 次更新不足以说明学会了病程或准确报告。两份时间报告很相似；真实／交换／全零 slots
的报告 CE 分别为 2.2311／2.2171／2.5435，交换条件甚至稍低。这个小样本诊断不能证明
模型已经学会利用正确患者状态或时间方向，需要正式训练后的对照评测。

## GPU 验证未完成的原因

**2026-09-16 更新：CUDA 已恢复，RTX 4090 上的两阶段短测与全部验收已通过，**
训练 allocated 峰值为 2.363 GiB（batch 1，报告预算 64 tokens）。详见
[GPU 短测记录](0916_unified_medworld_gpu_smoke.md)。以下保留 09-15 当时的故障记录。

GPU 3 的 NVML 查询显示空闲，但新 PyTorch 进程无法初始化 CUDA，返回 unknown error。
独立进程对 GPU 0／4 调用 CUDA driver 的 `cuInit(0)` 同样返回错误 999；GPU 3 的
数字编号和 UUID 两种选择方式都失败。该问题发生在模型运算之前。

本轮因此改用 CPU 完成真实模型验收，没有重置 GPU 或中止其他任务。
尚无融合版 GPU 峰值显存和吞吐量数据。驱动恢复后，应使用 README 的 `--smoke --gpu`
命令补测，再启动正式预算训练。
