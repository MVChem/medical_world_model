# 2026-09-16：融合版 GPU 短测通过

CUDA 已恢复。GPU 3（RTX 4090，24 GiB）完成了 Qwen3.5-0.8B 融合模型的两阶段短测，
退出码为 0；昨日初始化错误 999 本次没有重现。其他卡仅做 NVML 查询，未做计算压力测试。

## 环境与配置

- GPU UUID：`GPU-53a129cb-b94d-0cfd-db55-530f5896c2be`，驱动 595.84。
- PyTorch 2.11.0+cu129，CUDA runtime 12.9；8 个 CPU 线程。
- 使用本地真实 Qwen3.5-0.8B／V-JEPA 权重和真实数据。
- batch size 1，梯度累积 1；报告监督 64 tokens，观察文本 96 tokens，生成 24 tokens。
- Stage 1 四次更新，分类／报告／分割／SR 各一次。
- Stage 2 四次更新，最后一次含当前分类 replay；EMA 更新四次。

运行目录：
[qwen35_08b_smoke_gpu_20260916_0855](../code/medworld/runs/qwen35_08b_smoke_gpu_20260916_0855/)。
其中保留配置、数据审计、源代码副本和 SHA256 清单、环境信息、checkpoint 与日志。
这些运行产物继续由 Git 忽略。

## 显存与速度

本次为每个 optimizer step 单独记录 PyTorch 显存峰值，避开重载检查中同时存在两份模型的情况。

| 阶段 | 最大 allocated | 最大 reserved | 更新耗时 |
|---|---:|---:|---|
| Stage 1 | 2.121 GiB | 2.252 GiB | 四项任务分别 5.49／3.51／1.56／0.45 秒 |
| Stage 2 | 2.363 GiB | 2.527 GiB | 普通更新约 3.25 秒；含 replay 约 4.05 秒 |

`allocated` 是 PyTorch 活跃张量占用，`reserved` 包含其缓存；不包含全部 CUDA 驱动开销。
耗时包含该训练步骤的数据准备，不含随后验证／checkpoint 保存；首次操作也有初始化开销。
重载审计中同时驻留原模型与恢复模型，allocated 峰值为 **4.155 GiB**，不作为训练需求。
这些数值只对应上述短序列、小 batch 配置；正式配置的长度／batch 增大后需重新测量。

## 验收结果

- 所有训练与验证 loss 有限，无 OOM。
- 单独报告 CE 的梯度范数：online slots 2.461、World Model 输出层 5.226、文本投影 7.623。
- target 状态无梯度；online／target 共享 473 个冻结参数张量，EMA 独立维护 96 个可训练参数张量。
- 完成四次 Stage 2 更新后，EMA 计数恰为四，online／target slot query 差异范数为 0.01998。
- 正负时间间隔都能生成文本；保存到 CPU 的状态张量重载后解码完全一致。
- 重新构造模型、恢复 checkpoint 后解码完全一致；预测状态最大绝对误差为 0。
- 11 项单元测试再次全部通过。测试结束后进程退出，GPU 3 恢复空闲。

汇总与完整检查见 [gpu_test_summary.json](../code/medworld/runs/qwen35_08b_smoke_gpu_20260916_0855/gpu_test_summary.json)
和 [smoke_audit.json](../code/medworld/runs/qwen35_08b_smoke_gpu_20260916_0855/smoke_audit.json)。

报告生成受 24-token 短测预算截断。本轮验证 GPU 上的计算、梯度、EMA 和保存恢复；
不构成病程预测准确性或临床报告质量结论，未启动正式预算训练。
