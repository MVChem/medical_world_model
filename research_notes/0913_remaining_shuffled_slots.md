# 09-13：补齐其余模型的打乱 slots 对照

在上午 [冻结多层视觉 slots 实验](0913_frozen_multidepth_slots.md) 完成后，补跑 Qwen3.5-0.8B、Qwen3.5-9B、Qwen3.5-27B-FP8、MedGemma v1-27B 的 shuffled slots。每个模型分别训练 segmentation 和 ×4 super-resolution，共 8 项。上午的 Qwen4B、MedGemma4B shuffled 结果直接复用。

## 固定设置

- 沿用上午冻结的训练源码、视觉 slots 缓存、4,096 张训练图与患者互斥划分。
- 每项 20 epochs、seed 20260913、有效 batch 8、microbatch 4、学习率 3e-4；仅训练 decoder，固定使用最终 epoch。
- 同一 split 内将 slots 分配给其他患者的图像。打乱映射沿用上午实现和 seed，并核验同 split、不同患者。
- 显式以原 `frozen_slots_20260913` 为 `--data-run`，保留与原基线相同的数据路径合同；以新目录作为 `--run`，输出和 checkpoint 独立保存。
- 最多同时使用 4 张 GPU；启动前重新核查空闲计算进程，并沿用原队列的每卡锁、工作进程完成记录及断点恢复。

## 结果和来源

新运行：`code/medworld_dense_baselines/runs/frozen_slots_shuffled_remaining_20260913`。

正式队列于 **2026-09-13 15:44:05（北京时间）** 启动，coordinator PID 1606830。首批 GPU 4／5 分别训练 Qwen0.8B 分割／SR，GPU 6／1 分别训练 Qwen9B 分割／SR；其余四项自动接续。启动前 8 项 CPU 预检均通过：decoder 初始化哈希与原基线一致，donor 映射与原打乱对照一致，所有配对均为同 split、不同患者。队列 dry run 确认仅新增 8 项，合并报告 13 行且没有 provenance 警告。此处是启动快照，实时状态见下方入口。

- [合并报告](../code/medworld_dense_baselines/runs/frozen_slots_shuffled_remaining_20260913/REPORT.md)：9 行原结果加 4 行新 shuffled 结果。正负差值均相对原 image-only 基线。
- [新增 8 项队列状态](../code/medworld_dense_baselines/runs/frozen_slots_shuffled_remaining_20260913/status.json)。历史 24 项不计入新增完成数。
- `parent_reference.json` 固定原源码、协议、缓存、指标、逐样本记录及完成证据的 SHA256；新结果再次核对基线合同、初始化和各 epoch 样本顺序。
- 原始实验目录保持原结果；新报告明确标注哪些结果复用自上午，哪些来自本次补跑。

报告给出单 seed 的患者配对 bootstrap 区间，不覆盖训练随机性。MedGemma 两个检查点的视觉塔与 slots 缓存相同，其结果不能作为模型规模效应的证据。

## 恢复命令

```bash
/home/data2/chk/workspace/2026/.venv/bin/python -u \
  code/medworld_dense_baselines/runs/frozen_slots_shuffled_remaining_20260913/source/frozen_slots_supplement_queue.py \
  --run code/medworld_dense_baselines/runs/frozen_slots_shuffled_remaining_20260913 \
  --source code/medworld_dense_baselines/runs/frozen_slots_shuffled_remaining_20260913/source \
  --controls qwen08b,qwen9b,qwen27b_fp8,medgemma27b \
  --max-gpus 4 --gpu-order 4,5,6,1,0,2,7
```

正在运行的 coordinator 持有排他锁，重复启动会拒绝。真实完成状态以队列与报告为准。
