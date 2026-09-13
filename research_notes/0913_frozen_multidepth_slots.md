# 09-13：冻结多层视觉 slots 的分割与超分对照

目标：2026-09-13 20:00（北京时间）前得到首版结果，判断第 5–8 个 slots 对 segmentation 和 ×4 super-resolution 是否有额外收益。

## 本次固定的定义

按 [09-12 新方法图](0912_multiscale_slots_figure_prompt.md)，状态共 8 个 slots，前四个来自视觉语言融合路径，后四个来自同一个 VLM 自带的 vision encoder。**Classification 和 diagnosis 都可以读取全部 8 个 slots**，取代此前分类只读前四个的任务分配。本轮只做密集任务，实际提取并评测第 5–8 个视觉 slots；不运行前四个融合 slots，也不把旧版最后语言层 slots 当作本次表示。

四个视觉 slots 分别取视觉 Transformer 1/4、1/2、3/4 深度和最后一个 block 的输出，每层对空间 tokens 做均值汇聚，再无参数统一到 1,024 维。具体 block 层号与通道处理记录在各模型 `slots_model.json`，缓存来源记录在 `slot_contract.json`。没有可训练的 slot embedding、状态 adapter 或 LoRA；vision encoder 始终冻结。每张输入图像得到自己的四个向量，“固定”指取点、汇聚规则和编码器参数固定，并非所有图像使用同一组常数向量。

四个向量没有天然二维网格。Decoder 通过带空间位置的图像 queries 读取 slots，并融合图像 CNN 特征进行分割或超分。槽位到 decoder 的投影属于任务头，可以训练，不改变缓存中的 slots。本次固定池化是新设计的一个简单实例，不能冒充已经训练好的状态表示。

## 对照和预算

| 组别 | 图像输入 | 条件 | 更新参数 |
|---|---|---|---|
| Image-only | 分割原图；SR 为 LR | 四个零向量 | Decoder |
| Image＋frozen slots | 同上 | 同一输入图的第 5–8 个视觉 slots | 相同 Decoder |
| Image＋shuffled slots（补充） | 同上 | 同 split、其他患者图像的 slots | 相同 Decoder |

Image-only 使用同一套 decoder 结构与初始化，仅将条件置零；每项任务共享一个基线。六个条件模型分别为 Qwen3.5-0.8B／4B／9B／27B-FP8，以及 MedGemma-1.5-4B 和 MedGemma v1-27B。补充 shuffled 训练预设为 Qwen4B 和 MedGemma4B，安排在主表之后。

只加载每个检查点的原生视觉塔，记录其实际参数量和权重指纹。检查点的 4B／27B 是整套 VLM 的规模，**本实验没有运行语言模型，因此不能将差异单独解释为语言模型变大带来的收益**；如视觉塔权重相同，需明确标注。

每组固定 20 epochs、4,096 张训练图、有效 batch 8、AdamW、学习率 3e-4、cosine decay（最低为初始值的 10%）、weight decay 0.01、相同随机种子和样本顺序。分割使用 BCE＋soft Dice，SR 按新 method 使用 valid-pixel MSE。所有主结果取第 20 个 epoch，不按测试结果选择 checkpoint。原 09-12 的任务头和最后语言层特征仅作为历史实验保留，本轮全部训练新 decoder。

## 数据和指标

直接复用 `dense_20260912/data` 中已固定的病人互斥划分：4,096 train／249 validation／447 test；另有 Montgomery 138 张外部分割测试图，不参与训练和选择。

- Dice pseudo：CXAS 的右肺、左肺、心脏伪标签一致性。
- Dice human：Montgomery 人工两肺分割；不含心脏。
- SR：×4 抗锯齿 bicubic 降采样后的 uint8 LR。视觉塔和图像分支使用同一份 LR，HR 只用于目标。报告 valid ROI 上 PSNR、SSIM。
- 主表报告各模型分数、相对 image-only 的差值；按患者对逐样本差值 bootstrap 给出区间。单 seed 区间只反映测试患者抽样，不覆盖训练随机性。

已有相同测试集 Bicubic ×4 参考为 29.7421 dB／0.8998 SSIM，来源是旧运行的 `bicubic_metrics.json`。它不是冻结 slots 的实验结果。

## 运行和结果入口

新运行：`code/medworld_dense_baselines/runs/frozen_slots_20260913`。最多同时占用 4 张 GPU，只分配无计算进程的空闲卡；开始检查时 2／3／7 空闲，其余卡有其他工作。

正式队列于 **09:29:55，北京时间** 启动：GPU 2 为分割 image-only，GPU 3 为 SR image-only，GPU 7 为 Qwen0.8B slots 提取。主表 20 项工作加补充打乱对照 4 项；GPU 空闲时继续调度，源码在启动前已冻结。这个时间记录只说明已启动，是否完成以报告为准。

- [自动更新结果表](../code/medworld_dense_baselines/runs/frozen_slots_20260913/REPORT.md)
- [队列状态](../code/medworld_dense_baselines/runs/frozen_slots_20260913/status.json)
- [固定协议](../code/medworld_dense_baselines/runs/frozen_slots_20260913/protocol.json)

提取、训练、队列、报告分别位于 `code/medworld_dense_baselines/frozen_slots_{extract,train,queue,report}.py`。运行前冻结源码到新目录的 `source/`；检查、缓存、权重与指标各自记录来源。未完成和失败项目保留状态，不填零；主表与 shuffled 补充表分开。

验证完成：5 项提取协议测试、9 项 decoder 测试及原数据的 7 项协议测试；六个视觉塔分别完成真实 HR／LR／Montgomery 前向；Qwen0.8B 独立视觉加载与完整原生模型的视觉读出逐元素一致；两个任务、三种条件的 GPU 短训练与保存恢复检查通过。所有输入数组已重算 SHA256 与原 manifest 对齐，伪标签另计算指纹。这些检查不作为效果分数。

视觉塔核验发现：**MedGemma 1.5-4B 与 v1-27B 的视觉权重逐元素相同**，三个真实输入的 slots 也完全相同；每个视觉塔为 416,866,032 个参数。这两行是相同视觉表示的重复验证，不能作为两种视觉容量的证据。Qwen9B 与 27B 的视觉权重不同。

查看状态或在 coordinator 退出后恢复队列：

```bash
/home/data2/chk/workspace/2026/.venv/bin/python scripts/project_status.py

/home/data2/chk/workspace/2026/.venv/bin/python \
  code/medworld_dense_baselines/runs/frozen_slots_20260913/source/frozen_slots_queue.py \
  --run code/medworld_dense_baselines/runs/frozen_slots_20260913 \
  --source code/medworld_dense_baselines/runs/frozen_slots_20260913/source \
  --max-gpus 4
```

运行中的 coordinator 持有排他锁，重复启动会拒绝；恢复时采用已有工作进程和该运行自己的源码、配置、断点。
