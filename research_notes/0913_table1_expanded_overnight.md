# 09-13 Table 1：扩大纵向训练集与状态消融

本轮将原始 MIMIC-CXR＋MIMIC-IV 纵向训练对从 6,000 扩大至 **16,000 对／7,188 位患者**，仍采用每位训练患者至多 4 对的确定性选择。验证集仍为 **230 对／62 位患者**，测试集仍为 **297 对／94 位患者**；准备程序逐条核对新旧验证和测试清单完全一致，并核对旧训练集合包含于新集合。共 28,202 张不同观察图像；已有冻结 V-JEPA 特征按图像 ID 与路径复用，新增图像用同一权重和预处理提取。

运行目录为 `code/medworld_table1/runs/overnight_20260913`。`jobs.json` 给出实际命令、依赖、完成标志和所需更新数；`configs/*.json` 是运行配置；`source/` 与 `source_manifest.json` 保存代码副本及 SHA-256。本页是协议记录，尚不表示长任务已完成。

## 三个可比训练条件

| 条件 | 未来预测读取的状态 | 训练与评估处理 |
|---|---|---|
| `slots` | Qwen3.5-0.8B 产生的 8 个学习状态 slots | 当前源图、当前报告和源时点 EHR 构造状态，再传入 horizon predictor |
| `no_slots` | 全部有效图像、报告与 EHR token hidden states | 编码器不增加学习 slots；预测器和报告解码器保留完整 token 序列并应用 padding mask |
| `shuffled` | 同一官方 split 内另一位患者的源状态 slots | 确定性选择不同患者的源证据；本例 horizon、未来目标和图像分类 replay 保持本例对应关系 |

三个条件均从 **09-09 同一份 image-only Stage-1 checkpoint** 初始化共同参数，重置优化器后开展相同数量的 Stage-2 更新。该 checkpoint 只用过旧训练 split；`no_slots` 仅移除原 `encoder.slots` 参数。各条件建立自己的冻结目标编码器。每个条件的 `initialization.json` 保存原 checkpoint 的 SHA-256、原更新步数以及移除参数列表。

正式设置为 microbatch **32**、梯度累积 **1**、各条件 **2,400 次 optimizer 更新**，即每个条件累计 76,800 对输入、约 **4.8 遍**扩大后的训练集。精确词表交叉熵按 128 tokens 分块，避免旧 microbatch 2／累积 4 的小批次开销。无 slots 条件在 GPU 3 的 20 次额外热身后更新平均 **5.09 s／step**、中位数 4.87 s，约 **6.28 pairs／s**；最初新 kernel 编译耗时单独保存在 `benchmark_summary.json`，未混入稳态估计。新特征提取 batch 为 96。

这里的 8 个 slots 沿用旧 Table 1 的同质状态编码器；并非 Table 2 新试验的 4＋4 多层视觉 slots，不能合称同一模型结构。`shuffled` 是跨患者状态错配，也不同于在单份状态内部重排 slot 顺序。

无 slots 条件的源、目标 token 长度不同，因此仅在 latent 对齐损失内部，对有效 token 序列自适应平均至 8 个区间。任务读出仍读取全部有效 token。预测器的 8 个共享位置参数被插值到实际 token 长度；这些是预测器位置参数，不是额外的编码器状态 slots。该差异应随结果报告。

## 梯度、输入与评分

训练保持未来报告 CE、finding BCE、冻结目标 latent 对齐和 image-only 当前 finding replay。`task_gradient_audit.json` 单独求未来报告 CE＋finding BCE 对在线 encoder slots、视觉 adapter 与 LoRA 的梯度，排除 latent 和 replay 后仍要求非零有限梯度。冻结目标不参与反向优化。GPU smoke 已验证 full-token 和跨患者错配条件可前向、反向及保存 checkpoint；CPU 测试验证 padding 不影响有效 token 的预测与梯度，且 donor 始终属于同一 split 的不同患者。

Qwen 原始基座权重和 V-JEPA 视觉提取器仍冻结；在线更新的是 Qwen encoder／decoder LoRA、视觉 adapter、状态 slots、预测器及 finding head。这里“任务梯度回传至 VM”指实际进入 VM 的可训练 LoRA 与 adapter，不表示全参数微调 Qwen 或重新训练已缓存的 V-JEPA。

推理只读取源证据与 horizon；未来图像、报告及其 EHR 只用于训练目标和事后评分。EHR 同时要求事件时间、记录时间不晚于该源图时点。继续沿用“源报告已经可用”的回顾性假设，因为原始 CXR 不含可靠报告签发时间。未引入 Qwen-Gate 注释或未来区间事件。

测试评分输出既有的官方 CheXbert、RadGraph-XL partial F1，并使用与六模型基线相同的 `probability_metrics` 实现补齐 AP、AUROC、Brier、ECE。连续分数来自训练的 finding head；参考的未知／不确定字段被排除，预测不能改变覆盖范围。另将三条件预测接入已有官方 GREEN prompt、tokenizer、parser 和本地 GREEN 模型，生成独立的 `green/<condition>/test/green_metrics.json`。

本 297 对测试集没有经审核的方向真值，因此 Direction F1 仍留空。旧六模型的独立 82 对方向评测不能移填到此处。六种 Qwen／MedGemma 的现有零样本结果仍属于各自旧协议；本页三个新训练条件仅使用 Qwen3.5-0.8B。

截止时间沿用本轮统一设置：训练配置在 **2026-09-14 06:45（北京时间）** 后保存并退出，队列在 **07:45** 停止本轮子进程，**08:00** 执行强制截止。状态 JSON 会区分完整更新预算与中断；未完成或评分失败的指标不会作为完成结果写入。
