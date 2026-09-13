# 09-13：Table 1／Table 2 的结果来源与实验时间

整理与核对日期：2026-09-13。下文时间均为北京时间（Asia/Shanghai，UTC+8），来自运行状态 JSON 的时间戳及已有执行记录。本次整理读取已有结果，没有重新训练或评分。

## 1. 聊天里那张表在哪里

聊天展示的 Qwen／MedGemma Table 1／Table 2 数值主要来自：

- [完整结果矩阵 full_tables.md](../code/medworld_dense_baselines/runs/dense_20260912/preview/full_tables.md)：合并论文计划行、六模型零样本结果、后续训练任务头的结果。
- [密集任务结果与队列 preview.md](../code/medworld_dense_baselines/runs/dense_20260912/preview/preview.md)：六个模型 × 三种图像分支的分割／×4 超分，以及解剖定位和独立方向评测。
- [4＋4 slots／无 slots 最终比较 comparison_final.json](../code/medworld_stage1/runs/qwen08_noslots_20260911/comparison_final.json)：聊天最后那张两行的 slots 对照表，来自另一轮实验。

完整矩阵的生成逻辑见 [report.py](../code/medworld_dense_baselines/report.py)。它读取 `results/baseline_matrix_preview_20260912/table1.csv`、`table2.csv` 中的计划行与旧零样本结果，再读取 `dense_20260912/<model>/` 下的新指标，输出 `preview/full_tables.md`。分数的最终核对依据是各任务的原始 JSON。

论文当前本地源码是 [Table 1](../27cvpr/tables/table1_future.tex) 和 [Table 2](../27cvpr/tables/table2_downstream.tex)。截至本次核对，论文主表仍保留 `TBD`，运行结果没有自动回填；这些链接供本地工作区查看。

`runs/` 下的原始数据、预测和日志不随研究笔记上传；本页保存聚合分数和来源，便于日后查看。

## 2. 什么时候做了这些实验

| 实验 | 时间与完成记录 | 当时实际做的内容 |
|---|---|---|
| Qwen0.8B，4＋4 slots | 2026-09-11 12:56 启动；21:19:14 状态为 complete | 分类、疾病列表、三器官分割、×2 超分；24,000 步 |
| Qwen0.8B，无 slots 对照 | 2026-09-11 启动；23:29:35 状态为 complete | 相同视觉前端、共同参数初始值、样本顺序与 24,000 步；直接读取完整 tokens |
| 六个原始 Qwen／MedGemma 零样本模型 | 2026-09-11 晚间执行；2026-09-12 00:15:11 协调器完成 | Table 1 未来预测七项指标；Table 2 当前分类、报告；另列派生疾病列表 QA |
| 六个冻结 VLM＋独立任务头 | 2026-09-12 09:26:22 协调器启动；12:14:54 最后一项完成 | 分割、×4 超分、解剖区域定位，以及独立的 82 对方向预测 |
| 本次结果溯源整理 | 2026-09-13 | 核对已有 JSON、时间与表格来源，补写本页 |

时间证据：4＋4 的启动时间见 [09-11 执行记录](0911_stage1_slot44_run.md)，两组完成时间分别取 [slots 状态](../code/medworld_stage1/runs/slot44_20260911/joint/status.json) 和 [无 slots 状态](../code/medworld_stage1/runs/qwen08_noslots_20260911/joint/status.json) 的 `updated` 字段。零样本完成时间取 [coordinator_status.json](../code/medworld_baselines/runs/raw_models_20260911/coordinator_status.json) 的 `finished`；该文件的 `started` 是最后一段 `remaining_models` 协调器的启动时间，不能当作全部模型首次启动时间。

密集任务时间取 [launch.json](../code/medworld_dense_baselines/runs/dense_20260912/launch.json)、[status.json](../code/medworld_dense_baselines/runs/dense_20260912/status.json) 与 [queue.json](../code/medworld_dense_baselines/runs/dense_20260912/queue.json)。09:26 是本次协调器的启动时间，准备数据和共享缓存的工作已在此前进行。最终 **55/55 complete，0 failed**：1 个共享 V-JEPA 缓存、6 个 VLM 缓存、6 个方向推理、42 个任务头训练。42 份最终任务指标均记录 `epochs=20`。Qwen0.8B 的特征缓存曾重试一次，最终失败数为零不等于全程没有重试。

原定交付目标为 **09-14 08:00**，实际完成于 **09-12 12:14:54**。[09-12 实验小结](0912_recent_experiments_summary.md)的队列快照只截至当天 **09:56**，因此其中“在跑／待完成”的表述是历史状态。

## 3. Table 1／Table 2 分别评什么

| 表格 | 任务／维度 | 指标 |
|---|---|---|
| Table 1：未来预测 | 未来临床状态 | AP ↑、AUROC ↑ |
| Table 1 | 疾病变化 | Transition F1 ↑、Direction F1 ↑ |
| Table 1 | 未来报告质量 | RadGraph F1 ↑、GREEN ↑ |
| Table 1 | 概率可靠性 | Brier ↓、ECE ↓ |
| Table 2：当前下游任务 | 分类 | AUROC ↑、AP ↑ |
| Table 2 | VQA | Accuracy ↑、Micro F1 ↑ |
| Table 2 | 当前报告生成 | RadGraph F1 ↑、CheXbert F1 ↑ |
| Table 2 | 定位 | mIoU ↑、Acc@IoU≥0.5 ↑ |
| Table 2 | 分割 | Dice pseudo ↑、Dice human ↑ |
| Table 2 | ×4 超分 | PSNR ↑、SSIM ↑ |

这是指标设计，不表示所有计划模型和任务均已完成。六模型未来预测七项、当前分类／报告及派生 QA 的数值已保存在 [09-12 实验小结](0912_recent_experiments_summary.md)，原始来源为 `code/medworld_baselines/runs/raw_models_20260911/<model>/test/metrics.json` 和单独的 `green_metrics.json`。

原始未来预测为 297 对／94 位患者，允许源图、当前报告、源时点 EHR 和请求时间段。后续 Direction F1 使用独立的 82 对／154 个 finding-scope 字段，只允许源图、源报告和时间段。因此，完整预览中虽然把方向分数并排放在 Table 1，也不能称为同一测试集、同一输入协议的八项结果。

| 模型 | 独立 82 对 Direction F1 ↑ | 无效输出数／82 |
|---|---:|---:|
| Qwen3.5-0.8B | 0.0000 | 82 |
| Qwen3.5-4B | 0.0652 | 66 |
| Qwen3.5-9B | 0.2793 | 2 |
| Qwen3.5-27B-FP8 | 0.3635 | 0 |
| MedGemma-1.5-4B | 0.1933 | 52 |
| MedGemma-27B（v1） | 0.2933 | 0 |

来源：`dense_20260912/<model>/direction_metrics.json`。无效回答保留在评分中；Qwen0.8B 的零分包含全部回答无效这一因素。

官方 VQA 仍未形成这轮的正式结果，已有疾病列表 QA 单独报告。新定位任务使用 Chest ImaGenome 人工解剖区域框、26 类查询，不能记为 MS-CXR 病灶短语定位。

## 4. 09-12 完成的分割与 ×4 超分结果

这轮研究的问题是：六个公开 VLM 的冻结图像 token 表示，配合不同图像分支和新训练的任务头，能否支持密集任务？VLM 和 V-JEPA 均冻结；每项任务、每种分支独立训练 20 epochs，不使用 Ours 的训练后 checkpoint。

所有分支均读取对应 VLM 最后语言层的图像 token hidden states，统一为 `[64,1024]`。图像分支分别为原始图像、冻结 V-JEPA 特征、V-JEPA＋新训练 adapter。这里没有训练 4＋4 状态 slots。具体结构、优化器及划分见 [09-12 密集任务协议](0912_frozen_vlm_dense_baselines.md)。

| 模型 | 图像分支 | Dice pseudo ↑ | Dice human ↑ | ×4 PSNR（dB）↑ | ×4 SSIM ↑ |
|---|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | image | 0.9180 | 0.7538 | 32.0256 | 0.9101 |
| Qwen3.5-0.8B | vjepa | 0.9192 | 0.7494 | 22.5191 | 0.7481 |
| Qwen3.5-0.8B | vjepa_adapter | 0.9332 | 0.7523 | 26.3109 | 0.7954 |
| Qwen3.5-4B | image | 0.9131 | 0.7525 | 31.9545 | 0.9097 |
| Qwen3.5-4B | vjepa | 0.9160 | 0.7484 | 22.4570 | 0.7481 |
| Qwen3.5-4B | vjepa_adapter | 0.9309 | 0.7519 | 26.3279 | 0.7959 |
| Qwen3.5-9B | image | 0.9153 | 0.7511 | 31.9712 | 0.9098 |
| Qwen3.5-9B | vjepa | 0.9169 | 0.7461 | 22.5098 | 0.7482 |
| Qwen3.5-9B | vjepa_adapter | 0.9312 | 0.7508 | 26.3283 | 0.7971 |
| Qwen3.5-27B-FP8 | image | 0.9120 | 0.7534 | 31.9696 | 0.9098 |
| Qwen3.5-27B-FP8 | vjepa | 0.9158 | 0.7511 | 22.5017 | 0.7486 |
| Qwen3.5-27B-FP8 | vjepa_adapter | 0.9300 | 0.7515 | 26.3056 | 0.7961 |
| MedGemma-1.5-4B | image | 0.9133 | 0.7606 | 31.9503 | 0.9097 |
| MedGemma-1.5-4B | vjepa | 0.9142 | 0.7538 | 22.5044 | 0.7489 |
| MedGemma-1.5-4B | vjepa_adapter | 0.9281 | 0.7550 | 26.2967 | 0.7961 |
| MedGemma-27B（v1） | image | 0.9098 | 0.7563 | 31.9561 | 0.9096 |
| MedGemma-27B（v1） | vjepa | 0.9125 | 0.7564 | 22.3369 | 0.7493 |
| MedGemma-27B（v1） | vjepa_adapter | 0.9282 | 0.7540 | 26.2648 | 0.7959 |
| Bicubic，无训练 | LR 图像 | — | — | 29.7421 | 0.8998 |

分割／SR 共用 4,096 train、249 validation、447 test。Pseudo Dice 是 CXAS 左右肺与心脏三器官的伪标签一致性；human Dice 是 Montgomery 138 张外部图像的人工双肺测试，无心脏标签。SR 的全部输入路径共用同一份 ×4 降采样 LR，HR 仅作目标。`image` 分支有 bicubic LR 残差，两个 V-JEPA 分支没有像素／bicubic 旁路。

解剖区域定位使用独立任务头，在固定的 2,598 个测试查询／50 位患者上评价：

| 模型 | 解剖区域 mIoU ↑ | Acc@IoU≥0.5 ↑ |
|---|---:|---:|
| Qwen3.5-0.8B | 0.6172 | 0.7429 |
| Qwen3.5-4B | 0.6070 | 0.7348 |
| Qwen3.5-9B | 0.6045 | 0.7290 |
| Qwen3.5-27B-FP8 | 0.6043 | 0.7294 |
| MedGemma-1.5-4B | 0.6308 | 0.7671 |
| MedGemma-27B（v1） | 0.6150 | 0.7444 |

上述分数逐项对应 `dense_20260912/<model>/{segmentation_image,segmentation_vjepa,segmentation_vjepa_adapter,sr_image,sr_vjepa,sr_vjepa_adapter,grounding_image}/metrics.json`；Bicubic 对应运行根目录的 `bicubic_metrics.json`。模型目录 ID 依次为 `qwen08b`、`qwen4b`、`qwen9b`、`qwen27b_fp8`、`medgemma4b`、`medgemma27b`。

这轮点估计中，adapter 分支的伪标签 Dice 较高，但人工双肺 Dice 没有同步提高；超分中图像分支约 32 dB，两个 V-JEPA 分支低于 Bicubic。各分支保留的像素路径不同，不能把差异单独归因于 VLM 模型规模或 slots。

## 5. 与旧 slots 对照、新多层 slots 实验的关系

09-11 的 4＋4 slots／无 slots 对照使用 Qwen0.8B，允许当前报告辅助读出，SR 为 ×2。两组 final checkpoint 都为 24,000 步，评估样本与目标一致：

| 方法 | 分类 AUROC ↑ | 分类 AP ↑ | 疾病列表 Macro F1 ↑ | Dice pseudo ↑ | ×2 PSNR（dB）↑ | ×2 SSIM ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen0.8B，无 slots | 0.9189 | 0.9867 | 0.5209 | 0.6143 | 38.0069 | 0.9742 |
| Ours，4＋4 slots | 0.9663 | 0.9931 | 0.4776 | 0.6079 | 38.0361 | 0.9743 |

来源为前述 `comparison_final.json`，设置见 [无 slots 对照](0911_qwen08_noslots_baseline.md)。这两行没有填入新 ×4 结果列；协议不同，分数不可直接混比。

另有 09-13 冻结多层视觉 slots 对照，本地说明为 `research_notes/0913_frozen_multidepth_slots.md`，运行目录为 `code/medworld_dense_baselines/runs/frozen_slots_20260913`，使用视觉塔不同深度汇聚的第 5–8 个 slots，只训练新 decoder。它与本页 09-12 的最后语言层图像 tokens＋三分支实验是两次独立运行；本页没有包含它的结果。
