# 上一轮 Stage 1 下游任务指标汇总

整理日期：2026-09-11。实验：`overnight_20260910`，两个运行均已完成。以下数值直接从原始 `metrics.json`、`clinical.json`、`data_usage.json` 和状态文件汇总；本次没有重新训练或重新评分。

**这份结果属于上一轮“全部任务读取全部 8 个 slots”的实现。新的 4＋4 分配尚无训练结果。** 上一轮的 `diagnosis` 实际是 Findings/Impression 报告生成，本文统一标为“报告生成”；Fig1 的 Disease recognition 尚未单独实现。

[新的 slots 分配计划](0911_stage1_slot_allocation_plan.md) · [主指标 CSV](assets/stage1_20260911_metrics/main_metrics.csv) · [完整精度汇总与来源校验值](assets/stage1_20260911_metrics/summary.json)

**主结果**

| 任务 | 实际测试规模 | 指标 | 更新共享 encoder（joint） | 冻结共享 encoder 对照 |
|---|---:|---|---:|---:|
| Classification | 196 张 | macro AUPRC / AUROC | 0.9911 / 0.9715 | 0.9126 / 0.8182 |
| 报告生成（旧代码名 diagnosis） | 64 张 | CheXbert macro-positive F1 / RadGraph partial F1 | 0.6114 / 0.4444 | 0.2869 / 0.2736 |
| Segmentation | 236 张 | 三器官平均 Dice | 0.5588 | 0.5671 |
| Super resolution | 283 张 | PSNR / SSIM | 38.09 dB / 0.9731 | 38.20 dB / 0.9736 |
| Fig1 Disease recognition | 未评估 | 标准疾病列表指标 | 未单独实现 | 未单独实现 |

两个运行都使用 `checkpoint_final.pt` 评估。表中的“更新共享 encoder”就是之前简称的“更新 slots”：训练视觉 adapter、视觉位置参数、初始 tokens U 和 encoder LoRA，同时训练任务 decoder；V-JEPA 和 Qwen 基础权重冻结。对照则冻结整个状态 encoder，只训练四个 decoder。

因此，两列都使用患者自己的 slots。冻结 encoder 并不意味着每个患者的 S 相同；相同编码器处理不同图像／报告仍得到不同 S。这张表比较的是状态编码器参与训练的效果，不能当作“有 slots 对无 slots”的比较。

在这次测试中，joint 的分类和报告生成指标高于冻结 encoder 对照；分割平均 Dice 和 SR 指标略低。没有多次随机种子或对应差值置信区间，不能仅凭这些点估计宣称统计显著。

| 任务 | 指标 | joint 减去冻结 encoder 对照的绝对差值 |
|---|---|---:|
| Classification | macro_auprc | +0.0784 |
| Classification | macro_auroc | +0.1533 |
| Report generation | chexbert_macro_positive_f1 | +0.3245 |
| Report generation | radgraph_partial_f1 | +0.1708 |
| Segmentation | mean_dice | -0.0082 |
| Super resolution | psnr | -0.1083 |
| Super resolution | ssim | -0.0005 |

**数据与训练规模**

| 数据划分 | 图像 / 患者 | 分类池 | 报告生成池 | 分割有效池 | SR 池 |
|---|---:|---:|---:|---:|---:|
| train | 24,000 / 17,260 | 13,837 | 18,994 | 18,917 | 24,000 |
| validate | 320 / 320 | 161 | 246 | 251 | 320 |
| test | 283 / 283 | 196 | 235 | 236 | 283 |

这是从完整 377,110 张 CXR 候选中抽取的子集，并非全量训练。患者划分互斥，每个选定 study 最多一张图。分割采用质控后的有效池；旧总览中的 20,000 / 256 / 256 是候选数量。报告生成 test 池有 235 张，实际只生成并计算了前 64 张的临床指标。

joint 和对照都训练了 22,616 次 optimizer 更新，四个任务各 5,654 次。实际样本呈现次数分别为分类 90,464、报告生成 45,232、分割 90,464、SR 90,464，这些不是独立图像数。joint 的训练计算时间为 7.90 小时，对照为 4.88 小时，不等于包含准备、等待和评估的总墙钟时间。

**各项指标的含义**

Classification 使用 study 级报告提取标签，只有明确的 0/1 参与评估；不确定和未提及被屏蔽。13 个输出类别中，test 上有 10 类同时有阳性和阴性并进入 macro 均值。Fracture、Lung Lesion、Pleural Other 没有有效阴性，因此记为 NA。某些类别的有效样本明显偏阳性，AUPRC 的基准水平较高。报告已经输入 S，因此 0.9911 是本设置下的 macro AUPRC，不是纯图像诊断的 99.11% 准确率。

报告生成的 CheXbert 分数是本地实现的 `macro_positive_f1`：对参考报告中明确 0/1 的类别标签计算正类 F1，再平均参考集中存在阳性的类别；不应笼统写成与任意论文协议等价的标准 CheXbert 分数。RadGraph 使用 `radgraph-xl` 的 partial F1，按报告平均。两个模型均生成了 64 条非空、互不完全相同的文本；这些检查不能替代内容正确率。

Segmentation 的监督和评价对象是冻结 CXAS 的右肺、左肺、心脏伪标签，下面是与教师的一致性，不是人工 mask 准确率：

| 器官 | joint Dice | 冻结 encoder 对照 Dice |
|---|---:|---:|
| 右肺 | 0.5921 | 0.6032 |
| 左肺 | 0.5908 | 0.5988 |
| 心脏 | 0.4937 | 0.4992 |

SR 的输入为合成 LR，长宽各缩小 2 倍，重建后长宽各扩大 2 倍，即总像素 ×4。HR 长边上限 512，保持长宽比并只在有效区域评分。PSNR/SSIM 先逐图计算再平均。bicubic 在同一 283 张图上的结果为 **34.4239 dB / 0.9638**；joint 相对 bicubic 的 PSNR 差值为 **3.6644 dB**，不能将整个差值归因于 slots。SR 的 S 来自 LR 图像＋当前报告，属于报告辅助的合成降采样恢复。

分类和报告生成池均有有效提取报告。分割训练／测试中分别有 885／19 张没有有效提取报告，SR 中分别有 1,109／23 张；这些图像使用空文本输入，不能描述为每个样本都有图像＋报告。

**已有的固定表征线性 probe**

从两个训练结果分别提取 slots 并冻结，用相同的 1,024 张训练图像和 400 次更新重新拟合线性分类头；输入为全部 8 个 slots 经 LayerNorm 后展平。它衡量这两套表征在本分类数据上的可读出程度，不是新的诊断任务，也不是 4＋4 分组结果。

| 固定表征来源 | 测试图像 | 新线性头 macro AUPRC | 新线性头 macro AUROC |
|---|---:|---:|---:|
| joint | 196 | 0.9872 | 0.9670 |
| 冻结 encoder 对照 | 196 | 0.8522 | 0.7492 |

旧运行已经保存了清零和跨患者交换 slots 的推理干预结果；它们保存在原始 JSON 和本页的机器可读汇总中。本次仅归档现有数据，新计划不安排这些消融。

**分类逐类支持度与结果**

| 标签 | 有效阳性 / 阴性 | joint AUPRC / AUROC | 对照 AUPRC / AUROC |
|---|---:|---:|---:|
| Atelectasis | 57 / 3 | 0.9985 / 0.9708 | 0.9793 / 0.6608 |
| Cardiomegaly | 58 / 11 | 1.0000 / 1.0000 | 0.9443 / 0.7555 |
| Consolidation | 15 / 9 | 0.9860 / 0.9741 | 0.9334 / 0.8593 |
| Edema | 35 / 24 | 1.0000 / 1.0000 | 0.9164 / 0.9077 |
| Enlarged Cardiomediastinum | 11 / 5 | 0.9723 / 0.9273 | 0.8872 / 0.7636 |
| Fracture | 9 / 0 | NA / NA | NA / NA |
| Lung Lesion | 15 / 0 | NA / NA | NA / NA |
| Lung Opacity | 64 / 6 | 0.9954 / 0.9531 | 0.9610 / 0.7917 |
| Pleural Effusion | 63 / 27 | 0.9948 / 0.9888 | 0.8667 / 0.8007 |
| Pleural Other | 2 / 0 | NA / NA | NA / NA |
| Pneumonia | 21 / 25 | 0.9666 / 0.9762 | 0.8202 / 0.8514 |
| Pneumothorax | 7 / 47 | 1.0000 / 1.0000 | 0.8231 / 0.9331 |
| Support Devices | 60 / 2 | 0.9974 / 0.9250 | 0.9948 / 0.8583 |

**原始文件与查看入口**

- [旧运行总览](../code/medworld_stage1/runs/overnight_20260910/REPORT.md)。其中的 Diagnosis 名称按本页更正理解为报告生成。
- 主实验：[metrics.json](../code/medworld_stage1/runs/overnight_20260910/joint/evaluation/metrics.json)、[clinical.json](../code/medworld_stage1/runs/overnight_20260910/joint/evaluation/clinical.json)、[data_usage.json](../code/medworld_stage1/runs/overnight_20260910/joint/data_usage.json)。
- 冻结 encoder 对照：[metrics.json](../code/medworld_stage1/runs/overnight_20260910/frozen_control/evaluation/metrics.json)、[clinical.json](../code/medworld_stage1/runs/overnight_20260910/frozen_control/evaluation/clinical.json)、[data_usage.json](../code/medworld_stage1/runs/overnight_20260910/frozen_control/data_usage.json)。
- 主实验已有样例：[报告生成记录](../code/medworld_stage1/runs/overnight_20260910/joint/evaluation/diagnosis.jsonl)、[分割样例](../code/medworld_stage1/runs/overnight_20260910/joint/evaluation/seg_00.png)、[SR 样例](../code/medworld_stage1/runs/overnight_20260910/joint/evaluation/sr_00.png)。
- 主要结果对应 checkpoint：[joint](../code/medworld_stage1/runs/overnight_20260910/joint/checkpoint_final.pt)、[冻结 encoder 对照](../code/medworld_stage1/runs/overnight_20260910/frozen_control/checkpoint_final.pt)。
