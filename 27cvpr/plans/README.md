# Table 1 / Table 2 精简计划版

2026-09-11 更新：两张表均为 **10 行**，不显示 General VLMs、Medical VLMs 等方法类别标题。Table 1 已按用户确认固定为 **四个评价维度、每组两个指标，共八列**。已写入论文正文，并与独立两页预览共用同一份 TeX 源码。所有待跑分数为 `TBD`。

- 完整论文：[`../main.pdf`](../main.pdf)。
- 两页预览：[`table1_table2_plan.pdf`](table1_table2_plan.pdf)。
- Table 1 源码：[`../tables/table1_future.tex`](../tables/table1_future.tex)。
- Table 2 源码：[`../tables/table2_downstream.tex`](../tables/table2_downstream.tex)。
- 此目录 `tables/*_plan.tex` 仅转引共享源码，避免两份表格不一致。

## Table 1：10 行、四组八个指标（已确认）

| 选择 | 方法 |
| --- | --- |
| 通用 VLM ×2 | Qwen2.5-VL-7B；LLaVA-v1.6-Mistral-7B |
| 医学 VLM ×2 | LLaVA-Med-v1.5-7B；CheXagent-8B |
| 表征预测 ×2 | BioViL-T + predictor；CheXWorld + predictor |
| 预测任务参考 ×2 | Copy Current；Finding-transition prior |
| 同规模 baseline | Qwen3.5-0.8B direct |
| 完整方法 | MedWorld-JEPA |

| 宏观评价维度 | 指标一 | 指标二 |
|---|---|---|
| Future clinical status | Finding macro AP ↑ | macro AUROC ↑ |
| Disease progression | Transition macro F1 ↑ | Direction macro F1 ↑ |
| Future report fidelity | RadGraph partial F1 ↑ | GREEN ↑ |
| Probabilistic reliability | macro Brier score ↓ | classwise ECE ↓ |

Transition F1 对有真实支持的“征象×新发／消退事件”F1 等权宏平均。Direction F1 在参考确认异常前后持续存在的字段上，评价改善／稳定／加重，按疾病和侧别绑定至源检查。两个指标均计入合格稳定病例上的误报；各自报告参考覆盖率。Direction 的标签规则与人工复核仍待完成。

Onset／Resolution 分解、CheXbert F1、Future R@1、程度误差和 horizon 分层放补充分析，不增加主表列。四组八指标是当前执行标准，后续工作以标签验收、评分接入及实验为主。设计依据见[指标方案与文献核对](../../research_notes/0911_future_state_evaluation_redesign.md)。

四个公开 VLM 使用零样本预测提示词，并标记上标 ZS。表征模型需训练融合、预测器和读出；0.8B direct 与 Ours 用同一训练队列。这些训练设置在正文和表下注明，不以分组标题占据表内空间。

Finding-transition prior 是新计划的简单统计预测参考：仅用训练集估计未来征象在“当前标签＋horizon”条件下的平滑阳性概率；当前标签未知时使用该 horizon 的边际先验。阈值在验证集固定。它输出征象列表与概率，可评价 AP／AUROC／Transition／Brier／ECE；Direction 和报告生成列留破折号。Copy Current 没有连续分数接口，AP／AUROC／Brier／ECE 留破折号，变化评分使用保持当前状态的预测语义。

VLM 的逐征象分数来自 Yes/No 条件似然，通常无需新增分类头。单 token 答案读取 logits；多 token 答案计算完整候选 log-likelihood。两候选必须都能评分，不能把 top-k 未返回视为零概率。AP 是非插值 Average Precision，AP／AUROC 评价排序；Brier／ECE 评价相同分数作为概率预测的表现。主表报原始概率；验证集拟合的校准后结果另报。ECE 按征象使用固定十个等宽箱后宏平均。参考掩码和支持类别在各模型间一致。

## Table 2：10 行、六任务

| 选择 | 方法 |
| --- | --- |
| 通用 VLM ×2 | Qwen2.5-VL-7B；LLaVA-v1.6-Mistral-7B |
| 医学 VLM ×2 | LLaVA-Med-v1.5-7B；CheXagent-8B |
| 视觉表征 ×2 | DINOv2；CheXWorld |
| 专用方法 ×2 | MAIRA-2；SwinIR |
| 匹配 baseline | Qwen3.5-0.8B no-slots |
| 完整方法 | MedWorld-JEPA |

| 任务 | 数据 | 主表指标 |
| --- | --- | --- |
| 疾病/征象分类 | MIMIC finding 标签 | macro AUROC、AP |
| 标准 VQA | MIMIC-Ext-MIMIC-CXR-VQA | 官方 Accuracy、micro F1 |
| 当前报告生成 | MIMIC-CXR | RadGraph partial F1、CheXbert macro-positive F1 |
| 病灶定位 | MS-CXR | 统一框匹配协议下的 mIoU、Acc@IoU≥0.5 |
| 器官分割 | 当前 CXAS＋待选人工掩膜测试集 | pseudo Dice、human Dice 分开 |
| 超分 | 同一胸片合成空间 ×4 退化 | PSNR、SSIM |

共 12 个结果列；多个指标/测试集不计为新增任务。时序变化理解不进入 Table 2。

DINOv2/CheXWorld 使用冻结视觉编码器加监督任务头。MAIRA-2 保留原生报告生成/定位接口，SwinIR 适配相同 CXR ×4 退化。no-slots/Ours 进行匹配的六任务适配。普通文本输出 VLM 的原生行不评测分割/SR；破折号表示本行不评测，不代表永远不能适配。

## 执行条件

1. 固定跨 Stage、跨任务患者划分，排查官方 VQA、MS-CXR 等测试患者与所有训练来源的重叠；披露公开权重可追溯的预训练重叠。
2. 固定 Table 1 源时点截止；额外读取的未来检验/治疗不能作为输入。统一征象 schema、方向参考和临床评价器；报告生成与 Yes/No 评分使用相同允许证据。新增 Direction／GREEN／AUROC／Brier／ECE 需按八指标协议接入并验证，补充检索使用共同候选。
3. 新六任务评测使用图像＋问题/短语，不向状态提供同次目标报告；因此需要新增读出与配套适配训练，不能搬入旧四任务报告辅助成绩。
4. 为人工掩膜测试确定数据集、split 和器官对应；与 CXAS Dice 分开。所有 SR 方法采用相同空间 ×4 退化、强度、裁边与 padding 规则。
5. no-slots/Ours 同步训练和更改解码器，保持任务监督和预算匹配。零样本、冻结表征、原生专用方法与监督多任务模型的训练差异在正文明确。
6. 固定定位热图转框、多个框匹配和无效输出规则；VQA 附录给出 Verify/Choose/Query、区域/全图、macro F1 和 Exact Match，主要差值按患者配对 bootstrap。

## 编译

在 `27cvpr` 下执行：

```bash
make
make plan-preview
```

候选模型与数据的官方资料见 [此前调研](../../research_notes/0911_table_expansion_proposal.md)。该调研保留较大的候选池；最终本版主表名单以上面的每表 10 行为准。
