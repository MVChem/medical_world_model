# Table 1 / Table 2 结果与执行计划

2026-09-15 更新：已把核验过的相关指标写入共享源码，包括 **Qwen3.5-9B** full-token、slots、shuffled state 三组新完成的预测与 GREEN，以及 CheXagent 的两表分类／报告指标和预测 GREEN。Qwen 零样本参考也使用 **Qwen3.5-9B**。Table 1 为 **11 行、四组八个指标**；Table 2 为 **13 行、六任务十二个指标**，均不显示方法类别标题。论文正文与独立预览共用同一份 TeX 源码。尚未完成的指标为 `TBD`；本行不评测的接口为破折号。

- 完整论文：[`../main.pdf`](../main.pdf)。
- 两页预览：[`table1_table2_plan.pdf`](table1_table2_plan.pdf)。
- Table 1 源码：[`../tables/table1_future.tex`](../tables/table1_future.tex)。
- Table 2 源码：[`../tables/table2_downstream.tex`](../tables/table2_downstream.tex)。
- 此目录 `tables/*_plan.tex` 仅转引共享源码，避免两份表格不一致。
- [全精度数值及来源哈希](../tables/results_20260914.json)（保留兼容文件名，内部快照日期为 2026-09-15）；[本次回填记录](../../research_notes/0915_table_results_filled.md)。

两张表的分数统一 **乘以 100、保留两位小数**（如 `0.855531` 显示为 `85.55`），PSNR 保留 dB 单位并显示两位小数。表注明确标示缩放规则；Brier／ECE 仍为越低越好。该规则仅影响展示，原始实验结果与汇总 JSON 保留原始尺度及完整精度。

## Table 1：11 行、四组八个指标

| 选择 | 方法 |
| --- | --- |
| 通用 VLM ×2 | Qwen3.5-9B；LLaVA-v1.6-Mistral-7B |
| 医学 VLM ×2 | LLaVA-Med-v1.5-7B；CheXagent-8B |
| 表征预测 ×2 | BioViL-T + predictor；CheXWorld + predictor |
| 预测任务参考 ×2 | Copy Current；Finding-transition prior |
| 同规模预测对照 | Full-token forecaster (Qwen3.5-9B) |
| slots 预测适配 | MedWorld-JEPA (Qwen3.5-9B slots) |
| 状态打乱对照 | Qwen3.5-9B (shuffled state) |

| 宏观评价维度 | 指标一 | 指标二 |
|---|---|---|
| Future clinical status | Finding macro AP ↑ | macro AUROC ↑ |
| Disease progression | Transition macro F1 ↑ | Direction macro F1 ↑ |
| Future report fidelity | RadGraph partial F1 ↑ | GREEN ↑ |
| Probabilistic reliability | macro Brier score ↓ | classwise ECE ↓ |

Transition F1 对有真实支持的“征象×新发／消退事件”F1 等权宏平均。Direction F1 在参考确认异常前后持续存在的字段上，评价改善／稳定／加重，按疾病和侧别绑定至源检查。两个指标均计入合格稳定病例上的误报；各自报告参考覆盖率。Direction 的标签规则与人工复核仍待完成。

Onset／Resolution 分解、CheXbert F1、Future R@1、程度误差和 horizon 分层放补充分析，不增加主表列。四组八指标是当前执行标准，后续工作以标签验收、评分接入及实验为主。设计依据见[指标方案与文献核对](../../research_notes/0911_future_state_evaluation_redesign.md)。

四个公开 VLM 采用零样本预测接口，标记 ZS；Qwen3.5-9B、LLaVA-v1.6、LLaVA-Med、CheXagent 均已有预测结果及 GREEN。训练模型使用 16,000 对训练、230 对验证、297 对测试（94 位患者）。9B 三条件已从新训练的共同 9B Stage-1 初始化完成 2,400 updates、effective batch 32、seed 42 的预测训练，测试和 GREEN 均完成 297/297；整批 8/8 作业于北京时间 9 月 15 日 04:55 完成。BioViL-T/CheXWorld 适配版已用相同 forecast 更新预算完成主要指标及 GREEN。它们冻结官方视觉编码器，接入共享融合、预测及读出模块，并沿用原有共同 Stage-1 初始化；不能称为原论文原生预测方法的完整复现。

**Table 1 的 9B slots 适配仍采用八个最后语言层查询，与正文计划的多层 4+4 slots 分开。** Full-token forecaster 保留有效图像／报告／EHR tokens，仍有共同预测器；它与原生 9B 零样本接口分开。跨患者打乱另列一行。9B slots 相比 full-token 的 Transition F1 为 28.51 对 16.03，AUROC 为 77.94 对 76.42，Brier／ECE 更低，但 AP、RadGraph 和 GREEN 略低。Shuffled 的 AUROC 降至 52.85、GREEN 为 8.87，297 对只生成 1 个不同报告，CheXbert F1 为 0；其 ECE 4.97 不能单独支持预测质量更好的结论。以上均为单 seed 点估计。原始 0.8B 数值和观察保留在[升级记录](../../research_notes/0914_qwen9b_forecast_upgrade.md)。

Finding-transition prior 仅用训练集估计未来征象在“当前标签＋horizon”条件下的平滑阳性概率，固定 Laplace α=1；当前标签未知时使用该 horizon 的边际先验。它读取准备好的结构化源标签，属于额外明确的源状态接口，不是 VLM 预测标签。AP／AUROC／Brier／ECE 已完成；阈值选择和 Transition 仍待评分。Direction 和报告生成列留破折号。Copy Current 没有连续分数接口，AP／AUROC／Brier／ECE 留破折号，变化评分使用保持当前状态的预测语义。

VLM 的逐征象分数来自 Yes/No 条件似然，通常无需新增分类头。单 token 答案读取 logits；多 token 答案计算完整候选 log-likelihood。两候选必须都能评分，不能把 top-k 未返回视为零概率。AP 是非插值 Average Precision，AP／AUROC 评价排序；Brier／ECE 评价相同分数作为概率预测的表现。主表评估未校准概率，最终指标按上述 ×100 规则展示；验证集拟合的校准后结果另报。ECE 按征象使用固定十个等宽箱后宏平均。参考掩码和支持类别在各模型间一致。

## Table 2：13 行、六任务

| 选择 | 方法 |
| --- | --- |
| 通用 VLM ×2 | Qwen3.5-9B；LLaVA-v1.6-Mistral-7B |
| 医学 VLM ×2 | LLaVA-Med-v1.5-7B；CheXagent-8B |
| 视觉表征 ×2 | DINOv2；CheXWorld |
| 专用方法 ×2 | MAIRA-2；SwinIR |
| 已完成密集任务对照 ×3 | Qwen3.5-9B 冻结视觉 slots；打乱视觉 slots；Image-only |
| 匹配 baseline | Qwen3.5-0.8B no-slots |
| 完整方法 | MedWorld-JEPA |

| 任务 | 数据 | 主表指标 |
| --- | --- | --- |
| 疾病/征象分类 | MIMIC finding 标签 | macro AUROC、AP |
| 标准 VQA | MIMIC-Ext-MIMIC-CXR-VQA | 官方 Accuracy、micro F1 |
| 当前报告生成 | MIMIC-CXR | RadGraph partial F1、CheXbert macro-positive F1 |
| 病灶定位 | MS-CXR | 统一框匹配协议下的 mIoU、Acc@IoU≥0.5 |
| 器官分割 | CXAS 三器官＋Montgomery 138 张人工双肺掩膜 | pseudo Dice、human Dice 分开 |
| 超分 | 同一胸片合成空间 ×4 退化 | PSNR、SSIM |

共 12 个结果列；多个指标/测试集不计为新增任务。时序变化理解不进入 Table 2。

DINOv2/CheXWorld 使用冻结视觉编码器加监督任务头。MAIRA-2 计划保留原生报告生成/定位接口，权重访问尚未解决；SwinIR 已适配相同 CXR ×4 退化。普通文本输出 VLM 的四个 ZS 行均已填写分类和报告指标，包括 9 月 14 日完成补跑的 CheXagent；分割/SR 留破折号。9B 冻结原生视觉塔的四个多层视觉 slots、打乱条件与 image-only 独立成行，不合并进 ZS 或完整 Ours。

主表密集任务统一使用 **4,096/249/447 张训练／验证／测试图，20 epochs、batch 8、seed 20260913**。人工双肺为独立 Montgomery 138 张图，无心脏标签。18,708 张扩大数据版本不混入本次主表。分类测试为 353 张；DINOv2/CheXWorld 独立分类头使用 13,681 张训练、160 张验证、20 epochs。报告生成测试为 507 张。原始分类字段 `macro_auprc` 已核验是非插值 AP。

原计划的 0.8B no-slots／完整 Ours 两行仍为 `TBD`，等待匹配六任务适配；旧报告辅助四任务结果和另外的两轮联合 dense pilot 不填入这些行。正式 VQA 不能以派生疾病列表 QA 代填；MS-CXR 病灶短语定位不能以 Chest ImaGenome 解剖区域定位代填。

当前只在所述协议内匹配数据、轮数／更新数和 GPU 型号，**没有匹配实际 GPU 小时或预训练数据**。SwinIR 4,096 张训练图版本训练加评估为 8.313 GPU-h，image-only SR 为 0.161 GPU-h，不得宣称等算力优势。所有数值为点估计，尚无配对置信区间。

## 执行条件

1. 固定跨 Stage、跨任务患者划分，排查官方 VQA、MS-CXR 等测试患者与所有训练来源的重叠；披露公开权重可追溯的预训练重叠。
2. Table 1 仅允许源时点截止前证据；已接入 GREEN／AUROC／Brier／ECE。Direction 在 297 对 cohort 上的标签规则和人工复核仍待完成；独立 82 对方向实验输入协议不同，不填入此列。补充检索使用共同候选。
3. 新六任务评测使用图像＋问题/短语，不向状态提供同次目标报告；因此需要新增读出与配套适配训练，不能搬入旧四任务报告辅助成绩。
4. 人工掩膜已确定为 Montgomery 双肺测试，与 CXAS 三器官 Dice 分开。所有 SR 方法采用相同空间 ×4 退化、强度、裁边与 padding 规则。
5. no-slots/Ours 同步训练和更改解码器，保持任务监督和预算匹配。零样本、冻结表征、原生专用方法与监督多任务模型的训练差异在正文明确。
6. 固定定位热图转框、多个框匹配和无效输出规则；VQA 附录给出 Verify/Choose/Query、区域/全图、macro F1 和 Exact Match，主要差值按患者配对 bootstrap。

## 编译

`populate_results.py` 从本次固定的原始指标重建共享表格，并校验样本数、训练轮数和 GREEN 完成状态。底部三行只读取 `code/medworld_table1/runs/qwen9b_ablation_20260914/`：必须核验 9B 配置、2,400 步训练完成状态、297 对生成与最终指标，才能回填已完成的临床指标；GREEN 单独要求完整 297/297。缺失或仍在训练的条件保留全部 `TBD`。

在 `27cvpr` 下执行：

```bash
python plans/populate_results.py
make
make plan-preview
```

候选模型与数据的官方资料见 [此前调研](../../research_notes/0911_table_expansion_proposal.md)。该调研保留较大的候选池；本版主表名单以上述 11／13 行为准。历史结果及不同协议的边界见 [09-13 结果溯源](../../research_notes/0913_table1_table2_results_provenance.md)。
