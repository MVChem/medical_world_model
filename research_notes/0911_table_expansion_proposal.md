# Table 1 / Table 2 扩展方案（2026-09-11）

这是候选实验设计，不是已完成结果。依据当前 `27cvpr/sections/6_results_analysis.tex`、Table 1 评测代码和官方模型/数据说明整理。

后续已按用户要求精简：Table 1 / Table 2 各 10 行，每类选两个参照，最后保留一个匹配 baseline 和 Ours；删除表内分类分组标题。新版已写入论文正文，统一源码为 `../27cvpr/tables/table1_future.tex` 与 `../27cvpr/tables/table2_downstream.tex`，最终名单见 `../27cvpr/plans/README.md`。下文保留此前扩展候选池供参考，不能将 14–21 行版本视为当前主表。

此前扩展阶段讨论的范围是：Table 1 约 12–15 个方法/训练设置；Table 2 采用分类、标准 VQA、当前报告生成、病灶定位、器官分割和空间 ×4 超分六类任务。用户明确移除时序变化理解，变化预测保留在 Table 1。随后方法行经过上述精简；两页计划表位于 `../27cvpr/plans/table1_table2_plan.pdf`。新增模型行与新增任务列各自回答不同问题，不把同一 VQA 的多个指标计成多个独立任务。

**09-12 执行入口：** 实际已完成的六模型零样本评测、4＋4／无 slots 比较与新密集任务队列见[实验小结](0912_recent_experiments_summary.md)。Table 1 最终四组八指标见[评价方案](0911_future_state_evaluation_redesign.md)；下文旧五指标和候选模型池保留为讨论历史，不作为当前执行清单。

## 当前基础

- Table 1：7 行、5 个指标，预测未观察到的随访状态/报告。输入为源时点胸片、当前报告、截止源时点的 EHR 和预测时间段。
- Table 2：6 行，分类、疾病列表识别、三器官分割、合成超分 4 类任务。
- 当前 Table 2 编码器包含同次报告；分类/识别属于报告辅助读出。分割参考是 CXAS 伪标签；SR 每个空间维度 ×2。
- `code/medworld_table1/evaluate_qwen9b.py` 已实现源时点输入的 Qwen3.5-9B 零样本报告预测接口。当前输出 `scores=None`，没有连续 finding 分数；已有通用评测器可复用，其他模型仍需各自的输入/加载适配。

## Table 1：模型与训练设置

主问题：相同源时点证据下，显式状态建模和未来预测是否优于直接 VLM 预测？

| 类别 | 候选行 | 如何接入 | 优先级 |
| --- | --- | --- | --- |
| 简单基线 | Copy Current | 复制当前报告/征象，保留稳定病情参照 | 必须 |
| 简单基线 | Finding transition prior | 仅用训练集按当前征象和 horizon 估计未来阳性概率；需实现 | 可选，成本低 |
| 同规模直接预测 | Qwen3.5-0.8B direct | 保留已有小模型直接预测对照；训练输入与预算匹配 | 必须 |
| 通用 VLM | LLaVA-v1.6-Mistral-7B | 官方 checkpoint + 同一预测指令；零样本适配接口 | 第一批 |
| 通用 VLM | Qwen2.5-VL-7B-Instruct | 同上；可再做同数据 LoRA | 第一批 |
| 通用 VLM | InternVL3.5-8B | 同上，模型家族不同 | 第二批 |
| 通用 VLM | Qwen3.5-9B | 复用本地已写的预测接口 | 第一批 |
| 医学 VLM | LLaVA-Med-v1.5-Mistral-7B | 指令生成未来报告；本任务零样本与微调分别标记 | 第一批 |
| 医学 VLM | MedGemma-1.5-4B-IT | 医学对话模型；可接预测指令，预测性能需实测 | 第一批 |
| 胸片 VLM | CheXagent-8B | 胸片专用指令模型；先验证预测 prompt 的服从情况 | 第一批 |
| 时序表征适配 | BioViL-T + matched predictor/readouts | 重新训练预测器/融合/读出；不是原论文直接未来预测成绩 | 第二批 |
| 医学表征适配 | CheXWorld + matched predictor/readouts | 重新训练；CheXWorld 原任务是表征学习，不是患者未来预测 | 可选 |
| 内部结构对照 | Frozen Stage 1 state + matched LWM | 匹配未来预测器和 readouts，明确与 Ours 的训练区别 | 必须 |
| 完整方法 | Ours | 当前拟议完整方法 | 必须 |

以上为 14 行候选，不承诺都已可运行。B 级适配行若引入相同的完整融合/slot 模块，应命名为 encoder-swap adaptation，不能当作独立复现原论文。保持行的差异真实，避免同一系统改名重复计数。

建议划分 Zero-shot 与 Task-adapted 两个 panel。参数规模与是否使用本任务训练集是两个独立因素：7–8B 零样本对比用于展示公开模型能力；同数据微调的强 VLM 与同规模 direct baseline 才是验证方法收益的核心。第一批微调优先 Qwen2.5-VL-7B、LLaVA-Med-7B 或 CheXagent 中至少一个医学模型。

参数列记录名义模型规格、实际总参数、可训练参数；7B/8B 名称可能主要指语言骨干。Ours 的 0.8B 是语言模型规格，不等于整套模型总参数。匹配训练样本和更新数，另报告 GPU 时间/显存；不能将相同更新数称为相同计算量。

### 外部方法如何放置

| 方法 | 核实到的能力 | 适合位置/限制 |
| --- | --- | --- |
| MAIRA-2，Vicuna-7B 骨干 | 当前报告生成、带框报告、phrase grounding；允许当前胸片和先前检查信息 | 优先 Table 2 报告/定位。原生接口不是未来预测，也不是任意问题通用聊天接口；进 Table 1 需单独做 forecasting adaptation |
| BioViL-T | 时序图文预训练、分类/定位等表征任务 | Table 2 时序变化或定位；Table 1 需附加预测器并重训 |
| CheXWorld，CVPR 2025 | CXR world representation，分类和分割等 | Table 2 表征组；Table 1 只能标适配 |
| RadZero，NeurIPS 2025 | 胸片分类、定位、零样本分割 | Table 2 专用参考；具体器官/病灶标签需要对齐 |
| EHRXDiff，CHIL 2025 | 从旧胸片和之后发生的医疗事件生成后续胸片，已有公开权重 | 相关性高，但其原始事件时间窗超过我们的源时点。截断输入并重新适配才可公平比较；图像输出还需统一临床读出，不能直接填报告指标 |
| LUMEN，ISBI 2026 / arXiv:2602.21142 | NVILA-8B、多图指令训练，差异问答和预测问答 | 候选相关工作。本文明确提到两图训练和指令数据待 PhysioNet 批准；本次未确认可下载官方完整 checkpoint。需核实预测题的图像时点，不能默认满足 source-only 协议 |

### Table 1 指标

保留现有 Future R@1、Finding AUPRC、Transition F1、RadGraph partial F1、CheXbert macro-positive F1。扩展优先选 onset F1 / resolution F1 的分解，以及现有四个 horizon 的分层结果，避免增加含义相近的文本指标挤满主表。

- 只要生成了符合任务的未来报告，就可以复用共同的 RadGraph、CheXbert 和报告征象解析器；R@1、Transition F1 同样需要统一解析和候选构造。
- AUROC/AUPRC 需要连续排序分数。可研究固定 yes/no 答案的条件对数似然，统一模板/候选归一化并在验证集确定规则；它是模型分数，不自动等于校准的患病概率。现有 `scores=None` 的行应保留缺失，不能拿模型写出的“80%”或解析后的 0/1 硬标签顶替。
- 固定未来候选集合、非本患者干扰样本、平分规则和 horizon。不要对 Ours 使用私有潜空间检索、对 VLM 使用另一种文本相似度再横向比较。
- 所有输入在源时点截止。未来图像/报告用于目标，目标窗口的检验、治疗、用药不能进入预测输入。
- 加一项病情保持不变的样本上的 false-change rate，有助于判断模型是否过度报告变化。各分层是同一个预测任务，不称新增任务。

## Table 2：已确定的六类任务

| 任务 | 数据/输入 | 主指标建议 | 新增工作和作用 |
| --- | --- | --- | --- |
| 1. Finding classification | 现有 MIMIC 标签；扩展标准 image-only 设定 | macro AUROC、macro AP | 保留；报告辅助设定单列，外部集评测另报 |
| 2. 标准 VQA | MIMIC-Ext-MIMIC-CXR-VQA；图像 + 问题 | 官方 Overall Accuracy / micro F1；分类型补充指标 | 用标准任务扩展当前 disease-list query；不仅换数据路径 |
| 3. 当前报告生成 | MIMIC-CXR；图像，可按协议给 indication/先前报告 | RadGraph partial F1、CheXbert macro-positive F1 | 不提供本次目标 Findings/Impression；检验状态是否保留完整征象信息 |
| 4. Phrase grounding | MS-CXR v1.1.0；图像 + 病灶描述 | mIoU；框输出协议可加 Acc@IoU≥0.5 | 新定位输出/训练，检验空间信息。热图与框方法需固定统一转换及评测定义 |
| 5. 器官分割 | 当前 CXAS；优先补人工掩膜外部测试 | Dice，附录逐器官 Dice/IoU | 当前结果应标明伪标签；先处理已发现的解码器问题，再比较扩展基线 |
| 6. 超分 | 同一 CXR 合成退化，主比较可新增空间 ×4 | PSNR、SSIM；LPIPS 可作补充 | ×4 即边长 1/4 输入，总像素 1/16；保留 ×2 参照，所有方法退化/裁边/灰度范围一致 |

VQA 是一个任务家族，建议按下面的既定切片展示：

| 切片 | 问题例子（示意） | 指标 |
| --- | --- | --- |
| Verify | 是否存在左侧胸腔积液？ | Accuracy，附 balanced accuracy / positive F1 |
| Choose | 异常位于左肺还是右肺？ | 按官方 answer-set 定义的 Accuracy；不能默认所有 choose 都恰好有一个答案 |
| Query | 列出右肺所有异常 | answer-set micro F1、macro F1、Exact Match |
| 解剖/区域切片 | 左肺、右肺、局部肺区的问题 | 按预先固定的问题 metadata 分组，不按结果选取 |
| Full benchmark / clinical subset | 全题型，或预先固定 presence/anatomy/attribute/abnormality 等 | Full 结果与诊断相关子集分别报告；图像朝向/性别题不能统称疾病识别 |

MIMIC-Ext-MIMIC-CXR-VQA 官方数据：377,391 QA；train/val 来自 silver，test 有 500 图像、13,793 QA，来源为 gold。官方 gold 仍应按 Chest ImaGenome 的标注定义描述，不夸大为所有疾病的独立临床确诊。当前自建 53 类阳性 disease-list QA 与该 benchmark 不同。

第二个 VQA 数据集优先候选 GEMeX（ICCV 2025）：包含开放、封闭、单选、多选四种问题，并有视觉/文字解释。但它同样来自胸片资源，跨 benchmark 不自动等于跨医院外部泛化，需检查患者重叠。可解释文本单独评价；不要把流畅解释或 attention 图当作正确定位证据。

### Table 2 模型组与排版

排版已在独立 PDF 中实现为一张六任务宽表，方法按零样本 VLM、表征适配、专用方法和多任务适配分组。任务覆盖矩阵同时说明哪些能力来自同一个 checkpoint。时序变化理解不进入本次计划。

- 通用 VLM：复用 Table 1 的 Qwen2.5-VL、LLaVA、InternVL 中 2–3 个。
- 医学 VLM：LLaVA-Med、MedGemma、CheXagent；报告和定位加入 MAIRA-2。
- 表征方法：DINOv2、MaCo、EVA-X、CheXWorld；BioViL-T / RadZero 按支持任务加入。已有候选 X-WIN 的 checkpoint 可用性仍需进一步核实。
- 匹配对照：当前 no-slots 与 Ours；这对对照保持共同前端/任务头，适合判断 slots 本身的效果。
- 专用任务参考：如监督 U-Net、SR 的 bicubic/专用网络；若用各自最优任务训练，应标为 task-specific specialist。

普通文本输出 VLM 没有原生像素掩膜/HR 图像输出。它们可以直接做 VQA、报告等；加入分割/SR decoder 后属于适配模型，需要训练。DINOv2 / CheXWorld 也不能原生回答自然语言问题，必须交代语言融合与读出。不适用处用破折号；尚未运行用 pending，两者有不同含义。

若给所有视觉编码器接同一个 Qwen、报告融合和 decoder，比较的是换视觉编码器后的共同系统；若让各方法采用原生最优用法，比较的是端到端能力。两种 panel 不混写成完全匹配的表征比较。

## 执行顺序与评测约束

1. 先固定一个贯穿 Stage 1 / Stage 2 / 所有任务的 patient split；对官方 VQA、MS-CXR 等测试患者，排查各阶段全部训练来源的重叠。公开权重的预训练重叠能查则披露，不能证明未见时不得称严格未见患者。
2. 第一批复用 Table 1 报告指标，接 2–3 个通用 VLM、2–3 个医学 VLM；先在验证集做小规模格式/输入验证，再冻结 prompt 和 parser 跑完整测试。模型格式错误率也要保留。
3. VQA 数据可用后，实现官方 answer ontology、负例/空答案、不同 semantic types 和评分协议；配套 image-only 输入训练。当前报告辅助检查点只能作为另一设定，不能把评测时删除报告直接冒称公平适配。
4. 新增 current report generation 和 grounding；优先固定轻量读出、冻结状态的 transfer 设定，用于检验已学表示。若 end-to-end 微调，则另标 supervised adaptation。测试问题措辞和定位框不得进入输入以外的目标泄漏通路。
5. SR 新 ×4 训练与分割解码器修正作为两条独立实验；两者都需对 Ours/no-slots 同步更改才能保持结构对照。新增 VQA 不改变旧 4-task 结果的身份；扩展训练后的模型以新配置标记。
6. 比较均报告相同患者上的结果；主要差值做 patient-level paired bootstrap。预先固定主指标、label vocabulary 和 macro 零支持类别策略；各类 support 与分层结果附录完整给出。不能在观察哪项上涨后才将其定为主指标。

按用户已确定的六类任务执行。独立医院分类测试、图文检索、图像质量鲁棒性、少标签曲线可作为后续分析，优先级低于标准 VQA、grounding 和任务协议修正。

## 官方来源

- Qwen2.5-VL-7B：https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- LLaVA-v1.6-Mistral-7B：https://huggingface.co/liuhaotian/llava-v1.6-mistral-7b
- InternVL3.5-8B：https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF
- LLaVA-Med：https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b
- MedGemma 1.5：https://deepmind.google/models/gemma/medgemma/ ，https://huggingface.co/google/medgemma-1.5-4b-it
- CheXagent：https://huggingface.co/StanfordAIMI/CheXagent-8b
- MAIRA-2：https://huggingface.co/microsoft/maira-2
- BioViL-T：https://huggingface.co/microsoft/BiomedVLP-BioViL-T
- CheXWorld：https://github.com/LeapLabTHU/CheXWorld
- RadZero：https://github.com/deepnoid-ai/RadZero
- EHRXDiff：https://github.com/dek924/EHRXDiff
- LUMEN：https://arxiv.org/html/2602.21142v1
- MIMIC-Ext-MIMIC-CXR-VQA：https://physionet.org/content/mimic-ext-mimic-cxr-vqa/1.0.0/
- VQA 原论文：https://proceedings.nips.cc/paper_files/paper/2023/file/0c007ebef1d11fd48da6ce4f54687db6-Paper-Datasets_and_Benchmarks.pdf
- MS-CXR v1.1.0：https://physionet.org/content/ms-cxr/1.1.0/
- MS-CXR-T：https://physionet.org/content/ms-cxr-t/1.0.0/
- GEMeX：https://openaccess.thecvf.com/content/ICCV2025/html/Liu_GEMeX_A_Large-Scale_Groundable_and_Explainable_Medical_VQA_Benchmark_for_ICCV_2025_paper.html
