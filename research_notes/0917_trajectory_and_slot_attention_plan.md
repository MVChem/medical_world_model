# 2026-09-17：Results 的 trajectory 主图与真实 slot attention 单栏图

## 同日后续：已替换论文占位

用户随后要求将这两个方向替换 Results 的旧候选图占位，并各开一个小节。已在 [Results 正文](../27cvpr/sections/6_results_analysis.tex) 的 Downstream tasks 之后新增：

1. **Patient trajectory**：使用[双栏 trajectory 占位](../27cvpr/figures/candidate_forecasting.tex)，图注明确真实胸片时间线、报告对照、finding 概率，以及固定 Day 0、多时间跨度的预测协议。
2. **Attention visualization**：将[单栏 attention 占位](../27cvpr/figures/candidate_slot_attention.tex)移入主文，图注明确当前胸片、S5–S8 的 2×2 真实权重叠加图和统一色标。

主文不再引用 clinical evidence、state retrieval、slot perturbation、spatial outputs 四个旧占位；补充材料不再单列 Candidate Visual Analysis。旧源文件和试图保留，两个新小节及图注均注明结果待制作。本次只调整论文结构、分析协议和图注，未新增预测结果或 attention 导出。

已运行 `make -C 27cvpr`，更新主文、补充材料 PDF 和页面预览；无未解析引用或排版溢出。两个小节编号为 4.4／4.5，两张图为 Figure 3／4。当前主文共 10 页，正文延续至第 9 页，后续仍需压缩至模板要求的 8 页正文；补充材料为 5 页。

下文保留初次记录时的方案与状态；其中“尚未替换论文占位”等表述是此次修改前的快照，论文放置以本节为准。

## 本轮确认

用户认可两种配图方向，并要求先写入 research notes：

1. **胸片时间线＋预测报告＋finding 轨迹。** 用户认为这张图有实际意义，计划用于 CVPR Results。
2. **真实 slot attention。** 用户希望增加一张单栏视觉小图；在讨论 attention、finding 归因与可解释性后，明确选择了真实 slot attention。

本轮记录选图方向、实现依据和制作步骤，尚未导出 attention、执行新的模型评测或替换论文占位。此前讨论的 finding-conditioned attribution 是备选分析，**不作为本轮已选定的小图方案**。其余历史候选方向保留，未据此全部取消。

前序记录：[09-14 候选图计划](0914_candidate_figure_plan.md)、[09-15 候选图复盘](0915_figure_review_and_next_steps.md)。最新实现以 [融合模型](../code/medworld/README.md) 为准；旧冻结视觉均值池化缓存不代表当前可学习 slot readout。

## 两张图要回答的问题

| 图 | 要展示的证据 | 可支持的解释范围 |
|---|---|---|
| Trajectory 主图 | 同一患者不同 Day 的真实随访与预测 finding／报告 | 给定当前观察和时间间隔，模型能否预测后续放射学状态变化 |
| Slot attention 单栏图 | 真实 visual-slot 查询对当前图像 patch 的汇聚权重 | 状态构造时，各深度的 visual slots 读取了哪些图像位置 |

两张图分别展示时间预测与状态的视觉读取。Attention 可以辅助理解模型，但“热图更集中”本身不能证明未来预测更准。若要把收益归因于 world model 或未来 latent 监督，还需匹配训练的预测 baseline 和组件消融。

当前方法的待验证假设是：当前任务监督使状态保留临床信息，未来状态监督进一步约束其时间变化。这里记录研究问题，不预写“比传统 VLM 更好”的实验结论。

## 图一：胸片时间线＋预测报告＋finding 轨迹

### 版式与读者视角

候选英文图题：**Longitudinal radiographic state forecasting on MIMIC-CXR**。

建议跨双栏，先选一个具有 4–5 次随访的测试病例。列为真实采集时间，上半部分放胸片与简短语义对照，下半部分放一至两个 finding 概率曲线。

| 内容 | Day 0 | 后续时间点 1 | 后续时间点 2 | 后续时间点 3 |
|---|---|---|---|---|
| 真实影像 | 当前胸片，模型输入 | 随访胸片，reference only | 随访胸片，reference only | 随访胸片，reference only |
| 参考语义 | 当前报告摘要 | 随访报告中的 finding／变化描述 | 同左 | 同左 |
| 本模型 | 编码当前状态 | 预测 finding／短报告 | 预测 finding／短报告 | 预测 finding／短报告 |
| 对照 | 相同源信息 | Copy Current／匹配 forecaster | 同左 | 同左 |

这是版式示意，不是已产生的预测。Day 0 定义为本病例选定的源胸片采集时刻，不自动等于入院日。后续 Day 根据实际时间差计算；可用小数天或同时标小时，不能将不规则随访改画成等间隔的实际观测。曲线按真实时间定位，连接线仅辅助阅读，不表示中间时点有参考标注。

### 预测协议

首版采用固定起点、多时间跨度预测：

```text
S0 = Encoder(current CXR, current report)
predicted_state(h) = WorldModel(S0, actual_delta_hours=h)
predicted finding/report = task readout(predicted_state(h))
```

所有未来点都从同一个 Day 0 出发；后续影像和报告只供参考核验。当前接口已支持连续的正／负小时数，见 [predictor.py](../code/medworld/predictor.py)、[infer.py](../code/medworld/infer.py)、[时间配对数据入口](../code/medworld/datasets/temporal.py)。选取训练所覆盖的时间范围，并记录各跨度是否属于正式配对评测之外的新增分析。

该图展示 **multi-horizon forecasting**；首版不把多个直接预测连线称为 autoregressive rollout。后续可以另做“Day 2 获得新观察后更新 Day 4 预测”，但这是讨论中的可选扩展，尚未选定。

Finding 曲线纵轴使用模型实际输出的存在概率；参考用独立标记表示阳性、阴性、不确定／未提及，不人为给参考报告赋一个概率。存在概率不等于严重程度，改善／加重应有明确的报告证据或经核验的变化标注。Copy Current 使用当前判断或报告；没有连续分数接口时不虚构其概率曲线。

### 本地数据可行性

本轮只读检查 [当前时间配对测试文件](../code/medworld_table1/data/linked_20260913_16k/test.jsonl) 及同目录 [observations.jsonl](../code/medworld_table1/data/linked_20260913_16k/observations.jsonl)：按 patient、hadm_id、view 分组，以一个 pair 的 target 等于下一个 pair 的 source 连接，统计每位患者可连接的最长序列。

- 当前测试文件共 **297 对、94 位患者**。
- **52 位**患者有至少 3 个连接的观察，**25 位**有至少 4 个；同一次住院及相同体位约束下计数相同。
- 较长的一条 AP 序列有 10 个观察，相对日约为 `0, 0.93, 1.39, 1.94, 2.92, 4.95, 5.96, 6.93, 7.94, 10.49`。
- 这些计数说明有候选序列，不表示已经选好病例、完成临床复核或取得轨迹预测结果。
- 旧数据附录的三个 `multi_interval/example_01–03` 病例，在当前 observations 中属于 **validate**。它们可以调试布局；正式测试结果图重新从 test 选例。

优先检查胸腔积液、肺水肿、肺不张的持续／变化；气胸可作为边界较明确但需仔细处理不确定表述的候选。兼顾稳定或预测失败病例，避免只按预测成功选图。体位相同仍需留意吸气、旋转与管路变化。

### MIMIC-IV 治疗信息

MIMIC-IV 有给药、处方、输液和 ICU 操作记录，项目也已有 CXR 与住院／事件时间线的关联。但当前融合模型实际输入是 **源胸片＋源报告＋时间间隔**；[model.py](../code/medworld/model.py) 中明确记录 `ehr_input: False`。

Trajectory 可以增加一行“观察到的治疗事件”，标注为 **retrospective clinical context, not model input**。预测前治疗史与预测区间内实际发生的治疗分别记录。若以后将治疗作为模型条件，需要另建输入协议并重新训练、评测；本轮不扩展为治疗效果或反事实方案比较。

来源：[MIMIC-CXR 相对时间说明](https://physionet.org/content/mimic-cxr/2.1.0/)、[MIMIC-IV 数据表说明](https://physionet.org/content/mimiciv/3.1/)、[本地病例关联说明](../code/MIMIC_example/README.md)。

## 图二：真实 slot attention 单栏小图

### 当前能提取的权重

候选英文图题：**Visual-slot attention across representation depths**。

当前 [StateEncoder](../code/medworld/encoder.py) 对后四个 visual slots 使用实际的注意力汇聚。对第 j 个选定视觉层，先把 patch 特征投影到 slot 维度，再与对应可学习 query 计算分数，经 softmax 得到权重：

```text
values_j = visual_readout_j(visual_layer_features_j)
attention_j = softmax(dot(values_j, slot_query_j) / sqrt(1024))
visual_slot_j = LayerNorm(sum(attention_j * values_j) + slot_query_j)
```

这正是需要导出的 attention；不是另外计算的 slot–patch cosine similarity，也不是分类遮挡归因。当前前向已计算权重，但尚未返回或保存，需要增加不改变计算结果的只读导出接口。

Qwen3.5-0.8B 配置下，四个 visual slots 为 **S5–S8**，对应原生视觉塔的 **第 3／6／9／12 层**，层号从 1 开始。最终图按实际 checkpoint 配置和层数标注，不把这组层号推广到所有模型。前四个 fusion slots 的语言查询取点是另一条路径，首版不将它们混入 visual-slot patch attention。

### 单栏布局

首版只做一个测试病例、一个已训练 checkpoint：

- 上方：一张小尺寸原始 CXR，附一行当前报告中的简短 finding 描述，作为病例背景。
- 下方：**2×2 的四张热图**，分别标 `S5 / Layer 3`、`S6 / Layer 6`、`S7 / Layer 9`、`S8 / Layer 12`；每张都叠加同一幅当前胸片。
- 共用色标、透明度和图像范围。文字只保留 slot／深度及必要病例说明，保证单栏尺寸可读。

此版式是制作建议，尚未看过真实权重后的最终排版。若需要多个病例或 Stage 1／Stage 2 对照，优先放补充材料，不把所有比较挤入一张单栏图。

### 提取与解释要求

1. 从该 checkpoint 的真实 visual readout 导出原始权重，核对非负性、有限性及每个 slot 的权重和。固定 eval 模式与预处理，检查开关导出前后的状态输出一致。
2. 核对原生视觉 token 的空间排列、patch 分组顺序、resize 与 padding，再恢复到图像坐标。不能未经检查直接把 token 序列 reshape 成二维热图。
3. 保存原始数组和显示变换。采用统一色标；若显示为相对均匀权重的倍数等变换，应明确标注，避免逐图独立拉伸造成“都很集中”的错觉。
4. 保留分散、背景响应或各深度相近的真实结果，不预设四个 slots 分别负责器官、病灶或固定尺度。图中的“多层”指表示深度，不自动代表不同空间分辨率。
5. 当前 visual slots 不读取报告，时间条件只进入 World Model。因此同一胸片更换报告或 horizon，不应改变这四张 encoder attention 图。图旁的 finding 文本仅作背景，不能称为该 finding 专属的 attention。
6. 该图解释当前状态的视觉汇聚。预测后的八个 slots 没有天然二维坐标；不能将当前 encoder 的热图标成“未来病灶位置”。若沿多个真实 Day 展示 attention，需说明每张都来自该 Day 的实际影像编码。
7. 普通 VLM 没有同样的可学习 slot readout，不能任意挑选某层 attention 当作严格匹配对照。Stage 1／Stage 2 或无未来 latent loss 等同结构比较，可在实验条件核对后另做；阶段差异还包含训练量变化。

真实 slot attention 能说明“读取了哪些位置”。它本身不是病灶定位标注，也不足以单独证明临床解释正确、治疗因果关系或未来预测优势。

## 文献借鉴记录

以下按本轮读取的本地论文版本记录，重点保留图的实际含义。

| 工作与图 | 实际展示 | 对本项目的借鉴 |
|---|---|---|
| [VLA-JEPA](https://arxiv.org/html/2602.10098v2)，Fig. 6；本地 PDF 第 10 页 | latent action tokens 对 image tokens 的 attention；比较 LAPA、UniVLA 与 VLA-JEPA，使用预训练、未下游微调的 checkpoint | 真实 token-to-image 热图的紧凑版式，是本轮 attention 图最直接的参考；其图不等于对因果机制的独立验证 |
| [V-JEPA 2](https://arxiv.org/html/2506.09985v1)，Fig. 8–10 | 机器人目标距离曲线、候选动作对应的能量曲面、执行序列 | 通过模型预测距离与实际行为检验动力学；当前无治疗动作输入，不照搬其动作搜索图 |
| [VL-JEPA](https://arxiv.org/abs/2512.10942)，Fig. 3 | 在对齐编码器、数据和训练条件下比较 embedding prediction 与 token prediction | “为什么优于 VLM”的论证需要控制训练差异，不能只看热图 |
| [TC-JEPA](https://arxiv.org/abs/2605.03245)，Fig. 3–5 | 规模扩展、组件消融、patch–word 相似度及特征预测误差 | 定量分析与可视化配套；Fig. 5 的文本对应机制与本模型的 visual query readout 不同 |
| [X-WIN](https://arxiv.org/abs/2511.14918)，Fig. 5–6 | 模拟／真实 CXR 的解剖 patch 特征相似度，以及 CT 解码重建 | 可借鉴热图叠加方式，但特征相似度不能改称 attention |
| [CLARITY](https://arxiv.org/abs/2512.08029)，Fig. 5–8 | 主文风险分层及治疗轨迹；附录 latent→MRI 的辅助解码、患者时间线卡片 | 借鉴 Day 时间线、真实观察与预测的并列展示；本轮不新增 MRI／CXR 生成 decoder |
| [MeWM](https://openaccess.thecvf.com/content/ICCV2025/html/Yang_Medical_World_Model_ICCV_2025_paper.html)，Fig. 4–7 | 真实／生成肿瘤 CT、风险分布与误差、生存曲线、医生与模型协作病例 | 用不同图分别说明模拟质量与决策用途；其风险热图不是病灶归因 |

图数回顾：TC-JEPA 主文 5 图，其中 Results 3 图；CLARITY 主文 6 图，其中实验区 2 图；MeWM 主文 7 图，其中实验区 4 图。MeWM 的本地文件为早期 arXiv 版本，引用正式论文时另核对版本。

CLARITY Fig. 8 的病例临床信息已与公开 MU-Glioma-Post 表中 `PatientID_0014` 对应，模型按论文在公开 MU 数据上训练；该例具体 train/test 归属及图中两张 MRI 的原始来源尚未逐一核实。数据可行性与原始表入口见 [09-16 数据调研](0916_clarity_datasets_feasibility.md)。该病例仅作版式参考。

## 备选分析与后续步骤

以下是讨论中保留的扩展，不替代已确认的真实 attention 小图：

- **未来 finding 归因**：固定报告与时间，模糊当前图像，重算两条图像编码路径，分析未来 finding 分数变化。报告短语似然归因是另一种目标，不能与 finding 分类头解释混用。
- **MS-CXR 区域核验**：[MS-CXR](https://physionet.org/content/ms-cxr/1.1.0/) 提供当前影像的 phrase–box 标注；可用于当前图像的局部对齐分析，但不是“未来预测依赖区域”的金标准。
- **新观察后的预测更新**：对比固定 Day 0 预测与读入 Day 2 新观察后的预测；作为单独协议。
- **八槽未来误差矩阵**：按 slot×时间显示预测与实际随访 target state 的归一化误差；属于状态预测诊断，不是二维影像定位。

后续制作顺序：

1. 选定实际 checkpoint 和测试病例，记录其训练阶段、输入协议、图像路径、时间与报告来源。
2. 先导出一个病例的四组真实 visual-slot 权重，完成空间顺序和显示方式核验，再确定单栏版式。
3. 从测试序列核验 4–5 个时间点，生成固定 Day 0 的多跨度 finding／报告预测，再制作 trajectory。
4. 保存脚本、原始数组、逐样本输出及 provenance；图的最终结论按实际结果填写。
5. 看过真实产物后再修改论文占位和图注。当前 [forecasting 占位](../27cvpr/figures/candidate_forecasting.tex) 在主文 Results，[slot attention 占位](../27cvpr/figures/candidate_slot_attention.tex) 仍由补充材料引用；本轮未移动它们。

本轮交付为本笔记及索引更新，未启动新训练／推理任务，未更改运行中的实验或论文正文。
