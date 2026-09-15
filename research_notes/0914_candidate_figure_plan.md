# 09-14：候选图尝试与论文占位计划

## 本次决定

用户会尝试制作本次讨论中的所有候选图，完成后再比较实际展示效果，决定哪些放入论文。现阶段不确定最终入选图，也不预先写结果结论。

论文先为各候选图保留占位符，每张配一个简短图注即可。具体方案、制作想法和后续筛选记录写入 research notes；最终图号、主文／补充材料位置及版面安排待筛选后确定。

来源为已读取的 [ChatGPT 分享对话](https://chatgpt.com/share/6aa7aebf-3eec-83ee-80ac-7880c8c5fb6b)。对话先建议空间输出与纵向预测两张主图，随后提出临床证据定位、检索和条件扰动等替代方案。本次按用户最新决定，**所有候选方向都尝试，做完再筛选**；不沿用对话中“只优先做两种”或“检索不必做”的建议。依赖尚未实现接口的面板先保留计划。

## 候选图与制作边界

### 1. Clinical evidence encoded in the state

展示同一胸片、参考区域、匹配 baseline 与本模型输出，以及对应 finding 的预测。可选同一胸片的两个 finding，检查同一状态被不同任务读出时对应的影像证据。

- **Phrase grounding 路径：** 使用真实 grounding 预测框及 MS-CXR 参考框；参考框不是像素级病灶 mask。
- **分类归因路径：** 围绕经 slots 读出的 finding logit 做区域遮挡或模糊，以扰动前后分数差构造热图。所有图像编码路径接收同一扰动图像，沿用当前下游评价的 image-only 设置，不提供同次报告。它是 class-conditioned attribution，不能标成 grounding 结果或模型 attention。
- 参考区域、定位指标和高响应区域／随机等面积区域的遮挡对照需预先固定，不能仅凭热图外观判断证据是否可信。

### 2. State-based visual retrieval

同一查询胸片并列 fusion slots、visual slots、全部八个 slots 的 Top-3 真实检索结果，并标已有 finding 标签。先统一采用不含报告的状态输入，固定对应 slot 的相似度与聚合规则，检查全测试集的 Top-k finding 一致性。

查询与候选不能来自同一患者，各条件使用相同候选库；图片按真实检索排名展示，并检查体位、设备及外观的影响。两组 slots 是否有不同检索偏好是待检验问题，不预设 fusion 负责疾病、visual 负责解剖。该图评价临床相似性，不能证明精确空间细节可恢复。

### 3. Slot-condition perturbation in spatial tasks

固定已训练 decoder 与它的直接图像输入，只改变其实际使用的 slot 条件。分割和 SR 都尝试：

- **局部遮挡／模糊：** 仅扰动用于提取 slots 的图像，展示扰动位置、原始预测、条件扰动后的预测及差异图；如 decoder 只读取 visual slots，则仅替换该组。
- **跨患者替换：** 用另一患者的对应 slots 替换原条件，直接图像输入与 decoder 参数保持不变。

输出变化只说明 decoder 依赖 slot 条件。是否提供有益信息，还需比较相对参考的误差及区域／扰动对照。几乎无变化也保留为诊断结果，差异图使用统一色标。SR 的直接图像与 slot 提取始终只使用 LR，HR 只作参考。

### 4. Slot-conditioned spatial prediction

直接比较 Input、Image-only、Frozen slots、Jointly trained slots 和 Reference。分割展示轮廓及相同位置的局部放大；SR 展示相同裁剪、恢复结果、HR 和必要的绝对误差图。各条件统一显示范围及误差色标，保留 image-only 对照。

人工肺 mask 与 CXAS pseudo-label reference 明确区分。展示基于实际完成的匹配实验，不将旧 pilot 当作完整多深度 4＋4 模型。联合训练与冻结的差异仅回答联合优化的作用；纵向训练收益另需 Stage 1／Stage 2 及预算匹配对照。图注不提前写边界更准、细节更清晰等结论。

### 5. Image-grounded longitudinal forecasting

每例展示当前胸片及可用信息、预测 horizon、各方法的未来判断，以及真实随访胸片和核验后的参考变化。有连续评分接口的方法展示 finding 概率，有报告输出的方法展示短报告；Copy Current 没有连续概率接口。覆盖新出现、消退、稳定，保留 Copy Current、匹配 forecasting baseline 与本模型；持续 finding 的改善／加重须在方向标注核验后展示。

可加未来 finding 分数对当前胸片的归因面板，采用实际预测路径上的梯度或遮挡分析，并比较高响应区域与随机等面积区域的扰动。若使用当前报告和临床历史，归因时固定这些输入，说明热图只解释图像的影响。右侧随访必须标为 **observed follow-up, reference only**，不能排成模型生成胸片或未来分割的效果。

### 6. Slot-readout attention（有真实接口时）

作为小面板或补充候选，展示真实 slot readout 对空间 patch 的 attention 权重。只有实际接口使用可提取权重的 attention pooling／cross-attention 时才制作；平均池化加投影没有此类 attention，届时记录不适用，不额外制造注意力矩阵。

八个 slots 按来源和深度定义，没有预设疾病或器官位置。分析需标清是否经过本文 slot/readout 路径，不能将冻结视觉编码器的原始热图归因为 slot 训练或 world model 的效果。

## 论文占位映射

每类一个占位框与简短英文图注，图注标明 Planned。临床证据的两条实现路径、纵向预测的可选归因先作为各自图内的候选面板；最终选图后再确定面板组合和图号。

| 候选 | 当前暂放位置 | 占位源码与 LaTeX label |
|---|---|---|
| 纵向预测病例及可选未来分数归因 | 主文 Future prediction 后 | [candidate_forecasting.tex](../27cvpr/figures/candidate_forecasting.tex)，`fig:candidate_longitudinal_forecasting` |
| 临床证据定位 | 主文 Downstream tasks 后 | [candidate_clinical_evidence.tex](../27cvpr/figures/candidate_clinical_evidence.tex)，`fig:candidate_clinical_evidence` |
| 状态检索 | 主文 Downstream tasks 后 | [candidate_state_retrieval.tex](../27cvpr/figures/candidate_state_retrieval.tex)，`fig:candidate_state_retrieval` |
| Slot 条件扰动 | 主文 Downstream tasks 后 | [candidate_slot_perturbation.tex](../27cvpr/figures/candidate_slot_perturbation.tex)，`fig:candidate_slot_perturbation` |
| 直接空间输出比较 | 主文 Downstream tasks 后 | [candidate_spatial_outputs.tex](../27cvpr/figures/candidate_spatial_outputs.tex)，`fig:candidate_spatial_outputs` |
| 真实 slot-readout attention | 补充材料 [候选分析节](../27cvpr/sections/c_candidate_visual_analysis.tex) | [candidate_slot_attention.tex](../27cvpr/figures/candidate_slot_attention.tex)，`fig:candidate_slot_attention` |

上述位置只是便于当前稿件 review 的安排，不代表已选定入选图。具体图注以对应 TeX 为准，制作和筛选后统一更新。

## 共同展示原则

- 对话借鉴 X-WIN 式 patch 特征对应热图的展示目的；这里保留为参考思路。聚合 slots 没有天然二维位置，不能直接当作空间 patch，也不能只因维度相同就认定 slot–patch cosine similarity 能提供有效定位。
- 所有图来自真实输出和明确的 checkpoint／输入协议。固定病例选择规则，保留无改善或失败病例；先尝试全部候选，再按证据与展示效果确定入选和主文／补充材料位置。
- 当前阶段只添加占位符与简短图注，不表示候选实验已完成，不预填性能结论。论文 Results 入口为 [6_results_analysis.tex](../27cvpr/sections/6_results_analysis.tex)，补充材料入口为 [supplementary.tex](../27cvpr/supplementary.tex)。现有附录真实纵向病例图是数据示例，不能替代本次模型预测病例图。

## 占位稿检查

`make -C 27cvpr` 编译通过，主文 10 页（正文在第 8 页结束），补充材料 5 页。已核对主文 5 个占位框和补充材料 1 个占位框，更新 `27cvpr/preview/`。无未解析引用或 box overflow；主文第 6 页集中排放两张结果表，LaTeX 提示该页仅含浮动体。当前页数与图号均为保留全部候选时的工作稿状态。
