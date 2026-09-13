# MedWorld-JEPA Paper Plan

2026-09-08 更新 Results；保留 0907 的 Methods 整理。Table 1 评价 MIMIC-CXR 上的未来状态与临床报告预测，采用 Future R@1、Finding AUPRC、Transition F1、RadGraph F1、CheXbert F1 五项指标，并记录各方法的输出接口、可比范围和待定实现。本文作为后续整篇论文写作与讨论的工作稿，Introduction 的 paper plan 留待补充。

整理以当前 [Methods 正文](../27cvpr/sections/4_method.tex)、[Task Definition](../27cvpr/sections/3_problem_formulation.tex) 和 [新版 Fig. 1](../27cvpr/imgs/fig1.pdf) 为准，结合 [0906 方法讨论稿](0906_methods.md) 中的设计理由与问题澄清，沿用 [0829 设计 brief](0829_ideas.md) 中“科学问题 → 模块与公式 → 监督 → 验证”的组织方式。实验接口参照当前 [Experiments](../27cvpr/sections/5_experiments.tex) 和 [附录协议](../27cvpr/sections/b_protocol_details.tex)。

本文区分三种状态：**当前主方案**指已经明确并写入正文的设计；**待定实现**指接口已明确、具体配置尚未选择；**候选扩展**指曾讨论但尚未纳入主方案的方向。方法已经写入论文不等于训练与实验已经完成。最新接口变更同时记录在 [0907_methods_revision.md](0907_methods_revision.md)。

**本轮讨论确认的实验方向**单列于第 3 节：Table 1 同时评价未来状态预测及其解码报告，后续 Table 2 或图评价下游表征用途。五项指标与方法名单已写入计划；未来文本 decoder、疾病评分接口和评价空间仍需后续同步到 Methods、Experiments 与 Fig. 1。本次仅更新 paper plan，不代表正文、实现或实验结果已完成。

## 1. Introduction paper plan

待补充。

## 2. Methods paper plan

### 2.0 方法主线与写作组织

**MedWorld-JEPA 同时学习医学观测的状态表示，以及该状态在指定时间条件下的演化。** 当前主设定是：

> 当前影像与当前报告共同编码为状态；Latent World Model 根据当前状态和 horizon，预测真实随访影像与报告所对应的未来 latent state。

两个目标相互联系，需要分别验证：

- **Future state prediction**：预测具体患者在指定 horizon 下的未来状态，并通过文本 decoder 输出未来临床文本；Table 1 通过未来检索、latent 相似度、疾病评分、疾病变化和报告临床内容评价这一能力，检验是否超越当前状态的复制和群体未来先验。各指标所需输出不同，接口与可比范围见第 3 节，待同步正文。
- **Reusable state representation**：让任务监督和纵向监督共同塑造当前状态，支持分类、分割、指令式疾病识别和超分辨率等任务。其效用需要通过独立下游评价验证。

两者在方法中的地位并列。未来预测本身是核心任务；表征的下游收益也不能仅由一个更好的预测器来证明。

方法的逻辑顺序是：

**定义观测与状态 → 建立共享多模态编码 → 用任务监督训练状态 → 学习 horizon 条件化的未来状态预测。**

正文保留四个小节，Overview 与 Fig. 1 放在 Task Definition 之后：

| 正文小节 | 需要回答的问题 | 主要内容 |
|---|---|---|
| 3.1 Task Definition | 观测是什么，预测什么，时间条件是什么？ | 当前图文、真实随访图文、状态定义、两个目标、时间与证据边界 |
| Overview + Fig. 1 | 整体信息怎样流动？ | 两个时点的编码路径、任务读取、未来预测与监督 |
| 3.2 State Representation | 如何把图文变成可复用的状态？ | 共享编码、JEPA、adapter、输入 slots 与输出 state tokens、任务 decoder |
| 3.3 State Prediction | 当前状态如何预测未来状态？ | LWM 输入输出、真实随访目标、仅监督未来 S 的 latent loss |
| 3.4 Two-Stage Training | 哪些数据和损失训练哪些模块？ | 任务状态预训练、纵向预测微调、梯度路径、目标端更新 |

每一节先说明它在整个方法中的作用，再给出输入输出和必要公式。设计动机围绕医学状态与时间预测展开；具体 backbone、adapter 和 decoder 的配置放入实现设置，尚未确定的配置保留为待定。

### 2.1 Task Definition：定义医学状态与未来预测

#### 2.1.1 观测与主任务

任一时点 \(\tau\) 的观测记为：

\[
O_\tau=(X_\tau,C_\tau;m_\tau),
\qquad
m_\tau\in\{0,1\}^{2}.
\]

其中 \(X_\tau\) 是医学影像，\(C_\tau\) 是对应报告或其他允许使用的临床文本，\(m_\tau\) 表示两种模态是否可用；至少一种观测可用。

**当前主方案在当前时点与未来时点都使用影像—报告对。** 当前时点记为 \(t\)，实际随访时点记为 \(t^+>t\)。仅影像、仅报告或两端模态不同，在接口上可以处理；具体训练覆盖和实验组合仍待安排，不逐项宣称已经验证。

模型首先编码当前状态，再根据时间条件预测未来状态：

\[
S_t=E_\eta(O_t)\in\mathbb R^{K\times d},
\]

\[
\widehat S_{t,h}=P_\psi(S_t,h)
\approx S^*_{t^+}.
\]

这里的 state 指对**可用医学观测**的表示，不假定它包含决定患者后续病程的全部变量。研究对象是 observational follow-up dynamics，当前模型没有定义治疗干预、反事实或因果控制语义。

#### 2.1.2 当前状态、空间特征与预测目标

当前状态统一记为 \(S_t\)，由 VLM 输出的 \(K\) 个 state tokens 组成。影像经 JEPA 得到的 \(D_t\) 仅沿 adapter→VLM 路径构造状态；分割、超分等空间任务额外读取原始输入影像 \(X_t\)，在 task decoder 内部完成图像特征提取或恢复。

旧讨论中的

\[
R_t=(D_t^{\mathrm{optional}},S_t;m_t)
\]

仅保留为旧版本的编码结果记法。**当前正文直接使用状态 \(S_t\)，空间任务另接输入影像 \(X_t\)，不再引入统一的 \(R_t/Z_t\) 拼接状态或全局 \(W_D\)。** \(D_t\) 仍是状态编码的中间特征，但不直接送入 task decoder。

预测与目标严格区分：

- \(\widehat S_{t,h}\)：LWM 根据当前 \(S_t\) 和 \(h\) 输出的预测。
- \(S^*_{t^+}\)：真实随访观测经未来目标编码器得到的状态。
- 预测用帽号，目标用星号；不使用同时带帽号和星号的记法。
- 当前状态、预测状态和真实未来目标均采用 \(K\times d\) 接口；图中 \(K=8\)。

#### 2.1.3 Horizon 与实际随访时点

\(h\) 表示模型收到的时间条件，\(t^+\) 表示实际发生的随访时间，两者分别定义。

如果采用实际时间间隔，需明确对应的时间尺度与编码；如果采用 horizon bins，需明确区间、容差窗口和区间内目标选择规则。当前尚未选择，因此不直接假定 \(t^+=t+h\)。

现有附录保留了一套**拟议的 horizon-bin 评价协议**：先确定时间区间与窗口，再选择相应随访；预测器读取请求的时间区间，而不是事后观察到的精确随访时间。它是待确认的实验协议，不代表最终时间编码已经确定。

#### 2.1.4 当前输入的证据边界

当前分支只读取预测截止时间前已经可用的证据。报告的可用时间同样需要检查，不能仅依据影像采集时间默认最终报告已经存在。

未来影像与未来报告进入目标编码或评价，不进入本次预测的当前状态。一个较晚检查可以在训练划分内作为另一条独立的单时点任务样本，但这不改变每一条纵向预测的时间边界。

训练疾病标签或文本答案时，应从对应任务的状态输入中移除同一答案及直接透露答案的文本。附录中总结整段病程的叙述也不能整体作为起始时刻的输入。

#### 2.1.5 Overview 与 Fig. 1 的叙述

Overview 用一段话串起整图：

> 当前图文和真实随访图文沿用同一种多模态编码设计。JEPA 提取空间特征，经 visual adapter 转为 VLM 的视觉输入；报告经原生 tokenizer 与 embedding 转为文本输入。两种证据与 learned input slots 一起送入 VLM，同一组槽位位置的输出 hidden states 构成 state tokens。任务 decoder 从当前 state 读取信息，空间任务额外读取原始影像，由 decoder 自己提取图像特征；LWM 从当前 state 和 horizon 预测未来 state，并与真实随访经 VLM 编码的 stop-gradient 目标计算损失。第一阶段用任务监督建立状态，第二阶段学习状态转移并继续调整当前状态编码路径。

[Fig. 1](../27cvpr/imgs/fig1.pdf) 的可编辑源为 [fig1_v7.pptx](../27cvpr/ppt/ppt/fig1_v7.pptx)，与公式的对应关系如下：

| 图中路径或模块 | 对应含义 |
|---|---|
| 当前 image → JEPA → \(D_t\) → visual adapter → visual tokens | 当前视觉证据进入 VLM 的路径 |
| 当前 report → tokenizer + embedding → text tokens | 当前文本证据进入 VLM 的路径 |
| visual tokens + text tokens + learned slots \(U\) → VLM → \(S_t\) | 输入槽位与对应输出状态；图中都是 8 个位置 |
| \(S_t\) → task decoder，task query \(Q\) → task decoder | 状态先构建，任务条件在读取时加入 |
| 输入影像 \(X_t\) → task decoder | 从输入 X-ray 直接分支；分割读待分割图像，超分读 \(X^{\mathrm{LR}}\)，不从 JEPA 的 \(D_t\) 引线 |
| \((S_t,h)\) → Latent World Model → \(\widehat S_{t,h}\) | 当前主方案唯一的预测接口 |
| 真实未来图文 → 同构编码路径 → \(S^*_{t^+}\) | 真实随访提供未来状态目标 |
| \(\widehat S_{t,h}\) 与 \(\operatorname{sg}(S^*_{t^+})\) → latent prediction loss | 只监督未来 state tokens |
| Stage 1 / Stage 2 标注 | 任务预训练与未来预测微调的监督分工 |

新版图保留 task query \(Q\) 方框，示例为 “Segment the lungs”。旧 0906 讨论稿中“不再单独画统一 task query 方框”的建议已经被后续讨论和新版图更新。此处沿用新版图，但并不要求每个固定任务头都显式输入自然语言 query。

### 2.2 State Representation：建立共享多模态状态

#### 2.2.1 Shared Multimodal State Encoding

**当前观测和真实未来观测使用同一种编码设计与输出接口。** 先分别编码两个时点，再用预测器学习它们之间的转移。

| 可用模态 | 编码路径 | 可用输出 |
|---|---|---|
| 影像 + 报告，纵向主设定 | 影像与报告分别适配后共同进入 VLM | 融合后的 \(S_\tau\)，用于主要纵向状态预测及适用的全局任务 |
| 仅影像，空间任务主输入 | 影像 → JEPA → adapter → VLM | 由视觉证据得到的 \(S_\tau\)；同一输入影像另送空间 decoder |
| 仅报告，候选组合 | 报告 → tokenizer / embedding → VLM | 由文本证据得到的 \(S_\tau\)；无影像供空间任务读取 |

缺失模态可省略；批处理中使用 padding 时，通过 mask 标记不可用输入。缺少报告仍输出 \(K\) 个 state tokens；缺少影像时跳过 JEPA 与 adapter，也没有原始图像供空间 decoder 使用。当前单时点空间任务采用 image-only 状态输入，因此 Stage 1 包含相应 image-only 编码训练；其他缺失模态训练和评价仍待安排。

共享编码设计不等于已经确定严格共享参数。用户倾向同一套 VLM、adapter 和 slots 分别编码两个时点；是否采用严格同权重、EMA 或固定副本，在 Training 中保留为待定。

“双分支”指当前与未来分别编码，不指双向 attention 或未来→当前的反向时间预测。VLM 保留自身的 attention 方式。统一 token 形状提供接口兼容性，跨模态语义对齐与缺失模态性能仍需训练和验证。

#### 2.2.2 JEPA：提供空间视觉证据

采用预训练且冻结的 JEPA 视觉编码器 \(f_\theta\)，提取保留位置结构的单层空间 token grid：

\[
D_\tau=f_\theta(X_\tau)
\in\mathbb R^{N_D\times d_D}.
\]

\(N_D\) 是空间 token 数，\(d_D\) 是 JEPA 特征宽度。当前方案不再使用旧稿中的多尺度 \(D^{1:L}\) hierarchy。

JEPA 的作用是提供从上下文预测中学习的视觉表示；V-JEPA 2.1 的 dense feature 学习可作为保留空间结构的参考。当前方法使用已有视觉骨干，不把医学 JEPA 重新预训练当作第一阶段的内容。

\(D_\tau\) 经 adapter 构造 VLM 的视觉证据，不直接进入空间 task decoder 或主方案的 LWM。需要像素或局部空间信息的任务从输入影像 \(X_\tau\) 另开一条图像路径，其特征提取由对应 decoder 自己完成。

此前讨论的完整 CT volume→局部切片上下文学习，以及 CT 三维知识向 X-ray 的迁移，仍是后续研究方向；不能写成当前骨干已经具备或实验已经证明的能力。

#### 2.2.3 Visual adapter：连接 JEPA 与 VLM

可训练的 visual adapter \(a_\rho\) 把 JEPA 特征映射到 VLM 的 embedding width \(d\)：

\[
V_\tau=a_\rho(D_\tau)
\in\mathbb R^{N_V\times d}.
\]

其中 \(N_V\) 为送入 VLM 的视觉 token 数，可以与 \(N_D\) 不同。Adapter 可以只做宽度投影，也可以聚合或压缩 tokens；具体结构待定。

Projector、MLP、attention-based resampler 或其组合是实现候选，不提前限定必须使用 resampler。LLaVA 的 visual connector 可以支持这里的连接思路，但不决定本文最终配置。

空间 task decoder 不复用 JEPA 的 \(D_\tau\) 作为直接输入。它可以在内部对 \(X_\tau\) 做卷积、patch embedding、恢复网络编码或其他特征提取，再与 \(S_\tau\) 融合；这些图像处理与条件化参数属于该 decoder。原始图像输入指接收图像像素张量，允许必要的尺寸和强度预处理，并不要求网络内部完全不用特征。具体结构待定，不增加统一的全局状态拼接层。

文本则使用 VLM 原生的 tokenizer 与 embedding：

\[
T_\tau
=\operatorname{Embed}(\operatorname{Tokenize}(C_\tau)).
\]

Visual adapter 负责视觉输入，不替代文本的原生嵌入路径。

#### 2.2.4 Learned input slots \(U\) 与输出 state tokens \(S_\tau\)

这是本轮讨论后必须写清楚的关系：

\[
U=[u_1;\ldots;u_K]\in\mathbb R^{K\times d},
\]

\[
S_\tau
=E_\eta(O_\tau)
=g_\phi\!\left([V_\tau;T_\tau;U];m_\tau\right)\big|_U
\in\mathbb R^{K\times d}.
\]

\(\big|_U\) 表示取出输入 \(U\) 所在位置的输出 hidden states。可用视觉与文本证据排在 slots 之前，使这些位置能读取证据；缺少某一模态时省略相应输入。

\(\eta\) 汇总 adapter 参数 \(\rho\)、slot embeddings \(U\) 和 VLM 中选定的可训练参数；JEPA 参数 \(\theta\) 固定。

| 对象 | 在计算中的角色 | 图中 shape | 如何变化 |
|---|---|---|---|
| \(U\) | 输入侧的可学习 slot embeddings，跨样本共享 | \(8\times d\) | 由优化器更新模型参数 |
| \(S_\tau\) | VLM 在上述槽位位置的输出 hidden states | \(8\times d\) | 随观测与编码参数变化，每次前向重新计算 |
| \(\widehat S_{t,h}\) | LWM 预测的未来 state tokens | \(8\times d\) | 由当前状态、horizon 和预测器决定 |
| \(S^*_{t^+}\) | 真实随访经目标编码器得到的 state tokens | \(8\times d\) | 由未来观测及目标参数决定，本次 future loss 中 detach |

因此，learned slots 可以理解为**带有可学习 embedding 的输入槽位**；经过 VLM 后，同位置输出就是状态特征。它们不是与 \(S_t\) 无关的另一组输出，也不是八个输出之外再增加一个汇总 token。

\(U\) 与 \(S_\tau\) shape 相同，但数值和角色不同。\(U\) 不是患者专属参数，\(S_\tau\) 也不是每个患者单独存储并由优化器直接更新的自由参数。LWM 读取整组 \(K\) 个输出 tokens，不默认池化成一个向量。

图中采用 \(K=8\)，表示八个 \(d\) 维状态向量；这尚不是已验证最优的 slot 数。分类或回归 decoder 可以在内部聚合 tokens，但这种读取方式不改变共享状态本身的定义。

**当前 slots 不预先分配固定任务或临床语义。** “slot 1 对应分类、slot 2 对应分割，或一个任务对应多个 slots”的定向监督想法仍可研究，但尚未成为当前方法。若采用，需要另行定义 slot 子集、decoder 路由及相应验证，不能仅根据有八个 slots 就宣称已经形成八种功能。

共享状态的构建先于 task query 与 horizon。\(Q\) 和 \(h\) 都不进入这一轮状态编码；对相同的允许证据，可以复用同一状态完成不同任务或不同时间条件的预测。若不同任务允许的输入不同，则必须重新编码相应状态。

#### 2.2.5 Task-conditioned readout：根据任务读取状态

图中的 Task decoder 表示一组任务接口，不预设所有任务共享完全相同的网络或参数。任务索引记为 \(q\)，对应 decoder 参数为 \(\omega_q\)，需要时提供任务 query \(Q_q\)。

\(Q_q\) 表示“希望从状态中得到什么”，例如 “Segment the lungs”。它与输入 slots \(U\)、输出状态 \(S_t\) 都不同，在状态构建之后条件化 decoder。固定任务头可以隐含指定任务；多类别或多实例 query 的实现可以留在相应 decoder 内部。

| 任务 | Decoder 读取 | 输出与监督真值 | 任务损失 |
|---|---|---|---|
| 分类 | \(S_t\) | 类别或多标签预测，与类别标签比较 | CE / BCE |
| 分割 | \(X_t,S_t\)，按任务提供 \(Q_{\mathrm{seg}}\) | 预测 mask，与标注 mask 比较 | Dice 与 CE/BCE；组合待定 |
| 指令式疾病识别 | \(S_t,Q_{\mathrm{dis}}\) | 规范化疾病名称或标签列表，与标注序列比较 | Token-level CE |
| 超分辨率 | \(X^{\mathrm{LR}},S^{\mathrm{LR}}\) | 预测 HR 图像，与对应 HR 真值比较 | 按标量元素平均的像素 MSE |

分类通过任务头读取 \(S_t\)。指令式疾病识别通过语言 decoder 从 \(S_t\) 和指令生成标准化答案，不旁路读取原始图文；这里的“诊断”范围先落在疾病识别，不扩展为未定义的综合临床推理。

分割接口为：

\[
\widehat M_\tau
=H_{\omega_{\mathrm{seg}}}
(X_\tau,S_\tau,Q_{\mathrm{seg}}).
\]

这里 \(H_{\omega_{\mathrm{seg}}}\) 表示包含图像处理在内的完整任务网络：\(X_\tau\) 提供局部图像证据，\(S_\tau\) 作为学习到的状态条件，\(Q_{\mathrm{seg}}\) 指定目标。为说明接口，可将任一空间任务网络概念性地展开为

\[
F_q=b_{\xi_q}(X),\qquad
\widehat y^{(q)}
=d_{\zeta_q}(F_q,S,Q_q),\qquad
\omega_q=(\xi_q,\zeta_q).
\]

\(F_q\) 是 task decoder 自己从图像提取的特征，不是固定 JEPA 的 \(D\)；\(b_{\xi_q}\) 与 \(d_{\zeta_q}\) 也可以在实现中合为同一个网络。这个展开解释参数归属，不锁定必须采用两个独立模块。

LISA 提供了“由语言模型 hidden states 条件化图像分割”的相关先例；其内部也使用图像特征，并不支持网络无需特征提取的说法。本文先建立共享观测状态，再引入任务请求；LISA 的任务 token 不等同于本文通用 state slots，也不要求本文 decoder 复用 JEPA 特征。

#### 2.2.6 超分辨率：用像素恢复监督补充任务训练

超分训练从低分辨率影像出发，输出对应高分辨率影像：

\[
\widehat X^{\mathrm{HR}}
=H_{\omega_{\mathrm{SR}}}
(X^{\mathrm{LR}},S^{\mathrm{LR}}),
\]

\[
\mathcal L_{\mathrm{SR}}
=\frac{1}{N}
\left\|\widehat X^{\mathrm{HR}}-X^{\mathrm{HR}}\right\|_F^2.
\]

超分有两条来自同一 LR 观测的路径：\(X^{\mathrm{LR}}\) 直接进入恢复 decoder 的图像路径；同一 \(X^{\mathrm{LR}}\) 经 JEPA→adapter→VLM 得到条件 \(S^{\mathrm{LR}}\)。后一路保留状态编码，但没有将 \(D^{\mathrm{LR}}\) 直接送入恢复 decoder。当前超分状态采用 image-only 输入：

\[
O^{\mathrm{LR}}=(X^{\mathrm{LR}},\varnothing;(1,0)),
\qquad
S^{\mathrm{LR}}=E_\eta(O^{\mathrm{LR}}).
\]

HR 图像只作监督目标，不用于任一输入分支，也不通过 HR 派生报告或特征进入状态。配对可以通过下采样构造，具体退化方式待定。\(N\) 是目标中的标量元素总数，包含空间位置与通道；预测与真值使用相同强度尺度。

SR 的 MSE 已经明确；权重、退化过程和 decoder 结构尚未确定。这里引入像素级监督是为了约束任务所需的细节，其对疾病识别或未来预测的帮助需由实验判断。不笼统声称 JEPA 的 masking 目标必然无法学习细节。

#### 2.2.7 Task loss 到底优化什么

**优化对象是任务输出与真值的误差，并通过这项误差共同训练 decoder 和产生状态的编码路径。**

以任务 \(q\) 为例，损失经过：

\[
\mathcal L_q
\longrightarrow H_{\omega_q}
\longrightarrow S_i^{(q)}
\longrightarrow
\{g_\phi,\ U,\ a_\rho\}.
\]

这里的箭头表示反向梯度路径。Task decoder 内部的图像特征提取、恢复网络、状态投影与融合模块同时由任务损失训练；固定 JEPA 属于另一条状态编码路径。状态构建与任务读取是两个步骤，不意味着中间 detach。

每个任务使用允许的证据 \(O_i^{(q)}\)，形成：

\[
S_i^{(q)}=E_\eta(O_i^{(q)}).
\]

例如，疾病标签任务可能需要移除答案报告，SR 需要改用 LR 影像；这些任务对应的状态值不必相同。更新 \(U\)、adapter 与 VLM 参数后，下一次前向产生的 \(S\) 随之改变。当前没有给每个 slot 指定一个固定的监督向量，也没有将每例 \(S\) 当作独立参数优化。

空间任务可以通过直接图像路径完成较多预测，因此有通向 \(S\) 的梯度不证明 decoder 一定利用了 \(S\)。需要把相同图像网络在无有效状态条件、no-future 状态条件和完整模型状态条件下分别训练比较。这里移除的是有效 \(S\) 条件，图像网络的架构、训练预算与初始化策略保持匹配；不是只在测试时突然拔掉 \(S\)。

报告有两种用途，按样本和任务区分：作为观测参与编码，或作为文本监督答案。未来报告可以构造 latent target；第 3 节新增的未来文本任务还将其作为文本 decoder 的监督与评价参考。两者的损失路径分别定义，不能让真实未来报告进入预测分支。缺少文本标注时跳过相应语言损失，不把缺失文本训练为空答案。

### 2.3 State Prediction：预测真实随访的 state tokens

#### 2.3.1 Horizon-conditioned latent dynamics

当前主方案的预测接口为：

\[
\boxed{
\widehat S_{t,h}=P_\psi(S_t,h)
\in\mathbb R^{K\times d}
}
\]

LWM 读取当前整组 state tokens 和 horizon 的 embedding，直接输出给定时间条件下的未来 state tokens。

\(D_t\) 不作为额外 predictor 输入，但视觉信息仍经

\[
D_t\rightarrow V_t\rightarrow S_t\rightarrow P_\psi
\]

参与预测。Task query \(Q\) 只属于任务读取接口，模态标记 \(m_t\) 只用于编码接口；主预测器没有额外的 \(R_t\)、\(D_t\)、task query 或独立 transition query 输入。

DINO-WM 支持在预训练特征空间学习 latent dynamics 的思路；它不决定本文必须直接读取或预测空间特征。本文选择的预测空间是融合图文的 VLM state tokens。

当前定义的是给定 \(h\) 的直接预测。多时点记录可以产生多个训练 pair，不自动意味着已经实现多步时间 rollout。VLM 自身的自回归 token 接口与 LWM 沿时间递推是两个不同问题。

#### 2.3.2 真实随访作为目标

未来目标编码器记为 \(E_{\bar\eta}\)：

\[
S^*_{t^+}
=E_{\bar\eta}(O_{t^+})
\in\mathbb R^{K\times d}.
\]

主设定下，未来影像经 JEPA、adapter 进入未来 VLM，未来报告经原生文本嵌入进入同一 VLM；输出槽位 hidden states 构成目标。

有未来影像时得到的 \(D^*_{t^+}\) 只是视觉编码路径中的中间特征。**它不是 LWM 的独立输出或监督目标。** 未来仅有报告时，仍可以通过 VLM 得到 \(S^*_{t^+}\)；相应模态组合的训练覆盖与对齐效果需要另外验证。

未来影像、报告和未来模态可用性都不作为本次预测的输入。当前与未来采用一致的状态接口，使预测值与真实观测的编码值可以在相同 token 形状下比较。

#### 2.3.3 仅监督未来 \(S\) 的 latent loss

未来损失为：

\[
\boxed{
\mathcal L_{\mathrm{future}}
=\ell_S\!\left(
\widehat S_{t,h},
\operatorname{sg}(S^*_{t^+})
\right)
}
\]

\(\operatorname{sg}\) 表示 stop-gradient。损失比较预测的未来状态与真实随访的编码状态，不直接将当前 \(S_t\) 拉近未来 \(S^*_{t^+}\)。

如果采用按元素平均的 MSE，可写为：

\[
\ell_S
=\frac{1}{Kd}
\left\|
\widehat S_{t,h}
-\operatorname{sg}(S^*_{t^+})
\right\|_F^2.
\]

**这一 MSE 是候选 latent distance，尚未确定；它与已经确定采用 MSE 的 SR loss 分别处理。** 是否归一化、采用何种距离及相关超参数仍需选择。

输入 slots 所对应的固定输出位置提供 token 的比较顺序，不预先赋予各位置临床语义，也不假定它们对应固定解剖位置。当前不增加 slot-specific semantic targets。

未来 \(D\) loss 对应另一项空间特征预测任务，学习未来 \(S\) 不要求同时承担它。因此当前没有 \(\widehat D\)、\(\ell_D\)、\(\lambda_D\) 或 D/S 两项加权的未来目标；仅报告的未来样本也无需处理任何 \(D\) loss 分支。

#### 2.3.4 预测输出与可解码能力

“当前图文预测未来图文”在当前正文中首先表示**预测未来图文所对应的 latent state**。本轮讨论进一步明确：预测的 latent 通过文本 decoder 输出未来临床文本，并由疾病 readout 给出未来异常分数；Table 1 结合随访检索、异常评分与报告临床内容评价，详见第 3 节。当前正文尚未展开这些新增接口及其训练监督，后续需要补充；未来像素生成不属于本表任务。

未来状态评价只读取 \(\widehat S_{t,h}\) 与允许的任务条件。当前分割、超分 decoder 依赖相应输入图像 \(X\) 和 \(S\)，而预测器只输出未来 \(S\)，所以它们不能直接作为未来分割或未来图像恢复接口。

未来状态评价不能额外拼接当前或真实未来影像，也不能加入它们的 JEPA 特征；真实未来图像不能充当预测的输入图像。若要研究以当前影像和预测状态为条件的未来生成，那是另一项需要另行定义接口、目标与评价的任务。

#### 2.3.5 纵向数据与候选扩展的边界

主要纵向来源是 MIMIC-CXR 的同患者当前—随访观测对。已有 [附录病例图](../27cvpr/ppt/ppt/appendix_fig_v3.pdf) 展示变化、稳定随访和多次检查的形式；它们是实际数据示例，不是预测结果。可用 pair 数、不同时间间隔和模态组合的分布仍需统计。

保留以下候选方向，但不写成当前 LWM 的组成部分：

| 候选方向 | 需要另外定义或验证的内容 |
|---|---|
| 给 predictor 增加 \(D_t\) 输入 | 可写为 \(P_\psi(D_t,S_t,h)\)，输出仍只为未来 \(S\)；检验空间信息是否有额外帮助及是否降低对 \(S\) 的依赖 |
| 多步 autoregressive rollout | 预测状态如何递归输入、时间条件如何组合、训练目标与误差积累 |
| LSTM 等递归模块 | 具体状态更新与训练方式；缺失模态本身不要求使用递归结构 |
| Diffusion predictor | 加噪、去噪目标和采样过程；denoising step 与临床 horizon 分别定义 |
| 跨模态或缺失模态预测 | 输入采样、目标对齐和各组合的评价；接口兼容不等于已验证性能 |
| 未来影像生成 | 对应图像 decoder、监督与评价；不纳入当前 Table 1。未来文本预测已进入第 3 节实验计划 |

### 2.4 Two-Stage Training：先建立状态，再学习状态转移

#### 2.4.1 两阶段的动机

单时点任务标注告诉模型“状态需要保留什么信息”，纵向配对告诉模型“这些信息如何随时间变化”。因此先建立可用的多模态状态空间，再利用真实随访学习状态转移，并继续调整当前状态编码。

Stage 1 的 pretraining 指对后续 world modeling 的状态编码路径进行预训练，以已有 JEPA 与 VLM 权重为起点；不意味着从头训练整个 VLM，也不等同于医学 JEPA 再预训练。

这一安排希望减轻纵向训练同时学习临床表征和状态转移的负担。是否提高数据效率必须通过实验检验；目前没有确认单时点与纵向数据量的数量级差异。

#### 2.4.2 Stage 1：Task-supervised State Pretraining

使用单时点分类、分割、指令式疾病识别及超分数据，优化：

\[
\mathcal L_{\mathrm{stage1}}
=\mathcal L_{\mathrm{task}}
=\mathbb E_i
\left[
\sum_{q\in\mathcal Q_i}
\alpha_q\,
\ell_q\!\left(
\widehat y_i^{(q)},y_i^{(q)}
\right)
\right].
\]

\(\mathcal Q_i\) 是样本 \(i\) 的输入和标注支持的任务集合，\(\alpha_q\) 是任务权重，\(\widehat y_i^{(q)}\) 由该任务允许证据对应的状态和 decoder 产生。缺少某项标注时不计算该项损失。

训练参数包括 \(U\)、visual adapter、VLM 中选定的参数，以及各任务 decoder 内部的图像处理、状态投影和融合模块；JEPA 固定。LoRA 是可采用的参数高效适配方式，具体范围与配置待定。

任务监督通过输出误差促使状态保留临床任务需要的信息。这里使用下游任务形式作为预训练监督；最终评价需要独立的测试患者和数据，不能直接把预训练任务的训练表现当作可迁移性证据。

#### 2.4.3 Stage 2：Future State Prediction Fine-tuning

从 Stage 1 状态编码器初始化，加入 LWM，使用同患者的当前—未来观测对训练：

\[
\mathcal L_{\mathrm{stage2}}
=
\mathbb E_{(O_t,O_{t^+},h)}
[\mathcal L_{\mathrm{future}}]
+\beta\,\mathcal L_{\mathrm{task}}.
\]

\(\beta=0\) 表示纯纵向微调；\(\beta>0\) 表示保留任务监督。是否 replay、权重与混合比例尚未确定。

未来损失的反向路径是：

\[
\mathcal L_{\mathrm{future}}
\longrightarrow P_\psi
\longrightarrow S_t
\longrightarrow
\{g_\phi,\ U,\ a_\rho\}.
\]

它训练预测器，同时更新产生当前状态的 VLM 可训练部分、slots 和 adapter。未来 \(S^*\) 在该损失中 detach，JEPA 始终固定。

**独立 task decoders 不在 future loss 的计算路径中。** 如果不保留任务监督，它们内部的图像分支和状态条件化层都不会被 future loss 更新；若保留相应任务损失，则继续由这些任务损失更新。

| 参数或计算对象 | Stage 1 任务损失 | Stage 2 未来损失 | Stage 2 可选任务 replay |
|---|---|---|---|
| JEPA \(f_\theta\) | 冻结 | 冻结 | 冻结 |
| Visual adapter \(a_\rho\) | 更新 | 从当前编码分支更新 | 更新 |
| 输入 slot embeddings \(U\) | 更新 | 从当前编码分支更新 | 更新 |
| VLM 的选定可训练参数 | 更新 | 从当前编码分支更新 | 更新 |
| LWM \(P_\psi\) | 本阶段不训练 | 更新 | 不由独立当前任务损失更新 |
| 独立 task decoders 的图像分支、状态投影及融合层 | 更新 | 不由 future loss 更新 | 有对应任务时更新 |
| 每次前向产生的 \(S_t\) | 梯度经过，非独立模型参数 | 梯度经过，非独立模型参数 | 梯度经过，非独立模型参数 |
| 未来目标 \(S^*_{t^+}\) | 不作为该阶段未来目标 | 本次 future loss 中 stop-gradient | 不由独立当前任务损失直接读取 |

表中的“更新”以模块位于当前样本的有效损失路径上为前提。当前仅有报告时不经过 visual adapter；即使未来目标含有影像，该目标路径也已 stop-gradient，因此 adapter 不会从这一条 future loss 获得梯度。一个 batch 中其他带影像的当前样本或任务 replay 仍可提供 adapter 梯度。

目标输出的 stop-gradient 与目标编码参数如何随训练变化是两个问题，需要按下节区分。

#### 2.4.4 当前—未来编码器的参数更新

两端使用同一编码设计，但具体更新策略未定：

| 候选方案 | 两端参数关系 | 目标如何变化 |
|---|---|---|
| 严格共享权重 + stop-gradient | \(\bar\eta=\eta\)，同一套 adapter、VLM 与 slots 两次前向 | 未来端本次不反传，但目标随共享参数更新而改变 |
| EMA target | 目标维护独立参数副本，由 online 参数的移动平均更新 | 目标随 EMA 更新；系数与更新细节待定 |
| 固定 Stage 1 target | 目标使用冻结的第一阶段副本 | 同一随访的目标不再随 Stage 2 online 参数更新而漂移 |

用户当前倾向严格共享权重作为最简候选，但尚未正式锁定。EMA 与固定副本只共享编码设计，不应描述为实时保持相同权重。

Stop-gradient 阻止的是本次损失对目标输出的反向传播；它本身不能证明训练稳定，也不能保证避免表示坍塌。目标更新方式、是否保留任务监督以及训练稳定性需要共同确定和验证。

#### 2.4.5 与 VLA-JEPA 的联系与区别

VLA-JEPA 是 VLM 与 latent world model 结合及分阶段适配的重要参考。其预训练联合使用 Something-Something-v2 人类视频与 DROID 机器人数据，已经学习状态转移，随后适配具体机器人任务。描述时需要保留联合预训练这一事实。

本文借鉴分阶段学习和让 VLM latent tokens 位于预测损失梯度路径中的思路，但状态与训练顺序有区别：

| 比较点 | VLA-JEPA | 当前 MedWorld-JEPA |
|---|---|---|
| VLM tokens 的角色 | Latent action tokens，条件化状态预测 | 医学观测的共享 state tokens |
| 未来目标 | 视觉状态 | 真实未来图文经 VLM 编码的 \(S^*\) |
| 阶段安排 | 预训练时已学习转移，之后适配机器人任务 | 先用临床任务建立状态，再学习纵向未来预测 |
| 医学时间条件 | 不由该文替本文定义 | \(h\) 条件化 observational follow-up prediction |

本文没有沿用机器人 action 语义，也没有照搬 VLA-JEPA 的视觉 target。分阶段安排的具体理由应落在医学任务标注和纵向配对的分工上。

### 2.5 与实验计划的连接：哪些证据能支持 Methods

此处记录方法要求实验验证的内容，完整统计和数据协议继续见正文及附录，不将计划写成结果。

#### 2.5.1 Claim A：Patient-specific future prediction

检验模型能否预测实际患者的未来，至少需要与 persistence、population prior 和匹配预测器配置的静态表征对照比较，并分别报告 stable / changed pairs。

不同 checkpoint 的 latent spaces 可能不同，不能直接比较各自 hidden states 的 cosine。第 3 节的 Table 1 纳入五项评价：Future R@1 在共同临床 schema 中检索；Finding AUPRC 评价疾病分数；其余三项从生成报告计算。现有附录另有为每个 checkpoint 单独训练 observation decoder、将预测 latent 解释到共同 clinical schema 的协议，可保留为补充诊断实验。

若采用附录的冻结 schema readout 协议，这一 decoder 必须与预测器的**目标编码空间**匹配：

1. 固定该 predictor 对应的 target encoder \(E_{\bar\eta}\)。
2. 只使用训练划分内的单次检查 \((E_{\bar\eta}(O),y_{\mathrm{schema}})\) 拟合 observation decoder。
3. 不在该 decoder 的拟合中输入 longitudinal pairs、horizon 或测试标签。
4. 冻结 decoder，评价时仅从 \(\widehat S_{t,h}\) 读取未来 schema，不加入当前或真实未来的原始影像或 \(D\) 特征。

严格共享参数时，目标与当前编码空间一致；采用 EMA 或固定 Stage 1 副本时，要使用对应目标副本产生训练 tokens，不能默认拿 online 编码空间拟合的 readout 去解释另一空间的预测。

随访检索与非 persistence 变化识别已纳入第 3 节的 Table 1；为覆盖直接生成文本的 VLM，主表拟统一从生成报告抽取 schema 进行这两项评价。附录的 latent → 冻结 schema readout 仍可单独报告，不能与报告路径的成绩混填同一列。该冻结 readout 协议也不等同于文本 decoder 的训练协议；后者是否单独适配或与 LWM 联合训练，见第 3.4 节。

#### 2.5.2 Claim B：Reusable frozen current representation

移除预训练 task decoders 与 LWM，冻结当前编码路径；为每个下游任务重新初始化 decoder，包括图像分支与状态条件化层，使用独立数据评价。

全局任务读取 \(S_t\)，空间任务读取同一输入影像 \(X_t\) 与 \(S_t\)。不同状态 checkpoint 使用匹配的图像网络架构、初始化、预处理和训练预算；图像网络可以训练，冻结的是产生 \(S_t\) 的编码路径。这样避免把原有预测器或 task decoder 的训练优势混入表征结论。

病灶负担、疾病变化风险等任务可能更依赖纵向信息；普通器官分割可检验几何信息的保留，不预先要求所有任务都提升。由于 dense decoder 可以依赖直接图像路径，image-only 对照用于检查状态在图像网络之外的贡献；full 与 no-future 的比较则检验纵向训练是否进一步改善状态。

#### 2.5.3 Claim C：真实纵向对应的作用

现有实验计划通过跨患者目标打乱、同患者 source–target 对应打乱和 horizon 标签打乱，分别检查患者身份、时间对应和时间条件的作用。

对照应匹配模型容量、可见训练研究集合、更新预算和数据曝光，并报告实际计算开销。不能只凭完整模型优于较小模型，就把收益归因于真实时间结构。

当前所写的 shuffle、matching、horizon-bin 和统计设置均属于待落实协议。数据是否满足这些匹配条件及最终覆盖率需要报告。

#### 2.5.4 候选消融

优先围绕已明确的模块问题安排候选实验：

- \(K\in\{4,8,16\}\)：slot 数对预测和表征效用的影响。
- Task-conditioned 与 mean-pooled readout：比较任务读取方式，不把状态定义改成固定 pooled vector。
- Frozen / LoRA / partial unfreezing：检验 VLM 权重更新的作用；冻结 VLM 时，adapter 与 \(U\) 仍可能改变其输出。
- 在 LWM 输入中加入 \(D_t\)，但维持 \(S\)-only target：检验额外空间输入。
- Dense task decoder 的 image-only / no-future / full 对照：检验状态条件的作用及纵向训练的额外收益。
- Image-only 空间任务之外的缺失模态训练与评价：作为扩展决定是否开展。

现有附录已列出下面的 dense-task 对照。每一行都重新训练同架构 decoder，保留相同的图像处理与状态条件化层：

| Decoder 输入 | 对照含义 |
|---|---|
| \(X,S_{\varnothing}\)，\(S_{\varnothing}=0\in\mathbb R^{K\times d}\) | 从训练开始使用固定零状态，保留层结构和参数数量的 image-only 对照 |
| \(X,S_{\mathrm{no\mbox{-}future}}\) | 加入未经过纵向监督的状态 |
| \(X,S_{\mathrm{full}}\) | 加入经过真实纵向监督的状态 |

分割的 \(X\) 是待分割图像，SR 的 \(X\) 是 \(X^{\mathrm{LR}}\)。主对照的状态都从 image-only 观测编码，避免报告可用性成为收益来源；带报告的结果单独报告。各行匹配图像预处理、标签子集、训练日程、初始化与停止规则；SR 还匹配退化采样和倍率。Dice、PSNR、SSIM 均为待测指标，具体实现与统计报告待定。

两阶段是否比直接纵向训练更有效、SR 是否改善识别或预测、定向 slot 分工是否有益，也需要独立实验才能形成结论；目前没有把它们写为已验证收益。

### 2.6 引用安排与支持范围

以下是当前 Methods 已使用的引用。参考文献用于解释相关设计先例，不能替代本文组合方案的实验验证。BibTeX 条目见 [main.bib](../27cvpr/main.bib)。

| 文献 / BibTeX key | 放置位置与作用 | 需要保持的边界 |
|---|---|---|
| [I-JEPA，CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html)，assran2023ijepa | JEPA 视觉编码段：从上下文预测目标特征的学习原则 | 不据此宣称医学骨干已重新预训练 |
| [V-JEPA](https://arxiv.org/abs/2404.08471)，bardes2024vjepa | JEPA 视觉编码段：视频特征预测 | 不直接推导医学纵向预测效果 |
| [V-JEPA 2.1](https://arxiv.org/abs/2603.14482)，murlabadia2026vjepa21 | 保留空间 token grid 和 dense features 的动机 | 不等于本文已经完成医学空间迁移 |
| [LLaVA，NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html)，liu2023llava | Visual adapter：视觉特征连接语言 embedding 空间 | 不锁定本文 adapter 的具体结构 |
| [LISA，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Lai_LISA_Reasoning_Segmentation_via_Large_Language_Model_CVPR_2024_paper.html)，lai2024lisa | Task readout：LM hidden states 条件化图像分割 | 其内部图像特征不等于本文 JEPA 的 D；任务 token 不等同于共享 state slots |
| [DINO-WM，ICML 2025](https://proceedings.mlr.press/v267/zhou25t.html)，zhou2025dinowm | State Prediction：在预训练特征上学习 latent dynamics | 不支持“本文必须预测 D”的推论，也没有验证本文 VLM 状态目标 |
| [VLA-JEPA](https://arxiv.org/html/2602.10098v2)，sun2026vlajepa | Two-Stage Training：VLM/world model 结合与分阶段适配 | 保留人类视频 + DROID 联合预训练细节，区别 action tokens、视觉目标与本文状态 |
| [LoRA，ICLR 2022](https://openreview.net/forum?id=nZeVKeeFYf9)，hu2022lora | Stage 1：VLM 的参数高效适配候选 | 具体适配层、rank 等待定 |
| [MIMIC-CXR](https://www.nature.com/articles/s41597-019-0322-0)，johnson2019mimiccxr | Stage 2：同患者影像与报告的纵向数据来源 | 有数据来源不等于已完成 pair 统计和训练 |

此前讨论的 [NeuroVFM](https://www.nature.com/articles/s41591-026-04497-1) 可继续作为固定医学视觉编码器与语言模型连接方式的参考；[Neuro-JEPA](https://arxiv.org/html/2606.14957v2) 可用于医学体积表征的后续方向。它们不是当前 Methods 必须增加的模块；具体 connector 配方和医学预训练方向继续见 0906 讨论稿。

### 2.7 已明确的设计与待定实现

#### 2.7.1 当前已明确

| 项目 | 当前方案 |
|---|---|
| 两个核心目标 | Future state prediction 与 reusable state representation，分别评价 |
| 主要观测组合 | 当前图文 → 真实未来图文的 latent state |
| 当前 state | \(S_t\)，由 VLM 输出槽位 hidden states 构成 |
| 图中的 slots | \(U\) 为八个共享可学习输入 embeddings；\(S_t\) 为同位置八个输出，shape 均为 \(8\times d\) |
| 视觉路径 | 固定 JEPA → 单层 \(D\) → trainable visual adapter → VLM |
| 文本路径 | VLM 原生 tokenizer / embedding |
| Task query | 在 state 构建后进入 task decoder |
| 空间任务 | 原始 \(X_t\) 直接进入 decoder；SR 的图像与状态两路均只用 \(X^{\mathrm{LR}}\) |
| LWM 输入 / 输出 | 仅 \((S_t,h)\rightarrow\widehat S_{t,h}\) |
| Latent future loss | 仅比较预测 \(S\) 与真实随访编码的 stop-gradient \(S^*\)；新增文本 decoder 的监督单独定义 |
| 训练顺序 | Stage 1 任务状态预训练 → Stage 2 未来预测微调 |
| JEPA 更新 | 两阶段均冻结 |
| SR loss | 标量元素平均的像素 MSE |
| 当前正文与新增实验的衔接 | Table 1 纳入未来检索、Finding AUPRC、变化与报告临床内容评价；文本 decoder、疾病评分与评价空间待同步；不做未来像素评价 |

#### 2.7.2 需要进一步定下的内容

| 待定项 | 需要落实的决定 |
|---|---|
| 目标编码器更新 | 严格共享 + stop-gradient、EMA 或固定 Stage 1 副本；相应稳定性与评价空间 |
| Horizon | 实际间隔或时间区间、embedding 方式、容差窗口与真实随访选择 |
| JEPA 与 VLM 配置 | Checkpoint / revision、JEPA 特征层与网格、VLM hidden width |
| Visual adapter | Projector / MLP / resampler 等具体结构及输入输出 token 数 |
| VLM 训练范围 | LoRA 或部分解冻、适配层、rank 与相关配置 |
| Slots | 最终 \(K\)；是否研究任务定向分工，后者需另行定义与验证 |
| LWM 架构 | Predictor block、时间条件注入与输出映射的具体实现 |
| Task decoders | 分类、分割、疾病识别与 SR 的结构；内部图像处理 backbone、状态投影及条件化方式 |
| Future text decoder | 从 \(\widehat S_{t,h}\) 生成未来文本的条件化方式、初始化、训练及梯度范围；是否复用已有语言 decoder |
| Table 1 五项评价 | 共同检索 schema 与候选池、各方法疾病分数、Findings / Impression 范围、RadGraph / CheXbert 版本、标签及 uncertain 处理；方法名单见第 3.3 节 |
| Task labels | 类别体系、语言指令与答案格式、空间标注 |
| Loss | Latent distance 与归一化、segmentation loss 组合、各任务权重 |
| Stage 2 replay | 是否保留、\(\beta\)、batch 混合与任务采样比例 |
| SR 数据 | 退化方式、倍率、强度规范和任务评价 |
| 训练日程 | Optimizer、学习率、阶段长度、batch size、更新预算 |
| 数据与覆盖 | 训练/验证/测试患者划分、pair 数、模态组合、预处理与纳入排除 |
| 实验协议 | Claims A–C 的可行匹配子集、readout、schema、消融与统计报告 |

CT/NLST、MRI/OASIS 等扩展队列、医学 JEPA 再预训练、CT→X-ray 迁移、未来 dense generation 及完整缺失模态实验，均保留为后续方向，不纳入当前已完成结论。

### 2.8 后续写作与版本衔接

本计划保留旧 paper plan 的科学问题、模块动机和验证逻辑，但统一采用最新 Methods 的接口。后续改 Introduction、Related Work、Experiments 或图示时，以下区别需要同步：

| 旧稿或旧讨论中的表述 | 当前使用的表述 |
|---|---|
| 表征学习为唯一中心，future prediction 只是辅助目标 | 未来预测与可复用表征是并列目标 |
| Typed anatomy / pathology / lesion / trajectory slots 及固定路由 | 通用输出 state tokens，未预先分配任务或临床语义 |
| 输入 learned slots 与输出状态混为一组可直接优化的参数 | \(U\) 是共享输入参数，\(S\) 是对应位置的观测相关输出 |
| \(R=(D,S)\) 或 \(Z=[DW_D;S]\) 是所有模块的统一必读状态 | 状态为 \(S\)；空间任务直接读取原始 \(X\) |
| \(D\) 作为 segmentation / SR decoder 的空间旁路 | 原始 \(X\) 直接分支到 decoder，由 decoder 自己提取图像特征 |
| SR decoder 读取 \(D^{\mathrm{LR}},S^{\mathrm{LR}}\) | SR decoder 读取 \(X^{\mathrm{LR}},S^{\mathrm{LR}}\)，HR 仅作目标 |
| LWM 输入 \(D,S\) 或额外 transition queries，预测未来 D/S | 主方案仅 \(S,h\rightarrow\widehat S\) |
| 未来 D loss 或多分量空间目标 | 仅对未来 state tokens 做 latent loss |
| 未来端默认采用 EMA，或默认同构就等于同权重 | 共享编码设计明确，目标参数更新仍待选择 |
| Future loss 更新在线 JEPA 与独立 task decoders | JEPA 固定；独立 task decoder 的图像和状态条件化参数只由对应任务损失更新 |
| Task query 被删除或等同于 \(S_t\) | 新 Fig. 1 保留 \(Q\)，它在 decoder 端指定任务 |
| 预测 latent 等同于未来分割、SR 或报告生成 | 对这些输出需另外定义解码与监督 |
| 完整未来文本只列为可选扩展 | 本轮将未来临床文本预测纳入 Table 1；补充文本 decoder 和监督，不要求未来影像生成 |

当前正文对应文件：

- [Task Definition](../27cvpr/sections/3_problem_formulation.tex)：由 Methods 内部载入，对应正文 3.1。
- [Methods](../27cvpr/sections/4_method.tex)：Overview、Fig. 1 与正文 3.2–3.4。
- [Experiments](../27cvpr/sections/5_experiments.tex)：与方法一致的 Claims A–C 及候选消融。
- [Implementation appendix](../27cvpr/sections/b_protocol_details.tex)：实现待定项、时间与监督规则、目标空间匹配、评价协议。
- [Fig. 1 可编辑 PPT](../27cvpr/ppt/ppt/fig1_v7.pptx)：输入影像直连空间 decoder 的箭头与标签。
- [当前论文 PDF](../27cvpr/main.pdf)：可对照阅读 Methods 的完整英文表述。

后续 Introduction plan 可写入第 1 节，再围绕这里已经明确的任务定义、模块分工、两阶段动机和待验证问题组织论证。

## 3. Results paper plan：Table 1 未来状态与临床报告预测

### 3.1 第一个实验要回答什么

**给定当前影像、当前可用报告和 horizon，模型能否预测这个患者真实随访时的临床状态，并将其表达为准确的报告？**

Table 1 从三个层面回答这个问题：预测能否匹配真实随访（Future R@1），未来异常与变化是否预测正确（Finding AUPRC、Transition F1），生成报告中的临床信息是否正确（RadGraph F1、CheXbert F1）。保留简短表头，具体平均方式与计算对象在评价协议中定义。

Results 的暂定顺序是：

1. **Table 1：Future state and clinical report prediction on MIMIC-CXR。** 使用本节确定的五项指标，报告各方法适用的评价结果。
2. **Table 2 或后续图：下游任务评价。** 检验状态表征的用途，并包含 Stage 1 checkpoint 对照；具体任务与布局后续讨论。
3. **消融与分析。** 检验二阶段训练、LWM、真实纵向对应、时间条件和 decoder 的作用；补充 stable / changed 与不同 horizon 的结果。

**本轮主表不再放 BLEU-4、ROUGE-L；如需语言匹配结果，将其放入补充表。** 本表也不评价未来图像的 PSNR、SSIM、LPIPS、FID 或像素级分割结果。连接 MIMIC-IV 的生存预测属于另行定义的临床终点任务。

### 3.2 任务与预测接口

对同一患者的当前—随访观测对，本文的预测路径为：

\[
S_t=E_\eta(X_t,C_t),\qquad
\widehat S_{t,h}=P_\psi(S_t,h),\qquad
\widehat C_{t,h}=H_{\omega_{\mathrm{text}}}(\widehat S_{t,h},Q_{\mathrm{text}}).
\]

真实随访观测定义目标状态 \(S^*_{t^+}=E_{\bar\eta}(X_{t^+},C_{t^+})\)。该目标状态用于 latent 训练；报告评价比较 \(\widehat C_{t,h}\) 与 \(C_{t^+}\)。沿用第 2.1.3 节的时间定义，不在 horizon-bin 方案尚未确定时默认 \(t^+=t+h\)。

为计算 Finding AUPRC，另外定义每个异常 \(k\) 的连续预测分数：

\[
q_{t,h,k}=G_{\gamma,k}(\widehat S_{t,h}).
\]

该疾病 readout 与文本 decoder 是两个输出接口。直接 VLM 的报告路径为 \((X_t,C_t,h)\rightarrow\widehat C_{t,h}\)；其疾病评分接口也必须单独定义，不能默认一句生成文本已提供连续疾病分数。各方法可以使用不同内部实现，但共享输入证据、目标异常集合与测试 pairs，并披露各输出接口的监督与训练预算。

文本目标采用 Findings、Impression 还是二者的固定组合仍待确认；所有方法使用相同范围与预处理。本文文本 decoder 与疾病 readout 的临床证据来自 **predicted future latent**，不旁路读取真实未来影像、报告或其编码。真实未来观测在训练时用于 latent target、文本或疾病标签监督，在测试时仅用于评价。Teacher forcing 限于训练，测试生成不读取真实未来文本前缀。

未来文本预测不要求重建未来图像；latent 仍是多模态 state tokens。现有指令式疾病识别只定义疾病名称或标签列表，不能直接视为已具备完整报告生成能力。

### 3.3 Table 1 的五项指标与方法

#### 3.3.1 指标总览

主表采用以下五项指标，均为越高越好；F1 的平均方式写入定义，不延长表头。

| 表头 | 用一句话解释 | 计算所需输出 |
|---|---|---|
| **Future R@1** | 在一组候选随访中，预测结果能否把该患者的真实随访排第一？ | 生成报告及候选真实报告，经同一抽取器转换到共同临床 schema |
| **Finding AUPRC** | 对每种异常，能否把未来真正有该异常的患者排在高分位置？ | 每个异常的连续预测分数与真实未来标签 |
| **Transition F1** | 异常的新出现、消失等变化事件预测对了吗？ | 当前、真实未来与生成未来报告的异常标签 |
| **RadGraph F1** | 生成报告中的临床实体及关系与真实报告一致吗？ | 生成未来报告与真实未来报告 |
| **CheXbert F1** | 生成报告中各种异常的有无，与真实报告一致吗？ | 同一 CheXbert 抽取的预测与真实未来标签 |

F1 同时考虑“预测出来的有多少是真的”（precision）和“真实存在的有多少被预测到”（recall），即 \(F1=2PR/(P+R)\)。几个 F1 的计算对象不同；但它们也不是彼此独立的证据，尤其 CheXbert F1 与标签转移 F1 共享未来标签。所有文本抽取指标均受抽取器误差与报告记录范围限制。

#### 3.3.2 Future R@1：真实随访检索

**评价患者特异性：模型预测的内容能否区分该患者的实际随访与其他患者的随访。**

为覆盖直接生成报告的 VLM，主表拟使用统一的“报告 → 临床 schema”路径。固定抽取器 \(T\)，得到预测 schema \(\widehat z_i=T(\widehat C_i)\) 和候选真实 schema \(z_j=T(C_j)\)，用预先固定的距离排序。初版采用可评价异常字段上的平均类别不匹配率；预测未提及或 uncertain 不得通过缩小评价掩码获益。

每个 query 的候选池暂沿用附录方案：**1 个真实随访 + 31 个其他患者的真实随访**。负例匹配 horizon bin、检查条件、粗粒度当前状态与真实标签可评价范围；具体分层、可行样本量与纳入覆盖率需要验证。候选池使用固定随机种子生成，所有方法共享；不能根据某模型预测结果选择负例。

将检索分数定义为距离的负值。若最高分并列，真实随访属于 \(r_i\) 个并列第一候选之一时记 \(1/r_i\) 分，否则记 0 分；对 queries 取平均即 Future R@1，范围为 \([0,1]\)。报告总分，并补充 stable / changed 分组和检索子集规模。

共同 schema 中，不同患者可能具有相同异常组合。并列必须保留；R@1 表示候选池内的临床随访匹配能力，不等于完整的患者身份识别，也不能单独证明变化预测正确。附录的“predicted latent → 冻结 schema readout”是另一条评价路径，应独立标注，不与主表报告路径混用。

#### 3.3.3 Finding AUPRC：未来异常的评分能力

**逐异常评价未来阳性患者能否获得更高预测分数。** 对固定异常集合 \(\mathcal D\)，取预测分数 \(q_{i,k}\) 与真实随访标签 \(y^+_{i,k}\)，改变判定门槛，得到 precision–recall 曲线。

主实现拟采用 non-interpolated Average Precision（AP），再对异常等权平均：

\[
\mathrm{Finding\ AUPRC}
=\frac{1}{|\mathcal D|}\sum_{k\in\mathcal D}\mathrm{AP}_k,
\qquad
\mathrm{AP}_k=\sum_n (R_{k,n}-R_{k,n-1})P_{k,n}.
\]

表头沿用 AUPRC，表注注明采用 macro AP；不与梯形积分 PR-AUC 混用。取值为 \([0,1]\)，越大越好。AP 需要连续疾病分数，不要求已校准概率，也不评价概率校准；不同阳性率的数据子集不能只凭 AP 大小判断难度，应同时给出病种支持数与阳性率。[AP 实现说明](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)

本文可用 \(G_\gamma(\widehat S)\) 输出分数；开放 VLM 可采用明确训练的疾病 readout 或固定的标签评分接口。GPT-5.6 的拟议方案是另行通过固定结构化 prompt 获取逐异常数值评分，并标注其为模型报告的分数，而非已校准概率。具体接口在验证集确定，报告各方法使用的标签监督、提示与预算。接口未实现前属于待测；最终只生成硬标签而没有连续评分的方法，此格标“—”。

**不能使用 CheXbert 对一句生成文本的解析置信度代替患者未来疾病分数。** 单份报告抽出的有／无标签也不提供完整的置信度排序。本项与 PDF 的 Next-change AUPRC 不同：这里从预测未来状态评价未来异常；后者从冻结的当前状态评价后续变化风险。

#### 3.3.4 Transition F1：异常变化事件

**评价从当前到随访发生了什么变化。** 主表统一从报告标签构造转移，不混用本文疾病评分头与其他方法的文本输出：

\[
e_{i,k}=(y^t_{i,k},y^+_{i,k}),\qquad
\widehat e_{i,k}=(y^t_{i,k},\widehat y^+_{i,k}),
\]

其中当前、真实未来与预测未来标签分别由同一固定 CheXbert 对 \(C_t,C_{t^+},\widehat C_{t,h}\) 抽取。

| 当前 → 未来 | 事件 |
|---|---|
| 无 → 有 | 新出现 |
| 有 → 无 | 消失 |
| 有 → 有 | 持续存在 |
| 无 → 无 | 持续无异常 |

主分数拟采用**非 persistence 事件的 Macro-F1**：分别计算每个异常“新出现”和“消失”的 F1，再等权平均。统计使用全部可评价 pairs，因此稳定病例上误报变化也计入 false positives；不是只挑真实发生变化的病例。稳定状态与四类转移的完整结果放入分组分析。

评价掩码由真实当前与真实未来标签共同确定，不能因模型漏写某个异常而删除该条评价；uncertain、未提及和模型弃权的处理规则在评价前固定，并报告覆盖率。若只有二值有无标签，本项不能评价“轻度变重度”等改善／恶化；增加这些事件需要严重程度或人工变化标注。

本项按疾病及状态转移计分，不等同于“improved、stable、worsened”等变化词汇的匹配 F1。CheXGround 的 temporal F1 属于后者，可作相关评价参考，不能直接替代这里的定义。[CheXGround 评价协议](https://arxiv.org/html/2608.30758v1)

#### 3.3.5 RadGraph F1：报告临床实体与关系

**检查生成报告里的临床内容，而不只看字面措辞。** 用同一固定 RadGraph 抽取器和同一评分实现处理生成与真实未来报告，比较实体及其关系构成的计分条目。令两侧条目集合为 \(A_i\) 与 \(B_i\)，计算 precision、recall 及其调和平均 F1，再按预先固定的报告级聚合方式汇总。

它比单纯异常有无包含更多结构信息，例如异常与解剖位置的关联；但不是人工临床正确率，也不保证覆盖全部严重程度、时间或因果信息。需要固定 RadGraph checkpoint、评分变体、条目匹配规则和空集合处理，不能将不同实现都笼统称为同一成绩。[RadGraph 评价参考](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499844/)

#### 3.3.6 CheXbert F1：报告中的异常标签

**检查生成报告是否写对未来有哪些异常。** 用同一冻结 CheXbert 分别处理 \(\widehat C_{t,h}\) 和 \(C_{t^+}\)，在固定异常集合 \(\mathcal D\) 上计算每个异常的阳性 F1，再等权平均：

\[
\mathrm{CheXbert\ F1}
=\frac{1}{|\mathcal D|}\sum_{k\in\mathcal D}
\frac{2TP_k}{2TP_k+FP_k+FN_k}.
\]

表头写 CheXbert F1，表注写明为 **label Macro-F1**，不是 CheXbert embedding cosine。它从最终报告计算，不要求另训练疾病头；Finding AUPRC 则从模型疾病分数计算，两者可以不同。例如疾病头判断正确，并不保证文本 decoder 没有漏写该异常。[CheXbert 官方实现](https://github.com/stanfordmlgroup/CheXbert)

Finding AUPRC、Transition F1 与 CheXbert F1 的异常集合及适用标签映射应共同制定。当前 PDF 中的六种异常可作为起点，最终清单仍待数据核查；不默认把“未提及”全部视作“无”。各指标按其所需的真实状态定义可评价掩码，并报告有效样本数。没有真实阳性支持的病种／事件如何计分或记为不适用，需预先固定；宏平均分母不能随模型输出变化。

#### 3.3.7 主表方法与布局

主比较承接本轮的方法阵容：**Qwen3.5-27B、MedGemma-27B、GPT-5.6、LLaVA-Rad、UniRG-CXR、MedWorld-JEPA**。LLaVA-Rad 与 UniRG-CXR 是两个论文方法行；具体训练与评价配置仍待落实。

| Method | Future R@1 ↑ | Finding AUPRC ↑ | Transition F1 ↑ | RadGraph F1 ↑ | CheXbert F1 ↑ |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-27B | 待测 | 待测 | 待测 | 待测 | 待测 |
| MedGemma-27B | 待测 | 待测 | 待测 | 待测 | 待测 |
| GPT-5.6 | 待测 | 待测 | 待测 | 待测 | 待测 |
| LLaVA-Rad† | 待测 | 待测 | 待测 | 待测 | 待测 |
| UniRG-CXR† | 待测 | 待测 | 待测 | 待测 | 待测 |
| **MedWorld-JEPA** | 待测 | 待测 | 待测 | 待测 | 待测 |

“待测”表示计划中的评价，尚无结果；“—”表示按当前方法输出不适用，不表示零分。Finding AUPRC 的待测格以疾病评分接口落实为前提。

| Method | 本表中的角色与具体来源 | 适配与记录要求 |
|---|---|---|
| Qwen3.5-27B | 通用多模态直接预测；[官方 checkpoint](https://huggingface.co/Qwen/Qwen3.5-27B) | 输入当前图文与 horizon，适配未来报告和疾病评分；是否与本文 VLM backbone 相同需随最终配置确认 |
| MedGemma-27B | 医学多模态直接预测；[google/medgemma-27b-it](https://huggingface.co/google/medgemma-27b-it) | 使用多模态版本，明确当前报告与 horizon 的输入模板，以及纵向微调范围 |
| GPT-5.6 | 用户拟定的闭源 API 对照 | 名称暂记为 GPT-5.6；运行前核实可用 API model ID / snapshot，固定提示、推理配置、调用日期与输出缓存，不推定存在可用微调接口 |
| LLaVA-Rad† | Nature Communications 2025 的 CXR 报告生成方法；[论文](https://www.nature.com/articles/s41467-025-58344-x)、[代码](https://github.com/microsoft/LLaVA-Rad)、[权重](https://huggingface.co/microsoft/llava-rad) | 保留胸片编码器与语言生成架构，加入当前报告和 horizon，使用本文 pairs 适配未来报告预测 |
| UniRG-CXR† | 2026 预印本中的 SFT + RL 报告生成方法；[论文](https://arxiv.org/abs/2601.17151)、[权重](https://huggingface.co/microsoft/UniRG-CXR) | 从公开 checkpoint 适配本文任务；仅继续微调不等于完整复现其 GRPO 与奖励训练方案 |
| MedWorld-JEPA | 当前图文 → 预测未来 latent → 未来报告及疾病分数 | 固定文本 decoder 与疾病 readout 的训练日程，记录 latent target 更新方式 |

† 表示对原论文方法的任务适配，不是原文已报告“未观测未来影像条件下的报告预测”。它们的原任务读取待描述时点的影像；本表仅允许读取当前截止时间前的证据，不得把真实未来影像输入这些模型。

开放模型主比较拟使用相同纵向 pairs、目标报告及异常标签适配。GPT 若只通过 prompting 调用，应明确标为 API / prompting baseline，排版时与 longitudinally fine-tuned 方法分组；不能把训练监督的差异解释成纯架构优势。外部 checkpoint 的既有预训练并非完全匹配，同 backbone 和 matched controls 承担方法归因。

X-WIN 作为 encoder / representation 对照，暂不替换以上六个主比较方法。EHRXDiff 的未来影像输出与中间 EHR 输入不符合本表现有接口，保留为相关工作。AUROC、二分类 Brier 可作为 Finding AUPRC 的补充评价；C-index 与生存 Brier 需要另设时间到事件任务。

### 3.4 文本 decoder、疾病 readout 与训练方案的衔接

**保留 latent world modeling，不因文本评价而取消二阶段训练。** 当前 latent future loss 仍比较 \(\widehat S_{t,h}\) 与 \(S^*_{t^+}\)；文本 decoder 另外需要学习从状态读取报告内容。Latent loss 本身不会更新不在其计算路径上的 decoder。

可以考虑以下两种协议，最终主协议仍待选择，不能在论文中混用它们的监督描述：

| 协议 | 怎么训练 | 结果支持什么结论 |
|---|---|---|
| 单独适配文本 decoder | 完成 latent 训练后冻结状态编码器和 LWM；用训练集中的预测 latent 与真实未来文本训练 decoder | 评价固定预测表示在有监督文本读出后的用途；需计入 decoder 的纵向监督预算 |
| Stage 2 联合文本监督 | 在 latent future loss 和保留的任务监督之外，加入预测 latent → 真实未来文本的生成损失 | 评价联合优化的未来文本预测系统；明确文本损失是否更新 LWM、当前编码器及 decoder |

文本监督可写为：

\[
\mathcal L_{\mathrm{text}}
=-\sum_j\log p_{\omega_{\mathrm{text}}}
\big(c_{t^+,j}\mid c_{t^+,<j},\widehat S_{t,h},Q_{\mathrm{text}}\big).
\]

若采用联合方案，可在现有 Stage 2 目标中增加 \(\lambda_{\mathrm{text}}\mathcal L_{\mathrm{text}}\)；权重、梯度范围及训练日程待定。此时应描述为“latent prediction 与文本任务监督联合训练”，不能再声称整个系统的所有损失都在 latent space。

为进一步区分状态预测与解码误差，可另做冻结 readout 诊断：固定目标编码器，用训练集真实 target latent 与对应文本拟合 decoder，再冻结并分别读取真实 target latent 和 predicted latent。真实 target latent 含未来报告时，这个参考实验包含文本重建成分，应标为 observation-conditioned reference，不与可部署的未来预测方法混排，也不称为严格性能上界。该诊断与第 2.5.1 节的 schema readout 思路一致，但不是已经选定的主训练协议。

**Finding AUPRC 新增的 readout 也需要监督。** \(G_\gamma\) 可在冻结 encoder / LWM 后，用训练集预测 latent 与真实未来异常标签拟合；也可将异常分类损失加入 Stage 2 联合训练。需要分别记录它是否更新当前编码器、LWM 及自身参数，并计入标签与训练预算。若改用第 2.5.1 节的 observation-only readout，则只在单时点目标编码与标签上拟合，明确与纵向拟合的区别；不能把两种成绩混填而不标协议。

### 3.5 支持未来预测结论的对照与分组分析

- **Copy current report**：直接将当前报告作为未来文本输出，与其他方法使用相同文本范围和评价脚本，检验是否超越病情持续不变的假设。
- **Population transition prior**：仅用训练集拟合当前粗粒度异常与 horizon 条件下的未来分布，在共同 schema / 疾病评分实验中评价，检验收益是否超出群体转移规律。它不自动提供自然语言报告，未定义的报告指标标“—”。
- **同 backbone 直接 VLM**：若本文使用 Qwen3.5-27B，该主表行承担直接生成对照；若最终 backbone 改变，另补同 backbone 对照。
- **Stage-1 encoder + matched LWM / decoder / disease readout**：固定第一阶段状态编码器，训练匹配预测器及读出，与完整第二阶段适配比较。不能把未训练 LWM 的 Stage-1 checkpoint 当作现成预测模型，也不能将这个仍接受纵向监督的系统简称为完全 no-future。
- **预算匹配的 no-future encoder**：按 PDF 原协议，在相同训练检查集合与更新预算下继续单时点任务训练，再冻结并接匹配预测器与读出。它与直接冻结 Stage-1 的对照不同，用于排除额外训练曝光的解释。
- **w/o latent loss**：若采用 Stage-2 联合文本或疾病监督，保持架构、初始化与其他损失相同，只移除 latent future loss，检验 latent target 的额外贡献。
- **变化、稳定与 horizon 分组**：主表使用预先确定的评价 pairs，各指标报告其有效样本数；Future R@1 另报检索纳入子集。按真实标签定义 stable / changed，并按 horizon 分层，不根据模型输出挑选病例。

上述对照放在 Table 1 的独立 panel 或后续消融中，不替换六个主要方法。病例图并列当前报告、真实未来报告、各方法生成报告与异常评分，重点展示新出现、消失和持续存在的异常。只有存在严重程度或变化标注时，才进一步评价改善／恶化。

训练、验证和测试按患者划分；各方法共享评价 pairs、horizon 规则、目标文本和指标实现。模型选择及生成参数在验证集确定。若为主表报告区间，优先按患者 bootstrap，以保留同一患者多次检查之间的相关性；具体重复训练与区间报告方式待定。

### 3.6 写 Results 前需要落实的内容与参考

五项指标与六个方法的表格已确定为本轮计划；下列实现细节仍待落实，不预填数值：

| 项目 | 需要固定的内容 |
|---|---|
| 输入与目标 | 当前报告可用时间、Findings / Impression 范围、影像视图与预处理；禁止目标时点影像或报告进入预测输入 |
| Pairs 与 horizon | 患者划分、随访选择、时间编码、各 horizon 数量、stable / changed 定义及支持数 |
| Future R@1 | 共同抽取 schema、距离、32 候选池的可行匹配规则、随机种子、并列与子集覆盖率 |
| Finding AUPRC | 各方法连续疾病分数的来源、readout 或 prompt、macro AP 实现、标签监督与评分预算 |
| 三项 F1 | 异常集合、uncertain / 未提及 / 弃权、零支持类别、RadGraph 版本和评分变体、CheXbert checkpoint |
| 模型与训练 | 开放模型 checkpoint / revision、API 可用 ID / snapshot、微调或 prompting 分组、decoder / disease readout 训练协议、监督与更新预算 |
| 报告与统计 | 各指标有效样本数与量纲、患者 bootstrap、重复训练、horizon 与变化分组；“待测”和“不适用”分开 |
| 正文同步 | Methods 补文本与疾病读出；Experiments / Results / Fig. 1 同步五项评价，并修订原 Claim A 的检索和变化终点表述 |

Results 先报告未来状态与临床内容的表现，再用 persistence、population prior、直接 VLM 和 matched controls 解释增益。报告 F1 高不自动证明 reusable representation 更好。未来预测、状态表征用途与真实时间对应的贡献继续分别验证，不能提前写“显著提升”。

| 参考 | 支持本计划的内容 | 不应外推的结论 |
|---|---|---|
| [SMB-Structure，2026，§3–4](https://arxiv.org/html/2601.22128) | 冻结患者表示后用 linear-probe AUROC 比较 SFT、SFT + JEPA 与 curriculum，支持本文下游表征实验 | Probe AUROC 不直接测 predicted future latent 的准确度；其未来序列部分 masking 不等同于本文完全不读未来证据的预测接口 |
| [EHRWorld，2026，§5.3](https://arxiv.org/html/2602.03569v1) | 区分单步与多步预测，检查临床状态、变化病例和误差累积 | 本文当前直接预测 horizon，不自动具备递归 rollout；Retention Rate 不直接加入本表 |
| [VL-JEPA，§2、§4.5](https://arxiv.org/html/2512.10942v2) | 预测 embedding 后由文本 decoder 输出文本，解码与主体 embedding 预测训练可分开 | 其目标为文本 embedding，不等于本文多模态患者状态已自动具备相同解码能力 |
| [LLaVA-Rad，Nature Communications 2025](https://www.nature.com/articles/s41467-025-58344-x) | 提供有公开实现的 CXR 报告生成方法，作为适配 baseline | 原文已有影像条件下的报告生成成绩不能搬成本文未来预测成绩 |
| [UniRG-CXR，2026](https://arxiv.org/abs/2601.17151) | 提供 SFT + RL 的报告生成方法及公开 checkpoint，作为适配 baseline | 继续微调公开权重不等于完整复现 RL 方法；纵向报告生成仍读取待描述时点的影像 |
| [CheXGround，2026](https://arxiv.org/html/2608.30758v1) | 同时报临床报告指标与 temporal F1，说明需要单独检查变化信息 | 其 lexical temporal F1 不能直接称为本文按疾病状态转移计算的 Transition F1 |
| [AP 实现说明](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) | 明确 AP 的排序分数输入、非插值积分方式及 macro 平均 | AP 与梯形 PR-AUC 不同，疾病评分也不能用文本解析置信度替代 |
