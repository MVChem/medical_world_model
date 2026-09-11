# MedWorld-JEPA：Method 讨论整理

2026-09-06 整理，2026-09-07 根据最新正文和图像输入接口讨论更新｜供审阅。本文整理已讨论的主线，具体实现尚未确定的地方明确标出。

Method 暂按四节组织：**Task Definition → State Representation → State Prediction → Two-Stage Training**。

State Representation 首先单独介绍 **Shared Multimodal State Encoding（当前与未来状态的共享多模态编码）**，突出当前端与未来目标端的双分支结构。两个时点都允许仅影像、仅报告或影像与报告共同输入。

**当前主要设定：当前影像＋当前报告 → 未来影像＋未来报告的 latent state。**缺失模态保留为编码接口能力及后续可能的消融，当前不要求覆盖全部模态组合的训练与评价。

**本轮明确的接口：LWM 主方案接收当前 state tokens S_t 和 horizon h，只预测未来 state tokens，与真实随访经 VLM 编码的目标做 loss。任务 decoder 按任务读取 S_t，并在分割、超分等空间任务中直接接收任务输入图像 X，而非 JEPA 输出 D。** 当前与未来都保留影像经过 JEPA、adapter 进入 VLM 的状态编码路径；D 只在这条路径中提供视觉证据，不另接空间 decoder，也不是未来预测输出或独立损失目标。超分的图像分支接收 X^{LR}，状态也只由同一 LR 输入构造，HR 仅作监督目标。

## 1. Task Definition

**方法主体是一个 medical latent world model：表示当前医学状态，并预测它随时间的演化。** State Representation 建立模型的状态空间，State Prediction 学习该空间中的状态转移。

我们有两个需要独立验证、又相互联系的目标：

- **Future state prediction**：根据当前观测和给定的时间条件，预测未来状态。预测未来本身就是模型的核心任务。
- **Reusable state representation**：学习能够支持分类、分割、疾病识别和超分辨率等多种任务的状态表示。任务监督和未来监督共同参与表征学习。

这一节围绕图文→未来图文的主设定定义输入和输出，再简要说明缺失模态的兼容方式：

- **观测**：每个时点可以有影像 X、报告或其他临床文本 C，或两者都有，至少一种可用。当前和未来采用相同的观测定义，两个时点的可用模态可以不同；不要求每个纵向样本都是影像—影像配对。
- **当前状态**：VLM 根据当前可用观测输出 state tokens S_t，作为共享多模态状态。影像先经过 JEPA 得到空间特征 D_t，再经 adapter 进入 VLM。用模态标记 m_t 记录可用输入，统一以 E_η(O_t)=S_t 表示状态编码；D_t 是视觉路径中的中间特征，不另定义为 state 的组成分量。空间任务所需的局部证据由输入图像 X_t 直接交给 task decoder。使用 “state” 描述 S_t，不将其等同于完整的 patient state。
- **预测条件**：给定的时间跨度或 horizon，记为 h。
- **预测目标**：由真实随访影像、未来报告或两者经过 VLM 编码得到的未来 state tokens S*_{t⁺}。LWM 输出预测 tokens Ŝ_{t,h}，不输出未来 D。未来报告本身也可以构造状态目标，仅报告的随访可提供未来监督。实际随访时刻记为 t⁺，不要求恰好等于 t+h；具体时间条件的定义仍需确定。

编码接口兼容仅影像、仅文本等输入组合，便于后续利用不完整数据扩大训练规模。这些组合可以在 Results 中作为缺失模态消融，具体是否开展和如何训练留待实验安排，不作为当前主要任务逐项展开。

Task Definition 后用一段简短的 **Overview** 引出 [Fig. 1](../26iclr/imgs/fig1.pdf)，再展开后续模块。Overview 的主线可以写成：

> 我们提出 MedWorld-JEPA，一个同时建模当前状态与未来演化的 latent world model。同一套多模态编码流程分别处理当前图文与真实随访图文：JEPA 提取空间特征，经 visual adapter 接入 VLM，VLM 在 learned slots 对应位置输出共享 state tokens。任务 decoder 根据任务读取 state tokens；分割和超分同时接收任务输入图像，通过 decoder 自身的图像分支提取局部证据，并用 state tokens 提供语义条件。任务损失训练 decoder 及状态编码路径。Latent World Model 根据当前 S_t 和时间条件 h 预测未来 state tokens，与真实未来图文经 VLM 编码的 S*_{t⁺} 对齐。

Fig. 1 后续围绕当前图文与未来图文展示两条编码分支，箭头应明确：

- 当前 S_t 送入 task decoder 与 LWM；h 单独送入 LWM。
- 输入图像 X_t 在 JEPA 之前分出一路，直接送入空间任务 decoder；超分时这一路必须是 X^{LR}。D_t 只连接 visual adapter，不另接 task decoder 或主方案中的 LWM。
- LWM 只输出 Ŝ_{t,h}，与未来 VLM 输出的 S*_{t⁺} 计算 latent prediction loss；删除预测 D 与对应 loss 连线。
- 8 个 learned slots 是 VLM 输入中的可学习 embeddings；同位置输出的 8 个 hidden states 组成 S_t，不能另画成与 S_t 无关的一路信息。
- task query Q 不等于 S_t 或 U。保留新版图中的 task query 方框，如 “Segment the lungs”，它在状态构建之后条件化 task decoder，不进入状态编码器或 LWM；固定分类头等接口可由 head 隐含任务选择。图中的 task decoder 概括一组任务接口，不表示各任务必须共用全部参数。

缺失模态的兼容性可用简短注释说明。图的图像旁路、图注和正文应保持上述接口一致；图文双分支展示主要纵向设定，分割与超分的状态按下面定义使用 image-only 输入。

写作时必须保留两项能力的地位。不能把整篇方法仅定位为 representation learning，再把 future prediction 降为辅助 loss。Results 分别验证预测质量和表征的下游效用。

## 2. State Representation

### Shared Multimodal State Encoding：当前与未来状态的共享多模态编码

**当前状态和真实未来状态使用同一种编码流程与表示接口。** 对任一时点 τ，输入由可用影像 X_τ、文本 C_τ 及模态标记 m_τ 构成。VLM 的 learned slots 为三种输入组合提供固定数量、固定维度的 state tokens S_τ；有影像时，JEPA 的 D_τ 用于构造 VLM 视觉输入。空间任务另从其原始输入图像提取局部证据，该图像分支属于相应 decoder。

| 该时点可用的观测 | 编码路径 | 状态与任务输入 |
|---|---|---|
| 影像与报告（主要纵向设定） | 两种输入分别适配后共同进入 VLM | 融合后的 S，用于状态预测或适用的任务读取 |
| 仅影像 | 影像 → JEPA → visual adapter → VLM | 由影像得到的 S；分割与超分还将同一任务图像 X 直接交给 decoder |
| 仅报告 | 报告 → VLM tokenizer／词嵌入 → VLM | 由报告得到的 S；没有图像的样本不承担空间任务 |

分割和超分在当前方案中使用 image-only 状态；超分同时在状态编码路径与 decoder 图像路径使用同一 LR 图像，不引入 HR 图像或从 HR 得到、可能泄露目标信息的报告。上述输入安排定义训练协议，不表示相关效果已经得到验证。其他缺失模态组合仍是可选实验。

当前分支编码当前可用证据，未来目标分支编码真实随访证据；两端的模态组合无需一致。未来报告直接参与未来 state 的编码，通过 latent prediction loss 提供监督。共享编码在这里指两个时点使用同一种多模态状态定义与编码设计；用户倾向同一套权重分别前向计算，目标端的具体更新方式在 Training 中讨论。

写作中用“当前—未来双分支共享编码”说明这里的“双向”。它不要求 VLM 采用双向 attention，也不意味着已经训练未来→当前的反向时间预测。当前与未来分别编码，本次预测的当前分支只读取预测时刻已经可用的信息。

共享 slots 提供统一的 latent 接口；跨模态语义是否充分对齐仍需训练与验证。利用同一时点的图文配对、交替使用不同输入组合，是可考虑的训练安排，尚不将额外一致性 loss 写成已确定目标。

### JEPA 提供视觉表征

JEPA 是需要强调的核心视觉模块。我们重视的是通过上下文预测学习视觉结构的思想，希望视觉表示保留超出单一诊断标签所需的信息。

一个重要动机是：训练时利用完整 CT volume 的上下文监督局部切片，使局部观测的表示包含更丰富的空间信息；进一步探索将 CT 中的三维知识用于 X-ray 表征。医学 JEPA 的重新预训练和这些空间迁移实验尚未完成，当前可以解释动机与设计方向，之后依据实际实验补充预训练方法及结论。

影像路径从 JEPA 输出的空间特征 D_τ 开始，当前与未来均适用。现阶段采用单层空间 tokens 的表述，保留它们的空间结构，不沿用旧稿复杂的多尺度 hierarchy。仅报告的观测跳过视觉编码路径。

### Visual adapter 连接 JEPA 与 VLM

连接关系为：**影像 → JEPA → visual adapter → VLM**。

当前与未来的影像路径都使用这一连接设计。文本使用 VLM 原生 tokenizer 与词嵌入进入 VLM；visual adapter 负责 JEPA 视觉特征的适配。

统一使用 **visual adapter / connector**，不提前限定必须有 resampler：

- Projector 可以用线性层或 MLP 转换特征维度，通常保持 token 数量。
- Resampler 可以用可学习 queries 和 attention 聚合视觉信息、压缩 token 数量。
- 可以采用已有的 Transformer、projector 或两者组合；连接模块的结构不作为主要创新点。

### Learned slots 与 VLM 输出的 state tokens

**S_τ 就是 learned slots 对应位置经过 VLM 后输出的 hidden states 集合。** 需要区分可学习的输入 embeddings 和依赖观测的输出：

\[
U=[u_1;\ldots;u_K]\in\mathbb R^{K\times d},
\qquad
S_\tau=g_\phi([V_\tau;T_\tau;U];m_\tau)\big|_U
\in\mathbb R^{K\times d}.
\]

其中 V_τ 为经 adapter 适配的视觉 tokens，T_τ 为原生文本 embeddings，d 为 VLM hidden dimension。视觉和文本输入可各自缺省，但至少一种观测可用。

- **U：learnable slot embeddings。** 是跨样本共用的模型参数，由训练更新；不是每个患者单独保存的一组参数。
- **S_τ：state tokens。** 是该次输入经过 VLM 后，在 U 对应位置得到的输出，随患者和观测变化。S_τ 是计算结果，不是额外存储并独立优化的模型参数。
- **K=8 的含义。** 图中暂用 8 个 slots，因此 S_τ 是 8 个 d 维向量，形状为 8×d。它不是默认池化为一个向量，也不是额外于 slots 的第 9 个 token。当前 S_t、预测 Ŝ_{t,h} 与未来目标 S*_{t⁺} 都采用相同的 K×d 接口。最终 K 仍是实现参数，8 不代表已验证的最优数量。

slots 不预先指定 anatomy、pathology、lesion 等语义职责，也不为每个 slot 提供独立的语义真值。任务损失和未来损失通过 VLM 输出的 S 反向传播，促使这些位置形成有用的表示。

共享状态先于任务指令和预测 horizon 构建。S_τ 是三种输入组合共用的状态接口；D_τ 是影像可用时进入 visual adapter 的中间空间特征。无需再以 R_τ=(D_τ^{optional},S_τ;m_τ) 定义复合状态，也不需要将 D 与 S 统一投影拼接成一个全局表示。

LWM 主方案只接收 S_t 和 h。分类读取 S_t；疾病识别读取 S_t 与 instruction；分割、超分直接读取各自的输入图像 X 与 S。空间 decoder 可包含图像编码器或恢复 backbone，再通过 attention、调制或其他条件化方式引入 S；图像特征提取及条件化模块都归属 decoder，其结构后定。这里的图像特征由 decoder 自己从 X 计算，不是从 JEPA 的 D 接来的旁路。缺报告时仍输出 K 个 S tokens；缺影像时不计算 D。若使用占位 tokens 批处理，须通过 mask 标记缺失。

### Task decoder 为 state tokens 提供监督

Task decoder 是表征学习的一部分。**优化目标是任务输出与真值之间的误差；该误差共同训练 decoder、slot embeddings U、VLM 的可训练参数和 visual adapter。** 它不是仅训练 decoder，也不是给 U 或每个输出 slot 指定一个应逼近的固定向量。

每个样本只承担其输入和标注支持的任务。分割与超分训练使用 image-only 状态及相应空间目标；仅报告样本可通过适用的语言任务或纵向状态预测参与训练。所有分支只接收该任务允许的证据，不把答案或直接泄露答案的文本放入状态输入。

| 任务 | Decoder 输入 | 输出与监督真值 | Loss 的基本方向 |
|---|---|---|---|
| 分类 | S_t | 类别或多标签概率，与类别标签比较 | CE / BCE |
| 分割 | X_t、由该图像构造的 S_t；按任务需要提供 Q_seg | 预测 mask，与标注 mask 比较 | Dice 与 CE/BCE，具体组合后定 |
| 指令式疾病识别 | S_t、instruction | 规范化疾病名称或标签列表，与标注序列比较 | Token-level CE |
| 超分辨率 | X^{LR}、仅由同一 LR 图像构造的 S^{LR} | 预测高分辨率图像，与对应 HR 真值比较 | 像素级 MSE |

统一记任务索引为 q、任务请求为 Q_q、相应 decoder 为 H_{ω_q}。全局任务可写成 H_{ω_q}(S_t)，分割写成 H_{ω_seg}(X_t,S_t,Q_seg)；需要指令时再给 decoder 提供 Q_q。task query 是任务条件，与 VLM 输出的 S_t 不同。疾病识别的语言 decoder 从 S_t 读取状态，不直接旁路读取原始图文；具体 decoder 结构仍待确定。

各任务损失构成：

\[
\mathcal L_{\mathrm{task}}
=\lambda_{\mathrm{cls}}\mathcal L_{\mathrm{cls}}
+\lambda_{\mathrm{seg}}\mathcal L_{\mathrm{seg}}
+\lambda_{\mathrm{dis}}\mathcal L_{\mathrm{dis}}
+\lambda_{\mathrm{SR}}\mathcal L_{\mathrm{SR}}.
\]

每个样本只计算其输入与标注支持的损失项。任务损失一方面更新 task decoder 内部的图像特征提取、恢复及 state-conditioning 模块，另一方面经 **task decoder → S_t → VLM／U／visual adapter** 更新状态编码路径；JEPA 在当前方案中冻结。原始图像 X 是输入数据，不是待优化参数。状态构建和任务读取分开，不意味着阻断经过 S 的梯度路径。

分割、超分保留直接读取 X 的图像分支，但有一条经过 S 的梯度路径不等于 decoder 一定充分利用 S。按现有附录协议，分别训练使用固定全零状态、no-future 状态与完整模型状态的 H(X,S)，图像分支、条件化层、初始化及训练预算保持匹配。全零状态从训练开始固定，构成 image-only 对照，不是仅在测试时移除 S。这里检验的是状态在图像网络之外的贡献，不再以仅读取固定 JEPA D 的 decoder 为基线；不预先保证所有空间任务都会强化共享状态。

“诊断”的范围暂落在疾病识别，不扩展成未经定义的综合临床推理。它与普通分类头的主要区别是指令和输出接口。

论述任务监督的必要性时，强调“显式强化临床任务需要的细粒度信息”。不笼统宣称 JEPA 的 masking 目标必然学不到细节。

超分辨率提供像素层面的恢复监督。训练输入是低分辨率图像，目标是对应的高分辨率图像；配对可以先通过下采样构建。LR 图像一路直接进入恢复 decoder，另一路经固定 JEPA、adapter 和 VLM 构造 S^{LR}。恢复 decoder 利用 X^{LR} 的图像证据与 S^{LR} 的语义条件，让超分 loss 同时训练自身图像分支和产生 state tokens 的可训练编码路径。高分辨率图像只作 target；不把它或基于它得到的报告用于构造 S^{LR}。是否改善疾病识别或未来预测需要实验验证。

超分损失确定为预测高分辨率图像与对应真值之间的 MSE：

\[
\widehat X^{\mathrm{HR}}
=H_{\omega_{\mathrm{SR}}}(X^{\mathrm{LR}},S^{\mathrm{LR}}),
\qquad
\mathcal L_{\mathrm{SR}}
=\frac{1}{N}\left\|\widehat X^{\mathrm{HR}}-X^{\mathrm{HR}}\right\|_F^2,
\]

其中 N 为目标图像的标量元素总数（包含空间位置和通道），预测与真值使用相同的强度尺度。该目标直接约束像素重建误差；其在多任务总损失中的权重留到训练设置确定。

### 文本作为状态观测与任务监督

报告有两种明确用途：**作为该时点的观测参与 state 编码**，以及作为文本生成等任务的监督答案。未来报告作为未来目标分支的观测，即可提供 latent state 监督；它不要求同时存在未来影像，也不要求额外生成整份报告才能参与训练。

有适用的文本任务标注时，计算对应语言 loss；没有文本标注时，跳过该项，缺失文本不作为“正确答案为空”的语言训练样本。训练生成某份报告或疾病标签时，不把同一答案及直接泄露答案的文本预先放入状态输入。疾病识别的接口约束已明确为从 S 和 instruction 解码，具体网络实现留到训练设置中确定。

## 3. State Prediction

预测模块称为 **Latent World Model（LWM）**。当前主方案接收 S_t 与时间条件 h，只输出预测的未来 state tokens：

\[
\widehat S_{t,h}=P_\psi(S_t,h)\in\mathbb R^{K\times d}.
\]

**D_t 不作为主方案中 LWM 的直接输入；LWM 不预测未来 D。** JEPA 的视觉信息仍通过 D_t → visual adapter → VLM → S_t 进入预测路径。S_t 是 LWM 获取当前观测信息的接口，h 则指定预测的时间条件。真实随访只用于目标编码或评价，未来模态的可用性不作为默认的预测输入。

直接加入 D_t 可以作为后续消融：P_ψ(D_t,S_t,h) 可能补充 S 压缩时丢失的空间细节，也可能降低预测器对 S 的依赖；这两点都需实验判断，不宣称必然发生。即使开展该输入消融，输出仍可仅为未来 S，它不要求恢复未来 D 的预测目标。

“预测未来影像／文本”在当前 Method 中首先指预测对应观测所表示的未来 latent state。若需要输出实际图像或完整报告，还需定义相应 decoder 及监督；现有超分和疾病标签接口不能自动等同于未来影像生成和完整报告生成。

### 时间演化与预测机制

当前定义的是给定时间跨度 h 的直接未来状态预测。编码接口允许当前与未来使用不同模态，例如由当前影像编码的 S 预测未来报告编码的 S；这表示 latent 层面的跨模态预测，不等于已经生成对应影像或文本。时间方向仍是当前→未来。

- **Autoregressive rollout**：若将预测的未来状态继续作为下一次输入，逐步预测后续状态，才形成多步自回归 rollout；该实现和训练目标尚未确定。VLM 自身的自回归 token 接口与 LWM 是否按时间递推是两个层面。
- **LSTM**：可以作为递归建模状态演化的实现选择；支持缺失模态本身不要求采用 LSTM。
- **Diffusion**：需要另外定义加噪、去噪目标及采样过程；去噪 step 与临床随访时间跨度 h 分别定义。当前 latent prediction loss 不自动构成 diffusion 模型。

### 纵向数据

MIMIC-CXR 可以提供同一患者的 current–future 观测。当前[附录第一张图](../26iclr/ppt/ppt/appendix_fig_v3.pdf)已有三类示例：A 为约 20.1 小时后的明显变化，B 为约 24 小时后的稳定随访，C 为四次连续检查。它们说明了 pair 和多时点序列的构建形式；最终可用配对数量仍需统计。

纵向训练单位是同一患者在两个时点的可用观测，支持影像、报告或二者的不同组合。具体数据集内各类组合的数量留待统计，不将用户对可用性分布的判断写成已验证的数据结论。

先从两时点预测定义基本任务，多时点序列可以提供多个训练 pair，不据此默认已经实现多步 rollout。

### 未来目标与 latent loss

未来目标分支使用真实随访的可用观测生成 S*_{t⁺}：仅影像、仅报告和图文同时可用均可编码。有未来影像时，JEPA 提取 D*_{t⁺}，经 adapter 进入未来 VLM，用于构造 S*_{t⁺}。**D*_{t⁺} 是视觉编码路径的中间特征，不是 LWM 的独立预测目标。** 未来影像与报告不能进入本次预测的当前状态。

符号统一为：Ŝ_{t,h} 表示预测，S*_{t⁺} 表示真实随访经 VLM 编码的目标。目标不加预测帽号，预测不加目标星号，避免 Ŝ* 同时表达两种角色。

未来损失只比较这两组 K×d tokens：

\[
\mathcal L_{\mathrm{future}}
=\ell_S\!\left(\widehat S_{t,h},\operatorname{sg}(S^*_{t^+})\right).
\]

若采用按元素平均的 MSE，可具体写成：

\[
\mathcal L_{\mathrm{future}}
=\frac{1}{Kd}\left\|\widehat S_{t,h}
-\operatorname{sg}(S^*_{t^+})\right\|_F^2.
\]

latent 距离及是否先做归一化仍待确定；上述 MSE 是候选形式。比较的是预测的未来状态与真实随访编码的状态，而非直接拉近当前 S_t 与未来 S*_{t⁺}。同一套 slots 提供对应的输出位置，但不预先为各位置规定临床语义。

**不保留未来 D loss。** 它对应额外的空间特征预测任务，并非学习未来 S 所必需。未来仅有报告时，同样可以监督 S，无需为 D 构造或选择任何损失项；缺失模态训练覆盖仍需另行验证。

目标编码器如何更新由 Training 定义；共享编码流程不预先锁定优化方案。预测状态的评价只读取 Ŝ_{t,h} 及允许的任务条件，不额外拼接当前图像、真实未来图像或任何 D 特征。当前分割、超分 decoder 还需要对应任务的输入图像；仅有预测的未来 S 时，不能直接复用这些接口来宣称已生成未来密集输出。未来分割或未来图像恢复需要另外定义相应接口与监督。附录中总结整段病程的文字不能整体作为起始时刻输入。

## 4. Two-Stage Training

两阶段是当前主要训练方案：**先建立具有临床含义的状态空间，再学习状态随时间的演化。**

这里的 pretraining 指为后续状态转移学习预训练共享状态表示：以已有 JEPA 和 VLM 权重为起点，学习产生 state tokens 的编码路径及任务接口。它不等同于重新进行医学 JEPA 预训练，也不意味着从头训练整个 VLM。

### Stage 1：Task-supervised State Pretraining（任务监督下的状态预训练）

使用较广泛的单时点分类、分割、指令式疾病识别及超分辨率训练数据，优化前述任务损失：

\[
\mathcal L_{\mathrm{stage1}}=\mathcal L_{\mathrm{task}}.
\]

训练参数包括 task decoders、slot embeddings U、visual adapter 与选定的 VLM 参数；空间 decoder 自身的图像特征提取、恢复及状态条件化模块随相应任务更新，JEPA 保持冻结。通过改变这些参数，下一次前向得到的 S 会随之变化；不将每个样本的 S 当作独立参数训练。具体 VLM 适配范围、adapter 结构与任务实现留到训练设置确定。

按各样本的可用观测和标注组织任务；具体数据集、输入组合覆盖及影像模态留到实验设置中说明。

任务监督为共享 state tokens 提供临床内容，使其保留疾病识别、病灶定位和范围描述等任务需要的信息，为后续预测提供具有临床含义的当前状态。这里使用的是下游任务形式的训练监督；最终表征评价仍使用独立的测试数据。

### Stage 2：Future State Prediction Fine-tuning（未来状态预测微调）

从第一阶段模型初始化，主要利用 MIMIC-CXR 等数据中的当前图文—未来图文观测对微调状态编码器，同时训练 Latent World Model。“Fine-tuning” 指继续调整第一阶段得到的状态表示；新加入的预测模块学习从 S_t、h 到未来 S 的映射。

未来损失的反向传播经过 **LWM → 当前 S_t → VLM／U／visual adapter**，更新预测器与当前状态编码路径。未来 S* 经过 stop-gradient，不接收该次预测损失的梯度；JEPA 继续冻结。因此第二阶段既学习状态转移，也进一步塑造共享状态。

为表示是否保留部分任务监督，可写：

\[
\mathcal L_{\mathrm{stage2}}
=\mathcal L_{\mathrm{future}}+\beta\mathcal L_{\mathrm{task}}.
\]

β=0 表示纯纵向微调，β>0 表示保留任务监督。当前尚未确定 β 或混合比例；若不保留任务 loss，独立的 task decoders，包括其图像分支和状态条件化模块，不在 future loss 的计算路径中，不会由该项损失更新。保留相应任务 replay 时，它们才接收相应任务 loss 的梯度。

**缺失模态训练与消融留待后续决定。**若开展这些实验，可利用自然缺失样本或在完整样本中随机省略当前端模态；future loss 按实际可用目标计算。输入采样比例、不同模态目标的对齐及具体实验组合届时确定，现阶段不将这些安排写成必需训练步骤，也不预先声称各组合性能已得到保证。

现阶段不要求重新训练医学 JEPA，视觉骨干暂时固定。新接入的 visual adapter 随状态编码路径训练，其具体结构、参数配置和 VLM 适配范围仍待确定。

**当前—未来双分支的参数更新仍待确定。**用户倾向同一套 VLM、adapter 和 slots 分别编码两个时点；最简候选是严格共享权重，未来目标端 stop-gradient。这样未来端不接收该次 loss 的梯度，但目标仍会随着共享权重的更新变化。EMA 目标分支或固定第一阶段副本保留为其他候选，它们共享编码结构而不保持实时同权重；暂不将任何一种写成已验证的稳定训练方案。

第二阶段是否保留第一阶段任务监督，以及混合比例和 loss 权重，留到具体训练设置中确定。当前 Method 主线明确纵向监督学习状态转移并继续更新产生 S 的 VLM、U 与 adapter；stop-gradient 本身不保证共享目标训练的稳定性。

两阶段的动机来自两类数据的分工：较广泛的单时点监督帮助模型学习如何表示当前状态，较有限的纵向配对监督帮助模型学习状态如何演化。我们希望先建立可用的状态空间，减少纵向训练同时学习临床表征和状态转移的负担；是否提高数据效率需要实验验证。目前不写“纵向数据少几个数量级”的定量结论，待最终数据统计后再决定。

### 与 VLA-JEPA 的关系

VLA-JEPA 是这一训练思路的重要参考。其预训练利用人类视频学习与状态转移有关的 VLM latent action tokens，实际训练同时使用 Something-Something-v2 和 DROID 机器人数据；之后在具体机器人任务上微调，机器人训练结合 latent world-model loss 与动作预测 loss。依据：[原文 §3.2–3.3、§4.1 与附录 A.2](https://arxiv.org/html/2602.10098v2)。因此，描述它时应保留联合预训练这一细节。

我们借鉴的是**先用较广泛的数据学习 VLM tokens 的表示，再利用目标任务数据进一步适配**的训练安排。两个方法的监督顺序有区别：VLA-JEPA 在预训练阶段已经学习状态转移，随后适配机器人控制；我们先用临床任务建立共享状态，再用纵向观测学习未来状态预测。我们的 tokens 表示医学状态，不直接赋予它们机器人 latent action 的含义。

Method 中可以说明受 VLA-JEPA 的分阶段训练思路启发，但两阶段的具体理由应落在医学数据和目标上：**先通过临床任务监督建立 world model 的状态空间，再通过真实随访学习该空间中的状态转移，并继续调整当前状态表示。** 这一安排服务于未来预测和可复用表征两项能力。

## 审阅时还需要定下的内容

1. **未来目标编码器的更新方式**：当前与未来共用多模态编码设计，支持仅影像、仅报告和图文输入；严格共享权重配合 stop-gradient、EMA 或固定副本仍需选择。
2. **Decoder 的具体实现**：输入分工已明确为分类读 S、疾病识别读 S 与 Q、分割读 X 与 S（可带 Q）、超分读 X^{LR} 与 S^{LR}；空间任务的 S 均采用 image-only 输入。疾病识别需经 S 解码，空间任务的图像路径与 S 条件化路径均参与任务训练。具体 decoder 图像 backbone、状态投影与融合方式仍需确定，不默认复用固定 JEPA 的 D。LWM 主方案读 S_t 与 h、只预测未来 S，D 输入仅保留为独立的可选预测器消融。
3. **时间条件**：h 表示实际随访间隔还是时间区间，以及它与真实随访时刻 t⁺ 的对应方式。
4. **后续实验设置**：adapter 结构与 VLM 适配范围、LWM 结构、最终 slot 数 K（当前图示为 8）、数据集和模态覆盖、第二阶段是否保留任务监督、超分退化方式、latent 距离及归一化、分割 loss、各项权重和超参数。超分 loss 已确定为 MSE；future loss 只监督 S，采用 MSE 还是其他距离仍待确定。

以上不影响先写 Task Definition、Overview 和主体结构。超分辨率已纳入任务集合；医学 JEPA 的再训练及空间迁移实验仍列为后续方向。

## 已查阅的连接模块参考

- [NeuroVFM](https://www.nature.com/articles/s41591-026-04497-1)：固定医学视觉编码器，每个 volume 的 tokens 经 Perceiver 压缩为 64 个 latents，再通过两层 MLP 接入 Qwen3-14B；先训练 connector，再训练 connector 与语言模型。可参考其连接方式，不需要照搬全部配方。
- [Neuro-JEPA](https://arxiv.org/html/2606.14957v2)：可参考医学体积 JEPA 的动机与表征学习；该文的 attentive pooling / classifier 属于任务读取接口，不是接 LLM 的 connector。
- [VLA-JEPA](https://arxiv.org/html/2602.10098v2)：参考其通过 VLM 的 learnable latent action tokens 条件化 latent world model、预测未来状态的设计，以及预训练后针对具体任务微调的两阶段安排。其预训练联合使用人类视频和 DROID 机器人数据；我们据医学数据特点采用“临床任务监督下的状态预训练 → 纵向未来状态预测微调”。主要参考价值在于 VLM 与 world model 的结合及训练思路。

## 本轮补充的 AI 顶会依据

以下文献分别支持模块设计和训练组织；它们没有直接验证本文以 VLM state tokens 为预测状态的医学方法。

| 文献 | 对当前设计的参考意义 |
|---|---|
| [LISA — CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Lai_LISA_Reasoning_Segmentation_via_Large_Language_Model_CVPR_2024_paper.pdf) | 将 LLM 的 `<SEG>` hidden embedding 经投影后与图像编码特征共同用于 mask 解码，可参考其利用语言模型 hidden states 条件化图像分割的思路。本文空间 decoder 的图像分支直接接 X，由其内部提取特征；该引用不要求复用 JEPA D，也不验证本方法的 SR 设计。它的 token 是任务相关的，不是本文通用 state。 |
| [DINO-WM — ICML 2025](https://proceedings.mlr.press/v267/zhou25t.html) | 在预训练视觉 patch features 上学习未来预测，可参考其 latent dynamics 思路和空间特征建模。它没有 VLM 状态分支，不能据此推导本文必须直接输入或预测 D；本文主方案通过 S_t 接收已融合的视觉信息。 |

VLA-JEPA 原文的未来 target 是 V-JEPA2 视觉状态，VLM latent action tokens 用于条件化预测器。本文以共享 VLM tokens 表示医学观测，只对预测的未来 S 与真实未来 S* 做 latent loss，不对未来 D 施加独立预测损失。这是自己的状态与目标定义，不能写成直接沿用其 target。
