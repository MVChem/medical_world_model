# MedWorld-JEPA v2：VLM 驱动的预测式患者表征学习

下面这版可以直接作为改稿 Agent 的总体设计 brief。

## 1. 核心科学问题

论文不应只回答“模型能否预测下一次随访”，而应回答一个更强的问题：

> **真实患者时间能否重塑模型对当前患者的表示？**

未来预测迫使当前表征保留那些能够解释患者后续演化的潜在临床因素，例如疾病阶段、活动性、病灶负担和发展趋势。因此，与这些因素共享信息的下游任务——疾病识别、严重程度、病灶负担和变化风险——可能从纵向预训练中受益。普通器官分割主要依赖静态几何，其合理要求不是一定提升，而是相对无 future loss 的容量对照保持非劣。

全文仍然围绕三个相互依赖的 Claim：

- **Claim A：**模型确实学会了 patient-specific future prediction，而不是复制当前状态或输出群体平均未来。
- **Claim B：**预测未来改善了 frozen current representation，而不只是训练出一个更好的 forecast head。
- **Claim C：**收益确实来自真实患者、真实顺序和真实 horizon 的纵向对应，而非额外参数、扫描暴露或训练计算。

## 2. 当前方法需要解决的两个缺陷

当前稿件实际上并不是一个真正的 **VLM-centered world model**。它使用固定医学文本编码器，将当前状态拆成 global、anatomy、pathology、lesion、trajectory 五组 token，再用 visibility mask、固定 task route 和 group-specific loss 人工规定各组职责；future loss 又主要通过 trajectory token 回传。Figure 1 因而更像“typed tokens + objective routing + multiple task heads”的工程系统。

这产生两个问题。

第一，VLM 没有进入 world modeling 的核心计算路径。报告只提供辅助文本监督，真正形成当前状态和预测未来的仍是 JEPA tokenizer 与专用 forecast modules。

第二，token 设计过度。五组 state tokens、四种 task queries、显式 attention routes 和多条专用 loss 会给人一种“每个任务单独加一个部件”的印象，不像一个统一、可扩展的 representation-learning principle。

新版应保留三个正确部分：

1. 医学 2D/3D JEPA 提供的 dense geometry；
2. 与任务和 forecast horizon 无关的 current patient state；
3. 将未来数据严格隔离在 EMA stop-gradient target branch。

同时整体删除 typed-token taxonomy、hard objective routing 和 trajectory-token 专用通道。

---

## 3. 只保留三类 token

### Evidence tokens

Evidence tokens 表示在时间 \(t\) 或之前允许使用的证据，包括 X-ray、CT、MRI、当前报告、既往检查、病史、时间戳和其他许可上下文。

医学视觉编码器首先产生多尺度 dense features：

\[
D_t^{1:L}=f_\theta(X_t;m_t),
\]

其中保留物理坐标、局部纹理、解剖边界和跨切片几何。随后由 resampler 将这些特征压缩为数量较少的 compact visual tokens \(V_t\)，再投影到 VLM hidden space。

高分辨率 \(D_t^{1:L}\) 不需要全部送入 Qwen。它们作为独立的 **dense spatial memory**，服务于分割、检测、定位和细粒度 future prediction。

### State slots

State slots 是一小组通用的 learnable latent tokens，例如默认使用 \(K=8\)：

\[
S_t\in\mathbb{R}^{K\times d_{\mathrm{VLM}}}.
\]

它们读取所有 admissible evidence，形成固定长度的当前患者状态。

这些 slots 不再命名为 anatomy、pathology、lesion 或 trajectory，也不预先假设每个 slot 应负责哪种医学概念。输入 VLM 时，它们是 learnable queries；经过 VLM 后，同一位置上的 hidden states 就是 patient-state features。因此“state query token”和“state feature token”不是两套 token，而是同一位置的输入和输出。

slot 的功能应由 dense JEPA、语言建模、grounding 和 longitudinal prediction 等联合目标自然诱导，而不是由 visibility mask 和 group-specific loss 人工声明。

### Task/transition queries

Task/transition queries 在 \(S_t\) 完成以后加入，用来指定模型此时需要做什么。

- 分类、回归、风险预测和检索通常使用一个 query。
- 分割使用一个语义 query 条件化 dense decoder。
- 多类别或多实例任务所需的多个 queries 放在 task decoder 内部。
- Forecast 使用一个或少量由时间间隔 \(h\) 条件化的 transition queries。

因此，low-level task 的复杂性由 decoder 承担，不需要继续增加 anatomy tokens、lesion tokens 或 organ tokens。

整体序列顺序为：

\[
\text{Evidence}
\rightarrow
\text{State slots}
\rightarrow
\text{Task instruction / horizon}
\rightarrow
\text{Query}.
\]

通过 causal 或 block-causal attention：

- state slots 可以读取当前及历史证据；
- state slots 看不到后续任务、horizon 和答案；
- task query 可以读取 state；
- task query 不会写回或污染 state。

由此，同一个 \(S_t\) 是 **task-independent、horizon-independent、可缓存复用**的当前患者状态。

---

## 4. 采用 VLA-JEPA 式的 VLM–world-model 范式

新版在范式上应对齐 VLA-JEPA，但不机械照搬其 action terminology。

关键不是“在 JEPA 后面接一个 Qwen”，而是：

> **让 Qwen 产生的 latent hidden states 位于 future prediction loss 的直接因果路径中。**

在 VLA-JEPA 中，VLM latent tokens 表示与动作和状态转移有关的信息，并条件化 latent world predictor。对应到医学场景：

- latent action tokens 改为 **clinical transition queries**；
- action-conditioned future state 改为 **horizon-conditioned future patient state**；
- future visual state 由独立 EMA medical encoder 提供监督。

Qwen3.5-9B 或同类模型应成为真正的 multimodal fusion backbone，而不是冻结的文本编码器或最后接上的 report decoder。医学 2D/3D encoder 先产生 dense spatial memory，resampler/projector 再将 compact visual tokens 投影到 Qwen hidden space。Qwen 读取 visual evidence、当前报告、允许的历史信息和 state slots，输出 \(S_t\)。

因此，完整患者表示不是一个 pooled embedding，而是：

\[
\mathcal R_t=
\left(
D_t^{1:L},
S_t
\right).
\]

Qwen 可以采用 LoRA 或部分解冻，但 future loss 必须能够更新：

- visual resampler 和 projector；
- learnable state slots；
- Qwen adapters 或解冻层；
- 在线医学视觉编码器。

否则 Qwen 仍然只是一个辅助语言模块，不能支持“future prediction reshapes the present VLM representation”这一核心论点。

为了使 VLM 成为真实贡献，预训练还必须加入真正的 language/instruction objectives，例如 current report generation、medical VQA、finding description、comparison description 或 region-grounded text。

需要严格避免文本泄漏：

- 生成当前报告时，不能把同一报告同时作为输入；
- longitudinal batch 可以把当前报告作为 \(t\) 时刻 evidence；
- 未来报告不能进入 online branch，只能提供 target-side event 或文本监督。

---

## 5. Horizon-conditioned latent world prediction

当模型收到 forecast instruction 和时间间隔 \(h\) 后，transition query 从当前 state 中读取与该 horizon 有关的变化因素：

\[
A_{t,h}
=
Q_\omega
\left(
S_t,
I_{\mathrm{forecast}},
e(h)
\right).
\]

这里必须明确区分：

- \(S_t\)：患者当前是什么状态，不含 requested horizon；
- \(A_{t,h}\)：在指定时间 \(h\) 内，患者可能如何变化。

独立 world predictor 使用当前 dense memory、state slots 和 transition query 预测未来 latent：

\[
\widehat{Y}_{t+h}
=
P_\psi
\left(
D_t^{1:L},
S_t,
A_{t,h},
e(h)
\right).
\]

真实未来扫描只进入 EMA target encoder：

\[
Y_{t+h}
=
\operatorname{sg}
\left(
f_{\bar{\theta}}(X_{t+h})
\right).
\]

预测目标不应只有一个 global vector，而应同时包含：

- global future latent；
- multi-scale 或 region-level future features；
- 可用的 clinical-event targets，例如 finding onset、resolution、improvement、worsening，以及病灶出现、消失或增长。

这样能够保留原稿中有价值的 **fine-grained future prediction**，但无需设计专门的 pathology slots 和 lesion slots。

一个关键修改是：

> **Stop-gradient 只施加在 target branch，不再切断在线 state 的纵向梯度。**

真实 future loss 应直接塑造视觉 encoder、Qwen 和通用 state slots；current-language、grounding 和 dense JEPA objectives 则负责维持临床语义与空间几何。否则 future learning 仍会被限制在一个专用 trajectory 通道中，Claim B 会继续显得不充分。

---

## 6. 下游表示不是一个固定的 1024 维向量

全局任务可以从 \(S_t\) 中读取一个 query-conditioned embedding：

\[
e_t^{(q)}
\in
\mathbb{R}^{d_{\mathrm{VLM}}},
\]

随后连接线性分类器、MLP、survival head 或 retrieval head。

核心方法不应声称完整 patient representation 是 1024 维。默认保留 VLM native hidden width。若检索或部署需要紧凑索引，可以额外投影到 512 或 1024 维，但这只是 task-specific projection。

分割任务也不能只拿一个 1024 维 embedding。它应使用：

\[
\widehat{M}
=
G_{\mathrm{seg}}
\left(
D_t^{1:L},
e_t^{(\mathrm{seg})}
\right).
\]

其中：

- semantic query 表示“需要分割什么”；
- dense feature pyramid 表示“目标位于哪些像素或 voxel”。

多类别或实例分割所需的 mask queries 属于 decoder 内部机制，不再上升为新的 state-token taxonomy。

---

## 7. 精简训练目标与实验逻辑

总训练目标可以收缩为四类：

1. multi-scale medical JEPA loss；
2. VLM report/VQA/instruction loss；
3. 可用的 grounding 或 segmentation loss；
4. authentic longitudinal pairs 上的 future latent、clinical-event 和 rollout loss。

缺少某种监督时只关闭相应 loss，不再建立“某个 loss 只训练某组 token”的 objective-routing table。

预训练主对照只需要四个版本：

| 预训练版本 | VLM | Future supervision |
|---|---:|---:|
| JEPA only | 否 | 否 |
| JEPA + VLM | 是 | 否 |
| JEPA + VLM + shuffled future | 是 | 错误对应 |
| MedWorld-JEPA | 是 | 真实纵向对应 |

Claim A 比较 persistence、population prior 和 matched-capacity static predictor，证明模型学到患者特异未来。

Claim B 完全丢弃 forecast outputs，只使用冻结的 \(\mathcal R_t\)。全局任务训练相同容量的 linear probe；分割训练相同容量的 query-conditioned decoder。病灶负担、病灶分割和 next-change prediction 是预期受益任务，器官分割做相对 no-future control 的非劣性检验。

Claim C 在模型结构、扫描 multiset、batch、更新步数和 FLOPs 匹配时，比较 true pairs、cross-patient shuffle、within-patient future-time shuffle、horizon shuffle 和 no-future control。

附加消融只保留少数关键变量：\(K\in\{4,8,16\}\)、query readout 对比 mean pooling、Qwen 冻结/LoRA/部分解冻，以及 global 对比 multi-scale future targets。不要再围绕五组 token 做大量 routing 和 masking ablation。

---

## 8. 改稿时的结构调整

Method 可以重组为：

1. **Multimodal Evidence and Generic Patient-State Slots**
2. **Read-Only Task and Transition Queries**
3. **Qwen Patient-State VLM**
4. **Query-Conditioned Downstream Interfaces**
5. **Future Latent Predictor and EMA Targets**
6. **Joint Multimodal and Longitudinal Training**

原来的 **Typed Current-State Tokens、Objective Routing、固定 CLS/SEG/RETRIEVAL 路由表，以及 trajectory-token 专用设计应整体删除。**

Figure 1 只需要三个模块：

1. 当前多模态 evidence 经 medical encoder 和 Qwen 形成 generic state slots；
2. 同一个 state 被不同 task queries 只读复用；
3. horizon-conditioned transition query 驱动 predictor，与 future EMA targets 对齐。

全文最终只需要传达一句话：

> **一个当前状态，多个只读请求，一个隔离的未来监督分支；通过预测真实未来，VLM 学会更有用地表示现在。**