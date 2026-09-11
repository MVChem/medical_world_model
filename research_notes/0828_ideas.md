## 方法总构想

**1. 为什么要这样做。** 现有医学基础模型通常把能力拆开：体积自监督模型学习解剖和三维结构，视觉语言模型学习疾病语义，纵向模型则为单一任务预测未来。我们希望学习一个统一的患者状态，使模型同时理解“患者现在有什么”以及“这些状态之后可能怎样变化”。当前稿件已经用体积 JEPA、语言对齐和纵向预测共同学习结构化 patient state；我们的改进，是把该状态显式拆成不同类型的 token，并通过 objective routing 赋予它们不同职责。

**2. Token 与 query token。** 2D/3D JEPA 编码器先产生保留局部几何和多尺度空间信息的 dense features，在其上学习 global、anatomy、pathology、lesion 和 trajectory tokens。这些是“状态 token”，表示患者当前包含什么信息；CLS、SEG、RETRIEVAL 和 FORECAST(h) 则是“任务 query”，表示模型要做什么。不同 query 从同一组状态中选择性读取信息：SEG 读取 dense、anatomy 和 lesion，分类读取 pathology 与 trajectory，FORECAST(h) 加入时间间隔后预测未来。这样，可复用表示是结构化 token 集合，而不是一个平均向量。

**3. Objective routing。** 体积 JEPA 主要塑造 dense 与 anatomy token，使模型学习跨切片连续性和三维关系；报告级与 finding-level 对齐塑造 pathology token，使其编码疾病类型、位置和严重程度；框、mask、区域文本或弱定位监督塑造 lesion token，使其表示病灶的位置、大小和形态；真实纵向 pair 或 triple 的 future objective 主要塑造 trajectory token。token 的功能不是由名称规定，而是由输入路径、注意力连接和组别特定损失共同形成。

**4. Future token 与 fine-grained prediction。** current branch 中的 future token 更准确地应称为 trajectory token。它只读取当前图像、报告和允许使用的既往信息，不读取真实未来，也不依赖具体 horizon，因此仍属于当前患者状态。未来预测迫使它保留能够解释后续演化的潜在临床因素，例如疾病阶段、严重程度、病灶负荷、活动性以及进展或缓解倾向。扫描噪声和重建参数通常不能稳定预测未来，而疾病阶段和病灶状态同时影响当前影像与后续变化；因此，与这些因素共享信息的疾病识别、严重度分级、病灶负荷估计和进展判断等任务可能受益。这里不声称所有任务都会提升：普通器官分割主要依赖 anatomy 和 dense features，合理目标是保持而非必然改善。

未来建模也不应只预测一个 pooled embedding，而应分别预测 pathology slot 的新发、持续、改善、恶化或消失，以及 lesion slot 的出现、消失、增大、缩小和形态变化；病灶数量变化可通过 set matching 表示 birth/death。trajectory token 因而成为共享的 dynamics condition，解释疾病级和病灶级状态如何演化。未来影像与报告只能进入 EMA target branch，当前分支不能看到未来，以避免信息泄漏。

**5. 三个核心 Claim。** Claim A：模型确实学会 future prediction，预测需优于 persistence 和 population prior，并在 future retrieval、onset/resolution、临床变化及多步 rollout 中保留患者特异性。Claim B：future supervision 确实重塑当前表示；加入 trajectory token 与 future loss 后，与疾病状态和演化因素相关的冻结迁移任务应提升，同时静态解剖能力不明显退化。Claim C：收益确实来自真实纵向结构；在架构、token 数量、扫描集合、训练步数和算力匹配时，真实时间配对应优于跨患者打乱、同患者时间或 horizon 打乱，以及“有 trajectory token 但无 future loss”的容量控制。只有三者同时成立，才能说明未来预测不是附加输出头，而是在真实患者时间约束下形成了细粒度、可迁移且具有临床意义的当前状态表示。