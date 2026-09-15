# 09-15：Table 1 保留原生 Qwen 图文通路的未来预测重做协议

## 已确定的设计与当前状态

用户选择：**forecast decoder 保留当前原图、可用当前报告／EHR 和 horizon，再额外接入预测的 future 8 slots**。先用 Qwen3.5-0.8B 验证执行，再验证 Qwen3.5-9B，最后执行匹配的正式训练与测试。

本记录冻结新的输入边界、对照含义及拟定预算。**新协议的正式训练与最终评分尚未完成，本记录不提供新分数。** 现有 Table 1 的 9B full-token／slots／shuffled 数字保留原始归属，属于 legacy prototype。它们不是本次 native decoder＋预测多深度 slots 的结果。Table 2 四任务短测也不能证明本次 Table 1 通路已跑通。

## 为什么需要重新训练

旧 9B Table 1 状态采用缓存 V-JEPA 特征、adapter、Qwen 语言模型，以及最终语言层的八个查询；decoder 从预测表示生成报告，没有保留原生 Qwen 当前图像通路。旧状态形状是 `8×4096`。旧 full-token 行是这一输入路径下的压缩结构对照，不能改名为原生 Qwen 图文 SFT。

新状态为 `8×1024`：四个 JEPA／语言融合深度 slots，加四个 Qwen 原生视觉深度 slots。新 decoder 又增加原生当前图文路径，状态定义、输入接口和损失路由均有变化。因此新实验使用独立运行目录，重新训练 Stage 1，不从旧八个最终语言层查询 checkpoint 初始化，也不将旧分数挪到新行。

原生零样本 Qwen 的 AP／AUROC／报告分数高于旧结构的部分结果，只说明当前结果存在差距；无法据此判定原生 Qwen 微调变差，或锁定某一个原因。

## 前向路径

设当前允许证据为 `O_t=(X_t, C_t)`，其中 `C_t` 包含当前报告与满足 source cutoff 的 EHR；`h` 为请求的 horizon bin。

1. **原生 decoder 分支：** 从当前原始图像文件建立保纵横比、黑边填充的 512 像素图像，使用 Qwen 原生 processor、vision encoder、图像 token／位置接口和语言模型。提示包含相同截断预算的当前报告、EHR 和 `h`。不以 JEPA 特征替代这条图像输入。
2. **附加状态分支：** 同一当前图像通过冻结 JEPA、可训练 adapter 和 Qwen 语言层形成四个 fusion slots，当前可用文本参与融合；Qwen 原生视觉塔的四个深度形成四个 visual slots。两组投影到共同宽度 1024，组成 `S_t∈R^(8×1024)`。JEPA 保持自身预训练输入预处理；原生 decoder 两组实验均使用相同 512 像素协议。视觉四槽不读报告／EHR。
3. **预测器：** `S_hat=P(S_t,h)`，输出相同形状的未来八槽。预测器输入只有当前状态与请求 horizon。
4. **融合到原生 decoder：** `A(S_hat)` 投影到对应 Qwen 语言宽度，在原生 user 消息末尾、assistant generation prompt 之前插入八个 soft tokens。原生当前图像／文本上下文完整保留。输出未来报告与未来 finding 概率。

原生视觉输入需保留标准图像占位、attention mask、位置和 processor 元数据。插入未来 soft tokens 后，训练与生成应使用一致的序列构造，不能绕开原生视觉塔。编码器与 decoder 使用独立 LoRA，允许共享不会更新的预训练 base 参数；可训练 adapter、查询、LoRA 和 finding head 不能意外共享为同一可变参数。

## 未来信息边界与训练目标

实际未来图像与报告只用于目标构建、loss 和最终评分。它们不能作为当前状态、预测器或 decoder 的可用证据。

- 未来 latent target：用**本次新 Stage 1 的固定快照**编码未来观察，整支 stop-gradient。Stage 2 期间冻结其 base、LoRA、adapter 和 readout；不采用 EMA。
- 未来报告：用于因果 teacher forcing 的 token CE；生成和 finding 预测不读取目标报告。
- 未来 findings：用于 masked BCE，未知／不确定标签按共同 schema 屏蔽。
- finding head：读取 assistant 生成前的最后一个 source prompt 隐状态，保证该位置不能 attend 到任何 teacher-forced future token。三个新训练条件使用同一监督 head 类型和初始化。
- 新模型 Stage 2 objective 为 `λS·latent_loss + λR·future_report_CE + λF·future_finding_BCE`，可按统一协议追加当前任务 replay。Native SFT 对照使用相同报告／finding 监督；它没有 predicted-state latent loss。损失权重与 latent 归一化必须在正式运行 config 固定。

Stage 2 的 report／finding loss 应能通过插入的 future tokens 回传到预测器与当前八槽读出。需验证这条梯度路由，防止未来分支被 detach。目标 encoder 的所有梯度应为空；仅有 stop-gradient 而 target 参数继续更新不满足本协议。

## 第一轮匹配对照

| 条件 | 原生 decoder 当前输入 | 附加状态上下文 | 作用 |
| --- | --- | --- | --- |
| Native Qwen SFT | 真正的当前图像＋报告／EHR＋horizon | 无 | 原生图文微调基线 |
| Native Qwen＋predicted 8 slots | 完全相同 | 真正当前观察得到的 `P(S_t,h)` | 检验新增状态预测分支的整体作用 |
| Native Qwen＋shuffled predicted slots | 完全相同 | 同 split、其他患者 source observation 的状态，经相同预测器和本样本 `h` | 检验附加状态与当前患者是否对应 |

Shuffled **只替换附加状态分支的 source observation**，即 donor 的当前图像与允许的当前报告／EHR；原生 decoder 继续读本患者真正的当前图像、报告／EHR 与 horizon。目标仍是本样本真实随访。Donor 在同 split 的其他患者中确定，不读取任何 donor 的未来观察，不依赖物理 batch 的随机打乱。

三组共用同 cohort、样本顺序规则、预训练来源、原生 decoder 与 finding-head 初始化、原生视觉处理、当前文本预算、预测 horizon、监督 finding schema、报告生成设置以及正式优化步数／有效 batch。参数量和实际 GPU 小时另行记录；相同步数不等于相同算力。

这一轮 native baseline 对比**包含额外 JEPA encoder、状态读出与预测器的整体变化**，不能把所有差异解释为 slots 压缩带来的收益。后续需补相同 encoder／原生 decoder 下的 current slots 和 full tokens 对照，区分预测 dynamics、附加状态与压缩作用。Shuffled 不能替代这两类消融。

## Stage 1、预算与推进顺序

新 Stage 1 独立训练分类／当前报告的状态读出，并训练 native 当前报告 decoder。当前报告仅作单时点报告训练目标，不作为该 image-based Stage 1 输入；随后从本次 Stage 1 保存 online 初始化、固定 target snapshot 以及三组一致的 native decoder／finding-head 初始化。数据应遵守既有患者 split 和测试患者排除规则。

该 Stage 1 是分类／报告子集原型，**不是完整六任务适配**。它既不复用旧最终语言层查询初始化，也不把 Table 2 的四任务短测当作正式 Stage 1 成绩。

| 阶段 | 预定优化步数 | 有效 batch | 状态 |
| --- | ---: | ---: | --- |
| 新 Stage 1 | 1,694 | 8 | 新 9B 正式队列已启动；实时进度见下 |
| 各条件 Stage 2 | 2,400 | 32 | 已排入共享 Stage 1 之后；train／val／test = 16,000／230／297 pairs，测试 94 位患者 |

Stage 1 使用 1,694×8，Stage 2 使用 2,400×32，保留旧协议的样本呈现预算；并不继承其训练结果。物理 microbatch 与梯度累积由真实资源检查确定，不能降低有效 batch 或预算后仍称为相同实验。正式 seed 采用 42；完整配置、数据和权重签名保存在新 run。

推进顺序如下：

1. **0.8B smoke：** 使用真实数据验证 Stage 1／Stage 2 前向、全部 loss 与梯度、未来信息边界、native SFT 和 augmented／shuffled、生成、保存重载及 resume；smoke 分数不进论文。
2. **9B smoke／压力检查：** 覆盖最大允许图文上下文、target 长度、optimizer step、目标与在线参数隔离和生成，测实际显存，再确定 microbatch／accumulation。
3. **正式匹配训练：** 独立新 Stage 1，再进行三组 Stage 2。保留每组训练预算、初始化签名、检查点和失败／恢复日志。
4. **完整验证与测试：** 同 230／297 pairs，全部生成、finding scoring、CheXbert／RadGraph／GREEN；完成前维持 pending。Direction 沿用尚未完成标签审定的事实，不能用其他 cohort 结果填充。

## 评分解释与可复用代码

原始 zero-shot Qwen 使用 `P(Yes)/(P(Yes)+P(No))` 作为 AP／AUROC／Brier／ECE 输入；新 native SFT 与 augmented 对照统一使用监督 finding head。两种来源应分别标明。即使指标 schema 相同，新训练 head 与旧 zero-shot Yes/No 仍不是相同概率读出消融。Transition F1 使用实际生成报告抽取的 finding 状态，不用概率阈值结果无说明替换。

- [现成 native direct baseline](../code/medworld_table1/direct.py) 已接训练和评测，可复用其原生 VLM 与报告 CE／finding head；旧代码采用 256 像素处理，需改为新三组共同 512 协议。
- [现有 Corpus](../code/medworld_table1/data.py) 在 `use_ehr=true` 时已将 EHR 放入 source context，不能误称该 direct 通路没有 EHR；但它强制加载 JEPA feature cache，原生数据路径应按实际需要拆开，并显式支持仅用于 loss 的 target image。
- [4＋4 原型](../code/medworld_multitask/model.py) 可复用状态分支，但其原有融合方法只有固定 prompt，尚需接入当前 report／EHR，并另接 paired forecasting、固定 target、horizon predictor 和 native augmented decoder。
- [既有 Table 1 评分](../code/medworld_table1/evaluate.py) 可复用同 cohort 的临床提取和指标定义；新模型适配必须独立标记 provenance，不能继承旧 checkpoint 或预测文件。

原始像素／文本／split 数据缓存可以复用。正在训练的视觉／语言 LoRA、adapter 或 slots 的输出必须随参数更新重新计算；冻结权重共享与冻结 JEPA 特征复用不等于缓存可训练分支的状态。

## 论文同步范围

本轮修改 task definition、method、experimental setup、results interpretation 和 appendix protocol，明确 native 当前证据＋预测状态的 decoder 接口、完整 Stage 2 loss、固定 target 与 legacy／pending 区别。图 1 已从可编辑 v7 另存为 [v8](../27cvpr/ppt/ppt/fig1_v8.pptx)，补充原生图文直连、future soft slots 到 decoder 的输入以及状态内的 4＋4 来源，历史 v7 保留。Table 1 脚注明确旧结果为 legacy、新原生通路结果 pending；数值与结果 JSON 保持不变。

**后续图示选择（2026-09-15）：** 用户指定现有 [fig1_v7.pdf](../27cvpr/ppt/ppt/fig1_v7.pdf) 作为论文 Fig. 1，已原样同步到 `27cvpr/imgs/fig1.pdf` 并重建 PNG 预览、调整图注。Fig. 2 沿用 v2；v8 保留为本地历史设计稿。此图示切换没有改动上述预测代码或正在运行的训练协议。

## 本次执行证据

- [实现入口](../code/medworld_native_forecast/README.md)，独立于历史 Table 1 目录。
- [六组 GPU smoke](../code/medworld_native_forecast/runs/smoke_20260915/SUMMARY.json)：0.8B／9B 各 native、slots、shuffled 均完成真实更新、生成、固定 target 检查和保存恢复；同一规模的三个 Stage 2 decoder 初始化哈希相同。
- [最终源码 0.8B 队列检查](../code/medworld_native_forecast/runs/queue_smoke_20260915/queue_status.json)：共享 Stage 1 加三个 Stage 2，4／4 完成。另完成数据、模型、恢复、CLI 和评分依赖的 28 项 CPU 测试。
- [0.8B 验证](../code/medworld_native_forecast/runs/verify_20260915/qwen08b_batch2.json)与 [9B 验证](../code/medworld_native_forecast/runs/verify_20260915/qwen9b_batch2.json)：不接 slots 时隐藏状态误差为 0，贪心生成与原生 Qwen 一致；只反传 report CE 时八个 query 和 predictor 的梯度全部非零。两组均通过 batch 1 和 batch 2。
- 压力检查使用真实图像／JEPA 特征，并重复文本 token 覆盖当前／未来各 800 context tokens、385 report-target tokens；这是执行检查，未计算临床效果。9B batch 2 在 optimizer 初始化和两次梯度累积后峰值分配 **22.27 GiB**、保留 **22.65 GiB**。固定 target 参数保持不变。
- 正式配置使用 [qwen9b_batch2.json](../code/medworld_native_forecast/configs/qwen9b_batch2.json)：microbatch 2，Stage 1 累积 4、Stage 2 累积 16。新的 [9B 正式队列](../code/medworld_native_forecast/runs/qwen9b_native_20260915/queue_status.json)已启动并确认真实 Stage 1 optimizer update 成功；**当前没有新的 Table 1 最终指标**。
- 队列仅选择空闲 GPU 2／3／4／7 并使用共享 UUID 锁。后续依赖为三组 Stage 2、各 297 对测试生成、CheXbert／RadGraph、统一 GREEN。评分器发现缺失或不完整输出会失败并保留真实状态；Direction 标签仍 pending。

源码修复记录：最初调度脚本命名 `queue.py` 遮蔽 Python 标准库，已改为 `launch_queue.py` 并补独立进程导入测试；左 padding 的批量生成位置与不完整 target checkpoint 的严格拒绝也已修复。早期 smoke 的签名格式与最终版不同，最终 0.8B 队列已重新验证；正式训练从新模型初始化，不载入 smoke 权重。
