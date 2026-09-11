# Stage 1：4＋4 slots 分配与四任务训练计划

日期：2026-09-11。状态：供审阅的下一轮计划；本文件创建时，新分组尚未实现，也未启动新训练。

目标是让同一组 8 个状态 slots 保留当前胸片和报告中的临床语义、解剖空间信息，并支持 classification、disease recognition、segmentation、super resolution。用户已确认下表的读取范围，并决定本轮先不做消融，只实现、训练和评估这一套配置。

上一轮实际结果单独整理在 [下游任务指标汇总](0911_stage1_downstream_results.md)。上一轮所有任务读取全部 8 个 slots，文本分支做的是报告生成，其指标不属于本计划的 Disease recognition。

**已经确定的状态分配和输入**

| 分组 | Slots | 每个样本的形状 | 希望保留的信息 |
|---|---|---|---|
| 临床语义组 | S₁～S₄ | 4×1024 | 疾病／征象、存在性、否定、不确定性及相关临床描述 |
| 解剖空间组 | S₅～S₈ | 4×1024 | 器官位置、形状、大小、边界、区域关系 |

| 任务 | 读取的状态 slots | Decoder 额外输入 | 输出 |
|---|---|---|---|
| Classification | S₁～S₄ | 无 | 13 个标签概率 |
| Disease recognition | S₁～S₈ | 诊断 query | 标准疾病／征象名称列表 |
| Segmentation | S₅～S₈ | 当前图像 | 右肺、左肺、心脏三个 mask |
| Super resolution | S₅～S₈ | LR 图像 | 重建的 HR 图像 |

临床语义组由分类、诊断监督；解剖空间组由分割、超分以及诊断中的区域问题共同监督。组内不预先规定某个 slot 必须对应某种疾病或某个器官。分组是监督与读取职责的约定，是否形成预期信息仍需通过任务效果判断，不能直接宣称已实现语义解耦。

**共享 encoder 与独立 decoder**

图像经冻结的 V-JEPA 2.1 ViT-B 提取图像特征，经过视觉 adapter；当前报告经过 tokenizer 和 embedding。视觉 tokens、文本 tokens 和 8 个可学习初始 tokens U 一起进入 Qwen3.5-0.8B，取最后 8 个位置的隐藏状态作为 S，形状 B×8×1024。

- U 是全体样本共用的可学习参数；S 是每个样本重新计算的状态。
- 冻结 V-JEPA 和 Qwen 基础权重；更新视觉 adapter、视觉位置参数、U、encoder LoRA 和各任务 decoder 的可训练参数。
- 四个 decoder 参数独立。文本 decoder 使用独立于 encoder 的 Qwen 0.8B LoRA。
- 原始 report 进入状态 encoder，不直接传给任何 task decoder。任务 query 在 S 产生之后传入对应 decoder。
- 在 decoder 入口执行分组读取：分类 `S[:, :4]`，诊断 `S`，分割和 SR `S[:, 4:]`。不额外隔离 VLM 内部不同 slots 的注意力；共享参数更新仍可影响全部输出 slots。
- Stage 1 使用当前 observation，不加入 treatment、horizon、未来图像／报告或 IV 配对条件，也不使用之前 Qwen 清洗的变化标签。
- SR 单独计算 `S_LR = Encoder(LR, current_report)`。HR 图像及其特征只用于监督，不进入该任务的 encoder 或 decoder 条件。

**数据组织**

以既有 24,000 张训练图像的 Stage 1 数据清单为起点，复用可核验的图像、报告、官方分类标签和 CXAS 伪标签。Disease recognition 需要加入新的问答标注，最终训练／验证／测试规模以关联和患者去重后的清单为准，不沿用上一轮报告生成的样本数冒充诊断问答数量。

| 任务 | 数据筛选与监督 |
|---|---|
| Classification | AP/PA，有有效当前报告且至少一个明确的官方 CheXpert 标签；标签 0/1 参与监督，-1 和空白屏蔽；沿用 13 类，不包含 No Finding |
| Disease recognition | 优先关联 MIMIC-CXR-VQA 的单次、单图疾病／征象列表问题，包括限定解剖区域的问题；保留标准标签集合，排除需要历史检查、IV 查询、设备／性别信息才能完成的题目；本版要求有效当前报告 |
| Segmentation | AP/PA，冻结 CXAS 产生右肺／左肺／心脏伪标签，通过已有面积及连通性质控；允许没有有效提取报告的图像使用空文本，单独记录数量 |
| SR | 使用所选 CXR 的全部视角，不要求分类标签、分割标签或 IV 给药记录；允许缺失报告并记录数量 |

图像、报告、任务标签必须对应同一 study／选定 image，所有任务使用统一的患者划分。每个任务从有相应监督的数据池取样，不要求一张图同时具备四项标签。

MIMIC-CXR-VQA 的训练标签主要是 silver，测试标签来自人工校正的 gold。获取并关联数据后，先核验其测试患者与本轮全部训练任务的重叠：保留 gold 测试患者时，应从所有任务的训练池中移除这些患者。不能只保证诊断任务自身的 train/test 不重叠。该数据在本轮尚未完成本地接入，不声明已获得可训练的诊断问答池。[数据说明](https://physionet.org/content/mimic-ext-mimic-cxr-vqa/1.0.0/)

首轮按既有语料规模实施；一个图像有多个问题时限制其在单个 batch 中的重复，并轮换问题。记录原始报告提取范围和截断比例。空答案列表只表示标注明确没有 query 所指的疾病／征象，不把缺失或未知标注改成阴性。

**四个 decoder 的首版实现计划**

以下是供审阅的实现细节；上面的 slots 读取范围已经确认。代码改动将写入新配置和新运行目录。

| Decoder | 计划结构 | 如何使用指定 slots |
|---|---|---|
| Classification | 沿用 13 个类别 query、128 维 attention 读取和标签输出层 | 13 个 query 仅对 S₁～S₄ 做 cross-attention，输出 B×13 的 logits |
| Disease recognition | Qwen3.5-0.8B 文本 decoder；状态连续前缀＋逐样本 query；自回归生成标准标签列表 | 直接使用全部 8 个状态 tokens，不接原始报告；替换旧的固定 Findings/Impression 提示和报告目标 |
| Segmentation | 沿用轻量 CNN 像素特征、3 个器官 query 和 mask embedding；保留 slots-only 的 32×32 粗 mask 辅助头 | 器官 query 和粗 mask 头都只读取 S₅～S₈；像素特征负责定位与边界细化 |
| SR | 在现有卷积残差／PixelShuffle 主体上增加空间 attention 读取；保留 bicubic 残差起点 | 将 S₅～S₈ 分别投影成 4 个条件 tokens，在两个降低空间分辨率的特征层，由图像位置的特征读取这 4 个 tokens，再上采样融合；避免仅汇总为一个全图条件向量 |

SR 的空间读取是本计划新增的实现提案，尚未编码。初始使用 64 维、4 头 attention；每个位置的 query 来自图像特征和位置编码，key/value 来自 4 个空间 slots。以低分辨率特征进行读取控制显存，原 LR 卷积分支继续负责局部细节。这不要求从 4 个 slots 独立恢复所有像素。

本轮 SR 固定沿用现有合成退化：HR 长边上限 512，保持长宽比，padding 仅用于 batch；带 antialias 的 bicubic 下采样。倍率明确为长宽各 ×2、总像素 ×4。只在有效矩形计算 loss 和指标。本版不扩展随机模糊／噪声任务，也不新增风格迁移任务。

**损失与优化**

| 任务 | Loss | 初始权重 |
|---|---|---:|
| Classification | 仅有效标签参与的 masked BCE，使用训练集正类权重 | 1 |
| Disease recognition | 只在答案 token 上计算 CE，query／状态前缀不计入目标 loss | 1 |
| Segmentation | 像素 soft BCE＋Dice，再加 0.2×粗 mask 辅助 loss | 1 |
| SR | 有效像素区域 MSE | 10 |

保留上一轮的 round-robin：分类 → 疾病识别 → 分割 → SR。每一步从对应任务池取 batch，只计算该任务 loss，更新共享 encoder 的可训练参数和本任务 decoder。初始 microbatch 为 8／4／8／8，梯度累积 2；显存检查后若调整 microbatch，用累积步数保持相同有效 batch。

基础配置沿用 LoRA rank 8、alpha 16、LoRA 学习率 5e-5、其余可训练参数学习率 1e-4、AdamW、bf16、梯度裁剪 1。报告输入预算 384 tokens；诊断 query 初始上限 96 tokens、答案上限 128 tokens，长度分布检查后固定，并记录所有截断。

完整标准答案超过预算时，应提高统一答案上限或明确排除该题并计数，不把截断后的疾病列表当作完整 ground truth。query 也需保留完整语义，避免截掉区域或候选条件。

首版拟定 24,000 次 optimizer 更新，每任务 6,000 次，每 400 步验证并保存恢复状态。该步数是训练预算提案，不是收敛承诺；执行时根据完成的数据清单和显存／吞吐检查记录实际配置，不沿用上一轮已经过期的时间截止参数。

新实验从原始预训练骨干和新初始化的可训练参数开始，不续训上一轮 checkpoint，以免继承与新增 gold 测试患者的训练重叠。已有固定图像特征、伪标签可按新的患者清单复用。记录每任务 loss、slot 路由处梯度和共享可训练参数梯度，排查无梯度或数值异常；不以其他组 U 的梯度必须为零作为通过条件。

**常规评估与交付**

| 任务 | 主指标 | 同时交付 |
|---|---|---|
| Classification | 逐类和 macro AUPRC／AUROC | 每类阳性、阴性、未知数量；不可评估类别标 NA |
| Disease recognition | 标准标签集合 micro／macro Precision、Recall、F1；集合完全匹配率 | disease-list 与区域限定问题分开统计；同义词映射、无法解析／词表外输出、空集合单独记录；query／标准答案／预测样例 |
| Segmentation | 每器官及平均 Dice／IoU，与 CXAS 伪标签的一致性 | 原图／教师／学生 mask，说明伪标签不等于人工 GT |
| SR | 有效区域逐图 PSNR／SSIM 的均值 | LR／预测／HR 对齐样例、尺寸与倍率记录 |

疾病列表统一名称并去重，以集合方式评估，不因词序不同扣分；只做规则可核验的名称映射，不用 LLM 评审代替标签指标。报告进入 S，因此分类与诊断结果按多模态临床信息读出解释。

本轮只训练这一套分配，不新增冻结 encoder、无 slots、打乱 slots、不同分配比例或不同 decoder 的消融实验。上一轮已经完成的对照仅作为历史结果归档。常规验证用于训练监测；保留各任务验证最佳 checkpoint 和 final checkpoint，主结果预先约定报告统一的 final checkpoint，测试集不用于选择模型。

交付新配置、数据清单和质控计数、可加载 checkpoint、四任务指标及预测样例。Disease recognition 单独命名，旧的 report generation 结果保持独立，不再混用。

**实施顺序**

1. 接入疾病问答标注，统一所有任务的患者划分，冻结本轮清单与答案词表。
2. 实现 4＋4 读取范围、逐样本诊断 query 接口和标准答案评估；调整分割辅助头的输入范围。
3. 实现 SR 的 4-token 空间读取，核验同源 LR 输入、倍率、padding 和高分辨率目标隔离。
4. 用实际样本做接口、梯度和显存检查，固定预算配置。
5. 在当时空闲 GPU 上启动一个主实验，完成常规评估并整理结果。当前审阅阶段仅写计划，尚未执行这一步。

**相关入口**

- [上一轮计划](0910_stage1_four_task_plan.md)与[本次整理的历史指标](0911_stage1_downstream_results.md)。
- 当前实现：[networks.py](../code/medworld_stage1/networks.py)、[train.py](../code/medworld_stage1/train.py)、[evaluation.py](../code/medworld_stage1/evaluation.py)。
- Disease recognition 数据参考：[MIMIC-CXR-VQA](https://physionet.org/content/mimic-ext-mimic-cxr-vqa/1.0.0/)。
- 任务 tokens 的设计参考：[TaskPrompter](https://openreview.net/pdf?id=-CwPopPJda)。这里只借鉴任务相关表征与共享信息的组织方式，不把其自然图像结果当作本项目已验证的收益。
