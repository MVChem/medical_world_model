# 09-15：Table 2 共享模型短测、baseline 含义与缓存范围

## 本轮决定

用户要求先用 Qwen3.5-0.8B 跑通 Table 2 底部对照，再扩到 9B。同时讨论了 SwinIR 对照的用途，以及 Table 1 full-token 为何使用 JEPA 特征。本轮新增独立的[四任务共享模型代码](../code/medworld_multitask/README.md)，保留所有既有实验和论文分数。

## 两类 baseline 必须区分

Table 1 的原生 Qwen 零样本行保留原生视觉编码器，通过图文提示预测未来报告，以 Yes/No 条件似然得到疾病概率。已有 full-token forecaster 则使用缓存的 V-JEPA 特征、adapter、Qwen 语言模型、显式预测器和读出；它是与 slots 预测器匹配的结构对照，**不是原生多模态 Qwen 直接微调的结果**。

其 AP、AUROC、RadGraph、GREEN 低于原生零样本行，而 Transition、Brier、ECE 有改善。视觉输入路径、读出和训练目标同时变化，不能据此断言“原生 Qwen 微调变差”或把差距归因于单一原因。要补原生 Qwen 微调 baseline，应保留原生视觉与图文接口，以同 cohort 和目标做匹配训练，再独立记录；不能改名后复用已有 JEPA-path 分数。

SwinIR 可作为超分任务专用的性能参照，但不能单独检验 slots 的增益。检验 slots 需要同 decoder、初始化、图像输入和预算下的 image-only／slots／shuffled 配对。检验 slots 能否帮助 SwinIR 则需另做 SwinIR／SwinIR＋slots。本轮实现并验证前一种控制，没有启动新的 SwinIR 训练。

## 新四任务原型

- 四个 fusion slots：冻结 V-JEPA 特征经可训练 adapter，读取四个 Qwen 语言深度。
- 四个 visual slots：读取 Qwen 原生视觉塔四个深度，训练查询、投影及 LoRA。
- 分类／报告读取全部八个，分割／SR 仅读视觉四个。报告仅为目标；SR 的两条输入分支均从相同 LR 构造。
- 同一 checkpoint 轮转训练分类、报告、分割和 ×4 SR。报告编码／解码共用一个 Qwen 语言基座及 LoRA，9B 保留独立预训练 `lm_head`。
- `full_tokens` 使用相同特征来源和任务头，去掉状态查询；`shuffled` 采用同 split 的其他患者；`image_only` 仅读图像并使用同一空间 decoder。

数据复用现有划分：分类 13,681/160/353；报告 22,646/307/507；分割与 SR 各 4,096/249/447，人工肺分割另 138 张。任意两个任务之间跨 split 的患者交集均为零。正式 VQA 和 MS-CXR 数据与适配尚未接入，这个原型也尚无多深度状态的 Stage 2 未来预测训练。

## 真实 GPU 执行结果

最终短测位于 `code/medworld_multitask/runs/smoke_20260915_v2/`，摘要为 `SUMMARY.json`。使用空闲 RTX 4090 与共享 GPU 锁，0.8B 通过后才启动 9B。每任务真实 batch 为 1，报告预测最多 16 个新 token；训练目标上限为 384 tokens。

| 模型／条件 | 通过任务数 | CUDA 峰值已分配显存 GiB | 最大保留显存 GiB |
| --- | ---: | ---: | ---: |
| 0.8B slots | 4 | 2.09 | 2.19 |
| 0.8B full tokens | 4 | 2.11 | 2.15 |
| 0.8B shuffled | 4 | 2.09 | 2.19 |
| image-only | 2 | 0.09 | 0.13 |
| 9B slots | 4 | 18.62 | 18.77 |
| 9B full tokens | 4 | 18.69 | 18.73 |

六组全部通过真实前向／反向、优化器更新、检查点保存重载及重载前后输出一致性。所有组的分割／SR decoder 初始化哈希一致。分类／报告确认两塔 LoRA、JEPA adapter 及八个查询有梯度；分割／SR 确认视觉 LoRA 和后四个查询有梯度，语言 LoRA、JEPA adapter 与前四个查询无梯度。CPU 数据边界测试 5 项及训练恢复测试 5 项通过。

这些结果明确标记 `table2_ready=false`，**只说明执行流程跑通**。没有开展完整预算训练或最终测试集评分，不能填入 Table 2。当前四任务通路还不能称为完整六任务世界模型。更大 batch、最长报告和正式预算仍需单独验证。

## 缓存到底缓存了什么

| 路径 | 缓存内容 | 实际训练范围 |
| --- | --- | --- |
| 已有 Table 1 三组 | 冻结 V-JEPA 特征、分词与数据 | Qwen 语言 LoRA、adapter、预测器、报告／finding 读出在线训练 |
| Table 2 冻结视觉 slots | 冻结视觉特征／多深度摘要 | 只训练下游任务头，未联合训练 9B 语言模型 |
| 原有在线 dense pilot | 原始图像数组 | 在线视觉／语言 LoRA、查询、decoder；两个任务各自训练 |
| 本轮四任务共享原型 | 原始 HR／LR 像素数组与标签 | 按任务在线重算所需分支，训练共享 LoRA、查询、adapter 和任务头；JEPA 固定 |

缓存冻结特征是允许的计算复用；缓存将要更新的模块输出，会使训练失去对该模块的梯度。本轮不读取预计算 JEPA 或 VLM 特征，空间任务只执行需要的视觉分支。冻结权重从本地模型缓存读取，与缓存中间激活是两回事。

## Slots 大小与实测速度

新原型完整状态是 `8×1024`，共 8,192 个数；实测返回 FP32，每图 32 KiB。空间任务仅读视觉 `4×1024`，FP32 为 16 KiB。保存为 FP16／BF16 后分别为 16／8 KiB。旧 Table 2 冻结视觉缓存本来就是 `4×1024` FP16，每图每分支 8 KiB；旧 Table 1 的 9B 语言 slots 则是 `8×4096`，不是本轮统一到 1024 维的状态。

新 `benchmark_slots.py` 在 RTX 4090 上排除初次模型加载，以 batch 1、3 次预热和 12 张真实图同步计时。完整 8 slots 前向平均：0.8B **161 ms／图**、9B **204 ms／图**；四个视觉 slots：0.8B **约 12 ms**、9B **约 21–22 ms**。包含图像预处理、传输、编码和读出；不含读盘、任务 decoder 或 backward。原始记录为 `code/medworld_multitask/runs/latency_20260915/{qwen08b,qwen9b}.json`。

旧冻结视觉提取另有记录：9B 的 456M 视觉塔 batch 1 为 21.15 ms／图，batch 8 为 155.94 ms／批；扩展队列的 HR＋LR 共 38,946 次编码提取用时 543.92 秒。它不加载 9B 语言模型，不能用其耗时代表完整八 slots。来源：`code/medworld_dense_baselines/runs/expanded_overnight_20260913/extract_batch_benchmark_qwen9b.json` 与 `qwen9b/slots_complete.json`。
