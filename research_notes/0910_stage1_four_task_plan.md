# 2026-09-10 四任务 Stage 1 夜间训练计划

历史记录：本计划对应已完成的 `overnight_20260910` 实验，其中 Diagnosis / clinical description 实际为报告生成。2026-09-11 确认的 4＋4 slots 分配及独立 Disease recognition 任务见[新计划](0911_stage1_slot_allocation_plan.md)，已完成指标见[汇总](0911_stage1_downstream_results.md)。

目标：训练 Qwen3.5-0.8B 的 8 个共享 slots，使其保留当前 CXR 与当前报告中的临床、空间和细节信息。2026-09-11 08:00 前交付可加载 checkpoint、各任务验证/测试指标、预测样例和训练记录。训练于 07:10 截止，为自动评估预留时间；这是有明确数据和时间预算的一轮训练，不承诺收敛或超过基线。

## 共同约定

- 状态 `S=Encoder(image, current_report)`，形状 `B×8×1024`。冻结 V-JEPA 2.1 ViT-B 和 Qwen 基础权重；更新视觉 adapter、8 个输入 slot tokens、encoder LoRA 和各任务头。文本 decoder 有独立 LoRA。
- 原始 report 不直接传给任何 task decoder；diagnosis teacher forcing 的目标前缀仅用于训练。分类和 diagnosis 从 slots 读取；分割和 SR 额外接收相应图像。
- 本轮仅 Stage 1，无 treatment、horizon、IV 条件和未来配对筛选；不用 Qwen 的变化标签。当前观察在本任务中是独立样本。
- 从完整 MIMIC-CXR 索引构造候选目录，沿用官方患者互斥 train/validate/test。今晚确定性抽取最多 20,000 张正位和 4,000 张其他视角训练图像，训练患者最多 4 张；验证和测试各最多 256 正位 + 64 其他视角，评估患者各最多 1 张。候选数、实际数、排除原因、标签覆盖和文件指纹均落盘。
- 同一 study 在所选集合最多一张图，报告提取标签始终注明 study 级弱标签。不会把数据子集训练描述为全体 CXR 已训练。
- 分类和 diagnosis 必须有有效提取报告；分割和 SR 保留没有有效 FINDINGS/IMPRESSION 提取结果的图像，这些样本的文本输入为空。其余样本均融合当前报告，缺失数量按任务记录在 data_usage.json，不用生成文本补齐。
- 验证只用于选 checkpoint，测试保留到训练结束。原始图像、报告及旧实验不修改。

## Plan 1 — Classification

数据：选定正位 AP/PA、有效当前报告、至少一项确定性官方 CheXpert 标签。13 个病理/设备标签；No Finding 只有显式阳性而无显式阴性，本轮不作为二分类输出。标签 0/1 参与 loss，-1 和空白屏蔽，不把“没提到”写成阴性。

模型：13 个类别 query cross-attend 所有 8 个 slots，再分别输出 sigmoid 概率；query 是任务头参数，不新增状态 slots。

优化：按类别有效标签归一化的 masked BCE，训练集计算正类权重并限制在合理范围；不因肺炎等不确定标签剔除整张图。输出逐类阳性/阴性/不确定/空白计数。

评估：逐类 AUROC、AUPRC 和有效数量；没有正例或负例时明确标为不可评估。报告融合进 slots，因此指标代表多模态临床信息读出，不能声称纯图像诊断性能。

## Plan 2 — Diagnosis / clinical description

数据：AP/PA 且当前报告含有效 FINDINGS/IMPRESSION，至少 5 个词。直接使用原始报告中被规则提取的段落，不用 Qwen 改写或构造新的医学断言。保留否定、侧别、位置和不确定性。旧提取器可能漏掉无标题段落，本轮记录提取范围，不宣称完整重建原始报告。

模型：沿用 slots 连续前缀 + Qwen0.8B 自回归文本 decoder，任务提示要求简洁 FINDINGS/IMPRESSION。输入报告预算 384 tokens；输出目标预算 192 tokens，截断数量单独报告。

优化：token CE，只从 slots 条件生成文本。该任务是多模态语义重建，和离散 classification 互补但不算独立影像诊断验证。

评估：验证 CE；最终保存无 teacher forcing 的生成文本、原报告和截断后目标，报告重复率、唯一生成比例及状态打乱敏感性。用已有 CheXbert / RadGraph 本地工具提供临床指标，工具失败时明确缺失，不用词汇指标冒充临床指标。

## Plan 3 — Segmentation distillation

数据：AP/PA 当前图像，由冻结 CXAS `UNet_ResNet50_default` 生成左右肺、心脏 3 个概率通道。模型有 159 通道，本轮先使用这 3 类。采用现有 teacher 的归一化和 512 方形采样协议，再把输出映射回学生图像的有效矩形；不把 padding 当器官或背景监督。

模型：轻量图像卷积特征 + 从所有 slots 读出的器官 mask embeddings，产生 3 张 256 分辨率 mask；同时增加只读 slots 的 32×32 粗 mask 辅助头。

优化：soft-target BCE + Dice，辅助粗 mask loss 权重 0.2。多通道独立 sigmoid，允许结构投影重叠。teacher 完全冻结，过滤明显空/满的器官预测，并记录保留比例；其余异常不能仅靠阈值视为已解决。

预览训练样例后发现断裂心脏和远处小块，因此另加几何质控：每个器官最大 8 连通区域至少占其二值前景的 90%，否则该图不参加分割监督；原始概率图不修改，其他任务照常使用该图。记录按视角剔除数量。该规则可能偏向排除严重异常影像，需要在正式实验中复核选择偏差。

评估：逐器官 teacher-agreement Dice/IoU，真实 slots、空 slots、跨患者打乱 slots 对照；保存可视化。伪标签匹配不等于人工标注准确率。200 张 PA 人工心肺标注是后续独立验证选项，本轮未下载/确认重叠前不承诺人工 GT 指标。

## Plan 4 — Super resolution

语音中同时出现“256 除以 4”和“长宽各除以 2”。已重新本地识别，仍有歧义；已提供可选澄清。默认遵循后一个具体描述：`scale=2`，`LR=(H/2,W/2)→HR=(H,W)`，总像素 ×4；不将其标成标准线性 ×4 SR。scale 可配置为 4。

数据：全部视角 CXR 都进入候选范围，不要求诊断标签、分割 mask 或 IV。今晚为吞吐量设 HR 长边上限 512，保留长宽比例、尺寸对齐倍率后用黑边组成 batch；不把胸片拉伸成正方形，也不统一成 256×256。指标仅在有效图像矩形内计算。记录原始/HR/LR 尺寸。后续可提高分辨率预算。

退化：从 HR 用带 antialias 的 bicubic 缩小得到 LR。`S_lr=Encoder(LR, report)`，单独提取 LR 的 V-JEPA 特征，禁止复用 HR features。decoder 只能接收 LR 和 S_lr，HR 只出现在 target/loss。报告来源于原图，因此称 report-assisted SR。

模型：轻量残差卷积重建网络，在多个 block 用 slots 生成 FiLM 条件；bicubic 上采样作为残差起点。

优化：有效像素归一化 MSE。先验证倍率、长宽比、padding mask 和 HR 不进入 encoder；本轮不加入 GAN。

评估：有效区域 PSNR/SSIM、bicubic 基线、空/打乱 slots 对照以及 LR/预测/HR 可视化。不把合成低分辨率恢复推广成真实采集分辨率改善。

## 联合优化与夜间执行

四任务 round-robin；各任务单独取有监督的 batch，不要求每张图都有四项标签。classification、diagnosis、segmentation、SR 的初始 loss 权重分别为 1、1、1、10（MSE 数值小），只在调试确认数值和梯度后固定。记录每个任务传入共享 encoder/slots 的梯度范数；warmup、梯度裁剪、bf16。

完整尺寸显存检查通过后，固定 microbatch 为 classification/segmentation/SR 各 8，diagnosis 4，梯度累积 2。全长输入 384 tokens、输出 192 tokens、HR 512 的检查峰值约 4.04 GiB（不包含长训练 AdamW 状态），各任务梯度均能到达 slots。

主实验更新共享 encoder。另一个同初始化、同四任务数据顺序和相同步数的对照冻结共享 encoder，只训练任务头，用于判断表示学习带来的收益。对照并非“无 slots 模型”。最终再拟合相同 frozen-slot 线性分类 probe，评估主实验与冻结对照的表征。

计划使用 GPU 1 训练主实验，GPU 5 训练冻结 encoder 对照，GPU 6/7 准备 V-JEPA features / CXAS 标签。避开已有进程和此前不稳定的 GPU 4；启动前再次检查占用。tmux 持久运行，定期原子保存 checkpoint，失败记录原因；支持恢复。实际设备与启动时间以 run 配置为准。

输出目录：`code/medworld_stage1/runs/overnight_20260910/`。最终 `REPORT.md` 包括四任务指标、对照、覆盖范围、训练耗时、截断/伪标签局限及样例路径。

参考：[MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)、[人工心肺 mask 子集](https://physionet.org/content/heart-lung-segmentations-data/1.0.0/)、本地 `code/ChestXRayAnatomySegmentation/` 与 `code/medworld_table1/`。
