# 09-17：Sparse slots 的空间任务、FeatUp 一致性与弱语义对齐

## 本轮授权与基线

用户希望 sparse slots 对分割、超分带来实际增益，并用 attention 解释改善区域。
本轮明确要求：先 commit 当前代码；按 FeatUp 思想尝试，语义／视觉对齐尽量利用
VLM 本身的能力，少用额外强监督；实验整体运行约 8 小时，明早 8 点前看到一版结果。
允许多张空闲 GPU，不用第四／第五张卡，不终止其他人的任务。

已将原有代码、笔记与论文调整保存为基线 commit **`cc72e88`**。
新实现放在 [medworld_spatial](../code/medworld_spatial/README.md)，独立于现有融合模型。
下载的临床表、模型权重、数据与运行产物未加入此次源码提交。

## 当前实现与问题

原融合代码的 spatial 路由只返回四个 visual slots，跳过 JEPA＋Adapter＋语言融合。
每个视觉深度汇聚为一个全局向量，原 decoder 在 32×32 上读取 slots 后上采样。
这使得“完整 JEPA／VLM sparse state 帮助空间任务”尚未被该路径直接验证。
此前微小提升也不等于这次联合可学习 readout 的最终效果。

本轮关注 slot／decoder 接口与辅助目标，先固定大骨干。使用 09-16 两卡完整训练的
**Stage 1** checkpoint，冻结 Qwen/JEPA 及前四个 image-only fusion slots；训练后四槽
的 query、readout 与新空间模块。不将这次试验称为完整 VLM LoRA 微调，亦不用于证明
Stage 2 时间监督带来空间收益。融合 slots 的输入不含同次报告。

## 已实现的路线

1. **多分辨率读取。** 图像特征在 32×32、64×64 两处通过真实 cross-attention 读取八槽，
   残差门控融合；两级之间使用图像引导的 3×3 邻域加权上采样。分割、SR 共享空间模块，
   分别接任务输出头。前四槽提供 JEPA／语言融合条件，没有新增 dense JEPA 旁路。
2. **FeatUp 思路。** 学习高分辨率特征场，按记录的同一 crop/zoom 变换特征和输入图像，
   将变换后的特征下采样，对齐冻结视觉塔在变换图像上的输出。每例缓存 identity＋两个
   固定随机视图；使用固定正交 64 维投影和 area 下采样，教师不可随学生一起坍缩。
   这是借鉴多视图一致性的实验版本，不是原论文 JBU／可学习下采样器的逐项复现。
3. **VLM 弱语义对齐。** 冻结原生 Qwen3.5-0.8B，分别询问 2×2 图块是否包含
   `lung tissue`、`the heart`、`ribs`、`medical tubes or lines`，读取 No/Yes 条件似然。
   这些响应是弱软目标，不能当作临床校准概率、器官 mask 或原生 attention。
   只对跨图块有响应差异且有阳性证据的概念启用监督，并排除缺少有效影像的图块。
4. **语义 attention 进入输出路径。** 原生语言模型的 text-only hidden states 经可学习
   query 投影，读取空间特征场。汇聚出的概念表示通过同一组 attention 回写空间特征，
   参与分割／SR。KL 对齐的是四个图块上的 attention 总质量，不制造细粒度伪 mask。
5. **损失。** 保留原任务 BCE＋Dice／像素 MSE；新增 `0.1 × feature consistency`、
   `0.05 × semantic KL`。SR 的像素 MSE 量级远小于分割 loss，新增两项再乘预设的
   `0.001`（实际权重 1e-4／5e-5），避免辅助目标压过像素重建；image-only 教师对照
   使用同样系数，不按测试结果选择。对齐不增加框或像素标注，但原有分割监督、checkpoint 的训练
   经历仍然存在，不能把整个系统称为无监督。

来源：[FeatUp §3](https://arxiv.org/html/2403.10516v2)、
[LISA §4](https://arxiv.org/html/2308.00692v3)、
[AnchorSeg §3](https://aclanthology.org/2026.acl-long.938/)、
[DINOSAUR](https://dinosaur-paper.github.io/)。以上方法提供设计依据，不保证本项目增益。
AnchorSeg 的 spatial response 也不是其 VLM 原生 attention，本轮不混用这些术语。

## 六组匹配条件

| 条件 | Sparse state | 特征一致性 | VLM 弱语义 |
|---|---|---|---|
| image_only | 所有槽值置零 | 无 | 无 |
| visual_slots | 四个 visual，fusion 位置置零 | 无 | 无 |
| slots | 完整八槽 | 无 | 无 |
| featup | 完整八槽 | 有 | 无 |
| featup_semantic | 完整八槽 | 有 | 有 |
| image_only_featup | 所有槽值置零 | 有 | 有 |

所有条件共享新 decoder 架构、同 seed 初始化、batch 顺序和原始任务目标。
最后一组用于检查教师监督／图像 decoder 自身是否就能获得同样收益。
保留旧头作为历史实现，不用它与新架构的分数差直接归因于 slots。

## 8 小时协议

- 运行目录：`code/medworld_spatial/runs/featup_8h_20260917/`。
- GPU 只从物理编号 **0、1、2、6、7** 调度；保守排除 3／4／5，覆盖“第四／第五张”
  按序数或索引理解的两种情况。先检查显存、利用率、活动进程，再由 worker 获取 GPU 锁。
- 启动器保存源码快照及 SHA256；准备、训练、评测和画图共享 8 小时预算。
  最迟 09-18 07:55 收尾，预留训练截止后的 20 分钟给评测与图件导出。
  已完成固定预算的任务可提前结束，不为凑时长重复测试。
- 请求每任务最多 4,096 训练例、128 验证例，完整可用测试集及 Montgomery 人工肺测试；
  按 `UnifiedData` 的全局患者隔离过滤后实际人数／样本数，以缓存清单为准。
- 每条件三个 seed（42／43／44）；12,000 updates，分割／SR 交替，各 6,000 步，batch 8。
  单任务每组最多读取 48,000 个训练样本。若截止时间导致预算不足，报告显式标注，
  不把不等步数结果混入匹配 seed 的增益结论。
- 测试集不参与优化、选 checkpoint、选阈值或筛选好看的病例。展示例由固定 ID hash 排序确定。
- SR 的 VLM、JEPA、语义教师与图像 decoder 均只读取同一 LR 影像；HR 仅作目标。

## 可视化与验收

实际导出四类权重：encoder slot→patch、decoder pixel→slot（两种分辨率）、
semantic query→空间特征、局部上采样的 3×3 权重。原始数组、输入、预测、目标和
清零 slots 后的输出都保存。Qwen 的 pre-merger patch 序列先还原 merge-block 排列；
不能直接 reshape。Encoder／semantic 图按相对均匀权重的倍数显示，组内统一色标。

预览对比固定验证病例的轮廓／重建和共同色标误差；热图集中程度不作为性能指标。
清零 slots 只作推理敏感性检查，不替代独立训练的 image-only 对照。

已通过 22 项 CPU 检查（原模型 16＋新模块 6）。真实 GPU 短测验证缓存 slots 与原模型
一致、权重归一化、两个任务到 slot query／semantic 模块的梯度，以及 checkpoint 重载
输出完全一致。短测只验证运行链路，不报告其极小样本分数为方法效果。

正式启动时间、截止时间、PID 见运行目录 `plan.json`／`launch.json`；实时进度为
`status.json`，结果为 `REPORT.md`／`aggregate.json`。后续结论依据真实完成的匹配结果。

## 实际启动

- 新实验代码 commit：**`85482d5`**；运行使用该提交源码的独立副本。
- 北京时间 **09-17 23:07:19** 启动，8 小时预算截至 **09-18 07:07:19**。
- 调度器 PID：`663631`；初始两个准备任务 PID `663834`／`663875`，分别在 GPU 0／1。
- 启动后已核对实际 GPU UUID 与进程；两项任务均完成首批真实样本，继续准备特征。
- 六个条件均通过 20-step 真实双任务 smoke 和精确 checkpoint 重载，另完成 batch 8
  的 SR 损失缩放／任务单独梯度检查。22 项 CPU 测试通过，队列也验证了忙卡拒绝和
  2 个准备任务＋18 个训练任务的计划。正式实验尚无最终性能结论。
