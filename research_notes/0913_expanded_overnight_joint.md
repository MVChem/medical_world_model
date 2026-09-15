# 09-13 扩大数据与 VLM 联合优化过夜实验

用户授权使用 4–5 张 GPU，截止 **2026-09-14 08:00 北京时间**，并要求尽量提高每张卡的实际计算利用率。新运行使用独立目录；已有 09-11/12/13 结果保留。本页记录实验协议，实时完成情况以调度状态与成功产物为准。

## 数据和对照

| 分支 | 数据 | 主要比较 |
|---|---|---|
| Table 2 冻结视觉表示 | 18,708 train（原 4,096）、249 validation、447 test；Montgomery 138 外部人工双肺 | 六模型 × segmentation/×4 SR × 正常 slots/跨患者 shuffled slots；共享 image-only decoder |
| Table 2 在线联合优化 | 同一扩展 dense 数据 | Qwen0.8B、Qwen4B、MedGemma4B 的在线 8 slots、完整 tokens、打乱 slots、冻结 VLM 对照；共同 image-only decoder |
| Table 1 未来预测 | 16,000 train（原 6,000）；230 validation、297 test，保留旧评估患者与每患者最多 4 训练对 | Qwen0.8B 的 8 slots、完整源 tokens、跨患者打乱源状态 |

六个冻结模型：Qwen3.5-0.8B/4B/9B/27B-FP8、MedGemma-1.5-4B/v1-27B。冻结实验只运行原生视觉塔；MedGemma 两个大小的视觉塔相同，不能把其结果解释为语言模型规模效应。在线联合优化先覆盖可在单卡训练的小模型；27B 冻结结果不代表 27B 联合优化已经完成。

Dense 扩展使用已经 QC 合格的 frontal 图和既有 CXAS 伪标签，保留原验证、测试和人工外部测试的行与 HR/LR 像素。所有 SR 图像和 VLM 输入均来自同一份 ×4 LR；HR 只作目标。打乱指同 split 中另一位患者的状态，训练与评估均执行，定义不同于只交换 8 个向量的位置。

冻结实验每组 20 epochs、有效 batch 8、microbatch 8、同 seed/初始化/样本顺序。主 seed 20260913 完成后再接可选 seed 20260914。Image-only 头在每任务/seed 下只训练一次并明确共享，它不是完整 VLM tokens 基线。

联合优化使用原生 VLM 的四个语言深度读出和四个视觉深度读出，通过任务损失优化 slots、读出投影、decoder 和 LoRA。两个密集任务均读全部 8 个 slots，这是本轮明确记录的路由变体。梯度审计要求每个 slot 和语言/视觉 LoRA 均有非零梯度，并验证优化器确实更新参数。冻结 VLM 对照保留可训练的 slots/readouts，隔离是否更新主干 LoRA 的影响。每项任务分别训练完整 2 epochs、有效 batch 32；Qwen microbatch 32，MedGemma microbatch 16×累积 2。各下游任务分别联合适配自己的 VLM 和 decoder，分割与 SR 不共享本轮更新后的 checkpoint。具体参数以 `plan.json` 和各任务 `contract.json` 为准。

Table 1 沿用原未来预测原型及共同 Stage 1 初始化，比较三种状态接口；它与 Table 2 原生 VLM 的新多层 8 slots 实现分别记录，不宣称已经训练出统一的未来预测与密集任务 checkpoint。Table 1 测试集仍缺合格的 Direction 标注，原有独立 82 对 Direction 分数另列，不能填成同 297 对的结果。派生疾病列表问答不填入官方 VQA 列。

## 资源、停止和结果

资源分配：GPU **0/1** 扩展冻结对照，**2/4** 在线联合优化，**3** 未来预测；最多同时 5 张卡。三条队列复用同一调度器，卡集合互斥，每卡还持有共享锁。只在无计算进程的空闲卡上启动，不抢占其他工作。

用户要求提高 GPU 利用率后，实测固定有效 batch 8 的 dense decoder 将 microbatch 4→8，吞吐约 150→265 图/秒。在线 VLM 改为按 batch 执行原生视觉与语言前向，逐图前向已移除；这些是短基准的性能结果，不是效果指标。未来预测也重新测量更大的物理 batch，并在三个对照中匹配最终预算。

调度器在 07:45 后不再启动新任务；训练按各自更早的截止时间或 SIGTERM 保存，独立 worker 在 08:00 强制停止本轮进程组。即使协调器退出，worker 自己仍执行截止时间。未完成预算的结果记录为 partial/未完成，不能冒充最终结果。GPU 采样每 30 秒保存实际计算利用率、显存与功耗。

- [合并 Table 1 / Table 2 与运行状态](../results/overnight_20260913/REPORT.md)
- [GPU 实时利用率](../results/overnight_20260913/gpu_utilization.json)
- [扩展冻结队列](../code/medworld_dense_baselines/runs/expanded_overnight_20260913/scheduler/QUEUE.md)
- [在线联合优化队列](../code/medworld_joint/runs/overnight_20260913/scheduler/QUEUE.md)
- [未来预测队列](../code/medworld_table1/runs/overnight_20260913/scheduler/QUEUE.md)

合并表导出 CSV、Markdown 和 LaTeX 表体；每个新指标必须通过本轮成功 worker 回执和产物 SHA256 核验，历史零样本结果显式标记为历史。论文主表原有六任务匹配协议仍是待完成计划，本轮带协议的实验预览不把不同队列硬合并成同一方法行。

本轮调度代码：[overnight_queue.py](../scripts/overnight_queue.py)，汇总：[overnight_report.py](../scripts/overnight_report.py)。实际执行的是各运行 `scheduler/source/` 和对应模型任务目录中冻结的源码。恢复应使用原命令、原计划和原源代码；截止时间后不会自行重新占卡。

## 正式启动记录

三队列已于 09-13 北京时间 16:32–16:36 启动，协调器 PID：dense 2004062（16:32:07）、joint 2015536（16:33:28）、forecast 2037768（16:35:59）。独立 guardian PID 2037769，GPU 监测 PID 1937685。共 66 项主任务及 26 项可选第二 seed 任务；任务含提取、训练及评分，因此数量不等于模型数。最新阶段与完成数从运行文件读取。

调度验证另覆盖独立 worker 截止、过期回执拒绝、不同测试集指标隔离。独立 guardian 还追踪本轮新 session 的后代，处理训练主进程已退出但仍遗留后代的情形；纯 CPU 测试确认只清理所属后代，保留无关进程。
