# Stage 1：0.8B，4＋4 slots 新运行

启动准备日期：2026-09-11。依据 [4＋4 slot plan](0911_stage1_slot_allocation_plan.md)。

2026-09-11 12:56，两个完整四任务周期的 smoke 训练及四任务评估通过后，已启动后台主训练协调器（PID 1564364），训练使用 GPU 0，早期评估预留 GPU 1。原始预训练权重重新初始化主运行；smoke 的验证样本更新不会继承到主模型。

官方问答接入问题已告知用户；为保留 14:30 预览窗口，首轮先使用下述本地派生问答。它是对数据接入的明确调整，不能声称完全复现官方 VQA 训练／测试设置。

## 配置与进度

骨干采用 Qwen3.5-0.8B 与冻结 V-JEPA 2.1 ViT-B，8×1024 状态。分类读取前 4 个 slots，疾病列表读取全部 8 个，分割与 SR 读取后 4 个。SR 新增两个低分辨率空间 attention 层，分别以 64×64、32×32 图像位置读取 4 个条件 tokens。

新代码位于 `code/medworld_stage1/slot44_*.py`，保留上一轮实现。计划训练 24,000 次 optimizer 更新，每任务 6,000 次；batch 为 8／4／8／8，累积 2。原始预训练骨干加新初始化可训练参数，不继承上一轮 checkpoint。每 400 步验证，保存各任务最佳和恢复状态；主测试使用统一 final checkpoint。

运行配置：[config.json](../code/medworld_stage1/runs/slot44_20260911/config.json)。实际启动及进度以 [运行报告](../code/medworld_stage1/runs/slot44_20260911/REPORT.md) 和运行目录状态文件为准；本文件本身不表示训练已经启动。

计划北京时间 14:05 在完整四任务周期末保存固定时间的早期 checkpoint，在独立 GPU 上评估验证集并生成四任务指标、训练曲线、问答／分割／超分样例，供 14:30 查看。正式训练继续。

## 问答数据的实际接入状态

官方 MIMIC-Ext-MIMIC-CXR-VQA 接口返回 403。现有 PhysioNet 账号可登录，但该项目仍显示需要签署 DUA。未代替用户签署协议，未获得官方 train/valid/test JSON。

为初版准备了本地 Chest ImaGenome 派生问答候选。训练和验证使用现有 silver scene graph，测试仅使用对应 gold 标注的图像。这里只取 disease、anatomicalfinding 两类的明确阳性关系，分别构造整图和解剖区域疾病列表。无明确阳性标注的情况排除，不把未知／缺失标签转成空答案；因此该候选不覆盖空集合识别，不能报告成官方 MIMIC-CXR-VQA benchmark。

| 划分 | 图像 | 患者 | 问题 | 整图问题 | 区域问题 |
|---|---:|---:|---:|---:|---:|
| Train | 14,399 | 10,525 | 89,927 | 14,399 | 75,528 |
| Validation | 172 | 172 | 1,047 | 172 | 875 |
| Gold test | 36 | 36 | 207 | 36 | 171 |

本轮所有任务共同排除 500 位 Chest ImaGenome gold 患者的训练暴露，从原 24,000 张训练图像中移除 251 张。图像、固定特征和 CXAS 伪标签按原始 index 复用，原缓存不改写。划分和来源见 [selection.json](../code/medworld_stage1/data/slot44_20260911_derived_v2/selection.json)。

接口核验后的训练池为：分类 13,681 张、分割 18,708 张、SR 23,749 张，疾病识别 89,927 题。问答 query 最长 19 tokens、答案含结束符最长 107 tokens，未发生 query／答案截断或长度排除。报告输入超过 384 tokens 的图像：训练 7/23,749、验证 0/318、测试 1/536。

2026-09-11 12:53，真实验证样本上的四任务梯度检查通过：分类的状态入口梯度仅在 S1–4，分割／SR 仅在 S5–8，疾病识别覆盖 S1–8；共享 encoder 有非零梯度。检查峰值分配显存约 3.90 GiB（不等于所有训练步骤的显存上限）。SR 的 HR 条件隔离与 compact checkpoint 重载一致性也通过。详见 [检查记录](../code/medworld_stage1/runs/slot44_20260911/checks/contracts.json)。

问答名称只做大小写和空白规范化，不用 LLM 重新判分。macro 指标在预先固定的疾病／征象词表上计算，另报告有 reference support 的类别宏平均、micro 指标、集合完全匹配、无法解析／词表外输出以及整图／区域分项。

分类与疾病识别使用当前报告辅助的状态；分割指标为 CXAS 伪标签一致性；SR 为长宽各 ×2 的合成退化恢复。早期验证与正式测试保持区分，旧报告生成成绩不用于本轮疾病列表指标。

## Table 2

论文已采用 DINOv2、MaCo、EVA-X、X-WIN、Qwen3.5、Ours 六行，四个任务、六列短指标 AUC／AP／F1／Dice／PSNR／SSIM。结果留空。随后按用户要求新增并启动了相同步数的 [Qwen0.8B 无 slots 四任务对照](0911_qwen08_noslots_baseline.md)；其余候选 baseline 的适配实验尚未执行。
