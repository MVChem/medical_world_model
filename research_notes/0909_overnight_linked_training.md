# 2026-09-09 夜间：原始 MIMIC-CXR + IV 数据训练

本轮用户已授权在新链接数据上启动训练，取代当天早些时候“先讨论、不训练”的阶段性决定。截止需求为 **2026-09-10 07:30（Asia/Shanghai）** 能查看训练结果。后续语音将“两块 GPU”的限制更新为“可使用所有空闲 GPU，始终排除物理 GPU 4”。

实时入口：[训练、评估与 Table 1](../code/medworld_table1/runs/linked_20260909_overnight/REPORT.md)。[协调器状态](../code/medworld_table1/runs/linked_20260909_overnight/runner_status.json)与各方法 `status.json` 是执行状态依据；本文记录方案，不预先宣称训练完成或优于旧版。

## 数据与输入

- 来源为 `mimic_cxr_iv_linked/runs/full_20260909` 的 `same_admission_6h_72h_acquisition_matched_image_qc_prior_ehr` 层级：同住院（含关联急诊）、相邻检查、6–72h、采集字段已知一致、自动图像质控通过，当前前存在临床记录。
- 从该层级 23,009 个训练 pair 按 pair ID 哈希取 6,000 对，每患者至多 4 对，共 3,767 位患者；不依据标签或变化筛选。验证 230 对 / 62 位患者，测试 297 对 / 94 位患者，为该层级全部验证和测试样本。沿用官方患者互斥 split。
- **不读取 Qwen-Gate 标签进行抽样、监督或评价。** 影像、报告与临床值来自原始 MIMIC，疾病监督来自官方 MIMIC-CXR CheXpert 弱标签。blank/uncertain 不当作阴性。
- EHR 选择 18 项血液化验和 14 项 ICU 监护字段，以及原始 eMAR 给药记录。化验/监护取前 48h 内每个 item 最近可用记录，给药取前 24h 最多 6 种药物的最近记录，保留 held / not-given 等原状态，不推断实际治疗效果。每条记录必须原生 patient/admission 匹配，事件时间与 storetime 同时不晚于该影像。
- 输入只序列化原始值、单位、状态和距离当前片的时间，不放入患者 ID、住院 ID、出院诊断、未来操作或结束时间。每一条入模临床记录保存原始表与 `_source_record`，见新缓存的 `ehr_audit.jsonl`。
- 12,135 个图像端点中，12,077 个有本轮选定的近期临床值；其余明确表示无选定值，不据此推断正常。EHR 384-token 独立预算，按 ICU / 化验 / 给药交替纳入完整条目，不截断单个数值。
- 原始清洗报告在缓存中完整保留，模型报告预算 384 token；12,135 份端点报告中 4 份超过预算。Copy Current 使用相同报告预算，不复制 EHR。评估参考仍用完整报告。
- 当前报告真实发布时间仍不可核实，本轮是回顾性“报告已可用”的预测。目标端图像、报告、该端 EHR 仅作监督；普通预测输入不包含未来数据。horizon 只使用 6–24h / >24–72h 分档，实际间隔用于审计。

## 模型与预算

沿用上一版 Qwen3.5-0.8B / V-JEPA 2.1 ViT-B、8 个 slots、rank-8 LoRA、4 层 width-512 LWM、独立报告 decoder 的实现，使用同一固定未来目标编码器策略及 `latent + CE + 0.5 BCE + 0.1 image-only replay BCE`。没有未经诊断就更换 decoder。

| 物理 GPU | 任务 |
|---|---|
| 0 | 0.8B MedWorld-JEPA：Stage 1 最多 1.5h，累计最多 6h；结束后本卡验证/测试 |
| 1 | 0.8B native Qwen direct：最多 6h；同样的当前报告、EHR、horizon；结束后本卡评估 |
| 5 | Stage-1 后启动冻结编码器对照，共享初始化并跟随主模型相同 Stage-2 batch、更新数、损失和学习率；启动前可用于只读解码检查 |
| 6 | Copy Current、旧/新 checkpoint 状态诊断、中途验证，顺序运行 |
| 7 | 独立本地 Qwen3.5-9B 零样本未来报告生成；完成后释放自有 vLLM，再做指标评估 |

物理 GPU 2、3 上原有 Qwen 语义筛选继续运行，不抢占；GPU 4 不进入任何 CUDA 可见设备列表。所有任务通过 UUID 定位物理卡，避免坏卡造成 ordinal 重排。9B 是预训练零样本对照，不声称本轮微调，也不填不存在的连续疾病头 AUPRC。

训练改用每个 epoch 无放回排列，减少短 Stage 1 反复抽到同一观测；主模型与冻结对照共享确定性数据顺序。每 400 步记录固定验证前 64 对的损失，并保存验证 CE 最优 checkpoint；每 45 分钟另存可恢复 checkpoint。累计 3h 的 checkpoint 沿用历史文件名 `checkpoint_4h.pt`，**实际保存时刻以 checkpoint 的 elapsed_seconds 为准，文件名不表示本轮为 4h**。

非跟随训练最晚 05:45 在 optimizer 边界结束，以留出生成与评分时间；冻结对照跟随主模型最终步数。最终主表使用最终 checkpoint，验证最优快照保留供诊断。本轮不依据测试结果选 checkpoint。

## 评估与问题定位

所有模型共享本轮验证/测试 pair；主表为完整 297 对测试结果，独立列出 Copy Current。使用真实 CheXbert 与 RadGraph，4 候选 Future R@1 记录实际覆盖，疾病 AUPRC 只来自可用的连续头。稳定/可判断变化组、每类支持量保存在 `metrics.json`，并记录完整报告去重数、最大重复次数和前五模板占比。

旧版 1,000 个测试输出中主模型只有 41 种完整报告，最大模板出现 572 次。新诊断在**固定验证患者**上比较当前 image-only state、当前多模态 state、预测未来 state、真实未来在线/固定 state、打乱和均值 state；真实未来读出包含目标报告，属于重建诊断，不能当预测成绩或严格性能上界。均值 state 是该诊断小样本的均值，只作干预，不是预设的训练集常量基线。打乱可能改变 horizon 对应关系，需联合读解。

初步 24 位不同验证患者的旧最终 checkpoint 诊断：预测 state 产生 6 种报告，最常见 15 次；真实未来在线 state 仍只有 8 种报告，固定目标 state 只有 5 种。打乱预测 state 后 finding BCE 从 0.2454 增至 0.4720，报告 CE 从 1.5055 增至 1.5222。说明疾病头确实使用了患者状态，但现有报告读出仍明显模板化，不能仅归咎于 LWM，也不能断言 slots 完全无用。上述为小规模诊断，尚不唯一定位原因。

另用同一真实 checkpoint、同一 state、同一前缀比较 cached decode 与完整前向的下一 token logits，检查生成实现是否存在明显不一致。结果见 `generation_consistency_old.json`，不将 BF16 数值差异直接判成实现错误。

已完成的生成自检为 2 个病例、每例 32 步，64/64 次最高分 token 一致，BF16 最大词表 logit 差异 0.8125。只覆盖这组前缀，不能证明所有解码路径均无问题。

进一步检查旧 **Stage-1 checkpoint**：同样 24 位验证患者的 image-only 当前报告生成全部为同一份“无急性心肺异常”报告；加入当前报告、打乱 state 或换成均值 state 仍生成同一模板，非空报告。疾病阳性 CheXbert F1 为 0。报告模板化在引入 LWM 前已经出现。与最终 checkpoint 对照，Stage 2 从正常模板转向少数异常模板，但没有恢复充分的病例差异。[Stage-1 诊断](../code/medworld_table1/runs/linked_20260909_overnight/diagnostics_old_stage1/report.md)、[最终 checkpoint 诊断](../code/medworld_table1/runs/linked_20260909_overnight/diagnostics_old_final/report.md)。

本轮同时改变配对范围、EHR、患者抽样、采样方式、Stage-1 时长和报告预算，不能把新旧成绩差异单独归因于临床连接。验证患者 62、测试患者 94，单 seed、小样本支持量限制仍存在；先判断是否学到病例相关输出，再设计更细的消融。

### 当晚已完成的基线（22:49 前）

| 方法 | 测试对数 | Future R@1 | Transition F1 | RadGraph F1 | CheXbert F1 | 不同报告数 |
|---|---:|---:|---:|---:|---:|---:|
| Copy Current | 297 | 0.3074 | 0.0000 | 0.2077 | 0.5761 | 见实时报告 |
| Qwen3.5-9B zero-shot | 297 | 0.3734 | 0.1525 | 0.1929 | 0.6151 | 297 |

两行 R@1 共享 77 对可构造 4 候选的查询，其余指标按完整 297 对及真值有效掩码计算。9B 没有连续疾病头，因此不填 AUPRC。9B 的 384-token 输出上限导致验证 19/230、测试 33/297 份输出以长度上限停止；原样保留并记录，未按测试结果修改提示或重跑挑选输出。9B 在报告疾病 F1 上较高，但 RadGraph 较低，不宣称整体优于 Copy，也未做显著性结论。

0.8B 主模型、直接模型和冻结对照均通过真实数据 smoke test；正式主模型/direct 已于约 22:40 启动。冻结对照的正式训练仍等待主模型 Stage-1 checkpoint。9B 专用 vLLM 已正常关闭并释放物理 GPU 7，原有 GPU 2、3 的筛选进程未停止。

## 复核与运行

- [版本化数据准备脚本](../code/medworld_table1/prepare_linked.py)、[配置](../code/medworld_table1/configs/linked_20260909.json)。旧缓存与旧 checkpoint 保留。
- 8 项测试通过，包含双时间戳截止、患者/住院隔离、原始报告目标保留、未来输入不进入预测、无放回采样、真实 Qwen 反向梯度和停止目标梯度。
- 协调器通过独立进程组启动，关闭聊天不应停止训练。源码快照、GPU UUID 计划、子进程命令/PID、模型输入哈希与日志保留在运行目录。
- 自动报告每 15 秒刷新状态，训练曲线约 5 分钟更新；训练与评估失败会显式保留错误，未完成的单元格不填虚构数值。
