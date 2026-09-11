# Table 2：2025–2026 顶会参照与扩展建议

核对日期：2026-09-11。状态：讨论稿，不是已确认的实验安排。本次查阅论文正文、主表、会议记录和官方仓库；没有修改论文、启动训练或生成本项目新分数。

后续决定：主表收敛为 DINOv2、MaCo、EVA-X、X-WIN、Qwen3.5、Ours 六行，采用 AUC／AP／F1／Dice／PSNR／SSIM 六列指标，论文已按此更新。下面的更多方法与八列布局保留为文献调研备选，不是当前训练名单。0.8B 的 4＋4 slots 新实验另见 [运行记录](0911_stage1_slot44_run.md)。

## 1. 主要判断

保留 classification、disease recognition、segmentation、super-resolution 四个任务。Table 2 可以扩成按方法排列、按任务分组的主表，补充近期胸片表征方法；不同任务的专用方法可放在独立 panel。完整方法比较、冻结表征读出、替换视觉编码器是不同实验，必须明确使用哪一种。

原候选 DINOv2、MaCo、EVA-X、X-WIN 可以作为起点。最值得新增的近期方法是 CheXWorld（CVPR 2025）、RadZero（NeurIPS 2025）和 AlphaRad（ECCV 2026）。HealthGPT（ICML 2025）是医学理解与生成的参照；它的原论文 SR 是 MRI 数据上的 ×4，不能直接填入我们的 CXR ×2 结果。

## 2. 论文实际怎么排表和评价

| 论文 / 会议 | 查阅位置与做法 | 对本项目的用途 |
|---|---|---|
| [CheXWorld, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Yue_CheXWorld_Exploring_Image_World_Modeling_for_Radiograph_Representation_Learning_CVPR_2025_paper.pdf) | §5.2、Table 1/2。分类覆盖多个数据集；分割和少样本共用分组表头。接线性头 / U-Net decoder，并全量微调编码器。区分部分文献引用分数与重跑结果，另标预训练资源不同的参考模型。 | 优先增加的同领域 world-model baseline。借鉴任务分组与多个测试条件，不照搬其全量微调协议作为我们已完成的设置。 |
| [X-WIN, CVPR 2026](https://arxiv.org/html/2511.14918v2) | §4.2、Table 1/2：冻结 encoder 的 linear probing，以及 4/8/16-shot 微调。比较通用视觉、CXR 视觉和 CXR 图文表征。另设投影/CT 重建实验。 | 适合表征比较；原文重建不是 CXR 超分，也不是患者未来预测。 |
| [RadZero, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/510e5c2fd5ad326f75594563f4ad5e0d-Paper-Conference.pdf) | Table 1 分类 AUROC；Table 2/3 grounding；Table 4 分割 Dice / pixel AUC。分割表注明 zero-shot 与 MGCA 1%/10%/100% 标签微调的不同设置。 | 补充医学图文和空间定位参照。若换成有监督任务头，必须标为本任务适配，不能继续称原生 zero-shot 成绩。 |
| [AlphaRad, ECCV 2026](https://arxiv.org/html/2609.01757v1) | Table 1 列测试来源；Table 2 分类；Table 3 grounding、phrase grounding、分割；Table 4 进一步按疾病分解。方法与 RadZero、CARZero 等比较。 | 很新的胸片图文候选。会议归属已在 [ECCV 官方录用表](https://eccv.ecva.net/Conferences/2026/AcceptedPapers) 检索到完整题名；录用表自注仍待出版方检查。 |
| [HealthGPT, ICML 2025](https://raw.githubusercontent.com/mlresearch/v267/main/assets/lin25n/lin25n.pdf) | Table 1 区分仅理解和理解＋生成模型；Table 2 模态转换；Table 3 在 IXI 上做 ×4 SR，比较 SRGAN、DASR、Real-ESRGAN、LIIF、BSRGAN，报告 SSIM/PSNR/MSE/LPIPS。 | 强调统一模型仍需与专用方法比较。可考虑适配 CXR disease-list / SR，但原文没有直接给我们这四任务的完整结果。 |
| [M4oE, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/e3b82f4c7ba93a88025cf97dca9edc83-Paper-Conference.pdf) | Table 1 用 Setting 区分 Single-Task / Multi-Task，按数据集和任务分组；GAMMA 同时报 glaucoma classification 与 optic-cup segmentation。主要报告 Accuracy、Dice；AUROC/AUPRC 等放附录。 | 借鉴表格组织与训练设置标注。乳腺/眼底多模态方法的直接移植成本高，优先作为实验设计参照。 |
| [GEMeX, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_GEMeX_A_Large-Scale_Groundable_and_Explainable_Medical_VQA_Benchmark_for_ICCV_2025_paper.pdf) | 胸片 VQA 包含开放、封闭、单选、多选四种问题，结合视觉和文本解释，并评价多种 LVLM。 | 启发 disease recognition 按问题类型 / 解剖区域报告。不是四任务模型，不能列成表征方法行；也不直接替换已计划的 MIMIC-CXR-VQA。 |
| [SynerMedGen, ICML 2026](https://arxiv.org/html/2605.08724) | Table 1/2 按解剖部位、模态转换方向分组，主表用 SSIM，PSNR/MAE 放附录；比较专用合成方法与 HealthGPT、UniMedVL。另做未见数据集评价。 | 借鉴专用/统一模型对照和外部测试；其 22 个合成方向不是我们的四任务。会议记录见 [ICML 2026 下载目录](https://icml.cc/Downloads/2026)，实现见 [作者仓库](https://github.com/piooip/SynerMedGen)。 |
| [MambaIRv2, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_MambaIRv2_Attentive_State_Space_Restoration_CVPR_2025_paper.html) | 专门的图像复原 / SR 方法，比较经典和轻量 SR 设置。 | 可作 SR 强基线，须在相同 CXR 退化、倍率、划分上重新训练或适配；原文自然图像成绩不是 CXR 成绩。 |

这些来源包含会议正文、官方 proceedings / program 和作者仓库。没有将 workshop 或只写着 submitted / under review 的论文当成主会录用证据。上述论文的比较协议并不统一：CheXWorld 全量微调、X-WIN 线性 probe、RadZero/AlphaRad 零样本评价应分清。

## 3. 推荐的 Table 2 布局

正文 caption 可用 **Downstream task comparison.** 方法名用短名，自己的方法写 Ours。排版用两层表头，合并同一任务的列。

| Method | Cls. | Cls. | Recog. | Recog. | Seg. | Seg. | SR | SR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|  | AUC | AP | All | Reg. | Lung | Heart | PSNR | SSIM |
| DINOv2 | | | | | | | | |
| MaCo | | | | | | | | |
| EVA-X | | | | | | | | |
| CheXWorld | | | | | | | | |
| X-WIN | | | | | | | | |
| RadZero | | | | | | | | |
| AlphaRad | | | | | | | | |
| **Ours** | | | | | | | | |

这是候选布局，不是承诺所有原方法原生支持所有列。主块若使用统一的重新训练解码器，应标为 adapted representation comparison，并在协议确定后删除不适合接入的方法。

- Cls.：AUC = macro AUROC；AP = macro average precision，明确有效标签和宏平均类别。
- Recog.：All / Reg. 分别是整图疾病列表与区域限定疾病列表，均报告标签集合 macro F1；恰好对应已有诊断计划中的两个问题子集。区域问答关联尚未完成，最终覆盖率需记录。
- Seg.：Lung = 左右肺 Dice 的平均；Heart = 心脏 Dice。逐器官和三器官均值放附录，不把 Dice 和 IoU 的重复展示当作新增任务证据。
- SR：PSNR / SSIM，沿用目前计划的 CXR ×2 合成退化，注明有效区域和强度范围。暂不为扩表加入新的 ×4 训练任务。

若希望主表更紧凑，可回到六列指标 AUC / AP / F1 / Dice / PSNR / SSIM，把 All/Reg. 和 Lung/Heart 分解放附录。八列版本扩展的是已计划任务内部的评价覆盖，不增加第五个任务。

可选的第二 panel：**Task-specific references**。HealthGPT 适配疾病列表与 SR；U-Net 作分割参照；MambaIRv2 作 SR 参照。只评价对应任务。专用方法与共享模型分别标注参数、训练预算和输入，不能把缺少某个输出接口记为零分。正式重跑前应先核查 HealthGPT 的 CXR 适配和标签评分接口。

## 4. 公平比较需要先定的协议

1. **输入。** 当前方案将报告输入状态，分类标签又源于报告。结果属于图文临床信息读出。若要与上述标准 CXR 影像诊断结果比较，建议另外建立统一 image-only 评价；所有模型都不能看到该图的目标报告，且我们需要训练/验证这种输入设置。现有图文方案不因本次讨论而自动更改。
2. **比较对象。** 外部 encoder 冻结、接同结构新 decoder，测的是表征读出；各模型共同微调测的是任务适配。不能只冻结对手、却把我们 joint 训练的旧数值填入同一协议。四任务本来就是 Stage 1 训练目标时，留出患者测试支持多任务泛化，不自动构成未见任务迁移。
3. **适配命名。** 给 DINOv2 / CheXWorld 等接我们的完整报告融合 VLM 和 8-slot 模块，主要构成视觉底座替换。若采用此设计，明确标为 adapted / encoder swap，不能称完整复现原方法四任务系统。不要把共同融合设置产生的 V-JEPA 行与实际上相同的 Ours Stage 1 重复计为独立方法。
4. **空间监督。** 目前分割对照的是 CXAS 伪标签。建议增加独立人工心肺 mask 测试，例如 [JSRT 对应的 SCR 标注](https://zenodo.org/records/7056076)，先核验图像/标注许可、匹配和评估范围。不能用教师自己生成的 mask 评价教师并把接近满分当作人工分割效果。
5. **统计。** 主表最终补有效测试规模和明确的区间。多随机种子均值±标准差与患者 bootstrap 区间回答不同不确定性，不混用。当前仅拟定，不启动额外实验。

## 5. 比单纯加行更值得补的结果

| 优先级 | 补充内容 | 作用 |
|---|---|---|
| 先做 | 按同一输入、训练和数据划分跑核心方法；扩大到完整合格测试池 | 得到可比较的四任务主表 |
| 先做 | Disease recognition 整图 / 区域两个子集；人工标注分割测试 | 提高疾病识别与空间任务的独立证据强度 |
| 随后 | 一个外部分类测试集，例如 CheXpert / VinDr；先核查外部 checkpoint 的预训练重叠 | 检查跨数据来源泛化，不把公开测试已见过的情况称为严格未见泛化 |
| 随后 | 一张四任务定性图：疾病概率、query/标准答案/预测、mask、LR/重建/HR/局部误差 | 让主表分数能对应到实际行为，按固定规则取样并保留失败情况 |
| 可选 | 低标签预算曲线 | 参照 CheXWorld/X-WIN 的样本效率评价；仍是原四任务，不增加任务类别 |

Stage 1 / Stage 2 的内部比较暂不重新加入本轮必须项。若论文最终要声称纵向训练改善当前表征，再设计训练曝光匹配的比较；只比较外部方法无法单独证明这项归因。

## 6. 接入状态与实施顺序

- [CheXWorld 官方仓库](https://github.com/LeapLabTHU/CheXWorld) 提供预训练模型和下游划分的 Google Drive 入口，适合优先核验加载。
- [X-WIN 官方仓库](https://github.com/RPIDIAL/X-WIN) 已有代码；本次 README 检查未找到明确预训练权重下载入口，不能据此保证可以立即接入，更不能据此断言全网没有权重。
- [RadZero 官方仓库](https://github.com/deepnoid-ai/RadZero) 已提供 [Deepnoid/RadZero 模型入口](https://huggingface.co/Deepnoid/RadZero)，[AlphaRad 官方仓库](https://github.com/jz5426/ECCV-2026-AlphaRAD) 已提供 [maxxyouu/AlphaRAD 模型入口](https://huggingface.co/maxxyouu/AlphaRAD)，且发布清单已勾选模型与示例代码。与 CheXWorld 一起优先做加载核验。
- [HealthGPT 官方仓库](https://github.com/ZJU4HealthCare/HealthGPT) 现包含 HealthGPT 与 HealthGPT-Pro 两部分；若对照 ICML 2025 论文，要锁定原 HealthGPT 配置，不能混用后续 Pro 版的能力说明。本次没有实际加载或测试这些模型。
- 先确定 Table 2 是图文读出还是另设 image-only 主比较，再核验 checkpoint 和做少量样本前向，最后固定名单和完整预算。不应先把七个外部方法全部写成已完成实验。

现有任务和数据边界依据：[最新 4＋4 slots 计划](0911_stage1_slot_allocation_plan.md)、[上一轮真实指标](0911_stage1_downstream_results.md)。本文件所有新表格单元格均为空；外部论文数值没有移入本项目结果。
