# Methods 改写说明

2026-09-07，根据现有正文及最新图像输入接口讨论更新。本说明与 [0906_methods.md](0906_methods.md)、[0907_paper_plan.md](0907_paper_plan.md) 对齐；Methods 正文见 [4_method.tex](../26iclr/sections/4_method.tex)，Task Definition 保存在 [3_problem_formulation.tex](../26iclr/sections/3_problem_formulation.tex)，由 Methods 内部载入。本版本取代旧说明中将 D/S 一起作为预测输出的描述。

## 本稿的主线

**定义医学观测的状态 → 用任务监督建立状态空间 → 预测真实随访在该空间中的状态。** Future state prediction 和 reusable state representation 是需要分别评价的两个目标。

- 主设定为当前影像、报告共同编码，预测真实未来影像、报告共同编码的 latent state。
- 当前与未来采用同一种编码设计：固定 JEPA 产生单层空间特征 D，visual adapter 接入 VLM，视觉和文本证据之后的 learned slots 对应位置输出固定数量的 S。D 是状态编码路径的中间特征。
- U 是跨样本共享的可学习输入 embeddings；S 是同一组位置的 VLM 输出 hidden states。图示二者均为 8×d，但 U 是模型参数、S 是随观测变化的计算结果。当前不为各 slot 预先分配分类、分割等职责；最终 K 仍需选择。
- 统一以 S 表示 state，不再将 D 与 S 拼成所有模块共用的复合状态。任务 query Q 在 state 构造之后条件化 decoder，h 单独条件化 LWM；两者均不参与状态编码。
- 分类读取 S；指令式疾病识别读取 S 与 Q；分割读取输入图像 X、S 及适用的 Q。空间 decoder 直接接收原始任务图像，在内部进行图像特征提取或恢复，并接受 S 的语义条件，不另接 JEPA 输出 D。
- 超分读取 X^{LR} 与仅由同一 LR 图像构造的 S^{LR}，预测 HR 并使用按标量元素平均的像素 MSE。HR 图像仅作目标；不将 HR 图像或其报告放入超分状态输入。分割和超分都采用 image-only 状态，其他缺失模态组合仍属可选实验。
- LWM 仅以 (S_t,h) 为输入，输出 Ŝ_{t,h}，与真实随访编码的 sg(S*_{t⁺}) 计算 latent loss。未来图像经 D*、adapter、VLM 参与目标编码，未来报告直接参与同一目标；不输出或监督未来 D，也不把未来的模态可用性提供给预测器。
- Stage 1 的任务 loss 训练 task decoders（含其图像分支及状态条件化模块）、U、adapter 和选定的 VLM 参数；JEPA 冻结。Stage 2 的 future loss 更新 LWM 及当前状态编码路径；独立 task decoders 只在保留相应任务 replay 时更新。

## 本轮图与评价协议的对应修改

Fig. 1 的空间任务旁路从输入图像 X 分出、直接进入 task decoder，超分时使用 X^{LR}；D 仅连接 visual adapter。保留 Q、当前 S 到 task decoder 和 LWM 的分支、h 输入、Ŝ 与未来 S* 的单一监督接口。图文双分支展示主要纵向任务，dense-task 的 image-only 状态约束由正文与训练协议说明。

空间任务通过比较 H(X,S) 与同一图像分支的 H(X) 检验 S 的额外贡献，不能仅凭任务 loss 有经过 S 的梯度就宣称状态已被有效利用。预测评价的 readout 仅使用 Ŝ 与允许的任务条件，不拼接当前或真实未来图像，也不拼接 D。当前需要图像输入的分割、超分 decoder 不能据此直接解释为未来密集输出接口。

这些是方法与评价协议的定义，不是已完成性能的报告。结果表继续保留待填数值；附录病例不因本次接口修订改变。CT/NLST、MRI/OASIS 等扩展队列及缺失模态实验不作为已完成的主设定。

## 引用及其支持范围

| 引用 | 正文支持的设计 | 来源 |
|---|---|---|
| I-JEPA，`assran2023ijepa` | 从上下文预测目标特征的视觉学习原则 | [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html) |
| V-JEPA，`bardes2024vjepa` | 视频特征预测 | [原论文](https://arxiv.org/abs/2404.08471) |
| V-JEPA 2.1，`murlabadia2026vjepa21` | JEPA 保留空间结构并经 adapter 提供 VLM 视觉证据的动机 | [原论文](https://arxiv.org/html/2603.14482v1) |
| LLaVA，`liu2023llava` | 将视觉特征映射到语言 embedding 空间的 connector | [NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html) |
| LISA，`lai2024lisa` | 利用语言模型 hidden states 条件化图像分割；图像特征可由 decoder 内部的图像分支提取 | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Lai_LISA_Reasoning_Segmentation_via_Large_Language_Model_CVPR_2024_paper.html) |
| DINO-WM，`zhou2025dinowm` | 以预训练 patch features 学习 latent dynamics | [ICML 2025](https://proceedings.mlr.press/v267/zhou25t.html) |
| VLA-JEPA，`sun2026vlajepa` | VLM 与 latent world model 的结合、分阶段适配 | [v2 §3–4、A.2](https://arxiv.org/html/2602.10098v2) |
| LoRA，`hu2022lora` | 参数高效的 VLM 适配方式 | [ICLR 2022](https://openreview.net/forum?id=nZeVKeeFYf9) |
| MIMIC-CXR，`johnson2019mimiccxr` | 同患者影像与报告的纵向数据来源 | [数据集论文](https://www.nature.com/articles/s41597-019-0322-0) |

VLA-JEPA 的预训练联合使用 SSv2 和 DROID，已学习状态转移，然后适配机器人任务。本稿借鉴分阶段安排，但采用“任务监督建立医学状态 → 纵向状态预测”的顺序。它的 VLM tokens 表示 latent actions，未来目标是视觉状态；本文的 S 表示医学观测，未来报告参与 S* 的目标编码。LISA 的任务 token 不等同于本文的通用 state slots；其分割设计不要求复用 JEPA D，也不验证本方法的超分设计。DINO-WM 支持在预训练特征空间学习 dynamics 的思路，不能据此推出本文必须直接输入或预测 D。以上引用支持相关模块设计，不能替代本方法的实验验证。

## 仍需落实的实现

1. 目标编码器：同权重两次前向加 stop-gradient、EMA 或固定 Stage 1 副本。当前保留用户倾向的严格共享权重作为候选，没有锁定。严格同权重时，未来端本次不反传，但目标仍随共享参数更新而变化。
2. 时间条件：h 为实际间隔还是预设区间，以及如何关联真实随访时刻 t⁺。原稿的 horizon-bin 控制实验作为拟议协议保留。
3. 模块和损失：adapter、LWM、任务 decoder 的具体结构，尤其是空间 decoder 自身的图像 backbone 与 S 条件化方式；VLM 适配范围；slot 数；latent 距离和归一化；分割 loss；task 权重；超分退化方式。超分 MSE 已确定，future latent loss 的 MSE 仅为候选。Methods 定义接口与必要梯度路径，具体参数留在附录待定项。
4. Stage 2 是否保留任务监督：以 β 表达，β=0 或 β>0 尚未选择。
5. 相同 slot 位置规定预测和目标的 token 对应关系，不意味着已有各 slot 的临床语义标注；医学 JEPA 再预训练、CT volume→slice 上下文学习和 CT→X-ray 迁移仍是后续工作。

正文中的 TBD 刻意保留这些未定内容，避免用写作替代实现决策或实验结果。

## 复核范围

本说明已对照现有 Methods 的 X/S 任务接口、U/S 定义、预测公式和两阶段梯度路径整理。最终 PDF 的页数、引用与版式应以本轮正文和图更新后的实际编译检查为准；不沿用旧版编译结果作为本轮验证记录。
