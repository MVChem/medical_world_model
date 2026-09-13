# CVPR 2027 前文改写：以 medical world model 为主线

本次改写 `27cvpr/paper_config.tex`、Abstract、Introduction 和 Related Work，并补充必要参考文献。Methods、实验设置、Results 和两张主表保持原文。

## 标题

**MedWorld-JEPA: Learning a Medical World Model from Longitudinal Observations**

标题直接点明 medical world model，保留 MedWorld-JEPA 名称，并用 longitudinal observations 指明世界动态的学习来源。

## Abstract / Introduction 的叙事

1. World model 学习观测状态及其演化；在医学中，对应从当前检查预测后续检查可能揭示的患者状态。
2. 医学 world model 首先需要定义合适的世界状态：图像提供空间证据，报告提供临床语义，真实随访提供状态变化的监督。
3. JEPA 是学习这一 latent medical world 的原则。模型预测真实未来图像与报告的联合表征，在同一状态空间中学习演化。
4. 多深度 fusion / visual slots 构造世界状态；先用临床及空间任务确定状态内容，再用纵向预测共同适配状态编码路径和动态预测器。
5. 建立模型后再介绍能力：预测状态用于未来 finding / report readout；当前状态用于六项临床及空间任务。Dense tasks 仍以输入图像提供局部证据。

共享表征是 world model 的内部设计；全文的中心是医学世界状态与其动态如何通过 JEPA 学习。

## Related Work 的架构与来源

依据用户提供的[分享对话](https://chatgpt.com/share/6aa5ee41-4824-83e9-bb60-23d10eb6d8ff)，固定三个 subsection：

1. Clinical Multimodal Representation
2. Predictive Representation Learning
3. Medical World Models

新增 [BioViL](https://arxiv.org/abs/2204.09817)、[LLaVA-Med](https://papers.nips.cc/paper/2023/hash/5abcdf8ecdcacba028c6662789194572-Abstract-Datasets_and_Benchmarks.html) 和 [MAIRA-2](https://arxiv.org/abs/2406.04449) 的引用。明确说明 [VLA-JEPA](https://arxiv.org/abs/2602.10098) 的结构先例、[CLARITY](https://arxiv.org/abs/2512.08029) 的医学 latent dynamics，以及 [Clin-JEPA](https://arxiv.org/abs/2605.10840) 同时追求 EHR 轨迹预测与表征复用的目标。

## 稿件状态与验证

当前主表全部 TBD，因此前文陈述模型设计、学习目标和能力接口，保留实验结果位置。没有把旧原型数据写入新版，也没有加入首创或性能领先结论。

`make` 成功构建主文与补充材料；无未解析引用、LaTeX 警告或 box overflow。主文 9 页，正文在第 7 页结束，参考文献为第 8–9 页；补充材料 4 页。`preview/` 同步更新。
