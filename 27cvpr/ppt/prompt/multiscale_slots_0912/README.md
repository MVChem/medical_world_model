# 8 个多层 slots：方法小图 prompt

2026-09-12。按本次口述及后续纠正整理，供手动复制到 Image 2 / Image 2.5 生成方法小图。最新要求：视觉特征来自 **VLM 自带的 vision encoder**；增加轻量级 adapter，将视觉与视觉语言融合表示对齐；去掉 Early/Mid/Late 文字；进一步缩小图的布局，并保持 fig1 的风格和字体。

**当前使用：[PROMPT_SIMPLE.txt](PROMPT_SIMPLE.txt)，已原位更新。** 两个图像模型通用。上传 [fig1 风格参考](../../ppt/fig1_v7.png)，再整份复制该 prompt。参考图只用于字体和外观；新图的模型连接按本 prompt 绘制。

当前布局为：同一 VLM 内的两个特征来源 → **Lightweight adapter** → 两行各四个 slots → Task decoder。两路都经过 adapter，对齐后仍保留八个 slots。图幅约 2.7:1，面向约 85 mm 宽的论文小插图；缩小外边距、模块内边距与连线距离。移除取层文字、Semantic/Visual 行标题、decoder 内部图案和大括号；联合优化只用一行小字表示。

字体已从 [fig1_v7.pptx](../../ppt/fig1_v7.pptx) 的文本属性核对：模块标题为 **Comic Sans MS Bold**，短注释为 **Comic Sans MS Regular**，数学符号为 **Times New Roman Italic**，少量辅助技术文字为 Arial。颜色沿用浅蓝视觉模块（`#BFDFF9`）、淡紫 VLM 与浅绿 decoder（`#CFE7B5`，描边 `#3E6527`）。以当前 fig1 导出图作为外观参考。

以下两版仅为初版历史记录，其中 V-JEPA 来源标注已被本次纠正，不能继续作为当前架构的生成 prompt：

| 文件 | 用法 |
|---|---|
| [PROMPT_IMAGE2.txt](PROMPT_IMAGE2.txt) | 初版 Image 2 prompt，已由极简版替代 |
| [PROMPT_IMAGE2_5.txt](PROMPT_IMAGE2_5.txt) | 初版 Image 2.5 prompt，已由极简版替代 |

文件名仅区分交付版本，不依赖特定模型 API。当前版配合 fig1 参考图使用，以对齐字体和外观。

## 图中表达

- **S1–S4：融合语义来源。** 从 VLM 的视觉语言融合 hidden states 中，沿深度近似均匀选取四层，各对应一个 slot。
- **S5–S8：视觉来源。** 从同一 VLM 自带的 vision encoder 选取四层 hidden states，各对应一个 slot。
- **轻量级对齐。** 两路读出的表示都进入 adapter，映射到共同的 `d` 维表示空间，保留各自四个 slots 及其来源顺序。图中的单个 adapter 框抽象表示对齐／读出模块，不指定两路是否共享投影权重，也不额外引入对齐损失。
- **统一状态与训练。** 两组合为 `S ∈ R^(8×d)`，task loss 联合优化 adapter、slot 读出与任务 decoder。新视觉来源的骨干适配范围尚未指定，因此图中不加入冻结或全参数训练标记。空间任务所需的图像接口由完整方法说明承担。

“多尺度”表示网络深度上的语义和视觉层次。保留最初的浅层一层、中间两层、深层一层取点含义，但只画近似均匀分布的四个高亮条，不绘制 Early/Mid/Late 或层号。每层到单个 slot 的压缩／投影在轻量级 adapter 中抽象表达；具体结构仍需在实现中确定。

本次用户明确纠正了此前助手对“视觉部分”的理解：来源为 **VLM vision encoder**。新图 adapter 的作用是对齐两路抽取的 hidden states，不能把旧图中 V-JEPA 接入 VLM 的 visual adapter 直接当成此次模块。

## 与现有稿件及代码的关系

这是本次提出的**多层取点与对齐方案**。当前 [Methods](../../../sections/4_method.tex) 和旧训练路径仍采用“8 个可学习输入 slots → VLM 最后层对应位置的 8 个输出”。现有 4＋4 分组主要规定任务读取范围，并未实现这里的“4 个融合层＋4 个原生视觉层＋轻量级对齐 adapter”结构。本次交付为新图稿 prompt；已完成实验的结果不能作为此新结构的验证。

图中使用通用 `Task decoder Dq` 表达各任务的读取接口。slot 参数化、具体取层和最终任务路由需要与后续实现及 Methods 一并对齐。多层信息保留是设计目的，图中不附未经验证的性能结论。

## 建议英文图注

**Multi-depth state alignment and task-supervised learning.** Four hidden-state layers are sampled across the VLM's vision–language fusion pathway and four across its own vision encoder. A lightweight adapter aligns the two feature families into a common representation space, yielding four semantic and four visual state slots. Task supervision jointly trains the alignment/readout components and the task decoder.
