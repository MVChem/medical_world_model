# 09-12：8 个多层 slots 方法小图

最新纠正：后四个 slots 来自 **VLM 自带的 vision encoder**，并由一个轻量级 adapter 对齐视觉和视觉语言融合的 hidden states。用户还要求去掉 Early/Mid/Late 文字、缩小布局，并与 fig1 风格及字体一致。当前 [极简版 prompt](../27cvpr/ppt/prompt/multiscale_slots_0912/PROMPT_SIMPLE.txt) 已原位更新，Image 2 / Image 2.5 通用；生成时同时提供 [fig1 风格参考](../27cvpr/ppt/ppt/fig1_v7.png)，仅借用其外观。

已核对 fig1 的 PPT 字体：模块标题使用 Comic Sans MS Bold，短注释使用 Comic Sans MS Regular，数学符号使用 Times New Roman Italic。当前小图保留同一 VLM 的两种特征来源、各四个无文字标注的取层点、轻量级对齐 adapter、八个 slots、decoder 和小字联合优化标注。布局约 2.7:1，移除大括号、行标题和 decoder 内部图案，收紧留白。

初版英文绘图 prompt 作为历史记录保留，其中 V-JEPA 来源标注已被纠正；当前生成应使用上面的极简版：

- [Image 2 简洁版](../27cvpr/ppt/prompt/multiscale_slots_0912/PROMPT_IMAGE2.txt)
- [Image 2.5 详细版](../27cvpr/ppt/prompt/multiscale_slots_0912/PROMPT_IMAGE2_5.txt)
- [图稿说明与建议图注](../27cvpr/ppt/prompt/multiscale_slots_0912/README.md)

图示方案为：VLM 视觉语言融合路径取四层，同一 VLM 的 vision encoder 取四层；保留近似均匀覆盖深度的取点含义，但不在图中标层次名称或编号。两路经轻量级 adapter 对齐到共同的 `d` 维空间，各输出四个 slots，随后与 task decoder 联合优化。adapter 具体参数化和原生视觉骨干适配范围仍需在实现中确定。

这是本次新增的取层设计，与现有“八个 slots 均来自 VLM 最后层输出”的实现有区别。具体层号与每层读出结构尚未确定；本次未改训练代码、论文正文或已有实验结果。图稿说明中记录了这一差异，便于后续实现与稿件同步。
