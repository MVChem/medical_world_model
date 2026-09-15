# 09-14：四张候选图的首版实画

**09-15 复盘：** 本页记录的是已导出的试图，不表示原候选实验已经完成。临床图只分析现有 baseline，未经过本方法 state 或未来预测路径；现有材料不足，后续由用户继续尝试。修订后的研究目的和制作边界见 [候选图复盘](0915_figure_review_and_next_steps.md)。

本次需求是用已有数据、代码和实验产物先尝试画图。检查主文与补充材料后，确认共有五个主文候选占位和一个补充候选占位；其中下游分析对应临床证据、状态检索、slot 条件扰动、空间输出四张。

交付入口为 [四图 PDF](../27cvpr/figures/generated/candidate_figures_v1.pdf)、[总览](../27cvpr/figures/generated/candidate_figures_v1_overview.png) 和 [脚本与说明](../27cvpr/figures/README.md)。另外用已有保存预测制作了 [纵向预测试图](../27cvpr/figures/generated/forecasting.pdf)。这些是供选图的工作版本，原论文中完整方法的占位没有改成其他模型的试验结果。

## 实际完成的范围

- **临床证据：** 从 353 张测试胸片中，按固定 seed 与图像 ID 哈希排序选出同时有 Cardiomegaly、Pleural Effusion 阳性标签的一张。对原图的 8×8 网格逐块用 Gaussian blur 替换，每次重新经过冻结视觉编码器和已有最终分类头；画原始 logit 减去扰动 logit。DINOv2、CheXWorld 使用相同扰动图像；不提供报告。另比较最高响应四格与 16 组随机四格的分数变化。这是现有 baseline 的归因试图，不能叫本方法的病灶定位，也没有病灶参考框可评价。随机区域对照是同病例探索检查，不能作为独立验证。CPU FP32 重算与原 BF16 缓存分数有小幅差异，逐模型差异保存在 provenance，图中显示本次真实重算分数。
- **状态检索：** 使用已完成的冻结视觉 slots，比较前两层、后两层和全部四层；不是 fusion／visual／八 slots 的完整比较。所有条件采用同一个测试候选库并排除同患者，按真实相似度排序。额外提供全测试标签重合和体位一致性，以及相同候选库的随机期望；缺失或不确定标签不解释为阴性。
- **条件扰动：** 固定匹配 frozen-slots decoder 和直接图像，替换为同一测试 split 的不同患者条件。病例子集和 donor mapping 预先固定，再展示响应最小和最大的诊断例；不按误差改善挑图，保留 SR 几乎不变的结果。变化只能说明条件敏感度，不自动证明条件有用。
- **空间输出：** 使用已有 4,096 图、20 epoch 的匹配 image-only／frozen slots／shuffled-trained 实验，分别显示 CXAS 伪标签、Montgomery 人工双肺参考和 4× SR。旧联合训练 pilot 的训练预算不同，不并入这组匹配比较。
- **额外纵向图：** 采用已有 0.8B full-token／8-slot 保存预测，保留失败结果；使用固定 finding 的 report-label onset、resolution、persistence。自动标签和参考报告矛盾的候选剔除并记录。随访图标明 observed follow-up, reference only；这些不属于尚在训练的 9B 新结果，也不代表已经完成多深度 4+4 方法。

## 尚不能由现有材料完成的面板

临床图随机对照的“等面积”仅指输入画布面积相同。CheXWorld 的中心裁剪与原图 padding 会造成保留的有效影像面积、解剖区域面积不同，不能把它当成严格区域匹配的定位验证。图内与 provenance 均已补充该限制。

本方法的匹配 image-only 临床分类归因、MS-CXR 真实病灶参考框与 grounding 输出、fusion／八 slots 检索、匹配预算的完整联合模型空间输出，以及真实 slot-to-patch attention。冻结多层视觉提取使用均值汇聚；不能把 decoder 对聚合 slots 的权重重新标成二维病灶 attention。

本次未启动训练或下载模型，真实前向使用 CPU。所有图片、预测和状态来自本地实验，未用生成模型制作胸片、病灶或热图。各图的选择规则、权重和数据指纹、逐例数值及重建脚本都随图保存。
