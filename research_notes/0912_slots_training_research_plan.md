# 09-12：8 个 slots 的训练与密集任务研究计划

2026-09-12｜两次聊天的总结与后续计划。

拟继续以8个slots和任务解码器联合训练为主线：分类读前4个，诊断读全部8个，分割／超分读后4个，VQA后续接入。用BCE、答案交叉熵、Dice和像素MSE回传更新槽位嵌入、adapter、VLM LoRA与解码器，V-JEPA冻结；暂不加多层特征预测。

FeatUp的图像引导版本通过多视图一致性学习特征上采样；UPLiFT在分割中冻结DINO，用图像浅层特征引导局部注意力，以多尺度重建损失训练上采样器，再接线性探针。[1][2] 借鉴其分工，让slots提供语义、图像分支补细节；超分仅输入LR，HR作目标。

DINO与VLM视觉特征可采用相似读出，但训练目标、取点不同，需匹配比较。分开评测视觉编码器早中晚层与语言层图像位置的输出；8个slots无天然二维网格，应由带位置的查询读取。

设置仅图像、图像＋冻结slots、图像＋联合slots三组，冻结涵盖整个状态编码器；冻结与联合组从同一编码器检查点出发，统一患者划分、解码器与预算。另用同初始化的新头比较纵向训练前后的冻结表示，并加预算匹配的无未来监督对照，区分任务适配与世界模型收益。

论文与讨论来源（不计入正文）：

- [1] Fu et al. **FeatUp: A Model-Agnostic Framework for Features at Any Resolution**，ICLR 2024。[本地 PDF](../related_works/24-ICLR-FeatUp.pdf) · [原文 §3、§4.2](https://arxiv.org/html/2403.10516v2)。此处借鉴其图像引导上采样分支，论文另有逐图拟合的隐式版本。
- [2] Walmer et al. **UPLiFT: Efficient Pixel-Dense Feature Upsampling with Local Attenders**，CVPR 2026。[本地 PDF](../related_works/26-CVPR-UPLiFT.pdf) · [原文 §3.3、§4、§5](https://arxiv.org/html/2601.17950v2)。其图像超分实验上采样的是VAE latent，不能写成由DINO tokens直接重建图像。
- 聊天：[讨论 Slots 训练方法](https://chatgpt.com/share/6aa5018c-cc70-83e8-ad69-5660859ed2b0) · [大模型 Token 做超分辨率](https://chatgpt.com/share/6aa4ffce-9974-83ee-bac4-8d563b5e01e7)。本文的“8个slots”对应本次口述的“8个source”。
