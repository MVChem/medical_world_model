# 09-12：FeatUp／UPLiFT 与 VLM hidden states 下游代码核对

2026-09-12。承接 [8 slots 研究计划](0912_slots_training_research_plan.md)。本次阅读官方源码及论文，没有加载这些方法的完整权重、运行训练或复现分数。下文把源码可确认的行为、论文训练方案和对本项目的建议分开说明。

最相关的发现：**FeatUp／UPLiFT 适合参考二维特征上采样；LISA 更接近少量状态 tokens 控制分割；连续 VLM hidden states 也可以作为扩散条件，但这与生成离散图像码是两种接口。** FeatUp 已有 CLIP 视觉特征入口，不能把它概括成只适用于 DINO。

| 方法 | 实际读出的表示 | 图像如何进入输出路径 | 训练与输出 |
|---|---|---|---|
| FeatUp | DINO、DINOv2、CLIP 等视觉特征图 | RGB 图像引导局部滤波 | 冻结 backbone，训练上／下采样模块；输出高分辨率特征，另接分割／深度头 |
| UPLiFT | 默认 DINOv2-S/14 最后视觉层特征；另有 DINOv3、VAE 配置 | 浅层 CNN 从输入图像提取 guidance | 冻结 backbone，训练 CNN encoder、共享 decoder、Local Attender；输出特征或 VAE latent |
| LISA | 与 `[SEG]` 输出对齐的 LLaVA 最后语言层状态 | 独立 SAM image encoder 保留密集图像特征 | LoRA＋任务投影＋mask decoder 联合训练；输出 mask |
| S1-Omni-Image | S1-VL／Qwen3-VL 最后语言层序列，经 alignment 投影 | 编辑时输入图像另经 VAE，与噪声 latent 一同送入 DiT | 公开代码确认连续条件推理；论文后期冻结 VLM，训练 alignment＋DiT |
| HealthGPT | 早／晚 CLIP 特征进入 LLM；语言输出头生成图像码 | 图像作为自回归生成的输入条件 | H-LoRA／adapter 适配；VQGAN 将离散码解码成图像 |
| PURE | 自回归多模态模型的输出经词表头成为图像码 | LQ 图像与文本预先组成 token 序列 | 以目标序列交叉熵训练，图像 tokenizer 的 decoder 还原图像 |

**FeatUp：先把特征变密，再读任务。**

官方入口 `hubconf.py::UpsampledBackbone.forward` 同时把 backbone 特征和原图交给上采样器。核心 `JBUStack` 有四个独立的 `JBULearnedRange`，每级放大两倍。每级把原图池化到目标大小，从 RGB guidance 学习 range kernel，与空间 kernel 结合；先 bicubic 放大特征，再用 `AdaptiveConv` 做位置相关的局部滤波。实现还有 kernel fixup 和最终 1×1 残差投影，不能把完整实现简化为纯固定双边滤波。[入口](https://github.com/mhamilton723/FeatUp/blob/6b5a6c0e91f75e69194807128dcbc39c3084a30d/hubconf.py#L10) · [上采样实现](https://github.com/mhamilton723/FeatUp/blob/6b5a6c0e91f75e69194807128dcbc39c3084a30d/featup/upsamplers.py#L184)

训练的监督来自同一图像的随机平移、缩放、翻转：预测的高分辨率特征经过相同变换及可学习下采样后，应接近冻结 backbone 对变换后图像提取的低分辨率特征。`training_step` 在 `no_grad` 中提取特征；优化器只收集 upsampler、downsampler 和可选 uncertainty head 的参数。基础项为特征重建误差，可附加 uncertainty、CRF 等项；它不是先用分割标签训练这个上采样器。[训练与优化器](https://github.com/mhamilton723/FeatUp/blob/6b5a6c0e91f75e69194807128dcbc39c3084a30d/featup/train_jbu_upsampler.py#L118)

分割 probe 在 `train_probes.py` 中就是 `Conv2d(C, 27, 1)`，按有效标签计算交叉熵；深度分支输出一个通道。论文 §4.2 先在低分辨率特征上训练 probe，再冻结 probe、替换为上采样特征；另有端到端分割实验，不能混成一个训练协议。[probe 源码](https://github.com/mhamilton723/FeatUp/blob/6b5a6c0e91f75e69194807128dcbc39c3084a30d/featup/train_probes.py#L32) · [论文 §4.2／§4.4](https://arxiv.org/html/2403.10516v2)

按代码推导，DINO/16 的 224×224 输入得到 `[B,384,14,14]`，四级上采样成为 `[B,384,224,224]`。DINOv2/14 的初始网格为 16×16，原始四级输出会到 256×256；训练代码再插值回输入大小。不能只换 backbone 名称就假定输出空间尺寸不变。`CLIPFeaturizer` 调用的是视觉特征接口并排除 CLS，尚未经过语言模型的多模态融合。[backbone 映射](https://github.com/mhamilton723/FeatUp/blob/6b5a6c0e91f75e69194807128dcbc39c3084a30d/featup/featurizers/util.py) · [CLIP 入口](https://github.com/mhamilton723/FeatUp/blob/6b5a6c0e91f75e69194807128dcbc39c3084a30d/featup/featurizers/CLIP.py)

**UPLiFT：复用同一个两倍上采样器，局部重组特征。**

默认配置使用 DINOv2-S/14、384 通道；ViT wrapper 默认取最后一个 block 的输出并恢复 NCHW 网格，prefix tokens 单独返回。`UPLiFT.forward(img, x)` 的两个参数明确是输入图像与已有特征图。浅层 CNN 产生 guide，decoder 结合 guide 与当前特征，输出 Local Attender 所需的引导表示。默认 17 个邻域偏移；`LocalAttender` 生成每个目标位置的 17 个权重、做 softmax，再加权汇聚本级输入特征的邻居。输出通道保持为 backbone 通道数。[视觉取点](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/extractors/vit_wrapper.py#L136) · [上采样调用链](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/uplift.py#L597) · [Local Attender](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/uplift.py#L740)

与 FeatUp 的四套 JBU 参数不同，UPLiFT 多次调用同一个 decoder；默认 `enc_share=True`，图像 CNN 也只计算一次。训练从 448、224、112、56 像素图像提取固定 backbone 的对应特征，覆盖 1／2／3 步上采样及其中间输出，用 MSE 对齐相应尺度的教师特征；优化器只接收 `uplift.parameters()`。这里的多尺度是不同输入分辨率，**不等于预测视觉编码器早／中／晚三层**。[训练循环](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/train_uplift.py#L233) · [配置](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/configs/uplift_dinov2-s14.yaml)

图像超分走单独的 SD VAE 配置：图像编码成 VAE latent，UPLiFT 放大 latent，最后通过 VAE decoder 得到像素图。DINO 分支返回的仍是特征。公开 README 将完整下游评测指向 JAFAR／FM-Boost，当前仓库不能直接当成已配齐的分割 benchmark。另须注意 `UPLiFTExtractor.forward` 的推理包装把上采样也置于 `no_grad`；若用于我们的联合训练，应调用可微的核心模块，不能直接照搬这个包装。[解码与推理包装](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/uplift_extractor.py#L142) · [评测说明](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/README.md)

**LISA：VLM hidden state 作为分割条件，是 slots 最直接的参照。**

其前向可概括为：

```text
图像＋问题 → LLaVA 最后语言层 → 取与 [SEG] 输出对齐的状态
                              → MLP：D → D → 256 → sparse prompt
图像 → 冻结 SAM image encoder → image embeddings
sparse prompt＋image embeddings → 可训练 SAM mask decoder → mask
```

这不是把一个 hidden state 直接 reshape 成 mask。`text_hidden_fcs` 做维度投影，SAM prompt encoder 接收 `text_embeds`，mask decoder 同时读取稀疏条件和密集图像特征。默认损失为文本 CE＋2×mask BCE＋0.5×Dice。训练脚本冻结视觉塔及原有多模态 projector，使用语言侧 LoRA，并开放 `text_hidden_fcs`、`mask_decoder`、`lm_head`、`embed_tokens`。mask loss 能经投影回传到语言侧可训练参数。[任务读取与 mask decoder](https://github.com/dvlab-research/LISA/blob/3cb2d4301f1af4691bd4f3938335ef06e76f155a/model/LISA.py#L235) · [冻结与训练参数](https://github.com/dvlab-research/LISA/blob/3cb2d4301f1af4691bd4f3938335ef06e76f155a/train_ds.py#L160)

移植时必须重新做索引：源码的 `seg_token_mask` 用 `input_ids[:,1:]` 对齐 next-token 位置，并硬编码单张前置图像展开所需的 255 个偏移。不能把这段索引直接搬到 Qwen3.5／MedGemma。LISA 的任务 token 还受问题条件化，我们的状态 slots 在任务 query 之前产生，两者也不完全相同。[索引位置](https://github.com/dvlab-research/LISA/blob/3cb2d4301f1af4691bd4f3938335ef06e76f155a/model/LISA.py#L187)

**S1-Omni-Image：连续隐藏状态接扩散条件，支持医学超分／分割等图像编辑。**

公开实现的 `_forward_qwen3_with_alignment` 取最后语言层 hidden states；`AlignmentLayer` 支持 linear、MLP、MLP＋LayerNorm，把默认 5120 维映射到 3584 维，作为 DiT 的 `encoder_hidden_states`。编辑时，源图像另经 VAE 编码，和待生成的噪声 latent 拼接后送入 DiT，因此像素结构并非全部压在少量文本状态里。条件长度随序列变化，没有固定压缩成 8 个 slots。[alignment](https://github.com/ScienceOne-AI/S1-Omni-Image/blob/7cdb79d6481aaf09e21c406eb9f0527769d2ff0a/s1_omni_image/alignment.py#L19) · [隐藏状态读取](https://github.com/ScienceOne-AI/S1-Omni-Image/blob/7cdb79d6481aaf09e21c406eb9f0527769d2ff0a/s1_omni_image/modeling.py#L779) · [编辑与 DiT 条件](https://github.com/ScienceOne-AI/S1-Omni-Image/blob/7cdb79d6481aaf09e21c406eb9f0527769d2ff0a/s1_omni_image/modeling.py#L1085)

源码与论文有一个需要记录的差别：公开推理先生成 `output_ids`，再通过 `_encode_output_ids` 重新前向编码序列，裁掉模板前缀；这个调用没有再次传入 `pixel_values`。它不是直接复用首次图文生成时缓存的 hidden states，不能照论文措辞声称本次已核实“同一图文前向直接收集”的实现。[公开调用链](https://github.com/ScienceOne-AI/S1-Omni-Image/blob/7cdb79d6481aaf09e21c406eb9f0527769d2ff0a/s1_omni_image/modeling.py#L855)

训练范围来自论文，当前公开仓库以推理服务为主：先训练语言侧任务响应，再冻结底座训练 alignment；后期冻结 VLM／VAE，联合训练 alignment 与 DiT，使用 flow matching 和表征对齐损失。这个例子支持“冻结 VLM hidden states＋训练图像解码模块”的路线，不能据此说其像素损失微调了 VLM。[论文 §5](https://arxiv.org/html/2606.24441v1)

**HealthGPT／PURE：图像码生成路线。**

HealthGPT 的公开 demo 对理解任务选择 CLIP `hidden_states[-2]`，对生成任务选择 `hidden_states[1]`，都去掉 CLS，再经 `mm_projector` 送入 LLM。它明确用早层视觉信息服务生成，这对我们的层间特征比较有参考价值。[视觉取点配置](https://github.com/DCDmllm/HealthGPT/blob/30aa31e4c1ba903b60f1772aab839b3b7967afec/HealthGPT/llava/demo/utils.py#L59) · [视觉提取器](https://github.com/DCDmllm/HealthGPT/blob/30aa31e4c1ba903b60f1772aab839b3b7967afec/HealthGPT/llava/model/multimodal_encoder/clip_encoder.py#L35)

输出端增加 `<start_index>`、`<idx_i>` 等词表项，自回归生成离散图像索引；`idx2img` 查询 VQGAN codebook、恢复默认 32×32 latent 网格，再解码成图像。论文包含医学超分实验，但公开 README 主要承诺已发布 VQA 和重建所需权重，不能把重建 demo 当作完整 SR 复现。[生成 demo](https://github.com/DCDmllm/HealthGPT/blob/30aa31e4c1ba903b60f1772aab839b3b7967afec/HealthGPT/llava/demo/gen_infer.py#L99) · [图像码解码](https://github.com/DCDmllm/HealthGPT/blob/30aa31e4c1ba903b60f1772aab839b3b7967afec/HealthGPT/taming_transformers/idx2img.py#L83) · [公开范围](https://github.com/DCDmllm/HealthGPT/blob/30aa31e4c1ba903b60f1772aab839b3b7967afec/HealthGPT/README.md) · [论文](https://arxiv.org/html/2502.09838v3)

PURE 是有训练代码的自然图像 SR 例子。其 Chameleon 系模型把最后 hidden states 送入 `lm_head`，按移位后的目标 tokens 计算交叉熵；推理得到图像 tokens 后，通过 codebook＋图像 decoder 输出像素。可参考它如何组织 LQ／HQ 与文本的训练序列，但不能将其称为“冻结几个连续 hidden states，再训练轻量 SR head”。[语言输出头与 loss](https://github.com/nonwhy/PURE/blob/2c6ae90102167af87f38569bc45693268ca391cb/pure/model/chameleon/modeling_chameleon.py#L1657) · [图像解码](https://github.com/nonwhy/PURE/blob/2c6ae90102167af87f38569bc45693268ca391cb/pure/model/chameleon_vae_ori/image_tokenizer.py#L91) · [训练数据组织](https://github.com/nonwhy/PURE/blob/2c6ae90102167af87f38569bc45693268ca391cb/TRAIN.md)

**对应我们 8 个 slots 的建议。**

1. 比较 DINO 与 VLM 视觉塔时，保留各自真实二维网格、注明取层，使用可训练投影接同一种密集头；FeatUp 已有 CLIP 入口，说明相似接法在代码上成立，效果需要我们的数据验证。
2. 比较语言层图像 tokens 与末尾 slots 时，把它们当成不同表示。图像位置有可追溯网格；四个空间 slots 是压缩状态，不应直接冒充 2×2 图块。可借鉴 LISA，让空间／器官 query 从 slots 读取条件，再与图像特征融合。
3. 第一版继续任务 loss 联合优化状态编码路径与任务头。FeatUp／UPLiFT 的特征一致性可留作后续辅助监督，无需先照搬其整套预训练；使用官方推理封装前应核对 `no_grad`，避免意外截断梯度。
4. 分割／SR 维持“图像＋状态”的接口，并用 image-only、整个编码器冻结、联合优化三组判定状态的额外贡献。SR 的视觉条件均从同一 LR 图像取得。若要论证 slots 自身的空间信息，可再用 slots-only 粗分割读出；联合图像分支的高分不单独证明这一点。

源码链接已固定到本次检查的 commits；临时检出位于 `/tmp/medical_world_code_review_0912/`。本次新增阅读笔记，不改变现有训练代码和队列。
