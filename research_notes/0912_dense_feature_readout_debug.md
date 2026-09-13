# 09-12 分割／超分特征取点与 shape debug

目前两条实验都读取了语言模型输出。今天的 frozen VLM dense baseline 读取最后一层的 image-token hidden states；昨天的 Ours 4＋4 读取最后输出的可学习 slots。它们都不能直接称为 VLM vision encoder 的输出。

本次将用户提到的「VRM vision encoder」暂按 VLM 自带的视觉编码器理解。执行了 Qwen3.5-0.8B 的真实验证图前向，以及 Ours 第 24,000 步 final checkpoint 前向；没有训练或替换原实验的缓存、权重。

## 今天的 dense baseline

实际取点见 [features.py](../code/medworld_dense_baselines/features.py)：

```python
output = model.model(**inputs, use_cache=False, return_dict=True)
h = output.last_hidden_state[0]                  # 最后语言层输出
native = h[inputs['input_ids'][0] == token_id]   # 其中的图像 token 位置
aligned = align_hidden(native)                  # 空间采样 + 通道采样
```

这里没有可学习 state slots。缓存的 `[B, 64, 1024]` 是一个 8×8 网格：空间上按等宽分箱取中心位置，通道上也按分箱取样；并非学习到的投影，也不是平均池化。读取点的源文件与运行归档副本逐字节一致。

单张验证图的 Qwen0.8B 实测如下。Qwen 的视觉接口将图像的 patch 轴展平；同尺寸批次可以按 B 拆回，变尺寸批次需按 `image_grid_thw` 分段。

| 位置 | 分割条件图：512×512 | SR 条件图：128×128，经 processor 放大到 256×256 |
|---|---|---|
| `image_grid_thw` | `[1, 32, 32]` | `[1, 16, 16]` |
| 视觉第 12 个 block 后、merger 前 | `[1024, 768]` | `[256, 768]` |
| 重排为视觉 feature map | `[1, 768, 32, 32]` | `[1, 768, 16, 16]` |
| merger 后、进入语言模型前 | `[256, 1024]` | `[64, 1024]` |
| 语言模型最后一层的图像 tokens | `[256, 1024]` | `[64, 1024]` |
| 当前缓存／head 条件输入 | `[1, 64, 1024]` | `[1, 64, 1024]` |

merger 后与语言层输出虽然 shape 相同，数值并不相同。本次同图的两者平均绝对差分别为 3.4303、3.4362。直接视觉 API 的返回值与视觉模块 hook 捕获值逐元素一致。

[DenseHead](../code/medworld_dense_baselines/heads.py) 将条件输入从 1024 投影到 64 通道，恢复 8×8 网格并插值到 32×32。另一支来自原图 CNN 或 V-JEPA；融合输入为 `[1, 128, 32, 32]`，分割输出 `[1, 3, 256, 256]`，SR 输出 `[1, 1, 512, 512]`。三种分支均完成前向且输出有限；本次使用新初始化的 dense heads 检查 shape，不衡量模型效果。

V-JEPA 支路的现有 HR/LR 缓存都是 `[B, 576, 768]`，保留 24×24 网格。它与上表的 VLM 条件输入在 decoder 内融合，不能混为同一组 tokens。

## 昨天的 Ours 4＋4 slots

[StateEncoder](../code/medworld_common/qwen.py) 的路径是：

```text
V-JEPA 最后视觉层 [B, 576, 768]
 → 缓存时池化到 8×8 [B, 64, 768]
 → adapter + 可学习位置编码 [B, 64, 1024]
 → 拼接报告 tokens 与 8 个可学习 slots [B, 64+T+8, 1024]
 → Qwen 的 24 层语言主干与末尾 normalization
 → last_hidden_state[:, -8:] [B, 8, 1024]
 → 分割／SR 取 s[:, 4:] [B, 4, 1024]
```

本次验证样本 T=55，完整输入为 `[1, 127, 1024]`；直接比较确认返回状态等于 `last_hidden_state` 的最后 8 个 token。分割输出 `[1, 3, 256, 256]`、辅助粗 mask `[1, 3, 32, 32]`，SR 输出 `[1, 1, 512, 512]`。SR 读取 `lr_features`，改变 `hr_features` 不改变其状态。

“后 4 个 slots”是 token 位置，不是后 4 层，也不是四个确定的图像区域。这个模型构造时只保留 Qwen 的 `language_model`，原生 Qwen visual tower 没有进入 Ours 的训练前向；图像来自冻结 V-JEPA。Slots 是图文共同编码后的状态。

## 如果要直接取 VLM vision encoder

对于本机 Qwen 实现，下面的 API 已实测绕过语言模型：

```python
vision = model.model.get_image_features(
    inputs['pixel_values'], inputs['image_grid_thw']
)
patch_features = vision.last_hidden_state  # merger 前，Qwen0.8B 为 768 维
visual_tokens = vision.pooler_output       # merger 后，按图像拆分的 tuple，1024 维
```

如果目标是保留视觉网格，再接分割／SR decoder，可以优先验证 merger 前的 `patch_features`；merger 后输出也可以作为独立取点比较。哪种效果更好仍需训练验证，本次 shape debug 不能判断。

实现时不能直接把 `patch_features` 传进现有 `align_hidden()`：

- Qwen0.8B 只有 768 个视觉通道，现有代码强制从至少 1024 通道中取样，会抛出异常。应按实际视觉维度配置 head 或增加可训练投影。
- Qwen merger 前的 patches 按 2×2 块排列。以单图 `T,H,W` 网格为例，应先 `reshape(T,H//2,W//2,2,2,D)`，再 `permute(0,5,1,3,2,4)`，恢复 `[T,D,H,W]`；直接 `reshape(H,W,D)` 会打乱空间位置。
- 新取点需要独立的 feature contract、缓存和适配训练。已有 head 是在语言层输出上训练的，shape 一致并不能保证可以直接替换。
- SR 继续只由 LR 图产生视觉输入；当前 dense baseline 的长宽倍率是 ×4，Ours 4＋4 是 ×2。

六个模型的通道差异如下，来自本地 checkpoint 配置和原实验缓存元数据；本次只对 Qwen0.8B 执行了视觉前向。

| 模型 | 视觉主干通道 | merger／projector 后通道 | 原缓存中最后语言层图像 token 数 |
|---|---:|---:|---|
| Qwen3.5-0.8B | 768 | 1024 | HR 256／LR 64 |
| Qwen3.5-4B | 1024 | 2560 | HR 256／LR 64 |
| Qwen3.5-9B | 1152 | 4096 | HR 256／LR 64 |
| Qwen3.5-27B-FP8 | 1152 | 5120 | HR 256／LR 64 |
| MedGemma-1.5-4B | 1152 | 2560 | 256 |
| MedGemma-27B | 1152 | 5376 | 256 |

## 复现与限制

[debug 脚本](../scripts/debug_dense_features.py) 使用归档的 dense extractor／heads 和现有 Ours checkpoint，只保存 shape、有限性和数值比较元数据。

```sh
CUDA_VISIBLE_DEVICES=5 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
TOKENIZERS_PARALLELISM=false /home/data2/chk/workspace/2026/.venv/bin/python \
scripts/debug_dense_features.py
```

运行前选择空闲 GPU。结果保存在本地 [shapes.json](../results/feature_debug_20260912/shapes.json)。

缓存数值复现存在待定位差异：即使使用归档的 `load_vlm()` 和 extractor，本次 Qwen0.8B 重算与缓存也没有逐元素一致。HR/LR 最大绝对差为 0.796875／0.703125，平均绝对差为 0.0816675／0.0715670，展平后的余弦相似度为 0.999699／0.999774。尚未证明差异来源，不能声称精确复现了原缓存。取点结论来自归档源码和实际 hook，shape 检查、直接视觉 API 一致性、Ours 最后 slots 一致性均通过。
