# 09-13 DINOv2 / CheXWorld + heads：可执行对照

本轮使用公开预训练、冻结的视觉编码器，训练分类、分割和 ×4 SR 三个独立监督头。输出是最后层空间 patch tokens，不使用学习的 slots。代码是 `code/medworld_open_baselines/dinov2_{features,train}.py`；两个入口都接受 `--model dinov2_vitb14` 或 `--model chexworld`。

## 编码器来源及表示

- [DINOv2 官方仓库](https://github.com/facebookresearch/dinov2)，commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`；[官方 ViT-B/14 checkpoint](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth)，SHA256 `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`，86,580,480 参数。严格加载全部参数。518×518 输入，提取最终 `x_norm_patchtokens`，去除 class token，37×37×768 空间图通过无参数 adaptive average pooling 转为 8×8×768。
- [CheXWorld 官方仓库](https://github.com/LeapLabTHU/CheXWorld)，commit `090102758801dc097f53c49d135b835570c8d173`；使用官方发布 checkpoint 的 `target_encoder`，由 `chexworld_encoder.py` 严格加载，85,797,120 参数。沿用官方 resize-short256 / center-crop224 / grayscale3 / ImageNet normalization；14×14×768 最后层 patch 图同样无参数池化到 8×8。模型权重指纹写入 feature contract。其仓库没有顶层 LICENSE，不推断其为 MIT 等其他许可证。

两个编码器的缓存都是 `[N,64,768] float16`；交给任务头前右补 256 个零到 1,024 维。不将 64 个空间位置汇聚成四个深度 slots。两个编码器都冻结，CUDA 前向使用 bfloat16 autocast；每个样本独立提取并保存。

## 与 09-13 下午实验的可比性

密集任务直接使用已固定的 `dense_20260912/data`：4,096 train / 249 validation / 447 test，外加 138 Montgomery 人工肺分割测试图。数据、伪标签、human masks 和输入缓存都记录 SHA256。SR 编码器仅访问现成的 128×128 uint8 LR；HR 只作为重建目标。DINOv2 将 LR 放大到 518、CheXWorld 将 LR 按官方 transform 放大，并不能恢复被移除的 HR 信息。

`SpatialHead` 继承下午的 `FrozenSlotHead`，保留全部可训练模块，只把四个固定 depth positions 改为 8×8 固定空间坐标并允许 64 个 tokens。分割 221,715、SR 222,849 个可训练参数，与下午相同；在相同 seed 下，所有可训练参数初始化逐元素相同。跨注意力的 token 数从 4 增为 64，因此这不是严格 FLOPs 相等实验；编码器提取成本也分别记录。它是同数据、同训练步数、同监督头容量的 frozen-encoder 对照。

每个密集任务固定 20 epochs，effective batch 8，seed 20260913，AdamW / LR 3e-4 / weight decay .01 / epoch cosine 到 10% floor。4096 条样本每 epoch 512 次更新，共 10,240 次更新。分割 BCE+soft Dice，SR valid-pixel MSE；复用下午原 loss 与 scorer。只报告最后 epoch，不按测试结果选 checkpoint。原 image-only 控制可直接作为参照，不重复训练。扩大训练集时入口支持 `--data-run`，必要时可显式指定 `--pseudo-path`。

分类沿用 stage1 的实际 `Classification(1024)`、masked BCE 和 classification scorer，201,345 个参数；输入只有 64 个空间 tokens，不提供报告。使用原 stage1 cache 和同一 `selection.json` 患者 split overrides，获得 13,681 train / 160 validation / 353 test。验证和测试 ID 集合已与 raw-model C0 输入文件逐项核对一致。监督和评分都是官方 13 项 CheXpert labels；blank / uncertain 掩码，AP/AUC 仅对两种参考类别都有支持的疾病取宏平均。训练独立 20 epochs、effective batch 8，共 34,220 次更新；masked BCE 每次直接计算完整有效 batch，避免 microbatch 改变类权重。这与历史多任务训练的更新预算不同，需作为单独监督头协议报告。

此实现是公开冻结编码器加本项目统一任务头，不声称重新预训练了 DINOv2/CheXWorld，也不声称复现其论文专有分割或 SR 头。公共预训练数据与当前患者的潜在重叠未知。

## 启动与产物

首轮 run 为 `code/medworld_open_baselines/runs/comparators_20260913/dense_4096`。任务协调器负责训练 GPU；特征提取使用已有 `/tmp/medworld-frozen-slots-GPU-<uuid>.lock`，确认无计算进程、显存低于 512 MiB、GPU utilization 低于 5% 后启动。

```bash
# 对每个 MODEL=dinov2_vitb14 或 chexworld：
/home/data2/chk/workspace/2026/.venv/bin/python code/medworld_open_baselines/dinov2_features.py \
  --run code/medworld_open_baselines/runs/comparators_20260913/dense_4096 \
  --model dinov2_vitb14 --task dense --batch-size 8

/home/data2/chk/workspace/2026/.venv/bin/python code/medworld_open_baselines/dinov2_features.py \
  --run code/medworld_open_baselines/runs/comparators_20260913/dense_4096 \
  --model dinov2_vitb14 --task classification --batch-size 8

# --task 可为 segmentation、sr、classification：
/home/data2/chk/workspace/2026/.venv/bin/python code/medworld_open_baselines/dinov2_train.py \
  --run code/medworld_open_baselines/runs/comparators_20260913/dense_4096 \
  --model dinov2_vitb14 --task segmentation --epochs 20 --batch-size 8 --microbatch 4
```

每个模型目录中的 `dense_features_complete.json`、`classification_features_complete.json` 是训练的提取完成依赖。对应缓存及 contract 都有指纹。训练子目录为 `{segmentation,sr,classification}_spatial/`，写入 `contract.json`、`checkpoint.pt`（含 optimizer/RNG/epoch内游标）、`epochs.jsonl`、`progress.json` 和最终 `metrics.json`、逐样本预测文件。中断保留 checkpoint，未训练到 20 epochs 不生成冒充最终结果的指标。

GPU smoke 记录 `dinov2_assets/smoke/smoke.json`：两个编码器真实 HR/LR 前向、各两个 dense 任务单步反传、模型保存恢复逐元素相同、分类 masked loss 反传、SR 访问 HR 保护均通过。实际 trainer 在小型 CPU fixture 上的一步中断恢复，与不中断训练最终权重逐元素相同，记录在 `dinov2_assets/trainer_contract_smoke/verified.json`。这些是实现检查，不是效果分数。
