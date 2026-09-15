# 在线 8 slots 下游联合适配

这是从本地原始预训练 VLM 开始的新下游适配实验，不加载旧的 V-JEPA 世界模型检查点，也不包含未来预测预训练。用于验证分割和 ×4 超分损失能否通过 8 个 slots 更新 VLM 内的 LoRA。

前 4 个 slot 查询放在原生视觉投影、固定短提示之后，分别读取语言层 1/4、1/2、3/4、最后一层的对应查询位置。后 4 个 slot 查询分别注意力汇聚视觉塔这四个深度的原生图像 tokens。每个输出为 1024 维。语言和视觉取点均按 `ceil(depth * k / 4) - 1` 计算。

分割和超分在本轮均读取全部 8 个 slots，使同一个任务损失能够到达语言与视觉路径。这是新读出设置；旧实验“密集任务仅读后 4 个”的分数不能直接充当本轮对照。图片没有报告、诊断标签或其他患者文字输入。超分的 VLM 和图像分支都只读取 128×128 LR，512×512 HR 只作目标。

| 条件 | VLM 内 LoRA | slots 查询与投影 | 解码器上下文 |
|---|---|---|---|
| `joint_slots` | 训练 | 训练 | 本图像的 8 slots |
| `frozen_slots` | 冻结 | 训练 | 本图像的 8 slots；这是匹配的冻结 VLM 对照 |
| `shuffled_slots` | 训练 | 训练 | 同 split 内另一患者图像的在线 8 slots |
| `joint_full_tokens` | 训练 | 不使用 slot 查询；训练 token 投影 | 最终视觉层全部 tokens 与最终语言层 64 个视觉位置 |
| `image_only` | 不加载 | 不使用 | 零上下文；解码器参数、初始化与上述条件一致 |

LoRA 加在四个语言取点与四个视觉取点的注意力投影中。预训练基础权重保持冻结；“回传到 VLM”在这里具体指更新 VLM 层内的 LoRA，从而改变骨干有效变换，并非全参数微调。Qwen 使用 256×256 处理器预算；MedGemma 保持原生 896×896 处理器。Qwen 拼接的原生图像 tokens 在验证每个图像 grid 相同之后还原 batch，语言和视觉计算均按 batch 执行。

每个正式训练的首个优化步骤都会写 `gradient_audit.json`，检查 8 个输出 slots、8 个可学习查询的梯度范数，并检查语言/视觉 LoRA 的梯度和实际更新范数。冻结 VLM 对照保留查询与投影梯度，两个 LoRA 分支均不更新。`joint_full_tokens` 检查两个骨干 LoRA 分支，另记实际上下文 token 数。在线训练不读取任何特征或 slots 缓存。

训练使用固定患者划分、相同有效 batch 和同一确定性 epoch 顺序；默认保存最终训练预算的检查点。分割报告 CXAS teacher agreement 和独立 Montgomery 人工肺部标签 Dice；超分报告有效图像区域 PSNR/SSIM，协议复用密集任务基线。正式运行不使用 `--eval-limit`。

运行示例：

```bash
CUDA_VISIBLE_DEVICES=4 /home/data2/chk/workspace/2026/.venv/bin/python \
  code/medworld_joint/train.py \
  --data-run code/medworld_dense_baselines/runs/expanded_overnight_20260913 \
  --out code/medworld_joint/runs/overnight_20260913/qwen08b/segmentation_joint_slots \
  --model qwen08b --task segmentation --condition joint_slots \
  --epochs 2 --max-steps 0 --batch-size 32 --microbatch 32 \
  --validate-every 0 --deadline 2026-09-14T07:45:00+08:00
```

`checkpoint.pt` 只保存可训练参数、AdamW 状态、随机数和当前 epoch 的下一 batch 位置。相同命令可续跑；改动模型、数据、代码或预算会触发实验 contract 校验，必须新开输出目录。收到 SIGTERM/SIGINT 或到达带时区的 deadline 时，训练会在优化步骤之间保存并暂停。外部调度器另设 08:00 硬停止。

```bash
/home/data2/chk/workspace/2026/.venv/bin/python -m unittest discover -s code/medworld_joint -p 'test_*.py' -v
/home/data2/chk/workspace/2026/.venv/bin/python code/medworld_joint/report.py --run code/medworld_joint/runs/overnight_20260913
```

原生 Qwen 0.8B、4B 与 MedGemma 4B 已做真实 GPU 前向/反向 smoke；原始日志和检查点在 `runs/smoke_20260913`。27B 的端到端训练没有在单张 24 GB GPU 上得到验证，CLI 会明确拒绝该路径；模型规模对照中的 27B 继续使用独立冻结基线协议。
