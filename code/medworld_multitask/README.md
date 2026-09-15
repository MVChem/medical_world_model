# Table 2 四任务共享状态原型

先验证 Qwen3.5-0.8B 的执行与匹配对照，再验证 9B。当前提供分类、当前报告生成、分割和空间 ×4 SR；正式 VQA 与 MS-CXR grounding 数据和评测接口仍缺失。此目录的短测和训练 loss 不能回填成论文成绩。

## 状态与对照

同一个 checkpoint 轮转学习四项任务。四个 fusion slots 从冻结 V-JEPA 2.1 图像特征、可训练 adapter 与四个 Qwen 语言层读出；四个 visual slots 从 Qwen 原生视觉塔的四个深度读出。分类和报告读取全部八个；分割／SR 仅读视觉四个。报告是训练目标，不进入状态构建；SR 的两个输入分支均来自同一 LR 图像。

- `slots`：训练 4＋4 查询、投影、LoRA 和任务头。
- `full_tokens`：相同 JEPA／原生视觉输入与任务头，保留全部融合 tokens 和最终视觉 tokens；不使用 slots 查询。它是结构对照，不是原生 Qwen 的直接图文微调。
- `shuffled`：保持当前图像分支，将状态来源替换成同任务、同 split、其他患者的图像。
- `image_only`：仅分割／SR，使用相同初始化的解码器与零上下文。

报告 decoder 与状态编码器共享同一 Qwen 语言基座及 LoRA；9B 使用其独立预训练输出头，避免复制整个语言模型。JEPA 始终冻结。当前是 Stage 1 下游适配原型，尚未接入这套 4＋4 状态的未来预测训练，不能宣称验证了完整世界模型。

## 数据

| 任务 | Train | Validation | Test | 外部人工测试 |
| --- | ---: | ---: | ---: | ---: |
| 分类 | 13,681 | 160 | 353 | — |
| 报告 | 22,646 | 307 | 507 | — |
| 分割 | 4,096 | 249 | 447 | 138 |
| ×4 SR | 4,096 | 249 | 447 | — |

`data.py` 复用原始只读像素与标注数组，验证所有任务之间跨 split 患者交集为零。报告只返回在 `report_targets`，SR 从 LR 数组建立 PIL 和 tensor 输入。分类未知／不确定标签使用 mask，分割的伪标签和人工肺部测试分别处理。

## 执行

先选空闲 GPU，在项目根目录运行：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/data2/chk/workspace/2026/.venv/bin/python code/medworld_multitask/run.py \
  --smoke --model qwen08b --condition slots --out NEW_SMOKE_DIRECTORY
```

短测使用真实训练样本，每项任务做一次优化，并验证预测、检查点重载以及任务到各分支的梯度。结果明确标记 `smoke_only`、`table2_ready=false`。输出包含 decoder 初始哈希，用于匹配对照；不同患者 donor 不依赖 batch 大小。

`--steps N` 表示全部任务合计的优化步数，每步轮到一项任务；这不等于每项任务 N 步或 20 epochs。`--batch-size` 是物理 batch，`--accumulation` 指梯度累积，二者乘积为有效 batch。正式主表实验还需要固定各任务完整预算并接入最终临床与图像指标评估。不要用任意步数的原型去替代主表既定的 20-epoch 实验。

使用 `--resume PATH/checkpoint_latest.pt` 可恢复同一实验。checkpoint 保存可训练参数、优化器、随机数与每个任务的采样位置；冻结预训练模型按来源引用。新实验使用新目录，短测检查点不能直接转换为正式训练。

## 两类 baseline 的作用

同解码器的 image-only／slots／shuffled 控制 slots 的增益；full_tokens 控制状态压缩。SwinIR 是任务专用的性能参照，不能单独归因 slots 的作用。如需考察 slots 是否也能帮助 SwinIR，需要另做同设置的 SwinIR／SwinIR＋slots 配对实验。

Table 1 已有 JEPA-path full-token forecaster 同样属于结构对照；原生 Qwen 的直接图文微调是另一条尚需新增的 baseline，不能改名后沿用前者的分数。

## Slots 大小与提取耗时

新原型对两个 Qwen 规模均输出 `[B,8,1024]`，当前 FP32 为每图 32 KiB；空间任务只计算 `[B,4,1024]`，为 16 KiB。若保存为 FP16／BF16，体积减半。这个数值只包含状态，不包含模型参数或训练激活。

2026-09-15 在 RTX 4090 上，以 resident model、batch 1、3 张预热、12 张真实图像测得：0.8B 全部 8 slots 平均约 161 ms／图，9B 约 204 ms／图；仅读视觉四个，0.8B 约 12 ms，9B 约 21–22 ms。计时包括图像预处理和编码前向，不包括数据读取、任务解码、反向传播或首次加载。两个规模均使用相同 384px JEPA／256px Qwen 视觉预算。

可用 `benchmark_slots.py --model qwen08b --out NEW_RESULT.json` 重测；选择空闲 `CUDA_VISIBLE_DEVICES`，脚本自动使用共享 GPU 锁。原始记录保存在 `runs/latency_20260915/`。这些是新读出初始化下的延迟测量，不是质量指标，也不代表 Table 1 旧 8×4096 forecaster 的耗时。
