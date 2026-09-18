# Native Qwen + predicted future slots

Table 1 的新独立实验：保留 Qwen 原生当前图像、报告／prior EHR 与 horizon 输入，再检验预测未来八槽的作用。历史 `medworld_table1` 结果与检查点不属于此实现。

## 数据与模型

- 固定 cohort：16,000 train／230 validation／297 test pairs；测试 94 位患者；不重新选择或划分患者。
- 当前原图从原始文件读取，512² 保纵横比黑边填充，进入原生 Qwen vision encoder、图像 token 和多模态旋转位置编码。Source report 和 EHR 各自保留 384 tokens 的预算。
- 状态：冻结 JEPA 的 64×768 缓存特征＋当前文本形成四个融合深度 slots；当前原图经 Qwen 原生视觉四个深度形成四个视觉 slots；输出 `8×1024`。
- 预测器只读 `S_t,h`。投影后的八个未来 soft tokens 插入原生 user 消息末尾、assistant prompt 之前。
- Encoder 和 decoder 的 LoRA 独立，固定预训练参数共享存储。两者均适配四个采样语言深度和四个视觉深度。
- Stage 1：当前原生图像报告＋state-only 当前报告＋两路 finding BCE；输入不包含待生成的当前报告。它是分类／报告初始化，完整六任务 Stage 1 尚未实现。
- Stage 2：未来报告 CE＋监督 finding BCE＋逐槽 LayerNorm 后的未来状态 MSE。目标 encoder 是本次新 Stage 1 的固定快照；未来图文只作训练目标。

| Stage 2 条件 | Decoder 原图／文本 | 额外状态 |
|---|---|---|
| `native` | 本患者的当前证据 | 无 |
| `slots` | 相同证据 | 本患者的预测未来状态 |
| `shuffled` | 相同证据 | 同 split 其他患者当前状态，经本样本 horizon 预测 |

三组共享 Stage 1 decoder 与 finding-head 初始化、样本顺序和优化预算。所有概率来自同类型的监督 finding head；旧零样本 Yes/No likelihood 是不同读出。此比较检验额外状态预测分支整体作用；要分离压缩与 dynamics 的作用，还需同 encoder 下的 full-token／current-state 对照。

## 运行

在项目根目录，用 `/home/data2/chk/workspace/2026/.venv/bin/python`。以下输出目录必须是新目录。

```bash
python code/medworld_native_forecast/run.py \
  --config code/medworld_native_forecast/configs/qwen08b.json \
  --condition slots --smoke --out /path/to/new_smoke

python code/medworld_native_forecast/run.py \
  --config code/medworld_native_forecast/configs/qwen08b.json \
  --condition native --smoke --init-checkpoint /path/to/new_smoke/checkpoint_stage1.pt \
  --out /path/to/new_native_smoke
```

`shuffled` 使用相同命令与同一 Stage 1 检查点。先验证 0.8B，再对 `qwen9b.json` 运行。Smoke 每阶段仅一步，不是最终准确率评估，且不能初始化正式训练。

正式预算为 Stage 1 **1,694 updates × effective batch 8**，每个 Stage 2 **2,400 ×32**。物理 batch 1／2 均已验证。正式 9B 使用 batch 2，累积 4／16 次保持有效 batch；最长文本预算峰值分配 22.27 GiB、保留 22.65 GiB。队列先执行共享 Stage 1，再并行三个 Stage 2：

```bash
python code/medworld_native_forecast/launch_queue.py \
  --config code/medworld_native_forecast/configs/qwen9b_batch2.json \
  --out /path/to/new_formal_run --train --gpus 2 3 4 7 --max-parallel 3 --score
```

队列只占用检测为空闲并取得共享 UUID 锁的 GPU，不操作已有任务。`status.json`／`queue_status.json` 记录进度，`metrics.jsonl` 记录损失、梯度、显存和每步耗时。`--score` 会依次生成完整测试预测、运行 CheXbert／RadGraph 与 GREEN；临床评分完成后才能回填表格。

单个训练任务可用 `--resume /same/run/checkpoint_latest.pt` 在完整 optimizer step 边界恢复；重新传入原始 `--stage`、`--condition`、配置和同一输出目录。保存 optimizer、RNG、采样游标与冻结目标参数。Stage 1 转移会重置 Stage 2 optimizer。正式转移和恢复检查输入、预训练权重文件记录及训练源码签名，禁止静默混用协议。

## 验证与限制

### 从 latent 直接解码报告

`codec.decode_reports` 直接接收本 checkpoint 的 `[B,8,1024]` 浮点状态，返回每个样本的报告文本。它复用 Stage 1 的 `state_only_prefix`、slot projection 和 Qwen decoder，不需要重新传入图像或原始报告：

```python
import torch
from medworld_native_forecast.codec import decode_reports

model.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    current_state = model.state(source_batch)  # 当前图像＋允许的当前文本
    predicted_state = model.world(current_state, source_batch["horizon"])
    reports = decode_reports(model, predicted_state, max_new_tokens=384)
```

这里的 `model` 是已加载配套 checkpoint 的 `NativeForecast`；`source_batch` 只含当前允许证据。只重建当前图像报告时，用 `model.state(source_batch, image_only=True)`。解码器接受保存后重载的 CPU tensor，但状态必须来自与解码器配套的 encoder／predictor；相同形状不能证明不同训练、Table 2 checkpoint 或旧 slots 使用同一个 latent 空间。

这是独立的 state-only 读出。正式 Table 1 仍使用 `model.predict(source_batch)`，保留原生当前图文＋预测 future slots 的既定输入协议。报告 CE 已存在于 Stage 1／2；单纯 latent MSE 或张量保存成功不足以证明报告质量。当前接口也不提供纯文本→八槽→原文的无损往返，后四个视觉槽需要图像。

真实模型检查（自动取得空闲 GPU 锁，输出明确标记 diagnostic）：

```bash
python code/medworld_native_forecast/verify_codec.py \
  --checkpoint /path/to/slots/checkpoint_final.pt \
  --out /path/to/new_codec_verification.json --gpus 3 4 7
```

检查当前／预测状态的自由生成、与 Qwen 原生缓存生成的一致性，以及 tensor 保存重载后的报告一致性；同时记录 BF16 下缓存与完整重算的文本是否相同。它不计算报告临床指标。详见[两表 slots 与文本解码核对](https://github.com/MVChem/medical_world_model/blob/31c98ca14173620a3cec67da2defe559d05974da/research_notes/0915_slots_version_and_text_codec.md)。

```bash
python -m pytest -q code/medworld_native_forecast/tests
python code/medworld_native_forecast/verify.py \
  --config code/medworld_native_forecast/configs/qwen9b.json --out /path/to/new_verification.json
```

验证覆盖患者隔离、预测输入不含未来字段、shuffled source 边界、固定 target 参数独立性、M-RoPE、精确恢复、连续训练与断点恢复一致。GPU 检查额外对比原生 Qwen 的隐藏状态及贪心生成，单独验证报告 CE 回传到八槽与预测器，并以最长文本预算进行两次累积及 optimizer 显存压力检查。

冻结 JEPA 特征允许复用；训练中的视觉／语言 LoRA、adapter 和 slots 输出每步重新计算。缓存内容与原图路径在 `provenance.json` 中分别记录。所有运行内容只留在本地，生成文件可能包含研究数据，不能当公共示例发布。

设计与论文同步见 [09-15 协议](https://github.com/MVChem/medical_world_model/blob/31c98ca14173620a3cec67da2defe559d05974da/research_notes/0915_table1_native_forecast_redesign.md)。当前原型不补造 Direction 标签，也不替代尚缺的正式 VQA／grounding 或完整六任务实验。
