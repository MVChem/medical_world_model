# MedWorld：Table 1／Table 2 融合实现

当前目录是融合版本的开发入口。Qwen3.5-0.8B、4＋4 状态、两阶段训练、EMA、
双向时间预测和状态独立文本解码已接通。正式训练与临床报告质量评测尚未完成。

融合前实现保存在 Git 标签 `pre-table1-table2-unification-20260915`：
Table 1 位于 `../medworld_native_forecast/`，Table 2 位于 `../medworld_multitask/`。
本目录沿用它们的数据文件，模型与训练流程统一在这里实现。

## 核对实现

| 内容 | 代码入口 |
|---|---|
| 统一模型、当前任务 loss、时间预测 loss | [model.py](model.py) |
| 观察编码与 4＋4 slots | [encoder.py](encoder.py)、[adaptation.py](adaptation.py) |
| EMA 初始化、更新与恢复 | [ema.py](ema.py) |
| 带正负时间条件的 World Model | [predictor.py](predictor.py) |
| slots → 报告、分类、分割、×4 SR | [decoders.py](decoders.py) |
| 当前任务数据、双向配对、跨表患者隔离 | [datasets/](datasets/) |
| 两阶段训练、replay、断点恢复 | [train.py](train.py)、[runtime.py](runtime.py) |
| 两卡 DDP、计时训练、数据预取 | [distributed_train.py](distributed_train.py) |
| GPU 预留、代码快照、后台监控 | [launch_distributed.py](launch_distributed.py) |
| 同一 checkpoint 的两类评测 | [evaluate.py](evaluate.py) |
| 图文输入预测／保存状态／独立解码 | [infer.py](infer.py) |
| 可修改的原型配置 | [configs/qwen35_08b.json](configs/qwen35_08b.json) |

### 一个观察编码器，八个状态槽

状态形状为 `[B,8,1024]`，时间方向只送入 World Model。

- **前四槽**：冻结 V-JEPA2.1 ViT-B 的 64 个图像 token，经 adapter 后与观察文本、
  四个可学习查询拼接，进入 Qwen 语言模型；在第 6／12／18／24 层分别读出对应查询。
- **后四槽**：Qwen 原生视觉塔第 3／6／9／12 层的图像 token，经投影及可学习查询
  做注意力汇聚；这四槽不读取报告。层号从 1 开始，查询没有固定疾病／器官分配。
- Qwen 基座冻结；选定深度的语言／视觉 LoRA、JEPA adapter、查询及读出投影可训练。
  JEPA 始终冻结，特征在线从图片计算。
- 分类和报告读取八槽。分割／SR 只运行视觉分支并读取后四槽，同时保留图像网格输入。
  SR 的输入统一为 128² LR，512² HR 只作为监督。

### Online 梯度更新，target EMA 更新

Stage 1 训练当前分类、报告、分割、SR，按 optimizer step 轮流选择任务。
当前报告任务的编码器只看图像，报告只作监督。

Stage 2 从这个模型初始化 target。冻结的预训练基座在 online、target 之间共享存储；
所有可训练编码参数在 target 中各有独立副本，并关闭梯度：

```text
target ← m × target + (1 − m) × online
```

每完成一次 Stage 2 `optimizer.step()`，更新一次 EMA；不在梯度累积的 microbatch
之间更新。默认固定 `m=0.99`，可在配置修改。target 始终处于 eval 模式。
文本 decoder 有独立 LoRA，与 online 语言 encoder 共享冻结基座；decoder 不参与 EMA。

Stage 2 默认每四次更新加入一次当前任务 replay，四项任务轮流参与。
EMA、当前任务 heads 和 decoder 全部保存在同一 Stage 2 checkpoint 中。

### World Model 输出直接解码文本

```text
source_state = online(source_image, source_report)
target_state = stop_gradient(EMA(target_image, target_report))
predicted_state = WorldModel(source_state, signed_delta_hours)

loss = latent_weight × MSE(LN(predicted_state), LN(target_state))
     + report_weight × CE(text_decoder(predicted_state), target_report)
     + finding_weight × BCE(classifier(predicted_state), target_findings)
```

默认权重为 `1 / 1 / 0.5`。报告 CE 必须有正权重，`predicted_state` 不 detach，
梯度可进入文本 decoder、World Model 和 online encoder。

报告 decoder 的输入只有八个投影后的 soft tokens 和固定任务提示。
生成时不需要原图、原报告或目标报告。训练时目标文字只用于自回归 teacher forcing。
保存的状态必须搭配同一 checkpoint 的 decoder；CLI 用 checkpoint SHA256 校验，
不能仅凭 `[8,1024]` 的形状混用其他实验的状态。

### 正向与回溯使用实际时间

正向使用原配对的 `realized_gap_hours`；回溯交换两个观察，取负间隔。
例如 `+72` 小时预测三天后，`-72` 小时回溯三天前。
时间特征包含方向和连续的间隔幅度，没有沿用旧代码的离散 horizon 桶。
初始化时残差输出为零，正负方向的差别需要从双向监督中学习。

初版时间任务输入是**图像＋该观察的报告**，暂不包含 EHR；纯报告输入尚未实现。
前后观察使用相同的编码提示。`source_only=True` 的推理接口不会读取另一端的图像、
报告或标签；评测在预测完成后另行获取参考答案。

## 数据协议改变

合并两表后发现：Table 2 测试患者中的 53 位出现在旧 Table 1 训练池，另 1 位
出现在验证池。融合版优先保留测试集，剔除冲突的训练／验证行，不把行搬入测试集。
运行目录的 `data_protocol.json` 记录筛选与源文件哈希。

| 时间配对 | 原始对数 | 融合版保留对数 |
|---|---:|---:|
| train | 16,000 | 15,877 |
| validate | 230 | 227 |
| test | 297 | 297 |

默认启用双向，各池的有方向样本数是保留对数的两倍。当前四任务的样本数均保留。
这是新的训练协议，旧版分数不能直接作为融合版结果。当前数组按只读方式加载，
检查 shape/dtype；大数组的声明哈希来自已有 manifest，未逐次重新计算。

## 运行

在项目根目录执行。当前环境为 Python 3.12、torch 2.11.0、transformers 5.12.1，
还需要 numpy、Pillow、timm、scikit-learn、scikit-image，以及本地 Qwen／JEPA 权重和
`code/vjepa2` 第三方源码。程序从本地加载权重，不自动下载。

```bash
export MEDWORLD_PYTHON=/home/data2/chk/workspace/2026/.venv/bin/python

# 4 次当前任务更新 + 4 次时间预测更新，以及梯度／状态保存恢复检查
PYTHONPATH=code "$MEDWORLD_PYTHON" -m medworld.train \
  --smoke --gpu auto --out code/medworld/runs/smoke_01

# 默认预算是原型设置，尚不是正式论文协议
PYTHONPATH=code "$MEDWORLD_PYTHON" -m medworld.train \
  --config code/medworld/configs/qwen35_08b.json \
  --stage both --gpu auto --out code/medworld/runs/joint_01

# 恢复 optimizer、RNG、EMA、采样位置；使用原配置和同一 run
PYTHONPATH=code "$MEDWORLD_PYTHON" -m medworld.train \
  --resume code/medworld/runs/joint_01/last.pt \
  --out code/medworld/runs/joint_01 --gpu auto

# 也可先 --stage stage1，再转入新目录
PYTHONPATH=code "$MEDWORLD_PYTHON" -m medworld.train --stage stage2 \
  --init-checkpoint code/medworld/runs/joint_01/stage1.pt \
  --out code/medworld/runs/stage2_01 --gpu auto

# 同一个 checkpoint 评测当前四任务和正／负时间预测；加 --limit 2 可检查入口
PYTHONPATH=code "$MEDWORLD_PYTHON" -m medworld.evaluate \
  --checkpoint code/medworld/runs/joint_01/stage2.pt \
  --task all --split test --out code/medworld/runs/joint_01/evaluation_test --gpu auto

# 回溯三天，保存可独立解码的状态
PYTHONPATH=code "$MEDWORLD_PYTHON" -m medworld.infer \
  --checkpoint code/medworld/runs/joint_01/stage2.pt \
  --image /path/to/image.jpg --report-file /path/to/report.txt --delta-hours -72 \
  --save-state code/medworld/runs/joint_01/past_state.pt \
  --out code/medworld/runs/joint_01/past_report.json --gpu auto

# 不传图像／报告，仅从保存的 slots 解码
PYTHONPATH=code "$MEDWORLD_PYTHON" -m medworld.infer \
  --checkpoint code/medworld/runs/joint_01/stage2.pt \
  --state code/medworld/runs/joint_01/past_state.pt \
  --out code/medworld/runs/joint_01/decoded_report.json --gpu auto

OMP_NUM_THREADS=2 PYTHONPATH=code "$MEDWORLD_PYTHON" -m unittest discover -s code/medworld/tests -v
```

`--gpu auto` 使用本机既有的 UUID 文件锁选择空闲卡；也可指定编号。`--gpu cpu`
用于 CPU 验证，可搭配 `OMP_NUM_THREADS=8`。新运行目录须为空。
Ctrl-C／SIGTERM 会在当前 optimizer 更新边界保存。checkpoint 只保存可训练权重、
EMA、优化器和恢复信息；冻结参数通过原路径和内容哈希校验后加载。
恢复时须保持代码版本一致；程序检查配置、数据和预训练权重，不自动迁移旧目录 checkpoint。

评测提供分类 AUROC/AP、分割 Dice、×4 SR 的 PSNR/SSIM，以及报告与参考答案 JSONL。
时间任务分别汇总 forward/backward。分割默认测试为 CXAS 伪标签；人工两肺使用
`--task segmentation --split human_test`。CheXbert／RadGraph／GREEN 等临床指标尚待接入，
空报告和重复率只是生成诊断。

### 两卡训练一天

[两卡一天配置](configs/qwen35_08b_2gpu_day.json) 使用两个 DDP 副本同步训练同一个模型。
两卡读取同一打乱数据流中不同的样本；每次更新后梯度同步，Stage 2 各自更新相同的 EMA。
配置启用 BF16 autocast、并行图片读取和两步数据预取。Qwen 基座仍冻结，训练 LoRA 和任务模块。

| 任务 | 每卡 batch | 两卡有效 batch |
|---|---:|---:|
| 分类 | 96 | 192 |
| 报告 | 64 | 128 |
| 分割 | 96 | 192 |
| SR | 64 | 128 |
| 时间预测 | 64 | 128 |

梯度累积为 1。Stage 2 每四步额外加入一个当前任务 batch，按分类、报告、分割、SR 轮换。
`total_hours=24`、`stage1_hours=6`：模型准备好后开始计时，Stage 1 约六小时后在完整四任务
轮次边界切换，Stage 2 使用剩余时间。计时模式忽略 `stage1_steps` / `stage2_steps`，
验证和保存时间计入总预算；到期完成当前更新并保存，因此结束时间可能略晚。
每 50 步保存 `last.pt`，每 100 步在固定的各任务 16 个验证样本上计算 loss。
这些小样本验证用于运行监测，完整评测需另外执行；各阶段的最佳验证 checkpoint 单独保留。

```bash
# 示例使用物理 GPU 6、7 及其 NUMA 节点的 CPU 核。
# nohup 让关闭终端后训练继续；新运行目录须为空。
nohup env PYTHONPATH=code "$MEDWORLD_PYTHON" -m medworld.launch_distributed \
  --gpus 6,7 --cpu-base 32 \
  --config code/medworld/configs/qwen35_08b_2gpu_day.json \
  --out code/medworld/runs/qwen35_08b_day_01 \
  > /tmp/medworld_day_01_launcher.log 2>&1 < /dev/null &

# 恢复同一运行：读取原配置和 source/ 下的代码快照。
PYTHONPATH=code "$MEDWORLD_PYTHON" -m medworld.launch_distributed \
  --gpus 6,7 --cpu-base 32 \
  --resume code/medworld/runs/qwen35_08b_day_01/last.pt \
  --out code/medworld/runs/qwen35_08b_day_01
```

恢复保留原结束时间、各 rank 的 RNG、优化器、EMA 和已消费数据位置，要求仍使用两卡。
`status.json` 给出当前阶段、步数、开始/截止时间和心跳；`metrics.jsonl` 记录 loss、
每卡数据等待时间和峰值显存；`gpu_telemetry.jsonl` 记录显存、利用率和功耗。
向 `launch.json` 中的 `launcher_pid` 发送 SIGTERM，会在更新边界保存后退出。
运行目录包含代码快照及 SHA256 清单；后续编辑开发目录不会改变已启动的训练。

## 验证和范围

验收检查见 [tests/](tests/) 和 [smoke.py](smoke.py)，真实模型测试记录见
[融合实现验证记录](../../research_notes/0915_unified_medworld_implementation.md)。
2026-09-16 已补齐 [RTX 4090 GPU 短测](../../research_notes/0916_unified_medworld_gpu_smoke.md)：
两阶段、梯度／EMA、状态独立解码和 checkpoint 恢复均通过；batch 1、报告 64 tokens
时训练 allocated 峰值约 2.36 GiB，PyTorch reserved 峰值约 2.53 GiB。
训练步骤显存记录在 `metrics.jsonl`，重载审计的显存另记在 `smoke_audit.json`。
短测验证计算图、训练流程与保存恢复，不证明报告可靠或时间预测准确。
同日已启动 [两卡一天训练](../../research_notes/0916_two_gpu_day_run.md)：当前使用 GPU 6、7，
计划 Stage 1 六小时＋Stage 2 剩余十八小时；正式大 batch、四种 replay 和中断恢复已短测通过。
官方 VQA、grounding、纯报告输入和完整临床评测仍待实现；当前不宣称完成六任务实验。
