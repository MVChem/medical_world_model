# 2026-09-16：Qwen3.5-0.8B 两卡一天训练

**当前运行已于 16:13 改为物理 GPU 6、7，从 Stage 1 重新训练并重新计时 24 小时。**
下方 09:58 的 GPU 5、6 运行记录作为历史保留；新运行信息见末尾“GPU 6、7 重新启动”。

## 训练设置

用户要求两张卡尽量提高训练吞吐，先运行约一天。训练入口仍为 `code/medworld/`。
使用 GPU 5、6（两张 RTX 4090 24 GiB，同一 NUMA 节点），每卡一个 DDP 副本，
同步梯度训练同一个模型；各 rank 从同一打乱数据流读取互不重叠的样本。
CPU affinity 分别为 32–39、40–47；其他 GPU 的已有任务不作修改。

配置：[qwen35_08b_2gpu_day.json](../code/medworld/configs/qwen35_08b_2gpu_day.json)。

- Stage 1 约 6 小时，四项当前任务轮换；Stage 2 使用剩余约 18 小时。
- 时钟从模型、数据、DDP 初始化完成后开始，验证和保存计入预算。计时模式忽略 step 上限。
- Stage 1 在四任务完整轮次边界切换；总计时到期后完成当前 optimizer 更新并保存。
- 每卡分类/分割 batch 96，报告/SR/时间预测 batch 64；梯度累积 1。
- 两卡有效 batch 分别为 192/128/192/128/128。
- BF16 autocast，FP32 可训练参数；Qwen 和 JEPA 基座仍冻结，训练 LoRA、adapter 和任务模块。
- 报告及上下文长度均 384，词表交叉熵分块 128 tokens；图片并行读取 8 workers，预取 2 次更新。
- AdamW：LoRA 学习率 5e-5，其余 1e-4；梯度范数上限 1。
- Stage 2 target 从本次 Stage 1 训练后的 online 编码器初始化，EMA 0.99，每次 optimizer 更新一次。
- Stage 2 每四步加入一项当前任务 replay，四任务依次轮换。
- 每 50 步保存 `last.pt`；每 100 步各任务固定 16 个验证样本监测 loss，并保存各阶段最佳验证权重。
- 小验证集仅用于运行监测，论文完整评测另行执行；不改变训练/验证/测试患者隔离规则。

## 吞吐与显存选择

单卡先测试 batch 8、16、32、64、96，并使用真实的 384-token 数据。
时间预测 batch 64 的 warm peak allocated 约 8.9 GiB，batch 96 约 15.2 GiB；
后者在这次测试没有带来更高吞吐，所以选择 64，并给同时运行的 replay 留显存。
SR 的 cuDNN 首次算法选择临时显存明显更高：batch 64 冷启动约 14.7 GiB，
warm 约 7.2 GiB。显存选择同时考虑正常更新、冷启动和断点恢复。

单卡原始记录位于 `code/medworld/runs/throughput_20260916_2gpu_prep/`。

## 验证

16 项 CPU 测试通过，包括两 rank 动态任务/梯度累积与单进程全局 batch 的参数比较、
EMA、无重复数据分片、预取不提前推进 checkpoint 的消费位置，以及计时边界。
停止测试发现 torchrun worker 各自创建独立 session；launcher 已改为检查 `/proc` 中
的父 PID 后发送信号，并补充独立进程组回归测试，避免把进程组不同误判为非本次 worker。

小 batch 真实两卡运行 `qwen35_08b_ddp_smoke_20260916` 完成 Stage 1 四步、
Stage 2 十六步，覆盖四种 replay；EMA 更新次数为 16，参数两阶矩跨 rank 差异为 0。
该检查用于发现副本分歧，并非逐字节参数比较。训练、验证、checkpoint 保存和阶段切换均通过。

正式 batch 的两卡检查 `qwen35_08b_ddp_large_20260916` 完成 Stage 1 四步和
Stage 2 十六步。在 Stage 2 第 11 步保存退出，再从同一 `last.pt` 恢复完成第 12–16 步，
四种 replay 均覆盖。最终 EMA 更新次数 16、时间流消费 2,048 个有方向样本、replay index 4，
参数两阶矩跨 rank 差异为 0。最大 allocated 为 17.90 GiB，最大 reserved 为 19.81 GiB，
发生在恢复后的时间预测＋SR replay；进程返回码 0。

## 运行与恢复

`launch_distributed.py` 预留两卡、保存源代码快照和 SHA256 清单，并启动 torchrun。
`distributed_train.py` 保存每个 rank 的 RNG、全局数据消费位置、优化器及 EMA。
恢复必须使用同一运行目录和两个 rank，沿用原结束时间；运行中编辑开发目录不会影响快照。

- `status.json`：阶段、步数、心跳、开始与截止 Unix 时间。
- `metrics.jsonl`：loss、有效 batch、每 rank 数据等待时间及峰值 allocated/reserved 显存。
- `gpu_telemetry.jsonl`：每十秒的显存、GPU 利用率和功耗。
- `last.pt` / `stage1.pt` / `stage2.pt`：恢复点及阶段结束权重。
- `best_stage1.pt` / `best_stage2.pt`：小验证集 loss 最优的各阶段权重。
- `launch.json`：launcher/torchrun PID、GPU UUID 和启动参数。
- `launcher_status.json`：训练进程结束后的返回码。

向本次 `launcher_pid` 发送 SIGTERM 会请求在更新边界保存并停止。
24 小时时间预算以 worker 计时为主，launcher 另有到期宽限后的退出看护。

## 已启动的正式运行

- 目录：`code/medworld/runs/qwen35_08b_2gpu_day_20260916_095750/`。
- 后台 launcher PID `768706`，torchrun PID `768802`；launcher 已脱离启动 shell。
- Launcher 日志：`/tmp/qwen35_08b_2gpu_day_20260916_095750_launcher.log`。
- 模型准备完毕、计时开始：2026-09-16 **09:58:12**（Asia/Shanghai）。
- 计划 Stage 1 → Stage 2：2026-09-16 **15:58 左右**。
- 总预算截止：2026-09-17 **09:58 左右**，完成当前更新后保存退出。
- 使用新初始化的模型开始正式训练，没有接着短测权重训练。
- `code/medworld/runs/active_2gpu_day.json` 记录当前这次后台运行的位置和启动信息。

以上是启动记录，尚不是一天训练完成的结果。后续进度以运行目录里的心跳、日志和 checkpoint 为准。

### 启动后检查（10:02）

正式运行已到 Stage 1 第 59 步，首个 `last.pt` 保存于第 50 步，大小 212,305,131 bytes。
CPU 重载确认：可训练参数均有限、两 rank 的 RNG 齐全、world size 为 2、原 24 小时截止时间保留。
审计结果保存在运行目录的 `first_checkpoint_audit.json`。
前 46 步中，预热后平均更新约 4.04 秒，平均数据等待约 0.04 秒；
Stage 1 allocated 峰值 14.73 GiB、reserved 峰值 16.53 GiB。
GPU 计算时采样可到 99–100%，任务切换、同步及保存时利用率会降低；这不是全程平均值。

## GPU 6、7 重新启动（16:13）

用户反馈此前掉卡，要求改用物理卡 6、7 重新运行。检查时旧 launcher 和 workers 均已退出，
旧 `status.json` 停留在 Stage 1 第 1401 步；日志记录 12:08 收到 SIGTERM，
`last.pt` 和其他旧权重、日志均原样保留。仅凭此训练日志不判断掉卡的硬件原因。

- 新目录：`code/medworld/runs/qwen35_08b_2gpu_day_gpu67_20260916_161228/`。
- 物理 GPU 6：`GPU-119eacc5-5d83-69af-02e7-8ce6ce532b78`。
- 物理 GPU 7：`GPU-a1e1bb80-f4e0-7a4b-a80d-b9be8610bf40`。
- Launcher PID `467513`，torchrun PID `467658`；后台进程已脱离启动 shell。
- Launcher 日志：`/tmp/qwen35_08b_2gpu_day_gpu67_20260916_161228_launcher.log`。
- 新训练计时开始：2026-09-16 **16:13:16**（Asia/Shanghai）。
- 预计 Stage 1 → Stage 2：2026-09-16 **22:13 左右**。
- 总预算截止：2026-09-17 **16:13 左右**，完成当前更新后保存退出。
- 从预训练基座及新初始化训练参数开始，未读取旧训练 checkpoint；配置和 batch 保持一致。
- `active_2gpu_day.json` 已指向新运行，同时记录 `previous_run` 便于查找旧结果。

启动后通过 `nvidia-smi` 核对两 worker 分别落在上述 GPU UUID 上。
