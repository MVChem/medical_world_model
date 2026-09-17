# 与融合模型匹配的 zero-shot baseline

当前统一入口。只评测未经本项目训练的 Qwen3.5-0.8B、MedGemma-1.5-4B、Qwen3.5-4B／9B；
固定一张卡串行推理与临床评分。27B checkpoint 不满足本轮单卡 BF16 约束，不作量化替换。

旧 `medworld_baselines/runs/raw_models_20260911` 确实已完成六模型测试，但 Table 1 使用
EHR＋时间区间，Table 2 从原图重新预处理。本入口使用正在训练的融合版本的数据合同：

- Table 1 为 297 对／94 患者，正向和回溯各 297，分别汇总。仅源图像、源报告和有符号实际时间。
- 源报告内容与 `MedWorld.encode` 一样：包括固定 observation 前缀在内共 384 Qwen tokens。
- Table 2 分类 353 张、报告 507 张，输入直接导出训练读取的同一缓存像素，均不含报告。
- 所有报告贪心生成，最多 384 tokens。训练配置默认生成 64，正式比较必须覆盖成 384。
- Qwen 原生视觉大小与训练相同（256²）；MedGemma 使用自身原生处理器。共同输入像素相同，
  原生视觉计算预算不同；融合模型另有 384² JEPA 分支，不宣称计算量相等。
- 分类概率来自原始 Yes/No 条件似然，真实服务短测检查 token mask 未重归一化概率。
  参考标签、未知值掩码和宏平均支持集合固定。Ours 的 trained sigmoid head 与其分别注明。
- 使用 CheXbert、RadGraph partial F1、官方 GREEN；未来与回溯分开。固定分母，不筛掉失败样本。
- Direction 缺核验真值；官方 VQA、MS-CXR grounding 缺数据。分割、SR 的原生零样本接口为 N/A。
  历史训练 heads、正例派生 QA 和解剖框不填入这些空格。

`prepare` 对照训练目录 `data_protocol.json`，冻结全部样本、图像、权重哈希、配置和源码。
推理程序只读取 inputs，标签／目标报告放在独立 references 中。断点恢复检查源代码和输入哈希。
服务只绑定 localhost，队列只停止自己启动的子进程，遵守机器的 GPU UUID 文件锁。

在项目根目录执行：

```bash
PY=/home/data2/chk/workspace/2026/.venv/bin/python
RUN=code/medworld_zero_shot/runs/unified_20260916
TRAIN=code/medworld/runs/qwen35_08b_2gpu_day_20260916_095750
PYTHONPATH=code "$PY" -m medworld_zero_shot.prepare --run "$RUN" --training-run "$TRAIN"
nohup env MEDWORLD_PROJECT_ROOT="$PWD" PYTHONPATH="$RUN/source" \
  "$PY" -m medworld_zero_shot.run --run "$RUN" --gpu 2 --cpu-cores 16,17,18,19,20,21,22,23 \
  > "$RUN/launcher.log" 2>&1 < /dev/null &
```

恢复执行同一 `run` 命令。结果集中在 `REPORT.md`、`table1_forward.csv`、`table1_backward.csv`、
`table2.csv`，逐样本输出在各模型 `test/`，真实服务短测在独立的 `smoke/`。

本机 vLLM 0.22.1 导入时会访问已经故障的物理 GPU 4，且设备映射不接受 UUID。
`startup/sitecustomize.py` 只在本队列子进程中修正这两处设备发现逻辑，保留模型／kernel 实现和共享环境。
09-16 首次启动在推理前因此退出，原始 `source/` 保留；本次继续使用 **`source_v2/`**，
恢复时将上面 `PYTHONPATH` 改成 `$RUN/source_v2`。实际源码及哈希见 `launcher.json`、
`source_v2/execution_manifest.json` 和 `runtime_compatibility.json`。

训练结束后的对齐评测入口（同一 Stage 2 checkpoint 同时测两表）：

```bash
MEDWORLD_PROJECT_ROOT="$PWD" MEDWORLD_VLLM_COMPAT=1 \
  PYTHONPATH="$RUN/source_v2/medworld_zero_shot/startup:$RUN/source_v2:$RUN/source_v2/medworld_baselines" \
  "$PY" -m medworld_zero_shot.evaluate_ours \
  --run "$RUN" --checkpoint "$TRAIN/stage2.pt" --out "$TRAIN/matched_evaluation_test" --gpu auto
```

该入口采用训练冻结源码，检查测试 fingerprint，并使用相同临床评分器。不会自动挑选测试最佳
checkpoint。`stage2.pt` 路径按训练实际最终保存文件确认。当前训练仍在运行时不读取可变 `last.pt`。

协议检查：

```bash
OMP_NUM_THREADS=2 PYTHONPATH=code "$PY" -m unittest discover -s code/medworld_zero_shot/tests -v
OMP_NUM_THREADS=2 PYTHONPATH=code "$PY" -m medworld_zero_shot.audit --run "$RUN" --training-run "$TRAIN"
```
