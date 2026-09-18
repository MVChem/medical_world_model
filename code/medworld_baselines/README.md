# 原始 VLM 的 Table 1 / Table 2 评测

本轮按 2026-09-11 晚间需求运行本地 Qwen3.5-0.8B / 4B / 9B / 27B-FP8、MedGemma-1.5-4B、MedGemma-27B；不训练、不加载本项目 LoRA 或 Ours checkpoint。

- [实时报告](runs/raw_models_20260911/REPORT.md)
- [本轮协议](https://github.com/MVChem/medical_world_model/blob/31c98ca14173620a3cec67da2defe559d05974da/research_notes/0911_raw_model_baseline_sweep.md)
- [冻结配置与数据指纹](runs/raw_models_20260911/protocol.json)

Table 1 测试 297 对 / 94 位患者；Table 2 分类 353 张、报告 507 张。同次报告不会进入 Table 2 输入。另测 207 个 Chest ImaGenome 派生问答，单独报告。缺少可靠方向标签、官方 VQA 和 MS-CXR 数据的格子保留空值；原生 VLM 的分割、SR 接口记不适用。

`prepare.py` 固定输入和参考，并将二者存入不同文件。`infer.py` 只读取输入，记录原始回答与 Yes/No 条件似然。`score.py` 接入现有 CheXbert / RadGraph，计算参考掩码固定的概率指标。`green_eval.py` 使用官方 GREEN 的固定提示词、tokenizer 和解析器，采用本地 vLLM 批量推理。

`runner.py` 是持久调度进程：MedGemma-27B 使用物理卡 0/5/6/7；完成后 0/5 跑 Qwen27B-FP8、6 依次跑小 Qwen 和 GREEN、7 跑 MedGemma4B 后进行临床评分。已有 9B 本地服务继续处理 9B 请求。原先 GPU 1/2/3 的服务不被停止，历史异常 GPU 4 不使用。

```bash
PY=/home/data2/chk/workspace/2026/.venv/bin/python
$PY -m unittest discover -s code/medworld_baselines/tests -v
$PY code/medworld_baselines/report.py --run code/medworld_baselines/runs/raw_models_20260911
# See the status.json and report files in the relevant runs/ folder.
```

恢复推理使用本轮 `source*/infer.py`、相同 run/model/endpoint；协议、模型、提示词指纹不匹配会拒绝恢复。成功记录按任务/样本/征象去重；失败请求最多自动重试三次，再由调度器补跑失败项。输出解析失败不删除参考样本。每个完整面板才进入汇总指标，避免不同模型只比较各自成功的子集。

运行日志、数据、下载依赖和权重被 Git 排除；汇总报告不含病例原文。论文表格不自动回填。
