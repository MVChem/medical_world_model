# 09-15：回填 9B 预测对照与 CheXagent 结果

核对正式运行的完成状态、测试数量和最终指标后，本次新增 **32 个指标值**：Table 1 新增 28 个，现为 11 行、70 个已填指标；Table 2 新增 4 个，现为 13 行、42 个已填指标。已填写行沿用既定协议和数值展示规则：分数乘以 100、保留两位小数，PSNR 保留 dB。

- [两页表格预览](../27cvpr/plans/table1_table2_plan.pdf)
- [Table 1 共享源码](../27cvpr/tables/table1_future.tex) · [Table 2 共享源码](../27cvpr/tables/table2_downstream.tex)
- [全精度数值与来源 SHA256](../27cvpr/tables/results_20260914.json)：保留原文件名以兼容已有引用，内部 `snapshot_date` 更新为 `2026-09-15`。
- [重建脚本](../27cvpr/plans/populate_results.py)

## Table 1：新增四行结果

下表使用与论文一致的 ×100 展示尺度。Direction 的 297 对 cohort 标签仍待复核，该列继续为 `TBD`。

| 条件 | AP | AUROC | Transition F1 | RadGraph F1 | GREEN | Brier ↓ | ECE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-token forecaster (Qwen3.5-9B) | 83.09 | 76.42 | 16.03 | 17.11 | 17.83 | 11.64 | 9.42 |
| MedWorld-JEPA (Qwen3.5-9B slots) | 82.70 | 77.94 | 28.51 | 15.96 | 17.61 | 11.47 | 8.93 |
| Qwen3.5-9B (shuffled state) | 70.93 | 52.85 | 6.04 | 10.78 | 8.87 | 13.33 | 4.97 |
| CheXagent-8B ZS | 86.89 | 81.48 | 0.00 | 13.43 | 15.51 | 13.00 | 13.22 |

9B 来源为 `code/medworld_table1/runs/qwen9b_ablation_20260914/`。队列 8/8 作业成功，于 **2026-09-15 04:55（北京时间）**完成；共同 Stage 1 为 1,694 步，三个 Stage 2 均完成 2,400 步、effective batch 32。每组最终 checkpoint 生成 297 份预测，测试为 297 对／94 位患者；GREEN 各为 `status=complete`、`n=completed=297`、无无效输出。临床指标来自 `{condition}/evaluation_test/metrics.json`，GREEN 来自 `green/{condition}/test/green_metrics.json`。

这三个条件使用共同的新 9B Stage-1 初始化。slots 为八个最后语言层查询，full-token 保留同一预测器；它们与完整多深度 fusion/vision 4＋4 架构分开。历史 0.8B 分数保留在[升级记录](0914_qwen9b_forecast_upgrade.md)，不用于本次回填。

本次点估计中，slots 的 AUROC、Transition F1 和 Brier/ECE 优于 full-token，AP、RadGraph、GREEN 略低。打乱状态后 AUROC 接近随机，297 例只生成一种报告；其较低 ECE 不能单独证明预测更有用。这些结果尚无配对置信区间。

## CheXagent：两表结果与完成核验

来源为 `code/medworld_open_baselines/runs/comparators_20260913/zero_shot/chexagent8b/chexagent8b/test/`。恢复队列 `scheduler_chexagent_recovery0914/` 的 5/5 作业于 **2026-09-14 12:01（北京时间）**完成。正式推理回执为 complete，非 smoke；未来报告 297 份、当前分类 353 张、当前报告 507 份，最终临床指标齐全，GREEN 为 297/297。

Table 2 新填：分类 **AUROC 79.08、AP 87.92**；报告 **RadGraph F1 16.93、CheXbert F1 30.60**。派生疾病列表 QA 不替代正式 VQA，正式 VQA 和 grounding 继续留空。生成器已将 CheXagent 加入真实零样本结果映射，移除旧的下载等待占位。

## Table 2 训练规模与剩余项

主表分割／超分继续使用 **4,096/249/447 张训练／验证／测试图、20 epochs**；SwinIR 的既有 4,096 张版本 PSNR 37.27、SSIM 94.51 保持原值。今天 14:22 完成的 18,708 张 SwinIR 版本属于扩大数据实验，原始结果留在 `code/medworld_open_baselines/runs/comparators_20260913/swinir_18708/metrics.json`，不混入此主表。

真实缺项包括：Table 1 Direction、prior 的 Transition；Table 2 正式 VQA、MS-CXR grounding、MAIRA-2，以及完整六任务模型和 matched no-slots 两行。这次回填不将两任务 joint pilot 或冻结视觉摘要当作完整模型结果。

## 重建与核验

在项目根目录运行：

```bash
/home/data2/chk/workspace/2026/.venv/bin/python 27cvpr/plans/populate_results.py
make -C 27cvpr
make -C 27cvpr plan-preview
```

生成器核验 9B 配置、最终训练步数、checkpoint 路径、测试数量，以及完整 GREEN 状态；聚合 JSON 保存原始尺度与来源哈希。论文正文、两页预览和 PNG 使用同一份表格源码。

独立逐格核验通过：相对本次回填前恰有 32 个 `TBD` 变为数值，已有数值与行名未变；两份 TeX 与 JSON 的缩放、舍入一致，43 个来源 SHA256 均匹配原文件。所有 Table 2 分割／超分来源的训练规模仍为 4,096 张。

主论文（10 页）、补充材料（5 页）、独立表格预览（2 页）均编译成功；两张 PNG 已刷新并检查排版。最终编译日志没有未解析引用或 overfull/underfull box，主论文第 6 页为两张跨栏结果表，保留 LaTeX 的仅浮动体页面提示。
