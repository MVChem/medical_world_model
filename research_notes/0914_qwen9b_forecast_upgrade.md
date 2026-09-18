# 09-14：Table 1 底部三条件升级为 Qwen3.5-9B

用户要求将 Table 1 底部三行也改成 Qwen3.5-9B。本次指定为：

- Full-token forecaster (Qwen3.5-9B)
- MedWorld-JEPA (Qwen3.5-9B slots)
- Qwen3.5-9B (shuffled state)

现有底部三行实测来源为 **Qwen3.5-0.8B**，其分数不能随行名迁入 9B。新三条件需要共同的 9B Stage-1 初始化，原 0.8B checkpoint 的参数维度不兼容。计划保持 16,000/230/297 对 train/validation/test、94 位测试患者、2,400 forecast updates、effective batch 32、seed 42。此记录仅说明表格与评估映射，不宣称新训练已经完成。

三行当前均保留八个 `TBD`。生成器只读取新运行目录 `code/medworld_table1/runs/qwen9b_ablation_20260914/{no_slots,slots,shuffled}/`，核验 9B 模型配置、最终 2,400 步训练完成元数据、297 对完整生成和最终指标后回填临床分数。GREEN 从 `green/{condition}/test/green_metrics.json` 读取，独立要求 `status=complete` 且 `n=completed=297`。Direction 的 cohort 标签尚未完成，仍保留 `TBD`。

9B slots 适配继续采用最后语言层的八个学习查询。Full-token 条件保留有效图像／报告／EHR tokens 和相同预测器；shuffled 条件跨患者打乱源状态。这三个条件与 9B 原生零样本行及正文提出的多层 fusion/vision 4+4 完整架构分开。Table 1 其他八行及 Table 2 的数值保持原样。

## 历史 0.8B 结果

以下是原 0.8B 实验的单种子点估计，保留用于溯源，**不是 9B 结果**。Direction 尚无主表分数。

| 原 0.8B 条件 | AP | AUROC | Transition F1 | RadGraph F1 | GREEN | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-token | .8239 | .7851 | .3000 | .1562 | .1691 | .1177 | .0827 |
| 8 slots | .8303 | .7898 | .2400 | .1513 | .1738 | .1162 | .0820 |
| Shuffled state | .6968 | .5042 | .0604 | .0910 | .0819 | .1361 | .0594 |

原 slots 的 AP/AUROC 比 full-token 高约 .0064/.0046，但 Transition/RadGraph 较低。打乱条件 AUROC 接近随机，并对 297 对生成同一份报告。这些观察仅适用于原 0.8B 实验，不推断 9B 的结果。

原始指标仍保留在 `code/medworld_table1/runs/overnight_20260913/{no_slots,slots,shuffled}/evaluation_test/metrics.json`，GREEN 位于同运行根的 `green/{condition}/test/green_metrics.json`。原始文件与[此前回填记录](0914_table_results_filled.md)均未改写。当前共享源码、全精度聚合与自动回填映射分别见：

- [Table 1 TeX](../27cvpr/tables/table1_future.tex)
- 数值来源见表格各行的 TeX 注释；聚合 JSON 已在目录整理时移除。
- [回填生成器](../27cvpr/plans/populate_results.py)

## 正式补跑已启动

正式运行目录为 `code/medworld_table1/runs/qwen9b_ablation_20260914`，持久队列为其 `scheduler_training/`，启动协调器 PID 403325。队列包含共同 Stage 1、三组 Stage 2、三组完整测试，以及最后的 GREEN，共 8 个作业。源码快照、原始命令和校验哈希保存在该队列中；退出当前会话不会中止协调器。

共同 Stage 1 使用原相同的 11,332 个训练观测、1,694 次更新和有效 batch 8（microbatch 4 × accumulation 2，共 13,552 次样本呈现）。三组 Stage 2 各用有效 batch 32（4 × 8）、2,400 次更新。旧／新 tokenizer 对两个 cohort 的全部 report、EHR 及当前 decoder prompt 分词完全一致；旧特征缓存直接复用。原 0.8B checkpoint 不用于初始化 9B。

9B 保留 BF16 预训练参数及其独立输出头，没有量化。encoder、decoder 和 target 仅共享从不更新的基座权重；LoRA、自定义参数和模块状态保持独立。GPU 上已验证共享存储、online LoRA 可训练、target 全冻结。9B 四分片签名已接入训练和评估，原 0.8B 单分片恢复签名兼容性也已验证。

启动前验证：Stage 1 与三种 Stage 2 的真实 GPU 短测 4/4 完成，slots／full-token 的生成及临床评分短测 2/2 完成。最长 775-token source/target、385-token 报告目标、batch 4 的完整 loss、backward、AdamW 检查通过，峰值 reserved 21.75 GiB。正式训练使用共享 GPU 锁，允许空闲卡 3/4/5/6/7，避开已有任务及 GPU 0 的额外常驻显存。

这些短测仅验证执行和显存，不作为主表结果。查看[实时队列](../code/medworld_table1/runs/qwen9b_ablation_20260914/scheduler_training/QUEUE.md)；正式评估完成后运行表格生成器回填。
