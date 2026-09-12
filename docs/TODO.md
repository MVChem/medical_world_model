# 当前事项

项目入口：[README.md](../README.md)。历史任务原文已归档至 [task_history.md](../research_notes/archive/task_history.md)。

- [x] 核对 4＋4 slots 与无 slots 模型在第 3,176 步的同样本验证比较。
- [x] 核对两组 24,000 步 final checkpoint 的四任务测试结果、样本及训练预算匹配，见 [09-12 小结](../research_notes/0912_recent_experiments_summary.md)。
- [ ] 复盘疾病列表、分割和报告读出的失败样例；解释 slots 未在所有任务取得改善的原因。
- [x] 完成六个原始 Qwen／MedGemma 的零样本评测并归档聚合结果。
- [ ] 完成 09-12 冻结 VLM 的 55 项队列，核对 epoch 20 分割／×4 SR／解剖定位结果及独立 82 对方向评测。
- [ ] 完成官方 VQA 数据接入，区分本地派生问答与正式 benchmark。
- [ ] 接入 MS-CXR 病灶短语标注；当前 Chest ImaGenome 解剖区域定位单独记录。
- [ ] 按 Table 2 已讨论的方法补充其他 baseline 的适配与训练。
- [x] 确认 Table 1 四组八指标，并写入论文主表、正文和计划表。
- [x] 在原始 VLM 评测路径接入 Table 1 的 AP／AUROC／Transition／RadGraph／GREEN／Brier／ECE。
- [ ] 按已确认的 Table 1 指标完成方向标签验收，并统一 Ours／direct 的新评分接口与对应实验。
- [ ] 结果稳定后填写论文表格，保留验证／测试划分与 checkpoint 来源。

运行状态以 [实验索引](../experiments/README.md) 中的原始记录为准。新增实验须登记；文档中的计划完成后，在索引中链接执行记录。
