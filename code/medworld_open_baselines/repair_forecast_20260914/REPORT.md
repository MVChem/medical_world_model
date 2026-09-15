# 开源对比方法复现

更新时间：2026-09-14T10:31:59.116703+08:00

复用既有患者划分、目标和评分；训练预算、官方预训练来源与实际GPU小时分别记录。
4096与18708训练图的结果分开；Table1使用16000/230/297对。公开模型的ZS行保持零样本。

以下状态来自任务回执；未完成训练不填写最终指标。

| 任务 | 状态 | GPU | 日志 |
| --- | --- | --- | --- |
| repair_forecast_biovil_table1_eval | complete | [7] | [log](/home/data2/chk/workspace/2026/08/04/medical_world_model/code/medworld_open_baselines/repair_forecast_20260914/scheduler_recovery/attempts/repair_forecast_biovil_table1_eval-112dfed8-9d05-42e5-ae2b-3b12ca350e3f/worker.log) |
| repair_forecast_biovil_table1_green | complete | [7] | [log](/home/data2/chk/workspace/2026/08/04/medical_world_model/code/medworld_open_baselines/repair_forecast_20260914/scheduler_recovery/attempts/repair_forecast_biovil_table1_green-f7468b72-5b6f-468f-891f-b24406a2377d/worker.log) |
| repair_forecast_chexworld_table1_eval | complete | [3] | [log](/home/data2/chk/workspace/2026/08/04/medical_world_model/code/medworld_open_baselines/repair_forecast_20260914/scheduler_recovery/attempts/repair_forecast_chexworld_table1_eval-37ebac6c-658d-4371-8004-4fc6cf608da5/worker.log) |
| repair_forecast_chexworld_table1_green | complete | [3] | [log](/home/data2/chk/workspace/2026/08/04/medical_world_model/code/medworld_open_baselines/repair_forecast_20260914/scheduler_recovery/attempts/repair_forecast_chexworld_table1_green-8e5d84a0-ae0f-4b57-a432-cb91bac31f8e/worker.log) |

| 权重 | 下载状态 |
| --- | --- |

MAIRA-2官方权重当前需要HF账户访问授权，未授权状态不计为完成。

| 已完成指标文件 | 主要指标 |
| --- | --- |
| [repair_forecast_biovil_table1_eval](/home/data2/chk/workspace/2026/08/04/medical_world_model/code/medworld_open_baselines/biovil_runs/forecast_16k/evaluation_test/metrics.json) | {"chexbert_f1": 0.35132, "transition_f1": 0.22647, "ap": 0.84296, "auroc": 0.8124, "brier": 0.10748, "ece": 0.07725, "radgraph_f1": 0.14698} |
| [repair_forecast_biovil_table1_green](/home/data2/chk/workspace/2026/08/04/medical_world_model/code/medworld_open_baselines/biovil_runs/forecast_16k/green/biovil_t_adapted/test/green_metrics.json) | {"mean": 0.15369} |
| [repair_forecast_chexworld_table1_eval](/home/data2/chk/workspace/2026/08/04/medical_world_model/code/medworld_open_baselines/chexworld_runs/forecast_16k/evaluation_test/metrics.json) | {"chexbert_f1": 0.36681, "transition_f1": 0.23705, "ap": 0.83805, "auroc": 0.79556, "brier": 0.11728, "ece": 0.09037, "radgraph_f1": 0.13146} |
| [repair_forecast_chexworld_table1_green](/home/data2/chk/workspace/2026/08/04/medical_world_model/code/medworld_open_baselines/chexworld_runs/forecast_16k/green/chexworld_adapted/test/green_metrics.json) | {"mean": 0.12626} |

Finding-transition prior使用训练集拟合，源状态来自已准备的结构化finding标签；该输入接口单独注明。
Direction在297对测试集缺合格真值；官方VQA仍缺本轮正式数据。留空不等于零分。

运行的源码、命令、预算、checkpoint、逐样本预测与数据指纹保存在各任务目录。
