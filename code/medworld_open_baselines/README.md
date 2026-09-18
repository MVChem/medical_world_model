# Open Table 1 / Table 2 comparators

独立复现表中的公开比较方法，沿用原有患者划分、任务目标和评分。

- [总协议与限制](https://github.com/MVChem/medical_world_model/blob/31c98ca14173620a3cec67da2defe559d05974da/research_notes/0913_open_comparators.md)
- [实时队列与指标](runs/comparators_20260913/LIVE.md)
- [BioViL-T／CheXWorld future adaptation](biovil_chexworld_protocol.md)
- [DINOv2／CheXWorld heads](https://github.com/MVChem/medical_world_model/blob/31c98ca14173620a3cec67da2defe559d05974da/research_notes/0913_dinov2_chexworld_heads.md)
- [SwinIR](https://github.com/MVChem/medical_world_model/blob/31c98ca14173620a3cec67da2defe559d05974da/research_notes/0913_swinir_open_baseline.md)
- [原生CheXagent／LLaVA-Med](https://github.com/MVChem/medical_world_model/blob/31c98ca14173620a3cec67da2defe559d05974da/research_notes/0913_legacy_native_baselines.md)

`launch.py`冻结源码后启动共享GPU锁队列，`report.py`生成状态与完成指标。
训练数据、权重、第三方代码、日志和逐样本输出均保留在本地忽略目录。
