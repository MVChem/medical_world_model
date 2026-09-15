# SwinIR ×4 CXR 对比方法适配

入口：`code/medworld_open_baselines/swinir_train.py`。官方实现来自
[JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR)，固定 commit
`6545850fbf8df298df73d81f3e8cba638787c8bd`，Apache-2.0。
权重为官方 [v0.0 release](https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0)
的 `001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth`，SHA256
`4e78e33f22c1aa8a773db0cf4a7381bae97c2362c717f155439ebc690cbd9215`。
这是 DF2K 自然图像预训练的 classical ×4 SwinIR-M，不是灰度去噪或 GAN SR 权重。

模型保留官方 11,900,199 个参数，全部参与 CXR 适配。单通道 LR 重复为 RGB；
RGB 输出按三通道均值转回灰度，随后按共同 valid ROI MSE 优化。输入直接读取已冻结的
uint8 LR128 cache，预测整张 HR512；HR 只作为目标，不重新生成 LR，不随机裁剪、
不做数据增强，不添加图像 skip，不采用测试集选 checkpoint。

首轮复用下午 `dense_20260912/data` 的 4096 train、249 validate、447 test；
支持 `--data-run code/medworld_dense_baselines/runs/expanded_overnight_20260913`
切换到 18,708 train，验证和测试不变。训练预算为 20 epochs、有效 batch 8、
seed 20260913、AdamW 3e-4、WD .01、epoch cosine 到10% floor、梯度裁剪1。
每轮样本顺序与下午 frozen-slots decoder 完全相同；默认 microbatch1 后累积到8。
首轮 10,240 次更新、81,920 张训练输入；扩展版 46,780 次更新、374,160 张输入。

评分直接复用冻结于 `code/medworld_open_baselines/swinir_protocol/` 的下午协议：
逐图 valid ROI、输出clamp[0,1]、PSNR、skimage uniform7×7 SSIM、不裁边，
1,000次患者聚类bootstrap。保留逐样本指标，最后第20epoch才写最终 `metrics.json`。
模型结构和预训练不同，因此相同数据、epoch、更新数和GPU型号不等于相同FLOPs或耗时。
`contract.json`、`hardware.json`、`progress.json` 和 checkpoint 分别记录参数、
权重/源码/输入指纹、实际卡、累计训练输入、优化器更新和GPU训练步耗时。

checkpoint保存模型、AdamW、RNG、epoch内下一批游标和累计预算；SIGTERM/SIGUSR1或
可选 `--stop-at` 在完整更新边界暂停。没有默认截止。暂停不写最终指标，重跑相同命令恢复。
`--stop-after-steps` 是审计用暂停参数，退出124；正式训练不要设置。

```bash
CUDA_VISIBLE_DEVICES=<空闲GPU> /home/data2/chk/workspace/2026/.venv/bin/python \
  code/medworld_open_baselines/swinir_train.py train \
  --out code/medworld_open_baselines/runs/comparators_20260913/swinir_4096 \
  --microbatch 1
```

上述命令需由共同队列持有 `/tmp/medworld-frozen-slots-{GPU_UUID}.lock` 并确认卡空闲后执行。
不改共享Python环境；当前 torch2.11.0+cu129、timm1.0.24 可加载官方实现。

验证已完成：CPU真实LR128→HR512前向、实际ROI小块梯度更新、权重/优化器/RNG
保存恢复后输出逐元素一致、实际输入数组SHA256与manifest核对、患者划分互斥。
GPU7 同样通过真实前向和BF16短训练；随后一次完整LR128、有效batch8的实际优化器步
成功、无OOM，3.868秒/step，保存到 `runs/swinir_gpu_resume_smoke_20260913/` 后暂停。
粗略按首步推算首轮约11 GPU小时，扩展约50 GPU小时；这是未含验证的初始估计，
不是实测完成时长。检查产物位于 `runs/swinir_{cpu,gpu}_smoke_20260913/`。
这些短检查只验证实现，不作为Table2效果结果。
