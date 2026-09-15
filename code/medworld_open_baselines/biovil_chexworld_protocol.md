# BioViL-T / CheXWorld 的 Table 1 适配

2026-09-13 创建。这里使用公开预训练视觉编码器，训练与当前 Table 1 相同的 Qwen 融合、未来状态预测器和报告／finding 读出。名称应写成 **BioViL-T + matched predictor/readouts (adapted)**、**CheXWorld + matched predictor/readouts (adapted)**。这不是原论文自带的患者未来预测方法。

## 权重与实现

| 表征 | 官方来源 | 本地 checkpoint | 本地验证 |
|---|---|---|---|
| BioViL-T | [Microsoft 模型卡](https://huggingface.co/microsoft/BiomedVLP-BioViL-T)、[hi-ml](https://github.com/microsoft/hi-ml) | `biovil_weights/biovil_t_image_model_proj_size_128.pt`，官方 `v1.0` | 官方 MD5 `a83080e2f23aa584a4f2b24c39b1bb64` 一致；严格加载；CPU 输出 `[1,512,14,14]`，全部 finite |
| CheXWorld | [官方仓库](https://github.com/LeapLabTHU/CheXWorld)、仓库链接的 [Drive 目录](https://drive.google.com/drive/folders/1XdmQaNo0U2ilDEGYnLRz39Eywkom13BP) | `chexworld_weights/chexworld_pretrained.tar`，733,569,866 bytes | 严格加载 `model[target_encoder.*]`；CPU 输出 `[1,768,14,14]`，全部 finite |

BioViL-T 代码 commit 为 `b67c1d27c6b17d8e8ff01f8c507f3cabdb307388`，模型／代码标为 MIT。CheXWorld 代码 commit 为 `090102758801dc097f53c49d135b835570c8d173`；没有找到仓库顶层 LICENSE，部分继承文件保留 Meta 版权／许可证引用，因此不能把此仓库描述为 MIT。两份 checkpoint 的 SHA-256、来源与验证记录保存在对应 `*_weights/provenance.json`。全部推理使用本地影像，未向外部服务发送患者数据。

## 共同协议与差异

- 数据直接链接 `medworld_table1/data/linked_20260913_16k` 的不可变清单：16,000 train、230 validation、297 test，28,202 observations；不重新划分患者。三个 split 的 SHA-256 写入适配记录。
- 每个方法 2,400 次 optimizer 更新、batch 32、梯度累积 1，合计 76,800 对呈现。样本顺序、seed 42、学习率、损失权重、horizon、report/EHR token 上限沿用扩大版配置。没有复用旧截止时间；以完整更新预算为准。
- 视觉编码器冻结，空间特征平均池化到 8×8。BioViL-T 使用原生 512 维 static＋missing-prior 特征，CheXWorld 使用原生 768 维 target ViT-B 特征。报告融合、8 个状态 slots、horizon predictor 和报告／finding 读出相同。
- 两者从共同的旧 image-only Stage-1 checkpoint 迁移其余参数，**视觉 adapter 的全部参数重新初始化**，以免把 V-JEPA 特征空间的投影当成通用投影。BioViL-T adapter 输入维度为 512，CheXWorld 为 768，因此该层参数量略有差别。没有额外的 encoder-specific Stage-1 更新。
- BioViL-T 采用官方整图 min/max intensity remap、short-edge 512／center-crop 448、`[0,1]` 灰度复制三通道；CheXWorld 采用官方 short-edge 256／center-crop 224、ImageNet normalization。模型原生输入分辨率不同；更新次数与有效 batch 一致，**实际 GPU 时间不是预先强制相等**。运行记录保存 feature/training 秒数与 GPU 型号，报告时应同时给出。
- BioViL-T 只输入当前图像，使用官方缺失前次影像的 embedding。所有方法的 source 都只允许当前图、当前报告、源时点 EHR、horizon；未来图像／报告仅供训练目标。没有为 BioViL-T 引入额外的上一张片或未来片。
- `cache/vjepa_features.npy` 仅为现有 `Corpus` 的兼容存储名；`features.json`、配置和结果中的 `encoder`／`mode` 明确使用真实方法名。这里不含 V-JEPA 特征。

## 运行入口

在 workspace 根目录，Python 为 `code/medworld_table1/.venv/bin/python`。完整可调度的 8 个任务和成功条件见 `biovil_chexworld_jobs.json`；每个方法依次为 features → train → evaluate → GREEN。GPU 由外部共享锁队列分配。

```bash
code/medworld_table1/.venv/bin/python code/medworld_open_baselines/biovil_forecast.py prepare \
  --encoder biovil_t --output code/medworld_open_baselines/biovil_runs/forecast_16k
code/medworld_table1/.venv/bin/python code/medworld_open_baselines/chexworld_features.py \
  --encoder biovil_t --config code/medworld_open_baselines/biovil_runs/forecast_16k/config.json
code/medworld_table1/.venv/bin/python code/medworld_open_baselines/biovil_forecast.py train \
  --config code/medworld_open_baselines/biovil_runs/forecast_16k/config.json \
  --run code/medworld_open_baselines/biovil_runs/forecast_16k/train
```

CheXWorld 将 `--encoder` 换为 `chexworld`、输出根换为 `chexworld_runs/forecast_16k`。`prepare --steps --batch-size --base-config` 可配置预算和清单，默认匹配当前扩大版。`train --resume checkpoint_latest.pt` 沿用原配置恢复；`train --smoke-steps 2` 使用独立 `--run`，不得将其视作正式结果。

`evaluate` 重用已有 Table 1 的 source-only generation、CheXbert、RadGraph 和 AP/AUROC/Brier/ECE 评价。Qwen 生成留在 Transformers 5 进程，随后由 `biovil_score.py` 独立子进程加载 `metric_vendor` 的 Transformers 4，避免 RadGraph 的旧 tokenizer API 与生成环境冲突。`green` 使用已有官方 GREEN prompt/parser/checkpoint。297 对测试集没有经过裁定的方向标签，Direction F1 保留 unavailable；不能把另一个 82 对测试集悄悄混入同一行。

代码快照需要保持 `medworld_open_baselines`、`medworld_table1`、`medworld_common` 的相对目录结构，并链接固定 vendor／weights。原队列修改不在本适配代码范围内。下载的权重、vendor checkout、包含患者清单和预测的 `*_runs` 不应提交到 Git。

## 验证与状态

已完成两份公开 checkpoint 严格 CPU 加载和 finite 输出检查；BioViL forecast 的共同参数迁移、视觉 adapter 保持新初始化、target encoder 冻结、compact checkpoint roundtrip 均通过。GPU 任务的完成情况以 `feature_status.json`、`train/status.json`、`evaluation_test/metrics.json` 为准。代码／资产可用不表示正式 2,400 步训练已经完成。
