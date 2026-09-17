# MIMIC-CXR-VQA 官方数据下载与核验

2026-09-16 20:26（Asia/Shanghai）完成。PhysioNet 账号对该项目的授权已经生效，登录后文件请求返回 HTTP 200；此前 2026-09-11 记录的 DUA 阻塞已解除。

数据来源：[MIMIC-Ext-MIMIC-CXR-VQA v1.0.0](https://physionet.org/content/mimic-ext-mimic-cxr-vqa/1.0.0/)。

按用户指定位置保存于：

```text
/home/data2/chk/data/MIMIC/mimic-ext-mimic-cxr-vqa-1.0.0/
├── mimic-ext-mimic-cxr-vqa-1.0.0.zip
├── LICENSE.txt
├── SHA256SUMS.txt
├── download_verification.json
└── MIMIC-Ext-MIMIC-CXR-VQA/dataset/
    ├── train.json
    ├── valid.json
    └── test.json
```

压缩包 28,362,291 bytes（约 27.0 MiB）；官方文件解压后约 334.7 MiB。包内是问答标注，不包含胸片图像。

| 划分 | 问答条数 | 图像数 | 患者数 |
|---|---:|---:|---:|
| Train | 290,031 | 133,687 | 52,453 |
| Valid | 73,567 | 8,610 | 3,461 |
| Test | 13,793 | 500 | 500 |

共 377,391 条问答、142,797 张不同图像。ZIP CRC 检查通过，三个 JSON 和 LICENSE 均与官方 SHA256SUMS 一致；迁移到指定目录后再次核验通过。JSON 字段、答案类型、split 和条数检查通过。

以 `/home/data2/chk/data/MIMIC/MIMIC_CXR/files/` 为图像根目录，逐一检查标注中的 `image_path`：142,797 张图像全部存在且非空，缺失为 0。原始图像可直接复用。

机器可读核验记录：[download_verification.json](/home/data2/chk/data/MIMIC/mimic-ext-mimic-cxr-vqa-1.0.0/download_verification.json)。

本次完成原始官方数据的下载与核验。已有实验使用的 Chest ImaGenome 派生 QA 及其历史结果仍属于原协议；官方全量 VQA 的训练／评估接入和患者划分审计尚待执行。
