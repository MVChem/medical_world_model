# CLARITY 数据元数据审计（2026-09-16）

对应[研究笔记](../../0916_clarity_datasets_feasibility.md)。本目录只含公开小型表格和聚合审计，不含完整 MRI，也未提交 UCSF 数据申请。

| 文件 | 官方来源 | 本次下载字节数 |
|---|---|---:|
| `MU-Glioma-Post_ClinicalData-July2025.xlsx` | [TCIA 临床表](https://www.cancerimagingarchive.net/wp-content/uploads/MU-Glioma-Post_ClinicalData-July2025.xlsx) | 101,568 |
| `MU-Glioma-Post_Segmentation_Volumes.xlsx` | [TCIA 体积表](https://www.cancerimagingarchive.net/wp-content/uploads/MU-Glioma-Post_Segmentation_Volumes.xlsx) | 307,753 |
| `ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx` | [TCIA 临床表](https://www.cancerimagingarchive.net/wp-content/uploads/ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx) | 58,200 |

来源页面均将上述数据标为 CC BY 4.0；使用时须遵守集合页的数据引用要求：

- Yaseen et al. (2025), *University of Missouri Post-operative Glioma Dataset (MU-Glioma-Post), Version 1*. TCIA. [DOI: 10.7937/7K9K-3C83](https://doi.org/10.7937/7K9K-3C83)。
- Li et al. (2022), *I-SPY 2 Breast Dynamic Contrast Enhanced MRI Trial (ISPY2), Version 1*. TCIA. [DOI: 10.7937/TCIA.D8Z0-9T85](https://doi.org/10.7937/TCIA.D8Z0-9T85)。完整 cohort 还需遵循集合页对 ACRIN-6698 的引用要求：Newitt et al. (2021), *ACRIN 6698/I-SPY2 Breast DWI*. [DOI: 10.7937/TCIA.KK02-6D95](https://doi.org/10.7937/TCIA.KK02-6D95)。

[metadata_audit.json](metadata_audit.json) 保存来源 URL、文件 SHA-256、时间点／患者聚合统计及列名。复算：

```bash
/home/data2/chk/workspace/2026/.venv/bin/python \
  research_notes/assets/clarity_data_audit_20260916/audit_metadata.py
```

脚本依赖 `openpyxl`，只读取本地文件，不访问网络。候选 pair 由每人六个 MRI 列中的数值型相对日排序、去重得到；没有检查真实影像端点。表格里的全程进展、生存结局和未来治疗字段不能直接成为源时点模型输入。

本轮文件保留在工作区，未提交 Git。当前 `.gitignore` 并未忽略本目录；后续版本管理应区分研究笔记／审计脚本与下载的原始表格。
