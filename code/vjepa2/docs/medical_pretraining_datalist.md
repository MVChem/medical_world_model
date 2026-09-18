# Medical pretraining dataset shortlist

更新时间：2026-08-26

目标是给当前 V-JEPA 2.1 架构准备一个以 CT/MRI volume 为主、同时保留胸片能力的医学预训练数据池。这里优先考虑：能合法获得、下载步骤不复杂、总体体量适中、3D case 数量不太少。`NLST` 和 `CT-RATE` 全量不作为第一阶段数据。

## 结论

建议先使用下面这个组合：

| 模态 | 数据集 | 3D 规模 | 官方下载体量 | 下载难度 | 第一阶段建议 |
| --- | --- | ---: | ---: | --- | --- |
| Brain MRI | IXI | 近 600 人；每人最多含 T1/T2/PD/MRA/DTI 多序列 | 约 28 GB | 低：直接 HTTP，CC BY-SA 3.0 | 下载全部；作为健康脑 MRI 主体 |
| Whole-body CT | TotalSegmentator CT v2 | 1,228 scans | 23.6 GB | 低：Zenodo/Dropbox 直接下载 | 下载；临床来源、解剖和病理较多样 |
| Whole-body MRI | TotalSegmentator MRI v2 | 616 scans | 5.1 GB | 低：Zenodo/Dropbox 直接下载 | 下载；很小但能明显增加 MRI 部位和序列多样性 |
| Abdomen CT/MRI | AMOS22 | 500 CT + 100 MRI | 24.2 GB | 低：Zenodo 单包直接下载 | 下载；补充腹部、多中心、多设备、多期相 |
| Chest CT | LIDC-IDRI | 1,018 cases / 1,010 participants | 约 125 GB | 中低：无需资质审批，但需 NBIA Data Retriever | 下载；与最终胸片方向最相关的 3D 数据 |

这批公开 3D 数据合计约 **206 GB 下载量**，约 **4,062 个 subject/case-level 3D 样本**：

- CT 约 2,746 cases：LIDC-IDRI 1,018 + TotalSegmentator CT 1,228 + AMOS CT 500。
- MRI 约 1,316 subject/scans：IXI 约 600 + TotalSegmentator MRI 616 + AMOS MRI 100。
- IXI 的 T1/T2/PD/MRA/DTI 应分别视为 volume，因此实际 MRI 训练单元会明显多于 1,316。各序列并非每位受试者都齐全，生成 manifest 时按实际文件计数。
- 206 GB 是下载包/官方页面报告体量；解压、DICOM 转 NIfTI、重采样和缓存后应预留至少 2--3 倍空间。

这个规模足够先验证医学 JEPA 训练，不必为了“volume 数量”立即下载十几 TB 的数据。

## 本地已有数据

### MIMIC-CXR-JPG

- 本地状态：已找到完整图像集。
- 实际格式：**JPG，不是 PNG**。
- 本地盘点：377,110 张 `.jpg`，约 580 GB；metadata、official split 和 CheXpert-label CSV 均存在。
- 官方规模：377,110 张胸片，227,835 个 studies，65,379 位患者。
- 获取难度：高；PhysioNet credentialing、培训和 DUA 较繁琐。不过本地已经具备，不需要重新下载。
- 配置建议：代码和文档只使用 `MIMIC_CXR_ROOT`，不要把机器绝对路径写进可提交的配置。
- 训练建议：作为 X-ray 主数据源已经足够大。按患者划分，AP/PA/lateral 图像分别作为 `T=1` 样本；不要把 AP 和 lateral 当成连续时间帧硬 stack。

### CheXpert / CheXpert Plus

- 本地状态：当前 workspace 内没有找到 CheXpert 图像目录或其典型 CSV/目录结构。
- 容易混淆的文件：MIMIC 目录中的 `mimic-cxr-2.0.0-chexpert.csv` 只是用 CheXpert labeler 生成的 **MIMIC 标签**，不是 CheXpert 图像集。
- 官方规模：原始 CheXpert 为 224,316 张胸片、65,240 位患者；当前 CheXpert Plus 为 223,462 个图文 pair、187,711 个 studies、64,725 位患者，并提供 DICOM 和报告。
- 下载体量：当前 Stanford/Redivis 官方页面没有展示稳定的总字节数。不要把社区常见的“约 11 GB”误认为全分辨率数据；该数通常指旧的 320 px downsampled 版本。全分辨率版本需要按 portal 实际显示重新确认空间。
- 获取难度：中；需要 Stanford/Redivis 账户并接受数据条款，但通常比 MIMIC 的 credentialing 简单。
- 优先级：第二阶段。MIMIC 已覆盖大规模胸片，CheXpert 的主要价值是跨医院 domain diversity，不增加新模态，因此不应挤占第一批 3D volume 的磁盘预算。

## 可选的第二阶段数据

| 数据集 | 模态与规模 | 体量 / 门槛 | 什么时候值得加 |
| --- | --- | --- | --- |
| RSNA-MICCAI 2021 radiogenomics release（BraTS-derived） | Brain MRI；每例含 FLAIR/T1w/T1Gd/T2w | Kaggle 包 136.85 GB、约 400k DICOM 文件；需账户并接受 competition terms | IXI 之后补充脑肿瘤和病理变化。下载和预处理明显比 IXI 麻烦 |
| CheXpert / CheXpert Plus | 约 223k--224k 胸片，含跨医院图文信息 | 官方 portal 未公开稳定总字节；账户 + 数据条款 | 需要验证 X-ray 跨机构泛化时再加 |
| OASIS-3 | 1,378 人、2,842 MR sessions；含 T1/T2/FLAIR 等 | 需申请和 data-use agreement；体量随所选 session/序列变化 | 需要衰老、认知下降或纵向脑 MRI 时加入 |
| fastMRI brain | 约 7k brain MRI，含 T1/T1-post/T2/FLAIR 和 raw k-space | 申请 + data-use agreement；原始 k-space 较大且预处理重 | 需要更大脑 MRI 规模或研究 acquisition/reconstruction 时加入 |

如果 IXI 的健康脑分布不够，第二个脑数据优先选 BraTS-derived 病理数据，而不是继续堆健康人 MRI。

## 暂缓或排除

| 数据集 | 官方规模 | 原因 |
| --- | ---: | --- |
| NLST | 26,254 subjects、73,116 studies、约 11.3 TB | 体量过大，DICOM 下载和预处理都重；第一阶段不需要 |
| CT-RATE full | 25,692 chest CT volumes，扩展 reconstruction 共 50,188；约 21.3 TB | 方向匹配但全量太大，且 Hugging Face gated/research-only；只在验证方案后考虑定量抽取子集 |
| PadChest full | 超过 160k 胸片 | 仍是 2D X-ray，MIMIC 已经更大；下载与许可收益不如先补 3D |
| ADNI / UK Biobank | 大规模 MRI/多模态 | 申请、审批和使用条款较重，不符合“先选容易下载”的目标 |

## 建议的训练池组织

1. 以 subject/study 为单位划分 train/val，严禁同一患者的不同序列或不同视图跨 split。
2. CT/MRI 的一个序列是一个 3D volume；从同一 volume 沿固定解剖轴采连续 slice clip。不同 MRI 序列默认不要当成时间帧拼接。
3. MIMIC 胸片用 `T=1`；多视图信息留给下游 study-level aggregator，而不是伪造视频时间轴。
4. 先按模态采样，再按数据集采样。否则 377k 张 MIMIC 会完全淹没约 4k 个 3D case。初始可从 X-ray/CT/MRI = `40/35/25` 的 sample 比例开始，再用消融调整。
5. 建议第一轮只做 IXI + TotalSegmentator CT/MR + AMOS22 + LIDC-IDRI + 本地 MIMIC；直接下载新增约 206 GB，风险和预处理工作量可控。

当前仓库的通用 loader 主要读取视频或 JPG/PNG，不能直接把 DICOM/NIfTI 当训练样本。真正开训前还需要统一 DICOM/NIfTI 读取、强度窗/归一化、spacing/orientation 和连续 slice 采样；这些是数据适配工作，不要求改变 V-JEPA 2.1 encoder 架构。

## 官方来源

- [IXI dataset](https://brain-development.org/ixi-dataset/)；[IXI archive index](https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/)
- [TotalSegmentator CT v2, Zenodo](https://zenodo.org/records/8367088)
- [TotalSegmentator MRI v2, Zenodo](https://zenodo.org/records/14710732)
- [AMOS22, Zenodo](https://zenodo.org/records/7155725)
- [LIDC-IDRI, TCIA](https://www.cancerimagingarchive.net/collection/lidc-idri/)
- [MIMIC-CXR-JPG, PhysioNet](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)
- [CheXpert, Stanford AIMI](https://aimi.stanford.edu/datasets/chexpert-chest-x-rays)；[CheXpert Plus](https://aimi.stanford.edu/datasets/chexpert-plus)
- [RSNA-MICCAI radiogenomics, Kaggle](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/data)；[BraTS 2021 description](https://www.med.upenn.edu/cbica/brats2021/)
- [OASIS-3](https://www.oasis-brains.org/)
- [fastMRI](https://fastmri.med.nyu.edu/)
- [NLST, TCIA](https://www.cancerimagingarchive.net/collection/nlst/)
- [CT-RATE, Hugging Face](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
- [PadChest paper](https://pubmed.ncbi.nlm.nih.gov/32877839/)
