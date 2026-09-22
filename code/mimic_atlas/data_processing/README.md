# MedWorld 人工审核数据处理

从 Atlas 的全量 MIMIC-CXR / MIMIC-IV 表构建分类和时序清单，接入人工审核的
CXR 心肺与脑 MRI 分割。活动入口为 `code/data/medworld_0922`，只保存最终清单和
原始数据链接，不生成像素、掩膜或特征缓存。修改前的 MedWorld 已保存至 GitHub
提交 `b1cad08`。

## 数据与存储

| 分割来源 | 患者 | 有标注影像/体积 | 实际监督区域 |
| --- | ---: | ---: | --- |
| MIMIC 人工审核心肺 | 196 | 200 张 CXR | 双肺合并、心脏 |
| UCSF-ALPTDG | 298 | 596 个三维体积 | NETC、SNFH、ET、RC |
| MU-Glioma-Post | 203 | 594 个三维体积 | NETC、SNFH、ET、RC |
| Montgomery 人工肺掩膜 | 138 | 138 张 CXR | 双肺合并，仅外部测试 |

MU 共 596 次检查，其中 2 次缺少标准分割掩膜，保留在体积清单但不进入分割监督。
UCSF 的纵向差分掩膜不作为额外独立分割样本。MRI 共 1,190 个有标注体积，
不是 1,190 张二维图片。三个训练来源都按患者划分；同一患者的全部 MRI 时间点在同一划分。

这里的“人工审核”包括模型或半自动初稿经专家检查、修改的标注。活动 v2 数据
不读取 CXAS 伪标签，也不读取旧分类选集、旧时序配对或旧图像数组。
Montgomery 只复用原始人工掩膜和原始图像。历史 v1 加载接口保留用于旧实验。

原始数据统一放在 `/home/data2/chk/data`。新增公开标注实际目录为
`/home/data2/chk/data/heart-lung-segmentations-data/1.0.0`，项目入口
`code/data/heart_lung_human` 为符号链接。MRI 沿用该中央目录下的
`UCSF-ALPTDG` 和 `MU-Glioma-Post`。生成的数据集也通过 `code/data/medworld_0922`
链接到中央目录 `medical_world_model/medworld_0922`。

公开心肺标注下载后逐项验证发布者的 540 个 SHA256 校验值，原始 PNG 不修改。
每个 MRI 输入 T1ce 和目标掩膜均记录 SHA256，按需加载时校验。
来源说明：[心肺标注](https://physionet.org/content/heart-lung-segmentations-data/1.0.0/)、
[UCSF 论文](https://pubs.rsna.org/doi/pdf/10.1148/ryai.230182)、
[MU 论文](https://www.nature.com/articles/s41597-025-06011-7)。原始数据仍遵守各自许可和访问条件。

## 构建和检查

在仓库根目录使用已有 Python 环境；依赖见 `requirements.txt`，加载验证还需 PyTorch。

```bash
export PYTHONPATH=code
PYTHON=/home/data2/chk/workspace/2026/.venv/bin/python

# 已有数据集：构建成功后归档旧清单，再发布新清单。
$PYTHON -m mimic_atlas.data_processing --replace

# 实际 MedWorld 加载、患者隔离、图像/掩膜解码及 CPU 分割头前后向检查。
$PYTHON -m mimic_atlas.data_processing.validate

# 单独校验公开人工心肺标注，不重复下载。
$PYTHON -m mimic_atlas.data_processing.human_cxr --verify-only
```

构建先写临时目录。默认拒绝覆盖；`--replace` 只接受有 `.medworld-prepared`
标记的数据目录，并归档为 `medworld_0922_previous_时间戳`。项目入口为软链接时，
在其中央目标目录旁构建、归档，保留软链接。发布使用两次目录重命名，异常会回滚，
但两次重命名之间不是原子交换；重建应避开训练加载时段。

其他方案指定中央目录中的新输出，再建立项目软链接，并同步修改配置的数据入口。
`--pair-mode all` 导出全部合格时间组合；默认 `adjacent_random` 保留全部相邻对，
每患者再选最多 8 个非相邻对。`--max-patients 100` 只限制 CXR/IV 匹配规模，
分割和 VQA 仍完整处理。日志在 `runs/build_YYYYMMDD`，验证结果在
`runs/validate_YYYYMMDD/validation.json`，均不进入 Git。

## 匹配与采样规则

- 用精确 `subject_id` 连接 CXR 和 IV。默认要求被选图像的采集时间各自唯一落在
  同一住院区间，起点取 `min(edregtime, admittime)`，终点取 `dischtime`，边界包含。
  这比旧的“唯一共同住院”规则更保守：任一端本身存在住院歧义也会拒绝。
- 保留完整病人时间线；相邻配对不会跳过不可用的中间检查。按检查最早采集时间排序，
  拒绝并列时间和重叠的采集窗口。时间差使用实际选中图像的时间。
- 两端必须有相同 AP/PA 投照方向；共同 PA 优先。每个方向使用 Atlas 原有的
  最大尺寸图像选择规则，DICOM ID 用于打破并列。
- 默认间隔为 1 小时至 365 天；通过 `--min-gap-hours`、`--max-gap-days` 调整。
  `adjacent` 仅相邻；`random` 仅随机非相邻；`adjacent_random` 合并两者；
  `all` 导出全部合格的时间顺序组合。
- 随机子集在全部合格非相邻组合中按固定种子和标识的散列选择，无重复、可复现，
  内存中只保留每病人的有限候选。随机选择不依赖未来标签值或报告内容。
  时序监督要求两端存在非空报告和标签记录；标签缺失/不确定状态仍保留。
- `--linkage patient` 可显式允许同 IV 病人的跨住院/未匹配住院配对；默认不开启。
  所有检查的匹配状态和候选住院/ICU ID 保存在 `study_links.jsonl` 中。
- 使用官方 CXR 病人划分。VQA/分割的既有留出病人优先级为
  `human_test > test > validate > train`，冲突的分类/时序行被删除而不移动到测试集。
  MedWorld 再对所有任务做全局隔离检查。清单统计是导出数；验证结果是最终可用数。

## 清单与 MRI 解码

```text
medworld_0922/
  manifest.json                 # 规则、来源、计数、文件散列
  classification.jsonl          # 13 项分类标签
  segmentation.jsonl            # 所有人工分割：CXR 图像行 + MRI 切片行
  mri_volumes.jsonl              # 1,192 次 MRI 检查，保留四序列与缺标注状态
  mri_segmentation.jsonl         # 1,190 个体积的轴位切片引用
  study_links.jsonl             # CXR/IV 回溯审计
  temporal/
    observations.jsonl
    train.jsonl
    validate.jsonl
    test.jsonl
  images -> 原始 MIMIC-CXR/files
  iv -> 原始 mimic-iv-3.1
  vqa -> 官方 CXR-VQA/dataset
  segmentation/
    heart_lung_human -> 中央目录中的公开人工心肺标注
    montgomery -> 原始人工肺掩膜和图像
  mri/
    ucsf -> 中央 UCSF-ALPTDG
    mu -> 中央 MU-Glioma-Post
```

当前 MRI 模型输入为 **T1ce 二维轴位切片**；T1/T2/FLAIR 的原始文件引用保留在
体积清单，尚未作为四序列联合输入。图像和掩膜统一到 RAS 方向，保留物理像素比例，
放入 512×512 图像画布，掩膜最近邻缩放至 256×256。强度窗由输入图像的正值体素
1%/99.5% 分位数决定。切片范围只依赖图像，并检查没有遗漏范围外的标注前景；
范围内的无肿瘤切片也保留。默认 `--mri-axial-stride 1` 使用该范围内全部轴位平面。

六个输出通道固定为 `lungs, heart, NETC, SNFH, ET, RC`。CXR 仅监督前两通道，
MRI 仅监督后四通道，Montgomery 仅监督合并肺通道；未标注通道及填充区不进入损失。
新配置按三个训练数据集均衡抽样，MRI 在随机体积块内遍历切片，避免 CXR 被大量
MRI 切片淹没，也避免每张切片都重新解压体积。解码只用有限内存缓存。

分类保留 13 项 CheXpert 标签（去除 `No Finding`），值为 `-2/-1/0/1`，
缺失和不确定标签在损失中屏蔽。时序配对用真实时间差，支持反向负时间差样本。
IV 关联用于回溯，目前不作为 EHR 模型输入。分类与分割只接收图像，VQA 接收图像和
问题；时序 source-only 接口不读取目标图像或报告。

## MedWorld 配置与评估

新训练使用 `code/medworld/configs/medworld_0922.json`，配置了 `segmentation_channels=6`、
`segmentation_sampling=balanced_dataset` 和通用影像提示词。
六通道分割头不能直接续训旧三通道分割头检查点。

```bash
$PYTHON -m medworld.run_experiment \
  --config code/medworld/configs/medworld_0922.json \
  --gpus 1,2 --out code/medworld/runs/paired_YYYYMMDD
```

本次仅构建和验证数据，没有启动正式训练。完成训练后，slots 与 no-slots 两分支均须
评估分类、VQA、人工 CXR 心肺、UCSF/MU MRI 和 Montgomery 人工肺分割。
按数据集分别报告 mean IoU、Dice：MRI 在 256×256 评估网格上先跨切片汇总每个
体积的交并计数，再按体积平均；CXR 按图像平均。总体指标按数据集宏平均。
Native Qwen 没有分割头，分割结果为 N/A。

## 2026-09-23 本机生成与验证结果

全量构建和 MedWorld 实际加载检查已完成。全局患者隔离成立；200 张人工 CXR、
1,190 个 MRI 标注体积和 138 张 Montgomery 均保留，没有分割行因跨任务冲突被移除。

| 数据（计数单位） | Train | Validate | Test | Human test |
| --- | ---: | ---: | ---: | ---: |
| 分类（图像） | 143,592 | 1,255 | 2,475 | — |
| VQA（问题） | 280,441 | 72,825 | 13,793 | — |
| 时序（正向配对） | 104,967 | 930 | 1,848 | — |
| 人工心肺（CXR 图像） | 142 | 29 | 29 | — |
| UCSF（标注体积） | 416 | 88 | 92 | — |
| MU（标注体积） | 433 | 80 | 81 | — |
| Montgomery（CXR 图像） | — | — | — | 138 |

MRI 共引用 165,417 个二维轴位平面；没有把这些切片数当成独立标注体积数。
107,745 个正向配对由 64,800 个相邻对和 42,945 个随机非相邻对组成。
患者隔离移除了 9,590 条 VQA 训练问题和 742 条验证问题，测试问题保持 13,793 条。

验证遍历全部时序配对，抽查每个分割来源和划分的实际解码，并完成混合 CXR/MRI
六通道分割头的 CPU 前后向检查。没有载入基础模型或启动正式训练。完整结果见
[validation.json](runs/validate_20260923/validation.json)。

源代码测试：完整测试集 128 项和 35 项子测试通过；随后增加的两项软链接发布/回滚
测试及修改后的适配器测试均通过。
