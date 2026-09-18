# Glioma Atlas

UCSF-ALPTDG 和 MU-Glioma-Post 的本地 HTML 可视化项目。代码位于项目已有的 `code/` 目录。

- **完整页面**：[http://127.0.0.1:8766](http://127.0.0.1:8766)。服务运行在数据所在机器；远程访问可转发 8766 端口。
- **完整 HTML**：[code/data/glioma_explorer/runs/glioma_atlas.html](../data/glioma_explorer/runs/glioma_atlas.html)，入口约 1.17 MB，覆盖全部 501 位患者；按需读取旁边的 `full_data/`，无需 Python 服务，也可直接在聊天文件预览中运行。下载到其他机器时需一起保留整个目录。
- **预览**：[UCSF 桌面页面](../data/glioma_explorer/runs/desktop.png)、[MU 页面](../data/glioma_explorer/runs/full-mu.png)、[手机布局](../data/glioma_explorer/runs/full-mobile.png)。

## 实际可浏览的内容

两个数据集均已完整下载、校验。完整 HTML 已读取全部 **7,746 个 NIfTI**：501 位患者、1,192 个 MRI 时间点、4,768 个标准序列体积、1,190 份纵向分割及 1,788 个 UCSF 附加差分／标签文件。

| 数据集 | 影像 | 表格与图表 |
|---|---|---|
| UCSF-ALPTDG | 全部 298 人的双时间点 T1、T1 CE、T2、FLAIR；轴位／冠状位／矢状位；同步切片；分割开关和透明度 | 596 次检查；年龄、诊断、性别、分级、随访间隔；由原始 mask 计算的四分区体积；患者搜索和 CSV 导出 |
| MU-Glioma-Post | 203 人、596 个时间点的四序列 MRI；594 份分割；按原始时间点切换比较，保留单次检查和缺失标注 | 临床资料与影像编号匹配、随访时间线、扫描仪分布、分割体积与搜索／CSV 导出 |

完整 HTML 可以选择全部患者、所有本地时间点、四种 MRI 序列、轴位／冠状位／矢状位以及每一层切片；分割开关和透明度均可调整。MU 非连续时间点、单次检查及缺失 mask 均保留。此前只含两例、336 张采样切片的 HTML 已被替换。

“全部临床字段”展示所选患者的原始记录。“全部影像文件”可查看该患者的每个 NIfTI，包括 UCSF T1 CE−T1、纵向差分和差分分割。“原始表格”提供四个原始工作簿的全部 **12 个工作表**，保留标题行、字段编码、所有行列，支持搜索、分页和 CSV 导出。

HTML 本体不内嵌全体影像，`full_data/` 按患者／文件保存完整体积显示资源，共约 **10.95 GB**，是完整离线 HTML 的配套交付文件。打开页面只加载当前所需的几个体积，浏览器最多缓存八个体积，避免一次加载全部数据。实体仍在 `/home/data2/chk/data/glioma_explorer/runs/`，统一从 `code/data/glioma_explorer/runs/` 的符号链接入口访问。


## 启动

从仓库根目录执行，已验证环境为 Python 3.12：

```bash
PYTHONPATH=code /home/data2/chk/workspace/2026/.venv/bin/python \
  -m glioma_explorer.app --port 8766
```

默认只监听 `127.0.0.1`，不上传影像，也不需要 GPU。HTML、CSS、JavaScript 和 SVG 图表均在本地提供，无 Node 构建步骤、远程字体或图表 CDN。其他环境可安装本目录的 `requirements.txt`。

本次已启动用户级后台服务 `glioma-atlas.service`，可检查或停止：

```bash
systemctl --user status glioma-atlas.service
systemctl --user stop glioma-atlas.service
```

默认数据目录为 `/home/data2/chk/data`；可在启动前设置 `GLIOMA_DATA_ROOT`。**数据及生成文件只保存在数据目录，源码目录用符号链接引用。** 本机布局为：

```text
data/
  UCSF-ALPTDG/
    100001/
      100001_time1_t1ce.nii.gz
      ...
    ...
    UCSF_PostopGlioma_Table S1 R1 V5.0_UNBLINDED_FINAL.xlsx
    extraction_manifest.json
  MU-Glioma-Post/
    PatientID_0003/Timepoint_1/     # 四序列和 tumorMask，原始编号
    ...
    image_manifest.json           # 每个文件的 SHA-256、大小与 NIfTI 头信息
    matching_audit.json            # 影像与临床时间、mask 缺失审计
    MU-Glioma-Post_ClinicalData-July2025.xlsx
    MU-Glioma-Post_Segmentation_Volumes.xlsx
    MR_Scanner_data.xlsx
  glioma_explorer/
    runs/                         # HTML、截图、检查报告、参考预览
      glioma_atlas.html            # 完整入口
      full_data/                  # 全体积缓存与所有工作表，按需加载
      full_preview_status.json    # 全量原文件读取／缓存生成记录
      full_browser_checks.json    # 全部患者浏览器加载检查
```

项目统一在 `code/data/` 下建立目录符号链接，分别指向上述两个数据集及 `glioma_explorer`。HTML 的推荐入口为 `code/data/glioma_explorer/runs/glioma_atlas.html`；HTML 与旁边的 `full_data/` 一起通过目录链接访问，保证相对资源路径有效。原有 `code/glioma_explorer/runs` 入口也保留可用。仓库不保存数据实体或这些机器本地链接，导出和浏览器检查仍直接写入数据目录。在新工作区建立入口：

```bash
mkdir -p code/data
ln -s /home/data2/chk/data/UCSF-ALPTDG code/data/UCSF-ALPTDG
ln -s /home/data2/chk/data/MU-Glioma-Post code/data/MU-Glioma-Post
ln -s /home/data2/chk/data/glioma_explorer code/data/glioma_explorer
```

UCSF 直接从解压目录读取 NIfTI，运行时不依赖 ZIP。有界内存缓存服务近期病例，串行限制 MRI 解码的内存峰值。09-18 按用户要求解压外层 ZIP，保留标准 `.nii.gz` 影像格式；每个文件独立重读，核对原 ZIP 的大小、CRC32，并记录 SHA-256 到 `extraction_manifest.json`。确认浏览器正常后删除原始 ZIP，下载来源与原 ZIP 校验记录保留在数据目录中。若更新本地数据，重启服务以重建表格缓存。MU 通过原始 PatientID／Timepoint 文件名映射到临床表；`t1c/t1n/t2f/t2w` 分别显示为 T1 CE／T1／FLAIR／T2。时间点编号不重排，也不强制从 T1 开始；缺少 mask 时仅显示 MRI，体积为缺失值。

在其他机器首次准备 UCSF 时，可先运行以下命令完成解压与校验；它不会自动删除源 ZIP，也不会覆盖已有目录：

```bash
PYTHONPATH=code /home/data2/chk/workspace/2026/.venv/bin/python \
  -m glioma_explorer.extract_ucsf --archive /path/to/UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0.zip \
  --destination /home/data2/chk/data/UCSF-ALPTDG
```

## MU 下载方式与来源

官方入口为 [TCIA MU-Glioma-Post](https://www.cancerimagingarchive.net/collection/mu-glioma-post/) 的 Aspera 下载。本机公开授权成功，但 SSH 传输连接失败；09-18 找到 [sbandred/mu-glioma-post-raw](https://huggingface.co/datasets/sbandred/mu-glioma-post-raw) 第三方公开镜像，改用 HTTPS。

锁定 revision `f6acd4d7d19d35304dc4317d9af4bd25094eb6b9`，共 **2,978 个 NIfTI、11,890,059,719 bytes（11.89 GB／11.07 GiB）**。已逐目录核对 TCIA 的 596 个时间点，所有文件名和字节数与镜像一致；两个镜像表格的 SHA-256 也与已下载的官方表格一致。下载器对每个影像校验镜像 LFS SHA-256、gzip CRC 和 NIfTI 头。未取得官方逐文件哈希，因此不把这些检查表述为与官方副本逐字节比对。

```bash
# 可重复运行：跳过已通过校验的文件，续传 .part；直接保存 NIfTI，无外层 ZIP。
PYTHONPATH=code /home/data2/chk/workspace/2026/.venv/bin/python \
  -m glioma_explorer.download_mu --workers 8

# 本轮后台下载状态
cat /home/data2/chk/data/.mu_glioma_post_download/status.json
systemctl --user status mu-glioma-post-download.service
```

下载状态以 `status.json` 的 `phase`、`verified_files`、`verified_bytes` 为准，目录占用不代表校验完成。下载成功后生成 `MU-Glioma-Post/image_manifest.json`。本轮同时运行 `mu-glioma-post-finalize.service`：等待完整校验，然后检查全部本地路径／大小、运行数据测试、重建 HTML、重启页面和运行浏览器检查；最终结果写入 `.mu_glioma_post_download/finalization_status.json`，日志在同目录 `finalize.log`。单独复跑下载脚本时，完成后可运行 `python -m glioma_explorer.finalize_mu`，或手动重启页面服务以读取完整目录。文件保持原始 `.nii.gz` 格式，不产生需要清理的外层压缩包。

## 可视化口径

- **方向与空间**：NiBabel 转为 RAS+，采用神经学显示。完整 HTML 统一使用该患者首个本地时间点的网格，时间点选择不重新定义参考网格。轴位左侧标 L、右侧标 R；冠状位上 S 下 I；矢状位左 P 右 A。若网格不同，将显示数据重采样到首次分割的网格；掩膜只使用最近邻插值。显示考虑体素物理间距。
- **亮度**：每个标准 MRI 按正值强度的第 1–99.5 百分位设置显示窗。完整 HTML 保存该显示窗下的 8 位全体积缓存，保留所有空间体素与切片；它是显示数据，不能替代原始 NIfTI 用于定量强度分析或训练。标签按原始离散整数保存。差分图以绝对强度第 99.5 百分位设置对称窗，保留正负变化的显示。
- **默认切片**：选择首次分割中标签 1、2、3 总面积最大的切片，排除切除腔；不依据未来掩膜选取默认切片。切换切面后重新选择该切面的源时点默认层。
- **体积**：直接计数原始分割值 1 / 2 / 3 / 4，乘以仿射矩阵确定的体素 mm³，再除以 1,000 得到 mL。UCSF 对应 NCR、SNFH、ET、RC；MU 首类为 NETC。读取并检查 NIfTI 的 mm 单位，不把面积当体积，也不沿用临床表含义容易混淆的 `WT` 合计列。原始 mask 无该标签时体积为零。
- **UCSF 限制**：公开影像已配准到第二次扫描；本项目是回顾性数据浏览器，不是前瞻评测协议或临床预测系统。分区体积变化不直接等于临床进展。
- **MU 时间**：597 个数值型临床 MRI 时间项、654 条扫描仪表记录、官方摘要 596 个影像时间点分别保留。395 个候选相邻对来自数值天数排序、去重；其中 389 对有两端影像、387 对有两端 mask。593 个影像时间点匹配数值临床日期，另有 3 个影像时间点缺日期；4 个临床时间项无对应影像。详细 ID 在数据目录的 `matching_audit.json`。没有数值时间的患者仍保留在患者列表。
- **MU 体积**：各分区表独立显示记录数、中位数和四分位距；没有一致时间点键，不将行号当作时间顺序，不把缺失分区补零。
- **汇总**：年龄是 UCSF 首次扫描年龄／MU 诊断年龄；缺失年龄不进入直方图。诊断类别超过六种时，前五类之外的记录合并成“其他诊断（合并）”，保持患者总数。缺失分级单列“未记录”。

## Figma 设计来源

本轮 Figma 连接器没有启用。实际查看了下列公开设计稿预览，依据其中的侧栏导航、浅色卡片、留白、柔和配色及深浅内容分区实现 HTML；未访问私人设计，也未声称导入了 Figma 节点或组件。

1. [Healthcare Dashboard](https://www.figma.com/community/file/1026733583562048041/healthcare-dashboard/)：预览由 [Figma 官方 Dashboard 模板页](https://www.figma.com/templates/dashboard-designs/) 提供。借鉴指标卡片和明暗分区。
2. [Medical Doctor Patient Dashboard](https://www.figma.com/community/file/1433514985260691679/medical-doctor-patient-dashboard-template)，Reza Al Hasan：查看了作者的 [Dribbble 原始展示](https://dribbble.com/shots/25135884-Doctor-Patient-Medical-Live-Dashboard)。借鉴侧栏、数据表和选中状态。

查看过的参考预览存放在 `runs/design_references/`。这些图片只作设计参考，未作为网页素材发布。页面保留可点击的设计来源。

## 生成与检查

完整 HTML 的生成会读取每位患者的每个原始 NIfTI，并校验压缩显示数据的无损往返；原始输入不被改写。已生成且来源未变的体积可复用。`100075` 与 `100079` 的部分原始 MRI 头未填空间单位，已核对与同病例明确标为 mm 的文件具有相同 shape／affine，缓存记录单位依据；不修改原文件，也不将缺失单位无条件当作 mm。

```bash
# 生成完整 HTML 和按需加载的全体积文件；可断点复用已生成的患者。
PYTHONPATH=code /home/data2/chk/workspace/2026/.venv/bin/python \
  -m glioma_explorer.export_full --workers 12

# 数据、方向、标签插值及实际 API 检查。
PYTHONPATH=code /home/data2/chk/workspace/2026/.venv/bin/python \
  -m unittest glioma_explorer.test_data -v

# 逐人验证完整 HTML（501 人）、全部切面、原始表格及聊天沙箱预览。
PYTHONPATH=code /home/data2/chk/workspace/2026/.venv/bin/python \
  -m glioma_explorer.check_full

# 运行服务后，用 Playwright + 本机 Chrome 检查在线交互、移动布局与断网 HTML。
PYTHONPATH=code /home/data2/chk/workspace/2026/.venv/bin/python \
  -m glioma_explorer.browser_check
```

浏览器检查另需 `playwright`，API 测试使用当前环境已有的 `httpx`。检查报告保存在 [runs/browser_checks.json](runs/browser_checks.json)，截图和 HTML 也在 `runs/`，沿用仓库规则不纳入 Git。

数据来源：[UCSF](https://imagingdatasets.ucsf.edu/dataset/2)、[TCIA MU-Glioma-Post](https://www.cancerimagingarchive.net/collection/mu-glioma-post/)。详细研究结论见 [CLARITY 数据 Research Notes](https://github.com/MVChem/medical_world_model/blob/31c98ca14173620a3cec67da2defe559d05974da/research_notes/0916_clarity_datasets_feasibility.md)。
