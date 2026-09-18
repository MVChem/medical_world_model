# MIMIC Atlas

MIMIC-CXR × MIMIC-IV 的本地数据浏览项目：**React + Vite 前端、FastAPI 后端、原始 CSV 位置索引**。白色简洁界面，支持全量患者目录、完整检查时间线、影像对比、报告、临床原始记录、趋势图和离线 HTML。

## 启动与开发

首次安装并构建前端：

```bash
cd /home/data2/chk/workspace/2026/08/04/medical_world_model/code/mimic_atlas/frontend
npm ci --include=dev
npm run build
```

从 `code/` 启动后端；FastAPI 同时提供构建好的页面：

```bash
cd /home/data2/chk/workspace/2026/08/04/medical_world_model/code
# 新 Python 环境才需要安装；本机使用共享 .venv
.venv/bin/python -m pip install -r mimic_atlas/requirements.txt
.venv/bin/python mimic_atlas/setup_data.py --source /home/data2/chk/data/MIMIC
.venv/bin/python -m mimic_atlas --host 127.0.0.1 --port 8767
```

页面：<http://127.0.0.1:8767>；API 文档：<http://127.0.0.1:8767/api/docs>。开发前端时另开终端，在 `frontend/` 运行 `npm run dev`，访问 Vite 输出的地址；`/api` 自动代理到后端 8767。生产构建和离线 HTML 都不需要 Node 服务，浏览页面无需 CDN。

远程查看可使用 `ssh -L 8767:127.0.0.1:8767 your-server`。本机生产实例由用户服务 `mimic-atlas.service` 管理；状态、重启分别使用 `systemctl --user status mimic-atlas.service`、`systemctl --user restart mimic-atlas.service`。Python 日志写入 `runs/atlas_YYYYMMDD/server.log`。

CLI 默认选择目录名最新、完成且版本兼容的 `runs/patient_index_*/manifest.json`；也可通过 `--index-root` 指定。配置索引后，源文件变化或索引不完整会明确报错，不会悄悄回退扫描。未配置索引的旧版源表读取接口仍保留用于小数据测试；全库浏览应先准备下面的索引。

## 全局索引：只存位置，不存临床数据副本

本机 MIMIC-IV 已解压为 31 个 CSV，合计 90.52 GiB。最大的 `icu/chartevents.csv` 为 41.94 GB（39.06 GiB）。按用户要求，完整解压校验后已删除对应 `.gz`；普通 CSV 可以按字节位置直接读取患者记录。

```bash
# 从 code/ 运行；构建需要 g++，不需要 GPU
.venv/bin/python -m mimic_atlas.prepare_index \
  --output mimic_atlas/runs/patient_index_20260918_offsets --workers 2
.venv/bin/python -m mimic_atlas.verify_index \
  --index-root mimic_atlas/runs/patient_index_20260918_offsets
.venv/bin/python -m mimic_atlas \
  --index-root mimic_atlas/runs/patient_index_20260918_offsets --port 8767
```

构建扫描原始 CSV 一次。C++ 流式扫描器识别完整 CSV 记录，包括引号、逗号、引号内换行、CRLF 和末行无换行；只输出患者 ID、字节区间和记录数。同一患者分散在多个位置时保留多个区间。Python 在 SQLite 中建立患者查询索引，不改写源文件。

索引目录内包含：

| 文件 | 内容 |
|---|---|
| `tables/<表名>/subjects.sqlite` | `subject_id → start_byte, end_byte, row_count`；不含事件字段 |
| `tables/<表名>/metadata.json` | 字段名、源文件指纹、全表计数和抽样校验值 |
| `directory.sqlite` | 每个患者在哪些表中出现、各表记录数 |
| `cxr.sqlite` | 患者 / 检查 / 影像 ID、影像和报告相对路径、文件可用状态 |
| `manifest.json` | 源目录、版本、表目录、归集规则和构建结果 |
| `progress.json`, `build.log` | 构建进度和运行日志 |

**不保存 CSV/Parquet 临床记录副本、报告正文或影像像素。** 打开患者时通过 `seek()` 读取原始 CSV 的对应区间，解析完整字段并逐行验证患者归属。报告和影像从原始路径读取；小型 CXR 元数据和字典在服务启动时读入内存。已打开患者的解析结果在有界内存缓存中复用；淘汰、闲置过期或重启后重新读取。

当前全库索引约 **354 MiB（0.35 GiB）**，旧版 Parquet 数据副本缓存为 11.47 GiB，减少约 97%。三个患者各 24 张临床表的原始记录读取，三轮交错测试得到的每表中位数之和：

| 患者 | CSV 位置索引 | 旧 Parquet 缓存 |
|---|---:|---:|
| 10004606 | 206 ms | 1,019 ms |
| 12137189 | 224 ms | 1,038 ms |
| 10000032 | 68 ms | 334 ms |

这是**操作系统文件缓存已热时**的 Python 读取与解析测试，不等于首次冷盘读取或整个页面的加载时间。每轮逐表比较了原始字段、顺序和重复记录。详细结果保留在 [索引运行目录](runs/patient_index_20260918_offsets/README.md)。

构建可按完成的表恢复，`--workers 1` 可降低并发读盘。源 CSV 的实际路径、字节数和纳秒修改时间必须与索引一致；源表改变应重建到新目录。指纹不是整文件密码学校验。报告正文不在索引中，读取时取源文件当前内容；新增加的检查/影像需重新构建目录并重启服务。

## 内存管理

患者数据采用统一的 LRU 与闲置过期策略：默认最多保留 **4 位患者**，估算预算 **512 MiB**，最后一次访问后 **300 秒**自动释放；清理线程每 10 秒检查一次，即使没有新请求也会回收。基础临床表、报告/配对详情、扩展表原始行、归一化事件、字段/状态和任务引用一起移除。重新访问时从位置索引重新读取全部记录。

每次加载都有独立的有效标记。淘汰时取消排队任务，正在执行的旧任务即使随后完成也不能写回；再次打开同一患者不会收到旧任务结果。HTTP 响应和导出期间暂时保护正在使用的数据。超过人数上限且所有位置均在使用时返回可重试的 503。字典在所有患者间共享，避免重复持有整份 ICD 字典。

图像缓存只保存编码后的预览字节，最多 **64 MiB / 512 张**，同样闲置 300 秒释放。缓存键包含原图文件指纹、尺寸、质量和格式，源图改变后不会命中旧预览；原始/解码图片不驻留，最多同时进行两次解码。所有缓存只在内存中，不写临床数据或缩略图副本到磁盘。

患者预算是 Python 对象占用估算，**不是整个服务的 RSS 上限**：全库目录和共享字典是固定开销，读取/导出还有临时对象。单个超大患者或正在返回的响应可暂时超过预算，以保证记录完整；不会截断患者数据。释放对象后空间可被 Python 重用，操作系统显示的 RSS 不保证立即等量下降。

```bash
.venv/bin/python -m mimic_atlas --patient-cache-count 4 \
  --patient-cache-mib 512 --image-cache-mib 64 --cache-idle-seconds 300
```

`GET /api/memory` 提供缓存人数、估算字节数、图片缓存实际字节数、命中和淘汰次数。浏览器在服务缓存过期后自动等待该表重新加载。

## 导出产物与项目改名

项目目录和 Python 包已统一为 `code/mimic_atlas`，启动使用 `python -m mimic_atlas`。下游训练、病例审阅脚本和文档引用已同步更新。

原根目录中的 `example_output`、`linked_output`、`representative_output_10`、`representative_linked_output_10` 已完整迁到：

```text
runs/exports/legacy_20260918/
├── example_output/
├── linked_output/
├── representative_output_10/
└── representative_linked_output_10/
```

迁移逐文件校验内容与软链接，原始 JSONL/CSV、输入/目标契约和归档 HTML 均保留；历史 summary 中的路径作为生成时的来源信息保持不变。新 React **“导出记录”** 页面列出批次与文件，片段链接打开同一患者的当前影像/临床工作区，原文件可下载。

CLI 的新默认输出为 `runs/exports/generated_YYYYMMDD/cxr` 与 `linked`。`runs/exports/` 下一层或两层中含 `summary.json` 的导出批次会自动出现在该页面。导出清单流式分页读取，不在内存中长期缓存整份训练数据。用户显式生成的导出文件与可自动淘汰的浏览缓存分别管理。

## 患者归集与配对规则

规则写在 `prepare_index.py` 和索引清单的 `cxr.rules` 中：

- 只按完整、精确的 `subject_id` 归集患者，不根据时间接近或影像相似混合患者。
- 检查由 `(subject_id, study_id)` 标识，影像由 `dicom_id` 标识；冲突归属阻止发布索引。保留全部检查和投照体位，即使没有 split、标签、报告或本地影像。
- 报告从该患者目录中的 `s{study_id}.txt` 定位；缺失、空报告分别显示。
- IV 原始字段保持不变。缺失 `hadm_id` 的记录仍属于该患者，但不猜测住院；eMAR/POE 明细只使用同一患者的唯一父记录补充时间与住院归属。
- **索引只负责找全 A、B、C，不决定 AB、BC、ABC 或 AC。** 现有“影像配对”页是独立的相邻检查视图，不影响全量数据。

现有配对视图沿完整时间线选择严格相邻、同 AP/PA 的检查，间隔 1 小时至 365 天，不跨过中间检查。双方真实采集时间同时落在唯一住院区间 `[min(edregtime, admittime), dischtime]` 时才关联住院；未匹配和有歧义均保留。`study_id` 不是 `hadm_id`。

## 数据接入与全量范围

`setup_data.py` 校验后建立可重复使用的软链接；路径冲突时停止：

```text
code/data/
├── MIMIC        -> /home/data1/data/MIMIC
├── MIMIC_CXR    -> /home/data1/data/MIMIC/MIMIC_CXR
└── mimic-iv-3.1 -> /home/data1/data/MIMIC/mimic-iv-3.1
```

本机 `/home/data2/chk/data/MIMIC` 指向 `/home/data1/data/MIMIC`。普通读取接口支持 `.csv` 和 `.csv.gz`；新位置索引要求 `.csv`。可用 `--cxr-root`、`--iv-root`、`--pairs` 覆盖路径和精选入口。

| 范围 | 数量 |
|---|---:|
| CXR ∪ IV 患者 | 368,138 |
| CXR / IV 患者 | 65,379 / 364,627 |
| 同时有 CXR 和 IV | 61,868 |
| 仅 CXR / 仅 IV | 3,511 / 302,759 |
| CXR 检查 / 胸片 | 227,835 / 377,110 |
| 相邻影像候选配对 / 唯一共同住院配对 | 110,729 / 70,281 |
| train / validate / test 同住院配对 | 68,529 / 593 / 1,159 |

首页总览展示全库摘要；患者与配对目录提供搜索、筛选、分页和完整清单 CSV，不自动选择患者。以上是全库可用候选，未应用具体任务抽样，**不等于最终训练清单**。精选示例单独保留 7 位患者和 10 个注释片段。

进入患者后自动读取 **24 张患者级临床表**：7 张住院/ICU 背景表和 17 张扩展表，无需逐项加载。表清单中的源文件大小是服务器上的全库 CSV 大小，不是该患者的记录大小。仅 IV 的患者同样可查看所有相关记录。

| 分组 | 表 |
|---|---|
| 住院与 ICU 背景 | `admissions`, `transfers`, `diagnoses_icd`, `procedures_icd`, `icustays`, `procedureevents`, `inputevents` |
| 检验与生命体征 | `labevents`, `chartevents`, `outputevents`, `ingredientevents` |
| 微生物与常规测量 | `microbiologyevents`, `omr` |
| 处方、给药与药房 | `prescriptions`, `emar`, `emar_detail`, `pharmacy` |
| 医嘱与其他记录 | `poe`, `poe_detail`, `datetimeevents`, `services`, `hcpcsevents`, `drgcodes`, `patients` |

字典从原始数据补充项目和编码含义；`provider` / `caregiver` 是工作人员目录，不是患者记录表。

## 浏览和导出

- 首页为独立的“数据总览”（`#overview`），集中展示简介、全库规模、覆盖、划分和配对审计。“全量患者”（`#patients`）、“影像配对”（`#pairs`）和“精选示例”（`#featured`）只展示各自的目录与筛选。点击总览覆盖分组或 split 可跳转到对应的已筛选目录，刷新后仍保留该入口的筛选条件。
- “数据集简介与统计口径”说明 CXR/IV 内容、按患者去重后的并集与交集、患者/住院/检查/影像标识，以及可浏览候选与实际训练集的区别，附官方说明链接。在独立总览中默认展开，患者页默认折叠；统计来自当前全库目录，新导出的离线 HTML 保留导出时统计并明确其实际内容范围。
- 全部检查展示在患者时间线上；点击单次检查可查看所有投照，选择配对可比较两次影像、报告和四态 CheXpert 标签。
- 影像默认 512 px，可切换 960 / 1400 px；点击放大读取 1800 px。同步缩放、亮度、对比度、反相和拖动不修改原始影像。
- 临床背景页展示住院、ICU、病区、操作/输入事件和 ICD 编码，并提供 7 张原始表的分页、搜索、完整字段和 CSV。
- 扩展表支持患者全部记录、指定住院或影像前后 0 / 6 / 24 / 72 小时。项目按 `itemid + 原始单位` 分组；散点图只显示观测值，不插值、不换算单位。日期级、文本、界限值和无时间记录仍在明细中。
- 原始记录 CSV 导出包含全部筛选结果，不受分页和绘图采样限制；超过 1,500 个绘图点时明确显示抽样状态。
- JSON 导出患者浏览数据和加载概况。离线 HTML 内嵌当前配对、两次检查的可用影像（最多 1400 px）、报告、标签、临床背景，以及导出时已加载表中前后各 24 小时的完整记录；可断网直接打开。未嵌入表明确标记，单次检查不提供片段 HTML。

页面使用可取消的 API 请求。切换患者后旧响应不能覆盖新患者；临床背景稍后完成时更新住院选项，保留已选表和范围。图片以外的数据不再施加过小的传输限制，列表和原始记录仍分页以便浏览。

随访影像、报告及 IV 回顾性记录不是当前状态预测输入。ICD 诊断是出院编码，操作编码只有日期精度；同期治疗不能解释为影像变化的原因。原有 `forecast_inputs.jsonl` 输入边界保持不变。

## 项目结构

```text
mimic_atlas/
├── frontend/
│   ├── src/App.jsx                 # 应用、导航与目录加载
│   ├── src/components/             # 患者目录、查看器、报告与临床记录
│   ├── src/api.js                  # API、取消请求、导出
│   ├── src/charts.js               # 本地 SVG 时间轴与趋势绘制
│   ├── src/styles/                 # 白色设计样式
│   ├── package.json, package-lock.json
│   └── vite.config.js              # 开发代理与生产构建
├── backend/
│   ├── app.py                      # FastAPI 生命周期、CLI、页面交付
│   ├── api.py                      # 患者、表、影像与导出 API
│   ├── schemas.py                  # 请求与查询参数校验
│   ├── images.py                   # 原始影像读取与缩放
│   └── frontend.py                 # React 资源与离线 HTML 内嵌
├── app.py, __main__.py              # FastAPI 工厂与模块启动入口
├── data.py, cohort.py              # 患者数据、全库统计和关联
├── memory.py                      # 统一患者 LRU/TTL 与有界图像缓存
├── exports.py                     # 显式导出批次、分页片段与下载目录
├── clinical_tables.py              # 扩展表解析、筛选与数值序列
├── patient_index.py                # SQLite 查询与原始 CSV 定位读取
├── csv_spans.cpp, prepare_index.py  # 一次扫描建立位置索引
├── verify_index.py                 # 独立一致性校验和 API 验收
├── setup_data.py                   # 数据软链接
├── build_mimic_transitions.py       # 原有训练配对和显式导出
├── link_mimic_iv_context.py         # IV 读取与临床关联
├── tests/, browser_check*.py        # 合成数据及真实浏览器检查
└── runs/                           # 忽略的索引、日志和显式导出
```

保留训练数据准备和静态导出功能，命令与导入统一使用 `mimic_atlas`，详见 [配对与训练说明](docs/transition_pipeline.md)。`runs/`、源数据、软链接、`node_modules/` 和 `frontend/dist/` 均不进入 Git。

## 验证

先构建前端，再从 `code/` 运行：

```bash
.venv/bin/python -m pytest mimic_atlas/tests -q
.venv/bin/python -m mimic_atlas.browser_check --include-inputs
.venv/bin/python -m mimic_atlas.browser_check_cohort
.venv/bin/python -m mimic_atlas.browser_check_clinical
.venv/bin/python -m mimic_atlas.browser_check_react
.venv/bin/python -m mimic_atlas.browser_check_memory
.venv/bin/python -m mimic_atlas.verify_index \
  --index-root mimic_atlas/runs/patient_index_20260918_offsets \
  --url http://127.0.0.1:8767
```

测试依赖见 `requirements-dev.txt`；浏览器测试使用系统 Chrome/Chromium 或 Playwright Chromium，截图和结果写入 `runs/`。测试覆盖乱序/多段患者记录、多行 CSV、重复与缺失字段、患者隔离、源文件变化、自动加载、分页导出、延迟响应、临床背景迟到、离线 HTML 和 390px 手机布局。内存检查还覆盖人数/字节预算、无请求时的闲置释放、淘汰后的原始记录一致性、过期任务不能回写、已打开页面自动恢复、图片复用和旧导出导航。

## 设计与技术参考

沿用白色 Figma 设计参考：[shadcn/ui design system](https://www.figma.com/community/file/1203061493325953101)、[官方 Figma 目录](https://ui.shadcn.com/docs/figma)、[浅色 Dashboard](https://ui.shadcn.com/examples/dashboard)。此前 Figma 社区页在本机返回 403，视觉核对使用同体系官方预览；没有通过 Figma API 获取节点。界面现由 React 实现，不依赖付费设计资产。

结构参考：[React 自建应用](https://react.dev/learn/build-a-react-app-from-scratch)、[Vite](https://vite.dev/guide/)、[FastAPI 多文件应用](https://fastapi.tiangolo.com/tutorial/bigger-applications/)。数据与字段：[MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)、[MIMIC-IV](https://physionet.org/content/mimiciv/3.1/)、[检验](https://mimic.mit.edu/docs/iv/modules/hosp/labevents.html)、[ICU 记录](https://mimic.mit.edu/docs/iv/modules/icu/chartevents.html)。

概览页的检查间隔柱状图参考 [shadcn/ui Bar Charts](https://ui.shadcn.com/charts/bar) 的简洁卡片、直接数值标签与低对比网格。使用原生 CSS 绘制，从全库候选配对实时汇总六个时间段（左闭右开，末段含上限）；按配对计数，不按患者去重。

概览底部使用三张紧凑条形图：全部配对间隔、1 ≤ 间隔 < 24 小时的六段分布、全部 CXR 患者的独立检查次数分布（同一 study 的多张图像只计一次）。前两图按配对计数，第三图按患者计数，百分比分母分别显示在卡片右上角。
