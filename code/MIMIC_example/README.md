# MIMIC Atlas

本地 MIMIC-CXR × MIMIC-IV 交互式浏览器。默认展示全量患者及匹配统计，包含仅有 IV 的患者。支持全库配对清单、完整检查时间线、胸片对比、报告、检验和临床大表，以及内嵌真实影像的离线 HTML 导出。界面采用白色、简洁的 Figma 设计系统风格。

## 启动

在 `code/` 目录运行，使用共享 Python 环境：

```bash
cd /home/data2/chk/workspace/2026/08/04/medical_world_model/code
.venv/bin/python MIMIC_example/setup_data.py --source /home/data2/chk/data/MIMIC
.venv/bin/python -m MIMIC_example --port 8767
```

打开 <http://127.0.0.1:8767>。远程服务器可在本地电脑建立端口转发：

```bash
ssh -L 8767:127.0.0.1:8767 your-server
```

应用默认监听 loopback。已有环境包含依赖；新环境可用 `python -m pip install -r MIMIC_example/requirements.txt` 安装。

启动默认选择 `runs/patient_index_*/manifest.json` 中目录名最新的完整索引，也可用 `--index-root` 指定。首页只发送患者摘要；打开患者后按索引读取其背景，检验等表由用户明确加载。服务器日志位于 `runs/atlas_YYYYMMDD/server.log`。没有完整索引时保留直接扫描模式；配置了索引后，缺失或过期的索引会明确报错，不会悄悄退回全表扫描。

本机实例以 `mimic-atlas.service` 用户服务运行，独立于终端会话。可用 `systemctl --user status mimic-atlas.service` 查看，`restart` 重启，`stop` 停止。重启复用磁盘患者索引；首页汇总和旧版相邻配对视图仍在内存中构建。

## 提前准备全局患者索引

本次按用户明确要求允许在 `runs/` 生成患者索引和缓存。此授权只针对本项目的数据预处理产物；原始数据不修改，影像像素不复制，生成物不入 Git。

```bash
# 在 code/ 目录运行；再次运行相同命令会跳过已完成且源指纹一致的表
.venv/bin/python -m MIMIC_example.prepare_index \
  --output MIMIC_example/runs/patient_index_20260918
.venv/bin/python -m MIMIC_example \
  --index-root MIMIC_example/runs/patient_index_20260918 --port 8767
```

预处理包括本地 IV `hosp` / `icu` 全部 CSV（患者表、字典和工作人员目录），以及本地 `mimic-cxr-*.csv*`。每张患者表按 `subject_id` 聚集；所有列保持原始字符串，并额外保存源行序号。缺失 split 的检查仍保留，不会被训练划分规则过滤。目录定位精确行区间，文件以 Zstandard 压缩。字典和不含患者号的表作为全局表保留。

归集规则在 `prepare_index.py` 中实现，并写入 `manifest.json` 的 `cxr.rules`：

- 患者归属只用完整、精确的 `subject_id`；不会按患者编号接近、时间接近或相似影像合并患者。
- CXR 检查用 `(subject_id, study_id)`，影像用 `dicom_id`；保留所有检查、所有投照体位以及 A、B、C 各自的记录。重复/冲突的影像标识会阻止发布索引。
- 报告按患者目录中的 `s{study_id}.txt` 关联，保存全文，分别记录缺失和空报告。所有影像都检查路径存在性，不在预处理时解码图像。
- IV 的 `hadm_id`、`stay_id`、`emar_id`、`poe_id` 等原值保持不变；患者归集不要求这些字段完整，也不依赖影像可配对。网页需要明细时间时只使用同一患者内唯一父记录，缺失或歧义保持显式状态。
- **索引不选择 AB、BC、ABC 或 AC，不限制相邻检查、时间窗、split 或 AP/PA。** 现有相邻配对页是沿用的派生视图，不影响完整检查目录。跨 CXR/IV 的患者号关联与具体住院时间匹配分开处理。

默认同时处理两张表，可加 `--workers 1` 降低峰值内存。`progress.json` 保存各个活动任务的阶段，`build.log` 保存耗时。每张表单独完成后原子发布；中断时已完成表可复用，正在处理的表会重做。全局 `manifest.json` 仅在全部表和 CXR 目录完成后发布。源文件以实际路径、大小和纳秒修改时间校验，源表改变时必须使用新的输出目录；这是源指纹检查，不是整文件密码学校验。报告正文是预处理时快照，更新报告后也应重建到新目录。

每张表核对源行数、缓存行数和患者目录计数；构建时对患者 `10000032`、`10004606`、`12137189` 的原始字段序列计算 SHA-256，并与索引读取结果比对。测试另外覆盖乱序输入、跨 row group、重复行、前导零、换行文本、不同患者隔离、重启持久化，以及禁止任何源表扫描的 API 读取。

产物：`tables/<表名>/*.parquet` + `subjects.sqlite`，全局 `directory.sqlite`，CXR `cxr.sqlite`（检查、影像位置和全文报告），以及可审计的 `manifest.json`。API `GET /api/patients/{subject_id}/directory` 提供全部表的患者记录数；`GET /api/patients/{subject_id}/reports/{study_id}` 按需返回该患者该次检查的完整报告。

完整验收（不重新扫描大表 CSV；检查所有行区间和全部影像归属，并可测量在线 API）：

```bash
.venv/bin/python -m MIMIC_example.verify_index \
  --index-root MIMIC_example/runs/patient_index_20260918 \
  --url http://127.0.0.1:8767
```

结果保存在索引目录的 `verification.json`，包含实际大小、全部源行数、抽样字段校验，以及新患者的读取耗时、分页传输字节数和源表扫描行数。

本机全库索引已完成并通过验收；详细构建记录、占用空间和读取实测见 [运行结果](runs/patient_index_20260918/README.md)。

## 数据接入

`setup_data.py` 校验数据后创建以下软链接，可重复运行，遇到不同目标的已有路径会停止：

```text
code/data/
├── MIMIC        -> /home/data1/data/MIMIC
├── MIMIC_CXR    -> /home/data1/data/MIMIC/MIMIC_CXR
└── mimic-iv-3.1 -> /home/data1/data/MIMIC/mimic-iv-3.1
```

本机 `/home/data2/chk/data/MIMIC` 本身指向 `/home/data1/data/MIMIC`。原始数据不复制、不修改，软链接由 `code/data/.gitignore` 排除。

可覆盖位置和精选入口：

```bash
.venv/bin/python -m MIMIC_example \
  --cxr-root /path/to/MIMIC_CXR \
  --iv-root /path/to/mimic-iv-3.1 \
  --pairs MIMIC_example/representative_pairs_10.json
```

## 浏览与导出

- **全量首页**：默认展示 CXR ∪ IV 全部患者，每页 25 / 50 / 100 条摘要。按 `subject_id`、数据覆盖、官方 train / validate / test、是否有可用配对 / 同住院配对筛选，按检查数或配对数排序。仅 IV 的患者同样可打开，无 CXR split 不被分配成 train。
- **配对清单**：全库逐患者沿完整时间线查找相邻 AP/PA，校验文件存在、标签、split 和时间间隔，再按双方实际采集时刻匹配唯一共同住院。保留未匹配和歧义状态。点击任一行打开对应患者的准确配对；患者清单、配对清单均可导出筛选后完整 CSV。
- **数据分布**：查看 split、配对时间间隔、投照体位、阳性报告标签和排除原因。展示的是全库可用候选，未应用具体训练任务的抽样、标签变化阈值或实际导出名单，不等于最终训练集。
- **精选入口**：独立“精选示例”入口保留 7 位患者、10 个带注释的配对。进入患者后，展示其全部检查和所有符合配对规则的相邻片段；精选片段用 `✧` 标记。精选案例经人工挑选，不是随机训练队列。
- **影像时间线**：点击检查浏览单次影像，可切换该检查的全部投照图片；仅选择的影像会发送到浏览器；下拉框切回相邻同投照片段。配对对比固定使用经过验证的两张图像。
- **对比显示**：同步缩放、亮度、对比度、反相、放大后拖动；大图弹窗可用 Escape 关闭。左右方向键切换片段。显示调整不是图像配准，也不改变原始文件。
- **报告 / 标签**：保留报告原文，区分阳性、阴性、不确定和未提及；报告标签变化不解释为疾病严重程度变化。
- **IV 背景**：住院、ICU、病区转移时间轴，片段内操作事件和住院 ICD 诊断 / 操作。时间轴保留脱敏日期，区间随选中的片段裁剪，悬停可查看原始时间。
- **输入 / 用药**：默认不读取患者的 `icu/inputevents` 详情。在临床背景页点击“加载输入 / 用药”，仅为当前患者按需读取；或启动时加 `--include-icu-inputs`。未加载与没有记录分别显示。数值保留原始单位，不做跨单位合并。
- **JSON**：下载当前患者的浏览数据，包含当前临床加载状态；这是研究浏览导出，不是模型输入文件。
- **离线 HTML**：IV 加载完毕后，导出当前片段的 HTML。它内嵌两个检查的可用影像、报告、标签和已加载临床背景，可直接打开，无需服务或 CDN。导出影像长边最多 1,400 px；原始分辨率仍记录在元数据中。单次检查模式不提供片段 HTML 导出。

患者目录、CXR 检查/报告与临床原始记录已由预处理脚本持久化。临床表按患者聚集，SQLite 指向 Parquet 中的确切行区间；请求只读覆盖该患者的少量 row group，再截取精确行区间并验证每行 `subject_id`。浏览器只收到当前页和有限绘图点，重启或选择新患者都不需要重新扫描源表。

临床背景页的“患者全部住院记录 · 原始表”提供原有 7 张表的分页、搜索、逐行字段和完整 CSV，不依赖患者有无 CXR 或配对匹配。检验等 17 张表在独立页签中浏览。

### 远程流量

首页不自动选患者，不请求图像或临床详情。患者目录响应使用紧凑结构，报告及完整配对只在选择检查 / 片段后返回。JSON、HTML、JS 和 CSS 开启 gzip；无 CDN、外部字体或分析脚本。影像默认长边 512 px、WebP quality 60，可选择 960 / 1400 px；点击放大单独读取 1800 px JPEG。选中以外的检查不预取，关闭临床页签时停止轮询，数据读取期间只轮询小型状态。

本机 Chromium 首次访问的实测传输（含响应头）：首页约 **45 KB**，打开非精选患者的一对胸片额外约 **23 KB**；具体取决于患者、图像和选定清晰度。主动 CSV / HTML / JSON 导出不受预览大小限制。

### 全库关联结果

本地 MIMIC-IV 3.1 / CXR-JPG 2.0.0 真实扫描结果：

| 范围 | 数量 |
|---|---:|
| CXR ∪ IV 患者 | 368,138 |
| CXR 患者 / IV 患者 | 65,379 / 364,627 |
| 同时有 CXR 和 IV | 61,868 |
| 仅 CXR / 仅 IV | 3,511 / 302,759 |
| CXR 检查 / 胸片 | 227,835 / 377,110 |
| 可用相邻影像配对 | 110,729 |
| 唯一共同住院配对 | 70,281 |
| train / validate / test 同住院配对 | 68,529 / 593 / 1,159 |

患者匹配按 ID；住院匹配按时间区间，二者分别统计。配对使用原有 `find_candidates` 规则，检查双方图像及当前报告文件存在，但未全库解码图像或逐份解析报告。IV `patients`、`admissions`、`icustays` 缺失时显示覆盖统计不完整，不伪装成完整扫描。

当前覆盖本地 `hosp` / `icu` 的 **24 张患者级表**：原有 7 张临床背景表，加上 17 张扩展表。`provider` / `caregiver` 是仅含工作人员标识的目录，不作为患者记录表；相应标识保留在原始事件中。

### 检验、生命体征与其他大表

在“**检验 · 生命体征 · 更多**”页选择数据表。支持影像前后 0 / 6 / 24 / 72 小时、指定住院或患者全部记录；数值按 `itemid + 原始单位` 分组，散点图标出当前 / 随访胸片时刻，可悬停查看原值、参考区间、原始标记和录入时间。原始记录可全文搜索、分页、逐行展开，筛选后的 CSV 导出不受分页或绘图采样限制。

| 数据 | 原始表 | 展示要点 |
|---|---|---|
| 检验 | `hosp.labevents` + `d_labitems` | 项目、标本类型、原值/单位、参考区间、异常标记、`charttime` / `storetime` |
| 生命体征 / ICU 记录 | `icu.chartevents` + `d_items` | 心率、呼吸频率、血压、血氧、呼吸机参数及全部其他记录；可按项目筛选 |
| 排出量 / 输入成分 | `outputevents`, `ingredientevents` | 保留原单位，不与 inputevents 重复求和 |
| 微生物 / 药敏 | `microbiologyevents` | 标本、菌种、抗生素、药敏解释、原始备注 |
| 常规测量 | `omr` | 保留日期级精度和复合结果原文 |
| 处方 / 给药 / 药房 | `prescriptions`, `emar`, `emar_detail`, `pharmacy` | 订单与给药分别浏览；明细按 `subject_id + emar_id` 关联父记录 |
| 医嘱 | `poe`, `poe_detail` | 订单状态、字段明细；按 `subject_id + poe_id` 关联父记录 |
| 其他 ICU / 住院记录 | `datetimeevents`, `services`, `hcpcsevents`, `drgcodes`, `patients` | 日期型事件、服务转移、编码和锚定年龄等原始字段 |

默认所有扩展表均按需读取。完整索引提供加载前的患者记录数；点击后读取该患者的数据区间，最多同时读取两张表。可用 `--preload-tables` 为精选患者预载指定表。界面区分“已建索引 · 按需读取”和“已加载”；“全库源文件 · 服务器本地”的 MiB 大小不是患者数据量或浏览器下载量。

```bash
# 一次预载所有 17 张扩展表，同时读取原有 ICU 输入事件
.venv/bin/python -m MIMIC_example --preload-tables all --include-icu-inputs
# 只在界面明确点击时加载扩展表
.venv/bin/python -m MIMIC_example --preload-tables none
# 也可指定表；明细所需父表自动先加载
.venv/bin/python -m MIMIC_example --preload-tables labevents,chartevents,emar_detail
```

预处理通过 Arrow 流式读取源 CSV，保留全部原始字符串、重复行、缺失值和患者内源记录顺序；正确处理乱序患者、引号、逗号和多行文本。界面区分索引就绪、排队、读取、没有记录和读取失败。未准备索引的直接扫描模式仍可能需要几分钟。

缺失 `hadm_id` 的记录仍保留，可在患者全量或时间窗中查看，但不自动推断住院归属；指定住院只接受原始住院号或唯一父记录提供的住院号。缺失 / 歧义父记录的明细不会猜测时刻。只有日期的记录按整日与时间窗相交，保留在明细中，不绘成精确时刻测量点。`storetime` 不被替换成 `charttime`，也不暗示当时已可用于预测。

趋势图只绘制有限数值与精确时间的观测点，不插值、不换算单位；`<` / `>` 界限值和 NaN 不当作精确测量点。在线图超过 1,500 点时保留每段极值及端点，并明确显示采样数量，原始记录和 CSV 仍完整。日期级或无时间记录可在“患者全部记录”中查看。

离线片段 HTML 同时内嵌**导出时已加载表**中胸片前后各 24 小时的完整记录和原始字段。离线界面限定为此窗口，不声称包含患者全量数据；未加载表明确标为未嵌入，无时间记录不塞入时间窗。JSON 患者导出包含扩展表加载概况，批量原始行使用各表的 CSV 导出。

本机真实扫描验证：`labevents` 158,374,764 行、`chartevents` 432,997,491 行；精选患者 `12137189` 保留 608 条检验和 18,306 条 ICU 记录。数量来自本地源表扫描，不以官方旧版本统计替代。

## 关联与预测边界

`subject_id` 连接患者；两张胸片的实际采集时间必须同时落在唯一住院区间 `[min(edregtime, admittime), dischtime]` 才关联该住院。`study_id` / `dicom_id` 不是 `hadm_id` / `stay_id`。没有匹配和多重匹配会保留，不猜测住院。

检查按完整患者时间线排序。配对要求严格相邻、同一 AP / PA 投照、时间差 1 小时至 365 天，并验证标签、当前报告和图像存在；不跳过中间不兼容的检查。时间相同、采集窗口重叠等边界沿用原有筛选规则。其他检查仍可单独查看。

未来影像、报告、标签及 IV 回顾性记录不是当前状态预测输入；ICD 是住院出院编码，ICD 操作时间只有日期精度。同期操作 / 用药不证明其导致影像变化。原有 `forecast_inputs.jsonl` 的输入边界保持不变。

## 代码结构

```text
MIMIC_example/
├── app.py                         # FastAPI、受控影像访问、JSON / HTML 导出
├── data.py                        # 目录、按选择加载、患者索引接入
├── prepare_index.py               # 全库预处理、归集规则和一致性校验
├── patient_index.py               # SQLite 行目录 + Parquet 精确患者读取
├── cohort.py                      # 全库患者并集、配对校验、住院关联和统计
├── clinical_tables.py             # 17 张扩展表、Arrow 流式读取、查询与数值序列
├── web/                           # HTML / CSS / JavaScript，无构建工具和外部 CDN
├── setup_data.py                  # 校验与创建机器本地软链接
├── build_mimic_transitions.py      # CXR 数据读取、配对、纯内存准备和显式导出
├── link_mimic_iv_context.py        # IVTables 读取、可复用内存关联和显式导出
├── forecast_contract.py           # 模型输入契约
├── curated_pairs.json
├── representative_pairs_10.json
├── tests/                         # 合成数据测试，不依赖真实 MIMIC
├── browser_check.py               # 真实数据桌面 / 手机 / 离线交互验证
├── browser_check_cohort.py        # 全库筛选、非精选 / IV-only、网络流量验证
├── docs/transition_pipeline.md    # 原有训练队列和静态导出详细说明
├── verify_index.py                # 全表行区间、影像归属与真实 API 读取验收
└── runs/                          # 忽略的患者索引、运行日志、验收截图、主动导出的 HTML
```

重构后的 `prepare_dataset(config)` 返回内存中的 `BuildResult`，不写磁盘；`build_dataset(config)` 显式导出，保留原先的原子写入和校验。`load_iv_tables(...)` 与 `link_packets(..., tables=...)` 分开读取和关联，避免同一批数据重复扫描。原有命令和导入路径保持兼容，默认数据路径统一为 `code/data`。

原有静态输出目录继续保留，新界面通过原始数据或患者索引读取，不依赖这些旧输出。详见 [配对、训练与旧版导出说明](docs/transition_pipeline.md)。

## 验证

从 `code/` 运行：

```bash
.venv/bin/python -m pytest MIMIC_example/tests -q
# 服务启动后；--include-inputs 同时验证真实输入事件的按需加载
.venv/bin/python MIMIC_example/browser_check.py --include-inputs
.venv/bin/python MIMIC_example/browser_check_cohort.py
```

测试依赖见 `requirements-dev.txt`。浏览器检查优先使用系统 Chrome / Chromium，否则使用 Playwright Chromium（`python -m playwright install chromium`）。扩展表的独立真实数据检查：`.venv/bin/python MIMIC_example/browser_check_clinical.py`。

真实数据检查使用本项目的默认精选案例，结果写入 `runs/browser_YYYYMMDD/`，包括 `checks.json`、桌面 / 手机截图和 `mimic_atlas.html`。

## 设计和数据参考

当前设计参考 Figma 社区的 [shadcn/ui design system · Pietro Schirano](https://www.figma.com/community/file/1203061493325953101)，见 [shadcn 官方 Figma 目录](https://ui.shadcn.com/docs/figma)。使用浅色侧栏、细边框、低饱和状态标记、紧凑表格和克制的留白。Figma 社区页面在当前环境返回 403，因此视觉核对使用该设计体系的[官方浅色 Dashboard 预览](https://ui.shadcn.com/examples/dashboard)（[预览图](https://ui.shadcn.com/examples/dashboard-light.png)）；未声称通过 Figma API 读取设计节点。实现为本地 HTML/CSS/JavaScript，不依赖 React 构建或付费设计资产。

数据说明：[MIMIC-CXR-JPG 2.0.0](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) · [MIMIC-IV 3.1](https://physionet.org/content/mimiciv/3.1/)。病例导出遵循原数据访问限制，留在本地，勿提交到 Git。

字段含义与关联参考：[labevents](https://mimic.mit.edu/docs/iv/modules/hosp/labevents.html)、[chartevents](https://mimic.mit.edu/docs/iv/modules/icu/chartevents.html)、[prescriptions](https://mimic.mit.edu/docs/iv/modules/hosp/prescriptions.html)、[emar_detail](https://mimic.mit.edu/docs/iv/modules/hosp/emar_detail.html)、[poe_detail](https://mimic.mit.edu/docs/iv/modules/hosp/poe_detail.html)。读取实现参考 [Apache Arrow CSV streaming](https://arrow.apache.org/docs/python/csv.html)。
