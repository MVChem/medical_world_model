# 全量 MIMIC-CXR + MIMIC-IV 连接与清洗

独立于 `medworld_table1` 的版本化数据构建程序。参考 `MIMIC_example` 与 appendix 的同患者、同住院时间线连接，处理本地全部 CXR，而不是抽取 pilot 的 12,000 对。

一次完整运行会生成全部图像/检查的关联清单、全部 31 张 IV 表的患者相关 Parquet、纵向配对、自动图像质控、检查前临床记录覆盖、分患者互斥的候选子集，以及原始病例审阅页。

图像数与纵向对数是不同口径：原库 377,110 张图像，当前严格配对使用 99,695 张不同图像，组成 70,318 对。该数不是全库可用图像的上限。`audit_pair_funnel.py` 可重建逐步筛选计数；允许 AP↔PA 与所有正时间间隔后，有 136,385 对、170,668 张不同图像。Qwen3.5-9B 的双图＋双报告语义初筛、断点续跑和统计方法见 [VLM_SCREENING.md](VLM_SCREENING.md)。

## 运行

在项目根目录：

```bash
code/medworld_table1/.venv/bin/python code/mimic_cxr_iv_linked/run_all.py \
  --out code/mimic_cxr_iv_linked/runs/full_20260909
```

环境复用本地 Python，使用 NumPy、Pillow、PyArrow、Polars、pytest；无需启动 GPU、vLLM 或调用远端模型。默认本地数据路径为 `/home/data1/data/MIMIC/MIMIC_CXR` 与 `/home/data1/data/MIMIC/mimic-iv-3.1`，可用 `--cxr`、`--iv` 覆盖。

`build.py`、`link_iv.py`、`qc_images.py`、`availability.py`、`finalize.py`、`review.py` 也可分别运行，均接受 `--out`。IV 按表原子完成并记录输入指纹；图像 QC 按图片追加缓存。中断后重跑可复用已经完成的结果。改变 cohort 规则或源数据时使用新的输出目录。不要对同一目录同时运行多个构建程序。

## 规则与含义

1. **完整 CXR 目录索引。** 保留全部视角、每个 study 的所有图像 ID 和完整时间范围；同一图像按 `subject_id` 与时间分别连接严格住院窗口、住院＋该次入院前急诊窗口及 ICU stay。无法匹配、有多重匹配时保留状态，不猜 ID。官方 split 患者互斥。
2. **报告清洗。** 保留 Findings、Impression、合并标题；合并内容只保留一次。重复且不同内容的同名段落全部保留并标记。规范空白，保留不确定措辞、对既往检查的比较和原始报告路径/哈希。无支持章节时保留在全量索引但不进入图文配对。没有 256-token 截断，没有 LLM 改写。
3. **纵向配对。** 同患者完整 study 时间线的严格相邻检查；不跳过中间不合要求的 study。两端取相同 AP/PA（均可时优先 PA），同 study/view 多图按像素面积和 DICOM ID 确定性选一张。间隔 6h–30d，排除歧义时间戳和采集窗口重叠。标签值、变化与否不参与选择；缺失标签保留。
4. **IV 全表关联。** 扫描本地 hosp/icu 所有 `.csv.gz`。含 `subject_id` 的表保留 CXR 患者的全部可用历史；字典/provider/caregiver 表完整保留。原字段和原标识符保留，增加 1-based `_source_record`（CSV 解析后的数据记录序号，不是物理文本行号）。有 `hadm_id`、`stay_id` 的记录可用原生键连接；无住院 ID 的记录保持患者/事件级，未进行推测性住院归属。
5. **自动图像 QC。** 检查全部候选 pair 的 99k 左右独立图像端点：完整 JPEG 解码、原文件及完整解码像素 SHA-256、尺寸、近常量图像、dHash。全量索引中的其他图像仅完成存在性和关系检查。保守 QC 子集排除损坏/近常量、pair 两端像素完全相同，以及跨患者像素完全相同的图像；dHash 距离 ≤2 只标记待审，不删除稳定随访。
6. **当前时点前的临床覆盖。** 选取 labevents、emar、chartevents、inputevents、procedureevents、outputevents；要求原生 patient/admission 匹配，事件时间与 storetime 都存在且不晚于当前 CXR。统计同次住院内已有记录，单独输出各表数量。ICU chart 是广义 ICU 记录，不保证每项生命体征都存在。没有原生 hadm_id、没有完整时间的记录仍在 Parquet，未计入此严格子集。
7. **明确证据边界。** `iv/` 是回顾性数据库，含未来及出院信息，不能整体序列化为当前模型输入。ICD 出院编码、只有日期或没有可靠记录时间的内容用于审计。storetime 只提供记录可用性的代理，无法证明版本不可变。当前报告没有可用时间：带报告输入单列为回顾性假设；另一份输入视图不含当前报告。horizon 来自实际相邻间隔分档，未实现预先指定窗口后挑选目标。

`same_admission_6h_72h_image_qc_prior_ehr` 是供讨论的主候选集：同住院短期＋通过自动图像 QC＋至少一种临床事件在当前时点前已记录。其名称和数量不代表逐例临床审定或临床信息齐全。检查类型/体位已知一致的版本单列；元数据缺失不自动等同于图像差。

## 主要产物

| 文件 | 用途 |
|---|---|
| `summary.md` / `summary.json` | 完整统计、split 数量、限制和来源 |
| `review.html` | 既有 appendix/MIMIC_example 及新样本的两张原片、完整分节报告和按时间分开的临床记录 |
| `images.jsonl` / `studies.jsonl` | 全部图像、图文和临床 episode 连接索引 |
| `patients.jsonl` / `admissions.jsonl` / `icustays.jsonl` | CXR 患者的关联原始表记录；仅用于连接/审计 |
| `iv/hosp/*.parquet` / `iv/icu/*.parquet` | 全部本地 IV 表的筛选或字典副本；原始 CSV 不修改 |
| `iv/*/*.json` | 每表输入指纹、扫描/保留数量及字段列表 |
| `pairs.jsonl` / `linked_pairs.jsonl` | 全部合格 CXR 配对；后者增加图像与临床可用性标记 |
| `rejected_adjacent_pairs.jsonl` | 未进入图文 pair 的相邻检查及全部拒绝原因 |
| `image_qc.jsonl` / `exact_duplicate_groups.jsonl` | 图片解码与哈希审计；重复图像只隔离候选，不删除原文件 |
| `clinical_availability.jsonl` | 各 CXR 端点在 cutoff 前已有的临床记录计数 |
| `cohorts/<tier>/<split>.jsonl` | 各候选层级的配对 manifest；这些行含 target/audit 字段，不是直接模型输入 |
| `forecast_views/*_inputs_image_ehr.jsonl` | 当前图像、horizon、历史记录计数和查询键；不含当前报告 |
| `forecast_views/*_inputs_report_assumed.jsonl` | 增加当前完整报告，明确采用回顾性报告可用假设 |
| `forecast_views/*_targets.jsonl` | 未来图像、报告、弱标签及真实间隔，按 opaque pair_id 连接 |
| `example_regression.json` | 逐项核对现有 10 个 MIMIC_example；超出 30 天的示例会明确不纳入 |
| `native_key_audit.json` | 全表 patient/hadm/stay 原生键关联一致性检查 |
| `validation.json` / `provenance.json` / `source/` | 最终输出合同检查、环境/哈希与构建源码快照 |

临床值的实际训练序列化尚未规定；输入视图目前给出安全的计数和查询条件，完整值在关联 Parquet 中。审阅页展示经过字段白名单和时间过滤的节选，避免把最终输液总量/结束时间等回顾性字段带入当前证据。

`review_assets/` 指向原始 JPG，未生成或修改医学图像。用本地浏览器打开 `review.html`，或在输出目录运行本地 HTTP 服务查看。保留本地访问边界；输出属于 MIMIC 派生数据。

## 检查

```bash
code/medworld_table1/.venv/bin/python -m pytest code/mimic_cxr_iv_linked/tests -q
```

合成测试覆盖合并报告标题、重复不同段落、歧义住院、延迟/缺失记录时间、患者与住院隔离、带换行的 CSV、原生外键和流式 Parquet 过滤。

Qwen3.5-9B 可在之后辅助给语义复核队列打标签，但不适合猜测 `hadm_id`、补造报告可用时间或替代临床真值。本轮先确定可追溯的结构和时间资格。

可选的 `audit_report_tokens.py --out ...` 使用本地 pilot tokenizer 统计旧 256-token 预算会影响多少报告，仅作诊断，不更改新数据文本。该 tokenizer 为 Qwen3.5-0.8B 的已有本地 tokenizer，没有加载模型或执行生成。
