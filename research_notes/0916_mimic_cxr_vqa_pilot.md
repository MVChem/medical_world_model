# 09-16：GPU 0 MIMIC-CXR-VQA 第一版诊断问答评测

后续核对与复测见 [指标解释、文献方法及新结果](0916_mimic_cxr_vqa_literature.md)。
本页保留第一轮自定义词表 JSON 协议的结果，不代表原论文方法复现。

用户要求八小时内看到第一版结果，无需全量测试；模型为 Qwen3.5-0.8B、4B、9B 和
MedGemma-1.5-4B。仅使用 GPU 0，四模型串行运行，保留其他训练与服务。

**2026-09-16 22:10 已完成四模型推理并释放 GPU 0**，共 4,096 条成功响应。
完整结果和患者 bootstrap CI：[汇总报告](../results/mimic_cxr_vqa_pilot_20260916/README.md)。
所有 EM/micro-F1 已通过独立 scikit-learn 复算。

| 模型 | Diagnosis EM (%) | Diagnosis micro-F1 (%) | 截断数 / 1,024 |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 26.24 | 23.83 | 227 |
| Qwen3.5-4B | 29.98 | 21.12 | 19 |
| Qwen3.5-9B | 36.20 | 30.92 | 11 |
| MedGemma-1.5-4B | 17.19 | 15.12 | 62 |

无请求失败；约束生成后无效输出均由长度截断造成。模型仍有过多标签和答题指令遵循问题。
事后固定 no 输出的 Diagnosis EM/micro-F1 为 27.49/26.07，提示部分模型在此协议下
尚未超过简单常量输出。本轮适合检查接口和建立第一版基线，不宜直接作为最终诊断能力结论。

评测入口：[medworld_vqa](../code/medworld_vqa/README.md)。
运行目录：`code/medworld_vqa/runs/pilot_1024_gpu0_20260916_json/`。
结果和进度：[REPORT.md](../code/medworld_vqa/runs/pilot_1024_gpu0_20260916_json/REPORT.md)。

从官方 test 的 13,793 题中，按 semantic type × content type 的比例分层随机抽取
1,024 题、434 张图像 / 434 位患者，seed=20260916。选题不依赖答案，也不依赖模型输出。
保留 verify/choose/query、七种内容类型、否定和空答案。

输入是当前图像、问题和官方公开的固定 110 标签答案词表。没有报告、EHR、示例答案或微调。
共用 512² RGB 补边图；Qwen 原生处理器使用 512² 像素预算，MedGemma 使用自身处理器。
均为 BF16、temperature=0、max_new_tokens=192，关闭 Qwen thinking。
最终四模型协议统一启用公开 110 标签词表上的 JSON 数组约束解码。

首次自由生成的 Qwen-0.8B 在 1,024 题中有 416 条格式错误/截断（40.625%），
其严格 Diagnosis EM/μF1 为 17.53/18.00%。为避免将格式遵循混作诊断能力，
在其余模型开始推理前停止自由生成队列，保存旧协议和原始输出，另建上述 `_json` 目录，
四模型统一重跑。两轮样本、图像像素、提示词和词表相同；只增加解码约束。
这仍是探索性 pilot，不应将协议调整后的结果描述为独立确认性测试。

评分：答案标签集合 exact match、micro F1；分别报告题型、内容、区域/全图、空/非空答案，
另给出患者级 1,000 次 bootstrap 95% CI。格式错误、未知标签、截断和失败请求计错，
不允许将无法解析的输出转换成正确的空答案。采用本地显式集合评分，未宣称已对齐官方 evaluator。

在任何模型推理前完成的患者审计发现：

- 官方 VQA test 与自身 train、valid 的患者交集均为 0；train 与 valid 有 698 位患者重叠。
- 官方完整 test 的 500 位患者中，有 52 位出现在当前统一模型的时序训练集。
- 当前抽样有 43 位患者 / 88 题与该训练集重叠；与当前分类、报告、分割、SR 训练集的交集均为 0。
- 排除全部训练重叠患者后，VQA 子集为 936 题。
- 再排除 plane 和 gender 后，**Table 1 候选 Diagnosis 指标为 884 题 / 382 位患者**。

因此 `overall` 对应完整 1,024 题，`diagnosis` 对应无训练患者重叠的 884 题；
`diagnosis_all_sampled` 和 `vqa_train_disjoint` 另存，便于解释两种筛选的影响。
这些名称、过滤规则和样本在推理前固定，四模型完全一致。

当前论文 Table 1 是 future-state forecasting。新增列应写作
**Current-image Diagnosis (CXR-VQA, sampled)**，独立说明图像输入与评测队列。
这不是未来诊断 AP/AUROC 的替代值，也不是官方全量测试。
临床内容子集仍含器械和技术质量问题，不宜描述为纯疾病诊断。
未来加入 Ours 时需要相同输入、相同词表、相同 884 题与单独的 VQA 输出接口；本次未测 Ours。

患者审计针对当前统一训练协议。公开权重的预训练患者重叠仍未知。
所有逐题输出、图像与患者标识保存在忽略版本控制的本地运行目录，仅汇总表适合纳入论文。
