# 09-16：MIMIC-CXR-VQA 指标核对与按文献方法复测

用户要求解释第一版指标、核对别人如何做 VQA，并按公开方法重测，完成后补充 Markdown。
本记录接续 [第一轮 pilot](0916_mimic_cxr_vqa_pilot.md)。

第一轮的 EM、micro-F1 有明确含义，但“整份 110 标签词表 + 强制 JSON 数组”是本项目自行设计的零样本适配。
它没有复现原论文的训练，也没有执行原论文的评分代码，不能把当时的低分直接解释为模型的临床诊断能力。

## 这些指标分别在测什么

| 指标 | 计算对象和含义 | 本数据集中的用途 |
|---|---|---|
| Accuracy / Acc | 答对题数除以题数 | Verify 是 yes/no；Choose 是从题目选项中作答 |
| Set exact match / EM | 预测标签集合必须与参考集合完全一致，顺序不影响 | 第一轮本地实现；多答或漏答一个标签，整题记 0 |
| Label micro-F1 / μF1 | 全部题的标签 TP、FP、FN 累加后，计算 `2TP / (2TP + FP + FN)` | Query 可有多个答案，既惩罚漏答，也惩罚多答 |
| Sample-F1 | 每题先算 F1，再平均 | 与 micro-F1 权重不同；代码留存的辅助项，不作为本轮主指标 |
| 患者 bootstrap 95% CI | 按患者重采样得到指标区间 | 同患者多题相关，不能把每题当成完全独立病例 |

例如参考答案为 `{pneumonia, pleural effusion}`，只答 `{pneumonia}`：EM=0；
该题 TP=1、FP=0、FN=1，F1=66.7%。这不是文本 token F1，也不是 CheXbert 的疾病标签 F1。
参考与预测都为空时，集合 EM=1，但不会给 micro-F1 增加 TP。

VQA 没有一个所有数据集通用的评分法。[VQAv2 官方评测](https://visualqa.org/evaluation.html)
使用多位标注者答案的一致性评分和文本规范化；这里每题是一个参考答案集合，不能照搬那种多人共识分数。
医学 VQA 也需区分 yes/no、选择题、开放问答，以及开放题答案是短标签还是报告式文本。
长文本的词汇重合或语义相似度，不能自动等同于诊断准确率。

## 别人怎样使用这个数据集

首先核对数据身份：本地是
[MIMIC-Ext-MIMIC-CXR-VQA 1.0.0](https://physionet.org/content/mimic-ext-mimic-cxr-vqa/1.0.0/)，
不是 Medical-CXR-VQA、MIMIC-Diff-VQA，也不是另一个
[Visual Question Answering evaluation dataset for MIMIC CXR](https://physionet.org/content/vqa-evaluation-mimic-cxr/1.0.0/)。
即使都使用 MIMIC 图像，题目、划分和评分也不能混用。

| 来源 | 实际方法 | 评测/复现含义 |
|---|---|---|
| EHRXQA，NeurIPS 2023，§6.1 / Table 4 | 把图像和问题映射到 110 个答案标签，训练多标签 VQA 分类器；比较 PubMedCLIP、MedViLL、M3AE 等 | 报 Acc、micro-F1，另有依赖参考模型的相对 AUROC；不是直接给通用聊天模型提示词 |
| AOR，2025，§5 | 对 MIMIC-CXR-VQA 做任务微调，加入解剖区域与推理监督 | Verify Acc、Choose Acc、Query micro-F1；完整 test 为 13,793 题 / 500 图 |
| CheXagent / CheXinstruct 数据处理代码 | 图像 + 原问题，答案标签串成逗号分隔文本；gender 的 f/m 转为 Female/Male | 说明生成式任务如何组织；其处理器还丢弃空答案和部分全图异常题，不能直接当作完整 test 评测协议 |
| LLaVA 官方自定义评测指南 | 短答案 VQA 在问题后追加短答格式指令；使用 greedy decoding | 可用于把聊天模型适配到短答任务；它不是本数据集专属 evaluator |

原论文和 AOR 的训练、指标定义见
[EHRXQA 论文](https://proceedings.neurips.cc/paper_files/paper/2023/file/0c007ebef1d11fd48da6ce4f54687db6-Paper-Datasets_and_Benchmarks.pdf)、
[AOR 论文](https://arxiv.org/html/2505.02830v1#S5)。
生成式数据格式见
[CheXagent 数据处理器](https://github.com/Stanford-AIMI/CheXagent/blob/e4f31e6e517f442620fe816f90c846cbcb28b746/data_chexinstruct/dataset_processors/mimic_cxr_vqa.py)，
短答协议见 [LLaVA Evaluation.md（固定版本）](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/docs/Evaluation.md)。

作为量级背景，EHRXQA 的 M3AE* 在全 test 报告 Acc=69.2%、micro-F1=0.73；
AOR-t 报告 Verify/Choose/Query 为 80.48/71.96/65.05。
这两组均有任务训练，样本数和评分实现也不同，不能与本次四个原始 checkpoint 的抽样零样本成绩作公平排名。
要复现这些训练结果，需要使用训练/验证集训练分类头或做 VQA 指令微调，在验证集固定超参数，再评价未参与选择的 test。

原论文附录 E.1 / Table E15 给出的 M3AE* 设置是 ViT-B/16 + RoBERTa-base，
VQA 微调 50 epochs、batch size 64、学习率 5e-6；作者报告在单张 A6000 上微调约 16 小时。
因此原论文“训练后测试”和本次八小时内拿到原始模型抽样结果，是两种实验。
这些是作者的资源记录，不是对当前 GPU 0 的耗时估计。
见 [EHRXQA 补充材料](https://proceedings.neurips.cc/paper_files/paper/2023/file/0c007ebef1d11fd48da6ce4f54687db6-Supplemental-Datasets_and_Benchmarks.pdf)。

## 本次实际采用的协议

**采用 LLaVA 官方短答提示 + AOR 分题型指标定义；答案规范化是本地明确实现。**
没有找到 AOR 在所检出公开版本中的完整 MIMIC-CXR-VQA evaluator，不能声称逐行复现其官方评分。
也没有进行 AOR 的区域推理训练或原论文的多标签分类器训练。本轮是文献方法对齐的零样本 pilot。

四个模型共用以下输入；短答后缀逐字来自 LLaVA 的指南：

```text
<original question>
Answer the question using a single word or phrase.
```

同时输入当前胸片。没有报告、EHR、答案词表、few-shot 示例、金标准区域框或参考答案。
Query 的原问题仍保留“list all”等措辞；解析接受多个标签，不强制只输出一个标签。
但这条通用短答提示对多答案枚举是否充分，仍是适配限制，不能视为已验证最优提示。

- 完全复用上一轮 1,024 题 / 434 患者的选择、问题、参考答案和图像字节；seed=20260916。
- Verify 516 题、Choose 113 题、Query 395 题。保留原有 193 个空答案，不按答案删除困难题。
- 四模型：Qwen3.5-0.8B、4B、9B，MedGemma-1.5-4B-it；GPU 0 串行，BF16，无项目微调。
- 共用 512×512 RGB 等比例补边图；各自原生处理器；Qwen 固定 512² 像素预算、关闭 thinking。
- Temperature=0，最多 512 新 tokens；取消 JSON 和词表约束解码。
- 主报告为完整抽样的 Verify Acc、Choose Acc、Query label micro-F1，患者级 1,000 次 bootstrap。
- 另保留 936 题无当前训练患者重叠子集及 884 题临床内容子集，沿用先前已固定的排除规则。

完整抽样的 Verify 包含 267 个 no、249 个 yes，始终回答 no 的事后常量检查为 51.74%。
884 题临床子集的分母是 Verify 456、Choose 84、Query 344，不能与 1,024 题表混写。

文本解析在四模型正式复测前冻结：Verify 接受句首独立 yes/no；其他题接受规范标签的逗号、
分号、换行或 and 分隔列表、JSON 列表、基本项目符号，female/male 映射为 f/m。
独立 none/no/nothing/no abnormalities/no abnormality 视为空集。
不做医学同义词扩展，也不从段落任意搜阳性标签；非规范片段增加一个无效标签 FP，
同一列表中的规范标签仍可得部分分数。截断、空响应和失败整题计错。
因此 Query 的数值可能低估意思正确但措辞不同的答案；它是可审计的自动解析成绩，并非临床专家判读成绩。

协议调整记录：先按 CheXagent 原问题方式做了 Qwen-0.8B 试跑，发现模型输出大量长篇解释，
会让短答案解析主导分数，因此停止这一诊断试跑并保留 180 条响应（其中 8 条截断）；随后另建目录，
四模型统一采用上面的公开短答指令。该调整发生在探索性 pilot 中，不能称为从未查看 test 的确认性实验。
没有修改已执行的源码快照、覆盖旧预测，或按模型单独挑选得分更高的协议。

## 为什么不直接采用搜索到的最高分评分方式

核对 [Meissa 的 MIMIC-CXR-VQA 评分源码](https://github.com/Schuture/Meissa/blob/eb119115d2f5dfcfae772ba5f2466b4c603fac90/environments/continuous_tool_calling/eval/run_mimic_cxr_vqa.py)
发现，所固定版本用的是“任意一个参考标签是预测文本的规范化子串”就判正确：

- 参考 `pleural effusion`，预测 `No pleural effusion`，也会命中。
- 参考有两个标签，只写中一个，就能得到整题正确。
- 参考为空列表，任何输出都会判错。

这与严格集合 EM 或多标签 micro-F1 不等价。本次已原样执行该源码的两个纯评分函数，
仅展示同一批预测更换评分方法后分数的变化，不运行其 agent，也不将其当作主要指标。
这些是对已检出代码版本的观察，不能推断所有论文报告均由此版本产生。

实际执行该规则的无图像检查：每题都回答完整 110 标签词表，就能得到 **81.15%** 的
any-substring accuracy，恰好等于这批题的非空参考比例。这个反例说明高数值本身不能证明答案质量。

## 结果与 Table 1 使用

**2026-09-16 23:05:55（北京时间）已完成四模型，共 4,096 条成功响应；GPU 0 已释放。**

完整结果、旧新协议对比、患者级 CI 和评分敏感性检查见 [复测结果报告](../results/mimic_cxr_vqa_literature_20260916/README.md)。

完整 1,024 题抽样的分数如下，单位均为百分数。Query 使用上述保守标签解析。

| 模型 | Verify Acc（516 题） | Choose Acc（113 题） | Query label μF1（395 题） |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 64.15 | 21.24 | 11.17 |
| Qwen3.5-4B | 71.32 | 23.89 | 20.56 |
| Qwen3.5-9B | 73.45 | 27.43 | 37.21 |
| MedGemma-1.5-4B | 46.71 | 3.54 | 0.98 |

当前项目的 **884 题临床内容子集** 单列如下：

| 模型 | Verify Acc（456 题） | Choose Acc（84 题） | Query label μF1（344 题） |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 64.04 | 10.71 | 9.67 |
| Qwen3.5-4B | 70.83 | 11.90 | 18.16 |
| Qwen3.5-9B | 73.03 | 11.90 | 35.86 |
| MedGemma-1.5-4B | 46.49 | 4.76 | 0.68 |

这轮 Qwen 三个规模的 Verify 均高于旧 JSON 协议；Qwen-9B 在完整抽样的三项分数为本轮最高。
提示词、解码约束、输出解析和 token 上限同时发生变化，不能把增幅单独归因于短答提示；也没有做模型间显著性检验。

**MedGemma 的开放题成绩受解析影响尤其严重。** 1,024 条响应中，598 条包含无法按规范标签识别的文本，52 条截断；
在 Query 的 395 题中，有 385 条出现非规范文本或截断。它经常回答完整句子，例如“the lungs”而非分别枚举 left lung/right lung，
或用整句表示没有异常。本地规则不会自动把这些句子映射成金标准标签，故 0.98% 不能作为其临床能力排名依据。
同一批 MedGemma 文本换用 Meissa 的宽松规则得到 41.89%，但该规则存在上文已验证的缺陷；不能挑选较高者作为最终成绩。

**验证已通过：** 4×1,024 条响应完整且 ID 唯一；与上一轮题目、参考答案及图像字节一致；
运行协议/源码哈希未变；每模型 12 组分项经 scikit-learn 独立复算（总计 48 组）。
协议与解析单元测试共 8 项通过。上述验证检查实现和聚合，不等于人工医学语义审阅。

运行目录：`code/medworld_vqa/runs/literature_short_1024_gpu0_20260916/`。

Table 1 若增加该任务，建议列名为 **Current-image CXR-VQA**，采用 Verify Acc、Choose Acc、
Query micro-F1 三项，并标明 sampled / zero-shot。不要把三个不同分母的指标随意平均成“Diagnosis accuracy”。
884 题子集还包含器械和技术质量问题，不是纯疾病分类；当前图像问答也不等于未来状态预测。
目前 Table 1 的 future-state AP/AUROC 不应被这些数值替换。

所有逐题问题、图像和患者标识留在忽略版本控制的本地运行目录；研究记录和结果目录只含汇总。
预训练数据暴露未知；患者去重审计只覆盖已记录的当前统一训练协议。

代码入口：[medworld_vqa](../code/medworld_vqa/README.md)。


## 09-17：MedGemma 的格式遵循与评分适配

重新抽查原始输出后，需要区分“没有按短答要求回答”和“答案没有落在解析器规定的词表里”。
本轮提示只要求 single word or phrase，并未明确要求所有二分类题必须输出 yes/no、所有开放题必须使用那 110 个标签。
因此 598 条 noncanonical_text 不能解释为 598 次指令遵循失败。

MedGemma 平均输出 55.58 tokens，Qwen 三个规模分别为 3.15、6.93、7.41；MedGemma 有 52 条达到 512-token 上限。
这些观察说明它在这套统一提示下更容易生成长回答，但不能概括成其通用格式遵循能力差。

固定随机种子抽查非规范回答时，同时看到以下情况：

- `Normal.` 本身满足单词短答要求，但不会被本地二分类解析器映射成 no。
- 回答最后明确写出 `the answer is no`，也可能因不在句首而被记作非规范文本。
- `No abnormalities are observed ...` 与部分空参考含义一致，但未被当前严格空集规则接受。
- 还有与金标准不一致、漏答、多答或答非所问的情况，不能把所有扣分归因于格式。

这里的数字评价了模型、提示词和答案解析的组合。MedGemma 的 0.98% Query label micro-F1
不足以支持其诊断能力差的结论，也暂不适合作为正式 Table 1 的模型能力排名。
需要在验证集固定更可靠、适用于所有模型的答案规范化，再从保留的原始输出统一重算；当前结果和协议继续保留。
