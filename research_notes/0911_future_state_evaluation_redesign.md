# 从 Future State 出发重设计 Table 1

2026-09-11。**指标方案已由用户确认，并写入论文 Table 1。** 依据最初提出的“3–4 个宏观评价维度，每个维度 2–3 个指标”查阅近期论文后，方案收敛为四组八指标。用户确认后要求按此执行，后续以标签验收、评分实现和实验为主。表格、Results、附录协议与独立两页计划已同步；具体标签和评分实现仍需验证，不代表实验已完成。本次没有修改训练配置或启动实验。

**09-12 执行更新：** [原始 VLM 评测](0911_raw_model_baseline_sweep.md)已接入并完成除 Direction 外的七项指标，[实验小结](0912_recent_experiments_summary.md)保存六模型分数；新增 Direction 队列使用单独的 82 对子集。下文第 7 节保留 09-11 方案形成时对 `medworld_table1` 主路径的检查，不代表新增 `medworld_baselines` 路径仍缺这些评分实现。论文主表尚未回填。

## 1. 已确认：四个维度、每组两个，共八个核心指标

先确定 Future State 的哪些性质需要验证，再选择分数。本文把四类分别定义为：未来有什么、相对现在如何变化、能否表达完整临床内容、概率预测是否可靠。这四类是互补的评价问题，分数之间仍可能相关，不称四个独立任务或统计独立的能力。

| 宏观维度 | 回答的问题 | 已确认核心指标 | 所需输出与参考 |
|---|---|---|---|
| **Future clinical status：未来临床状态** | 指定随访时点有哪些异常？ | Finding macro AP ↑；macro AUROC ↑ | 逐征象连续分数；未来明确阳性／阴性标签 |
| **Disease progression：病情演变** | 哪些异常新发、消退，持续异常如何变化？ | Transition macro F1 ↑；Direction macro F1 ↑ | 当前与未来征象；经核验的疾病／侧别级加重、改善、稳定标签 |
| **Future report fidelity：未来报告的临床内容** | 预测状态能否表达出正确的征象、部位、程度及关系？ | RadGraph partial F1 ↑；GREEN ↑ | 预测未来报告；真实随访报告；冻结的评价模型 |
| **Probabilistic reliability：概率预测可靠性** | 模型给出的概率与真实发生频率是否相符？ | macro Brier score ↓；classwise ECE ↓ | 与第一组相同的逐征象概率及参考标签；固定校准协议 |

已写入的表头组织为：

```text
                      Future clinical status      Disease progression      Future report       Reliability
Method                    AP     AUROC          Transition   Direction    RadGraph   GREEN     Brier   ECE
```

每组固定两列。病情演变组将新发／消退合并为 **Transition F1**，保留 **Direction F1** 评价持续异常的改善／稳定／加重；前者检查有无状态改变，后者检查持续异常的程度变化。Onset F1、Resolution F1 的单独分解放附录。

**Degree MAE ↓（程度等级误差）**也放补充分析，不增加主表第三列。它需要可靠的程度等级参考，当前不应把规则抽取的词语等级直接称为临床严重程度金标准。是否开展应在查看方法比较结果前，依据标签覆盖与人工一致性确定。

这套设计保留文本读出，因为当前系统已有未来报告输出；它评价的是由预测 state 到临床内容的整个读出过程。单独归因于 latent 表征的质量，还需要控制解码器的诊断实验。

## 2. 近期论文实际如何评价

下面区分“原文实际采用的评价”与“本项目可借鉴的设计”。没有发现一套可直接照搬到本项目的统一四维标准。

### 顶会 world model 与生成模型

| 论文与正式状态 | 核对位置、原文评价 | 对本项目的启发与边界 |
|---|---|---|
| [Medical World Model，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Yang_Medical_World_Model_ICCV_2025_paper.pdf) | Table 1：放射科医师真假判别、FID、LPIPS；§4.2／Fig. 6：生存风险及 C-index；Table 2：治疗方案 Precision、Recall、F1、Jaccard。 | 生成保真度、临床结局、决策用途分别验证。我们可借鉴分层逻辑，但现有输入和目标不支持直接迁入治疗规划或生存指标；匹配历史方案也不等于证明最优治疗。 |
| [DINO-WM，ICML 2025](https://proceedings.mlr.press/v267/zhou25t.html) | [正式 PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/zhou25t/zhou25t.pdf)，Table 1：四个环境用 planning success rate，Rope／Granular 用 Chamfer distance；Table 3：由预训练编码器特征到环境真实状态的线性 probe loss。 | latent 模型通过可观测状态／行为验证。Table 3 测编码器表征，不能改称其“预测未来 latent 的误差”；控制任务的成功率也不能直接迁入胸片预测。 |
| [MRI Contrast Enhancement Kinetics World Model，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Kong_MRI_Contrast_Enhancement_Kinetics_World_Model_CVPR_2026_paper.html) | 正式 PDF §4.1.1、Table 1：空间 PSNR、SSIM、LPIPS、rMSE；时间 cSSIM，定义为相邻生成帧 SSIM 的均值。 | “单时点保真度／跨时点行为”可分开评价。cSSIM 测相邻帧相似性，不能单独证明真实动态预测准确，见下文反例。 |
| [CLARITY，ECCV 2026 已录用](https://eccv.ecva.net/Conferences/2026/AcceptedPapers) | [作者 v3 正文](https://arxiv.org/html/2512.08029v3)：Table 2 生存 C-index；Table 5 编码器对照同时报 C-index／Brier；Table 6 在相同编码器及生存头下比较目标 latent cosine distance；附录 Table 7 用辅助图像解码器报 PSNR／SSIM／LPIPS／FID。 | 同时评价任务用途、概率误差和表示误差；原文的 cosine 比较控制了评价空间。附录有图像解码器，不意味着我们的现有预测接口也能计算图像指标。 |
| [REVTAF，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_Learnable_Retrieval_Enhanced_Visual-Text_Alignment_and_Fusion_for_Radiology_Report_ICCV_2025_paper.pdf) | §4.1、Table 1：NLG 与 clinical efficacy 两组，前者 BLEU／METEOR／ROUGE-L，后者报告标签 Precision／Recall／F1。 | 分组应对应评价对象。该论文描述当前图像，并非源时点预测未见随访；文本重合分不能直接当作动态能力证据。 |

CLARITY 的会议身份已在 ECCV 官方录用名单核对；正文依据作者 v3，不能将其说成已核对出版社最终排版版本。CVF 两篇 world model 的正式 PDF 已读取；MeWM 本地原存文件是早期 arXiv，本文采用另行获取的 ICCV 正式版核对。

### Nature 系列的真实未来预测

| 论文与期刊 | 核对位置、原文评价 | 对本项目的启发与边界 |
|---|---|---|
| [Delphi-2M：Learning the natural history of human disease with generative transformers，Nature 2025](https://www.nature.com/articles/s41586-025-09529-3) | Fig. 2、Fig. 3、Methods 的 Performance measures and calibration：疾病预测 AUROC／average precision，预测与观测发生率的校准，以及生成轨迹的人群发生率与按预测时间展开的命中表现。 | 区分风险排序、概率可靠性和轨迹表现。其长期疾病事件与胸片某次随访的征象状态不是同一目标，不能直接套用时间或删失处理。 |
| [DT-GPT：Large language models forecast patient health trajectories enabling digital twins，npj Digital Medicine 2025](https://www.nature.com/articles/s41746-025-02004-3) | [正式 PDF](https://www.nature.com/articles/s41746-025-02004-3.pdf)，Table 1：scaled MAE；Methods／补充表：MAE、MASE、SMAPE、Spearman、异常高低值及趋势 AUROC；Fig. 3：KS 分布差异，另检查变量间相关性。 | 同一未来状态可以分“数值误差／趋势／分布关系”评价。其 scaled MAE 是按变量标准差缩放，不能与使用朴素预测误差作分母的 MASE 混淆。 |
| [SurvivEHR，npj Digital Medicine 2026](https://www.nature.com/articles/s41746-026-02709-z) | Fig. 4：next-event IEC 及多步退化；Table 2、Fig. 7：time-dependent concordance、Integrated Brier Score、Integrated NBLL 与各 horizon 校准图。Methods 将 NBLL 展开为 negative Bernoulli log-likelihood，Results 的命名有差异。 | “第一步预测好”与“多步可靠”分别验证。我们目前做 horizon-conditioned endpoint prediction，应采用该目标的 Brier，不能把按 horizon 分层直接叫作 integrated survival score。 |

### 专门补充：胸片结构与时间语义

| 论文与准确发表类别 | 原文评价 | 对本项目的用途 |
|---|---|---|
| [Libra，ACL 2025 Findings](https://aclanthology.org/2025.findings-acl.888.pdf) | §3.2、Table 1、附录 F.3：词汇、临床、Temporal Entity F1；最后一项由预定时间变化关键词的交集计算。 | 可借鉴“变化单列”。本项目需要将变化方向绑定到具体疾病／侧别；词语相同但疾病配错也应扣分。它使用已观测的当前／先前图像，不是 future-only forecasting。 |
| [RadEval，EMNLP 2025 System Demonstrations](https://aclanthology.org/2025.emnlp-demos.40/) | §2、§6、Table 2：统一报告指标，并检查其与专家错误计数的相关性，错误涉及漏报、误报、部位、程度、比较关系等。 | 用结构抽取与临床错误评价互相补充；新评价器在本项目变化病例上仍需核验。不是 EMNLP 主会研究论文。 |
| [LUNGUAGE，CHIL 2026](https://proceedings.mlr.press/v333/moon26a.html) | [正式 PDF](https://raw.githubusercontent.com/mlresearch/v333/main/assets/moon26a/moon26a.pdf)，§3–5：实体／关系／属性及时间组；Table 1 结构化提取，Table 2 与专家错误相关性，Table 3 single／sequential LUNGUAGESCORE。 | 与“部位、程度、变化分别评价”很贴近。是医疗机器学习专门会议，不能误称 ICLR／NeurIPS 录用；其 sequential reporting 也不是现成的未见未来预测协议。 |

版本提醒：LUNGUAGE 正式 CHIL 论文写 186 份纵向报告／30 位患者；当前 PhysioNet v1.0.0 页面写 80 份／10 位患者。论文中的 benchmark 总规模、已发布版本和各具体实验子集要分别核验，不能直接据较大数字宣称我们已经有对应标签。[发布页](https://physionet.org/content/lunguage/1.0.0/)

上表核心依据覆盖 2025／2026 年的顶会与 Nature 系列，另外三项专门文献按真实发表类别补充。没有将检索到的 under review、workshop 或未核验 arXiv 当作主会发表依据。

## 3. 从文献得到的两条关键判断

**时间一致性需要与真实变化比较。** MRI CEKWorld 的 cSSIM 按相邻帧相似度计算。按该公式推导，如果所有生成帧完全相同，理论上每对 SSIM 都为 1；但患者真实状态仍可能发生变化。因此，“更平滑／更相似”不能单独等同于“更准确”。这是本文依据公式的分析，不是原论文报告的实验失败。

**latent distance 需要固定评价空间。** CLARITY 的相关对照使用相同编码器和生存头；DINO-WM 还通过状态 probe 与行为用途验证表示。我们若将每个方法各自 latent 的 MSE／cosine 并列，模型间坐标系、尺度和压缩信息不同，缺少可比性。主表优先评价共同临床参考；在共同冻结空间中的表示误差可以作为补充诊断。

## 4. 八个核心指标的具体定义

### A. Future clinical status

对病例 i、征象 k，记未来明确参考为 y(i,k)∈{0,1}，预测分数为 p(i,k)。参考未知／未提及／冲突字段不参加二值评价，其掩码由参考固定，对所有模型一致。

- **AP**：逐征象计算非插值 average precision，然后宏平均。只有同时存在阳性和阴性参考的类别进入共同的排序评价集合。
- **AUROC**：逐征象计算阳性相对阴性的排序能力，再对相同类别集合宏平均。AP 和 AUROC 共同刻画排序，不能据此推断概率已校准。

所有模型报告同样的有效样本数、阳性率和类别集合。公开 VLM 的固定 Yes／No 答案条件似然可作为分数，但须冻结提示词、完整候选序列评分及归一化方式；不能拿生成的“80%”直接作为有效概率接口。Finding head 与文本回答似然属于不同读出路径，比较中要披露。

**附录候选 Degree MAE**：对固定疾病／侧别的明确参考程度，计算预测等级与参考等级的绝对误差。它补充 AP／AUROC 对“有病但程度不同”的盲点。不能把不同疾病的程度词直接当作同一生理量；需逐病种定义等级、间距假设和宏平均权重。参考程度明确而预测缺失或部位不符时，不能把该字段删掉后只报成功提取的 MAE；应固定惩罚规则并另报缺失率，或将此结果明确限定为带覆盖率的辅助分析。

### B. Disease progression

两项均由当前参考与模型预测未来构成预测变化，再与当前／未来真实配对比较。所有预测的解释以选定 t0 为参照，不能让生成器选择另一张旧片作为比较基准。

- **Transition macro F1**：沿用当前二值变化评价，逐征象计算新发（0→1）和消退（1→0）两个事件的 F1，再对有真实事件支持的“征象×事件”项等权宏平均。支持集合由参考固定、对所有方法相同；附录报告逐事件支持数及分数。它不是将所有事件的 TP／FP／FN 合并后计算的 micro F1；当新发与消退支持项数量不同时，也不等于两组宏 F1 的简单二等分平均。当前无且未来仍无时预测有，计误报新发；当前有且未来仍有时预测明确无，计误报消退。未提及不能视为明确消退。
- **Direction macro F1**：在参考确认前后持续存在、且变化方向可评价的疾病／侧别字段上，对 improved／stable／worsened 三类计算 F1，再按预先固定的疾病、侧别和类别规则宏平均。预测缺失、错误消退或不可解析不能缩小参考集合，至少计为对应真实类别的漏检。

Direction F1 是本项目拟定义的 macro F1 应用协议，不是宣称某篇论文提供了完全同名、可原封使用的标准。它补足 Transition F1 无法覆盖的 1→1 程度变化。例如“有积液→无积液”由 Transition 评价，“大量积液→少量积液”由 Direction 评价。

程度词区间完全分离可以提供方向证据；两个相同程度等级不保证没有等级内部的变化。稳定阳性需要明确、且比较对象可确认的证据。左右变化不同按区域分别计分；不能把“左改善、右恶化”压成整个人的一个方向，也不能将未提及默认为稳定。

必备补充结果：新发／消退的 F1 分解、参考稳定字段上的 **false-change rate ↓**，以及 Direction 子集的覆盖率与混淆矩阵。它们与主表两列同时查看，避免只对 changed 病例评分。Transition 与 Direction 的合格字段集合分别说明：前者要求前后有无状态明确，后者额外要求持续异常及方向证据明确。

### C. Future report fidelity

- **RadGraph partial F1**：沿用当前明确指定的 RadGraph-XL／RG_ER partial 口径，比较带断言状态的实体及其关系参与信息。它并非所有关系端点和关系类型完全匹配的 complete 分数，表注应准确。
- **GREEN**：使用冻结、可复现的评价器比较预测与真实未来报告，同时保存匹配征象和错误分类。其目标是临床内容误差，仍会受评价器能力影响，不能将分数直接解释为医生认可率。[原始指标定义：EMNLP Findings 2024](https://aclanthology.org/2024.findings-emnlp.21/)

本组与 A 组有语义交集，但粒度不同：A 检查预定义标签的排序，C 检查完整报告的表达，包括标签之外的内容。当前 CheXbert F1 可保留为补充及旧结果连续性对照；主表第二列用 GREEN 扩大错误类型覆盖。

评价必须统一 Findings／Impression 的目标范围。未来报告中的历史比较可能指向 t0 之外的检查：完整报告评分保留其标准定义，针对 t0 的变化正确性由 B 组单独核验，不把两者混成同一种真值。

### D. Probabilistic reliability

本组使用与 A 组完全相同的未来征象目标。它评价选定随访队列中的概率预测；受报告标签和随访选择影响，不能直接解释成一般住院人群的真实疾病风险校准。

- **Brier score**：每个征象按有效参考计算 mean[(p−y)²]，然后宏平均。越低越好。它评价总体概率误差，同时受区分能力和校准影响，不能简称“纯校准误差”。需与训练集估计的当前状态＋horizon 转移先验比较。
- **Classwise ECE**：按征象分别将 p 分箱，对每箱计算平均预测阳性概率与真实阳性比例的差的绝对值，再按该箱样本占比加权，最后宏平均。建议先固定十个等宽箱，配合样本量和可靠性图；它不是对“最自信类别”的 top-label ECE。采用的是对多标签事件概率的适配定义。[经典校准研究](https://proceedings.mlr.press/v70/guo17a.html)

ECE 会受箱数和样本量影响，低 ECE 也可能来自始终输出群体阳性率，所以必须与 AP／AUROC、Brier 一起看。按 horizon 分层报告不能推导出各组都校准，总体校准也不代表每位患者的个体不确定性已被完整刻画。

建议主表先报告原始概率结果，附录给所有方法在同一验证集、同一校准方法预算下的校准后结果。校准器和阈值只能在验证集拟合。Brier／ECE 并不要求 Bayesian 模型或多次采样，但需要实际可用的概率评分接口；只输出报告且没有该接口的方法留空并解释。

## 5. 一个病例说明四组为何互补

构造示例：当前“右侧大量积液、无气胸”；未来参考“右侧少量积液、出现气胸”。

| 预测行为 | 主要暴露的问题 |
|---|---|
| 保留积液存在，漏掉气胸 | 未来征象有漏检，新发预测错误 |
| 预测积液与气胸，但仍写大量积液 | 未来二值疾病可以正确，持续异常的改善预测错误 |
| 写少量积液与气胸，却把右侧写成左侧 | 需要部位、属性或报告临床内容评价 |
| 文字判断偶尔正确，但长期以接近 100% 概率预测经常不发生的事件 | 需要跨病例评价概率可靠性，单份报告无法判断校准 |

这些是解释性示例，未计算任何本项目新测试分数。

## 6. 哪些内容适合放在附录

| 候选内容 | 建议安排与理由 |
|---|---|
| Future R@1／R@5、MRR | 保留为检索诊断。当前六征象 schema 易并列，候选数量、匹配规则、合格率会明显影响分数；多加一个 k 不能构成新的宏观能力。 |
| 预测 latent 的 MSE／cosine | 同一冻结评价空间、同一 target 编码与归一化下做内部对照。不同模型各自空间不横向排名。真实 future state 的读出可作重建诊断，不当作合法预测成绩或无条件严格上界。 |
| 不同 horizon 的结果 | 对相同核心指标分层，并报告样本数及区间。当前不同 horizon 往往来自不同病例，曲线本身不能证明同一患者轨迹正确。 |
| 连续轨迹、趋势误差、KS／相关矩阵、CRPS | DT-GPT 式评估需要真实序列、数值变量或预测分布。当前只输出一个 horizon 的 state 时，不宜为了第四类而强加；若以后扩展，可重新定义。 |
| Future image PSNR／SSIM／LPIPS 或 future mask Dice | 需要真正由预测 state 生成未来像素／mask，并获得匹配参考。现有空间 decoder 需要对应时点原图，不能输入未来原图再称为 future forecasting。 |
| C-index、生存、治疗规划 | 需要新目标、时间／删失协议和相应输入假设，不是给现有随访报告评分多加一列。 |

所有核心指标都需要 matched direct、Copy Current／状态保持和转移先验等对照，但这些是归因实验与基线，不单独包装为第四个“评价指标”。这次讨论不扩大 Table 2。

## 7. 与现有代码的距离

| 项目 | 当前核对状态 | 接入前需要完成的工作 |
|---|---|---|
| Finding AP | [metrics.py](../code/medworld_table1/metrics.py) 已有 | 扩表前冻结类别与参考掩码；核验各方法连续分数接口 |
| AUROC | 当前主评分函数未返回 | 在相同参考集合上补充并验证；没有正负支持的类别不计 |
| Transition F1 | 当前逐疾病新发／消退事件已计算，并汇总为 Transition F1 | 保留该宏平均为主表一列；Onset／Resolution 分解放附录，保持稳定病例误报计数 |
| Direction／程度 | [semantic_metrics.py](../code/medworld_table1/semantic_metrics.py) 有独立原型；[语义评价笔记](0910_richer_semantic_evaluation.md) 明确尚未验收 | 核验参考比较对象、程度／侧别、标签覆盖与专家一致性；不能把原型输出直接填主表 |
| RadGraph／CheXbert | 现有评估流程已有 | 固定模型版本、报告范围与指标变体 |
| GREEN | 本轮检查的主评分实现未接入 | 固定本地评价器、prompt 与失败计分；在代表性的未来报告错误上做人工抽查 |
| Brier／ECE | 本轮检查的主评分实现未接入 | 先核验概率接口、有限值和标签对应，再实现分箱、宏平均与验证集校准 |

优先工作应是锁定四个评价问题和标签定义，并用参考样本审核“改善／恶化”是否可稳定标注。所有新指标的规则在模型比较前冻结。统计区间按患者配对 bootstrap；同一患者多个 pair 不当独立患者。未知参考、预测解析失败、各指标覆盖率和 per-class support 都需记录。

caption 已用逐列箭头替代 **“higher is better”**，Brier／ECE 越低越好。已同步至 [Table 1 源码](../27cvpr/tables/table1_future.tex)、[正文](../27cvpr/sections/6_results_analysis.tex)、[附录协议](../27cvpr/sections/b_protocol_details.tex)和[计划说明](../27cvpr/plans/README.md)。所有待跑分数仍为 TBD，未开展本方案下的新实验。
