# 2026-09-16 CLARITY 数据调研：脑 MRI 能否用于 MedWorld-JEPA

初次核对日期：2026-09-16；UCSF 下载信息于 2026-09-17 补充。重点是 **brain／脑肿瘤数据**。09-16 实际下载并核对了三个小型 XLSX；09-17 用户提供经网页操作获得的 S3 签名链接，并授权将 UCSF 压缩包下载到 `/home/data2/chk/data`。文件规模和传输说明见第 9 节。

**建议先接入 MU-Glioma-Post，随后用 UCSF-ALPTDG 做外部验证；ISPY-2 暂列第二阶段。** 两个脑数据集能支持“未来影像状态／肿瘤负荷预测＋当前肿瘤分割”的扩展实验，但不能直接接入现有胸片图文训练流程。公开数据没有已确认可用的逐次放射报告，当前代码还需要 MRI 输入、任务标签和缺失文本适配。以下“接入成本”和优先级是结合本项目作出的判断，不是数据提供方承诺。

## 1. 找到的是哪篇论文

准确名称为 **CLARITY**，不是 Clarify。本地文件：[26-ECCV-CLARITY.pdf](../related_works/26-ECCV-CLARITY.pdf)。该 PDF 是 arXiv:2512.08029v3，2026-07-06 版本，封面题目为 *Medical World Model for Guiding Treatment Decisions by Simulating Context-Aware Disease Trajectories*。arXiv 页面仍保留早期的 *…by Modeling Context-Aware Disease Trajectories in Latent Space* 题名，并标注 Accepted to ECCV 2026。[论文入口](https://arxiv.org/abs/2512.08029)

项目已在 [Related Work](../27cvpr/sections/2_related_work.tex) 和 [参考文献](../27cvpr/main.bib) 引用 `ding2025clarity`。本轮不改论文正文。

本地 PDF §4.1（第 10 页）明确列出：

| 数据 | CLARITY 报告的规模 | 在 CLARITY 中的用途 |
|---|---:|---|
| MU-Glioma-Post | 203 人、654 次 MRI 随访 | 脑肿瘤模型训练和内部评测 |
| UCSF-ALPTDG | 298 人 | 仅用 MU 训练后的外部零样本评测 |
| ISPY-2 | 985 人 | 独立训练乳腺癌模型，验证跨癌种适用性 |

这里的两个 brain 队列都是**治疗后纵向胶质瘤 MRI**，并非直接使用普通 BraTS 分割集。当前稿件附录中的 BraTS 迁移评测不能替代这个纵向实验。

## 2. 规模、下载和申请：一览表

| 数据集 | 公开版本与实际规模 | 影像下载体积 | 获取／申请要求 | 本项目优先级 |
|---|---|---|---|---|
| **MU-Glioma-Post** | 官方页 203 人、596 个时间点；数据论文写 594。当前临床表实测 203 人、597 个数值型 MRI 时间项，详见下节 | **11 GB**，预处理 NIfTI＋分割；临床表约 102 KB、体积表约 311 KB | TCIA 开放、CC BY 4.0；影像入口为 **Aspera**，表格可直接 HTTPS 下载。未发现项目审批或培训要求 | **第一优先**：数据小、已有多序列和分割；最适合先做脑 MRI 实验 |
| **UCSF-ALPTDG** | **298 人、每人 2 次，共 596 次 MRI／298 个纵向对** | **29.74 GB ZIP（27.70 GiB）**；ZIP 内文件合计 **30.18 GB**，09-17 实测 | UCSF 官方仓库；**非商业使用＋DUA**。09-17 用户完成网页操作后提供 S3 签名下载链接；本轮未独立核对后续表单和完整 DUA 文本 | **第二优先**：适合 MU 训练后的外部预测／分割评测 |
| **ISPY-2 Imaging Cohort 1** | **985 人＝719 ISPY2＋266 ACRIN-6698**；3,677 studies，43,356 series | **2.41 TB**；仅 719 人版本为 1.75 TB。二者包含重叠数据，不应相加当作完整队列所需体积 | TCIA 开放、CC BY 4.0；下载 manifest 后使用 TCIA Data Retriever／兼容的 NBIA 工具 | **第三优先**：治疗反应任务有价值，但下载和 DCE 预处理成本高 |

来源：[MU 官方 Data Access](https://www.cancerimagingarchive.net/collection/mu-glioma-post/)、[MU 数据论文](https://www.nature.com/articles/s41597-025-06011-7)、[UCSF 官方入口](https://imagingdatasets.ucsf.edu/dataset/2)、[UCSF 数据论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11294954/)、[ISPY2 官方 Data Access](https://www.cancerimagingarchive.net/collection/ispy2/)。

**申请容易与下载容易是两件事。** MU／ISPY2 的公开访问规则没有 MIMIC 式资质培训门槛，但 MU 需要 Aspera 客户端，ISPY2 有 TB 级传输负担。UCSF 的非商业 DUA 与验证码是已确认门槛；不能据此声称需要数周审批，也不能保证所有申请都即时放行。09-17 已取得可用的签名链接并验证 S3 支持 Range 续传；完整文件状态另见第 9 节。

TCIA 在 2026 年迁移 DICOM 基础设施，提供兼容旧 `.tcia` manifest 的新版 Data Retriever；NIfTI 仍使用 Aspera。应以当前集合页为准，不机械照搬 MU 数据论文中关于 NBIA 下载的泛化描述。[TCIA 迁移说明](https://www.cancerimagingarchive.net/evolving-the-cancer-imaging-archive-a-new-hybrid-infrastructure-for-open-science/)、[当前下载 FAQ](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=4555089)

## 3. MU-Glioma-Post：最适合先做，但需要清点时间轴

### 3.1 可以拿到什么

公开影像是四序列 **T1、增强 T1、T2、FLAIR**，提供去颅骨、配准、重采样后的 NIfTI，以及自动分割后经人工审核／修正的四类标签：NETC、SNFH、ET、RC（非增强核心、周围 FLAIR 高信号、增强组织、切除腔）。临床表包含诊断、分级、分子标志物、治疗起止时间和 MRI 相对诊断日。官方提供的是结构化临床资料，没有列出可直接用于报告生成的逐次放射报告文件。[MU 数据与标签说明](https://www.cancerimagingarchive.net/collection/mu-glioma-post/)

这正好补充本项目目前的胸片任务：既能学习随访状态变化，也能检验同一状态表示是否保留肿瘤空间信息。但只有 203 人，不能把切片数当成独立患者数，也不适合据此宣称大规模脑影像预训练。

### 3.2 实际下载临床表后的审计

本轮读取的是官方目前链接的 [MU-Glioma-Post_ClinicalData-July2025.xlsx](https://www.cancerimagingarchive.net/wp-content/uploads/MU-Glioma-Post_ClinicalData-July2025.xlsx)，只统计 `MU Glioma Post` 工作表中有效患者行及六个 MRI 时间列的数值单元格。

| 实测项目 | 结果 |
|---|---:|
| 不同患者 | **203** |
| 数值型 MRI 时间项 | **597** |
| 0／1／2／3／4／5／6 个数值型时间点的患者数 | **1／46／46／50／17／17／26** |
| 至少两个数值型 MRI 时间点的患者 | **156** |
| 按实际相对日排序、去重后的候选相邻时间对 | **395** |
| 候选间隔：最小／中位数／最大 | **3／77／1,109 天** |
| 时间列编号顺序与日期顺序不一致的患者 | **1** |
| 死亡标记为 1；具有数值型死亡时间 | **97；96** |

这些是**表格审计结果，不是已验证的训练样本量**。395 对只表示相邻的已记录数值时间，尚未证明每端都有四序列、分割及正确的时间点映射，也不保证是完整临床时间线中的相邻检查。非数值项未被推断为日期；正式入组需与影像目录逐一对齐。

由此可见，CLARITY 的 **654**、TCIA 摘要的 **596**、数据论文的 **594**、当前表格的 **597** 存在真实口径差异。本轮无法解释差异来源。不能直接把任何一个数写成我们最终入组数，也不要用 TCIA 的 **2,978 images** 当作 2,978 次纵向 MRI 检查。

### 3.3 两个容易被忽略的接入问题

**体积表不能按行顺序当成随访序列。** 实际读取 [Segmentation_Volumes.xlsx](https://www.cancerimagingarchive.net/wp-content/uploads/MU-Glioma-Post_Segmentation_Volumes.xlsx) 后发现，四个工作表的患者 ID 命名混用：部分有 `Post-treatment_X` 后缀，部分只有重复的患者 ID；没有独立时间点列。优先从每个影像文件夹的分割 mask 重算体积，再与明确的时间点配对；现成体积表适合交叉核验，不能直接认定第 n 行就是第 n 次检查。

**生存终点还不够干净。** 本轮临床表含死亡标记与诊断至死亡天数，但未找到独立的末次存活接触／删失日期列。97 个死亡标记与 96 个数值死亡时间也需核对。因而首个实验建议做影像／体积预测和当前分割；生存预测要先形成明确的删失与缺失处理协议。

三份原始小表、SHA-256、聚合统计和重算脚本保存在 [本地审计资产](assets/clarity_data_audit_20260916/README.md)；[统计 JSON](assets/clarity_data_audit_20260916/metadata_audit.json)。以上新增统计可直接复算。

## 4. UCSF-ALPTDG：适合外部验证，关注变化标签与配准

数据提供 **298 个两时间点病例**，四序列 MRI、各时间点肿瘤分区，以及 ET／SNFH 的专家纵向变化标注；临床数据包括治疗史、分子信息和时间信息。数据论文报告两次扫描间隔中位数为 **65 天**。这比从文本自动推断“变好／变坏”更适合做空间变化评价。[UCSF 数据说明](https://pmc.ncbi.nlm.nih.gov/articles/PMC11294954/)

**最自然的用途是 MU 内部训练／验证，UCSF 整体作为外部测试。** 先锁定 MU 上的 checkpoint、分割标签映射、体积变化阈值和评价方式，再测 UCSF。若用 UCSF 调参或适配，需另设 split 并改称适配实验；不能继续称为外部零样本验证。

需要在实验协议里处理三个细节：

- 原始构建在有多个可选随访时选取存在间隔变化的两次检查。其变化比例不代表自然就诊人群；每人只有两次影像，也不足以直接检验多步长期 rollout。
- 论文的预处理将影像配准到**第二次扫描的增强 T1**。这意味着公开当前端图像可能已受未来端几何信息影响。用于严格 source-only 预测前需核对变换／原始数据；如无法消除，应明确标注为预配准回顾性评测，不能声称完全前瞻。
- 该队列有 **62 人与早期 UCSF 术前胶质瘤数据重叠**。如果选择的 MRI 预训练模型或训练数据包含该来源，需要查重／披露；不能仅靠集合名不同就认定预训练与外部测试无患者重叠。

上述细节见 [UCSF 数据论文的队列与预处理部分](https://pmc.ncbi.nlm.nih.gov/articles/PMC11294954/)。其官方 [nnU-Net benchmark](https://github.com/rachitsaluja/UCSF-ALPTDG-benchmarks) 的变化分割模型会输入两端真实图像，可用作观察到变化后的检测参照，不能直接作为只看当前端的未来预测基线。

获取入口是 **UCSF 自有站点，不是猜测的 TCIA collection URL**：[dataset/2](https://imagingdatasets.ucsf.edu/dataset/2)，页面版本 1.0，DOI `10.58078/C21592`。09-16 浏览器点击 Download 后要求 reCAPTCHA；09-17 已通过用户接手网页取得下载链接，包大小和目录信息见第 9 节。非商业使用和 DUA 要求由[数据论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11294954/)及 [RSNA 数据卡](https://atlas.rsna.org/cards/48595a48-e1e6-4ab5-8844-e895a3482c01)交叉确认；没有足够信息报告所有用户通用的审批耗时。

## 5. ISPY-2：可以用，但不适合当前先全量下载

CLARITY 的 **985 人**对应 TCIA 的 *I-SPY2 Imaging Cohort 1* 完整 manifest；只下载集合首页的 719 人版本会缺少 ACRIN-6698 中的 266 人。官方完整包为 **2.41 TB**，7,575,549 个 DICOM image records；页头 **4.16 TB**不能理解为获得 985 人必须额外下载的体积。[ISPY2 Data Access](https://www.cancerimagingarchive.net/collection/ispy2/)

纵向 MRI 包括治疗前 T0、早期治疗 T1、中期 T2、治疗后术前 T3；并非所有患者都具备全部时间点。**单次 DCE 检查里的增强动态相位，与跨治疗阶段的纵向时间是两条轴**，构造世界模型样本时不能混在一起。影像既有原始序列也有派生增强／分析对象，不能把全部 series 直接当作训练样本。[官方影像协议](https://www.cancerimagingarchive.net/collection/ispy2/)

本轮实际下载的 [985 人临床表](https://www.cancerimagingarchive.net/wp-content/uploads/ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx) 有 **985 个不同 ID、10 列**：`Patient_ID, Arm, HR, HER2, MP, pCR, Age_at_Screening, Race, menopausal_status, ethnicity`。其中可直接用的终点是 **pCR**，不是完整的 OS／PFS 生存时间；表内也没有逐次放射报告和影像时间。时间轴需要与影像 manifest／DICOM 关联。官方另有 384 人的多特征 MRI 表，不能当作全部 985 人的完整特征。

适合新增的实验是“早期 MRI＋已知治疗组预测后续肿瘤负荷／pCR”。建议先选固定治疗时间对和必要序列的小队列，并预先固定筛选规则，再按需下载。全量 2.41 TB 在持续 **10 MB/s** 下理论传输约 **67 小时**，100 MB/s 约 6.7 小时，尚未计入连接开销、重试和解包；这些是算术估计，不是实测网速。原始包加转换文件和缓存，工程上可先按 **4–6 TB** 工作盘预算评估，实际应由选择的子集决定。

## 6. 对我们当前代码和论文的适配判断

项目论文允许图像＋可用临床文本，但当前融合实现仍以胸片为基础：[模型说明](../code/medworld/README.md)。实际 [temporal.py](../code/medworld/datasets/temporal.py) 使用 PIL 图像，并要求非空报告和六类 finding 标签；[空间 decoder](../code/medworld/decoders.py) 也是现有二维任务结构。所以“数据允许使用”并不等于“当前训练器已支持”。

| 本项目能力／任务 | 脑数据能否支持 | 必要调整 |
|---|---|---|
| 未来状态预测 | **适合，推荐作为扩展主任务** | 多序列 NIfTI 输入；保留真实随访天数；重新训练 MRI 时间分布下的预测器 |
| 当前分割 | **适合，已有四类空间标注** | 统一 MU／UCSF 标签语义与数值映射；加入背景的多类或明确定义的区域 heads；不沿用胸片通道定义 |
| 未来肿瘤负荷／变化预测 | **适合** | 从未来 mask 得到 ET／SNFH 等体积和变化监督；未来图像、mask、差分图只作目标／评价 |
| 自由文本报告生成／报告预测 | **没有已确认的真实报告监督** | 可用截止当前时点的结构化临床文本，或测试无文本设置；模板文本不能冒充真实报告 |
| 六疾病分类、胸片 VQA／grounding | **标签不匹配** | 新定义脑肿瘤任务与评价，不能把缺失标签补零复用现有表格 |
| 超分辨率 | **可构造辅助任务** | 需要单独定义 MRI 退化；不能把重采样后的 1 mm 图像自动当成真实高分辨率金标准 |
| 生存／治疗决策 | **可进一步研究，非首轮目标** | 先审计删失、事件时间、治疗可用时间和混杂；模型风险降低不等于真实治疗获益 |

具体接入建议：

1. **先建 MRI 数据适配层。** 正式体积预测采用能覆盖全体积的切片集合或 3D 编码。单一切片／拼图只用于快速流程检查，不能把得到的结果称为完整 3D 分割。裁剪或切片选择只用源时点证据，不能用未来肿瘤位置挑选输入。
2. **先做 image-only，再加入可追溯的结构化临床上下文。** 无报告时显式设置文本缺失，调整报告强制检查和对应 loss；不能为了满足接口生成“假报告”。来源不明时间的治疗／进展信息保持未知，不把整份全程临床表塞进当前输入。
3. **时间单位统一。** 新版 [predictor.py](../code/medworld/predictor.py) 接收小时并转换为连续天数特征，脑数据的相对日应正确换算。接口支持月级间隔，不代表胸片训练出的模型已学会月级脑肿瘤演变。
4. **先分患者再生成切片、pair 和双向样本。** 同一人的所有时间点必须进入同一 split；图像和状态预训练也不能包含声明为完全未见的测试患者。正向预测和负向回溯分开报告。
5. **保持本文的观察性随访定位。** 如果增加未来区间实际发生的治疗作为输入，应单列“给定治疗条件的预测”协议，并说明这些治疗何时已知；它与当前仅看源时点信息的预测不是同一输入条件，也不能据观察数据验证未实施治疗的反事实效果。

推荐扩展表只保留清晰的几项：当前分割 Dice／HD95、未来 ET／SNFH 体积 MAE、未来体积变化误差；如从体积定义方向标签，阈值在训练／验证集确定，并称为“影像体积变化”，不直接等同临床进展。和 copy-current／零变化、源时点体积回归、直接未来预测、带／不带临床上下文的 JEPA 比较。同一患者多对数据的置信区间按患者重采样，避免把切片当独立样本。

## 7. CLARITY 复现还缺什么

官方代码已经公开：[DingTianxingjian/CLARITY](https://github.com/DingTianxingjian/CLARITY)。但“公开数据＋公开仓库”尚不能保证一键复现论文中的 cohort 和分数：

- README 明确将 `clinical_latest.json` 和 MRI／backbone 权重列为仓库外资源；可选的 `features_output.csv` 还会改变患者纳入范围。仓库提供 [临床提取脚本](https://github.com/DingTianxingjian/CLARITY/blob/main/Predictor/dataset/extract_clinicial.py)，但它另依赖 `pidtime_combo.txt`，输出名也不是 README 所需最终 JSON。因此并非没有预处理代码，而是最终产物及完整映射仍待核对。
- 本次读取的提取脚本会将无死亡信息者的末次 MRI 用作删失时间；遇到死亡日在某次 MRI 之前的冲突，会把死亡日移到最后 MRI 后一天。这是代码里的处理规则，不是原始数据提供的真实生存终点。我们的实验不应未经核对直接继承。
- 官方 [pair loader](https://github.com/DingTianxingjian/CLARITY/blob/main/Predictor/dataset/dataset_glioma_all_pairs_text.py) 构造多种 `i<j` 配对，并按特征／生存信息筛选；所以本轮的 **395 个候选相邻对**不能当作 CLARITY 的训练 pair 数。
- 本地 PDF 的 train／validation／test 写成 **75%／15%／15%**，合计 105%；还同时描述五折交叉验证。需取得患者 split 或明确说明，不能自行改成一个合理比例后声称完全复现。
- PDF 附录写 MRI-CORE；本次公开 README 的默认视觉骨干为 BrainIAC。论文版本与代码默认配置有差别，正式 baseline 需记录具体配置和代码版本。

这些问题不妨碍我们建立自己的 MRI 扩展协议，但目前不宜承诺能直接复现 CLARITY 的原表。若进入对比实验，再向作者核对：MU 的 654 次随访清单、最终时间线／split、UCSF 标签映射与 ISPY2 的实际入组时间对。本轮未联系作者。

## 8. 建议执行顺序与尚未确认事项

**第一步：MU 小规模接入。** 从 [MU 集合页](https://www.cancerimagingarchive.net/collection/mu-glioma-post/) 的 Images and Segmentations 进入 Aspera，取得 11 GB 包；先验证 10–20 位患者的文件／时间／标签对应，再清点全部病例。可先预留约 50–100 GB 工作空间给原文件、转换与缓存，这是工程预算，不是压缩包大小。接入工作量中等：主要花在数据映射、MRI 编码和任务适配，申请不是主要障碍。

**第二步：UCSF 外部集。** 09-17 已取得下载链接，压缩包 29.74 GB；下载后重点核对第二时点配准和标签映射。完整 DUA 细则及是否存在人工审批的通用规则尚未独立核实，不把本次获取流程推广为固定审批时间。

**第三步：ISPY2 按需子集。** 若脑 MRI 扩展已经证明方法可迁移，再决定是否投入乳腺癌 DCE 队列。使用完整 985 人 cohort 对应的 [官方 manifest](https://www.cancerimagingarchive.net/wp-content/uploads/ISPY2-Cohort1-inclu-ACRIN6698-full-manifest.tcia) 定义母集，再按预先确定的规则选择时间点和序列，避免先下载 TB 级数据再决定任务。

对当前文章，最有价值且范围可控的新增证据是：**MU 上学习脑 MRI 的共享状态与未来变化，UCSF 上测试外部泛化，同时评价当前分割。** 它能扩展现有胸片实验证据；真实报告生成仍由已有图文数据承担。最终是否采用，取决于影像文件审计后的有效 pair 数、源时点可用信息和配准条件，而不只看公开页总人数。

## 9. 2026-09-17 UCSF 下载补充

用户接手 VNC 浏览器后提供了 UCSF S3 签名链接，并授权下载。VNC、专用浏览器和相关端口已按用户要求关闭。签名参数不写入研究文档。

对 S3 做 `Range: bytes=0-0` 请求，收到 HTTP 206 和总长 **29,738,537,232 bytes**；随后仅读取 ZIP 中央目录，取得以下信息：

| 项目 | 实测值 |
|---|---|
| 压缩包 | `UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0.zip` |
| 下载大小 | **29.7385 GB／27.696 GiB** |
| ZIP 内非目录文件 | **4,769** |
| 数字命名的患者目录 | **298** |
| ZIP 内文件总大小 | **30,179,666,034 bytes，约 30.18 GB**；影像仍多为 `.nii.gz`，不含进一步展开这些 gzip 文件的空间 |
| 临床表 | `UCSF_PostopGlioma_Table S1 R1 V5.0_UNBLINDED_FINAL.xlsx` |
| 完成后的目标路径 | `/home/data2/chk/data/UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0.zip` |

中央目录审计本身仅获取约 627 KB，不代表完整文件已下载或通过校验。传输使用同目录 `.zip.part` 和 aria2 续传控制文件；分段下载时文件逻辑大小可提前接近总长，**不能用 `ls` 显示的大小当作已下载量**。

运行状态：[status.json](/home/data2/chk/data/.ucsf_alptdg_download/status.json)。下载器成功后逐项检查 ZIP CRC，计算 SHA-256，再改为最终文件名，并生成同名 `.sha256` 和 `.metadata.json`。当前传输及校验是否完成，以该状态文件为准；本轮下载不自动解压。

09-17 续传改由用户级 systemd 服务 `ucsf-alptdg-download.service` 管理，使用 aria2 分段续传，避免聊天工具进程中断影响下载。可运行 `systemctl --user status ucsf-alptdg-download.service` 查看运行状态；服务成功结束后可被自动回收，最终结果仍保留在上述 JSON 与文件旁的校验记录中。

## 10. 2026-09-17 MU-Glioma-Post 下载实测

用户要求尝试另一个数据集，本轮选择约 11 GB 的 MU-Glioma-Post。通过 [TCIA 官方集合页](https://www.cancerimagingarchive.net/collection/mu-glioma-post/) 的公开 Aspera 链接，成功读取 package 1030，并获得下载 transfer specification；本次无需注册、人工审批或验证码。该结果仅说明这条公开链接的当前获取流程。

以下三个表已完整下载到 [/home/data2/chk/data/MU-Glioma-Post/](/home/data2/chk/data/MU-Glioma-Post/)，并记录 SHA-256：

| 文件 | 实测大小 |
|---|---:|
| `MU-Glioma-Post_ClinicalData-July2025.xlsx` | 101,568 bytes |
| `MU-Glioma-Post_Segmentation_Volumes.xlsx` | 307,753 bytes |
| `MR_Scanner_data.xlsx` | 25,875 bytes |

公开文件浏览 API 返回 **203 个患者目录**。抽查 `PatientID_0003` 可见 `Timepoint_1`、`Timepoint_2`、`Timepoint_5`；第一个时间点有 `brain_t1n`、`brain_t1c`、`brain_t2w`、`brain_t2f` 和 `tumorMask` 五个 NIfTI 文件。这也说明时间点编号可能不连续，后续应按实际影像与临床表建立映射。

**影像下载未成功。** 已在私有下载工具目录解包 IBM 官方 Transfer SDK 1.1.5，使用随公开链接取得的下载令牌试传一个影像。`ascp` 返回 `Unable to connect via SSH`，退出码 1，未保存任何 `.nii.gz`。服务返回的传输地址是 `144.30.235.113:33001`；本机直接连接被重置，经现有 HTTP 代理的 CONNECT 尝试也未获得 SSH 握手。尚不能据此区分本机出口、中间网络或远端服务问题。该公开账户的 `gateway_enabled=false`，站点 `http_gateway_url=null`，因此没有可直接改用的已配置 HTTP Gateway。

Aspera 页面显示的 **270.37 KB** 不能当作影像包真实大小：包内是指向真实影像的符号链接，抽查单个增强 T1 的 API 大小已达 **5,251,963 bytes**。本轮没有枚举全部文件求和，完整影像体积仍以 TCIA 官方标注的约 **11 GB** 为参考，尚未独立核验精确字节数。

当前状态与失败记录保存在 [/home/data2/chk/data/.mu_glioma_post_download/](/home/data2/chk/data/.mu_glioma_post_download/)。下一步需要可连接该 Aspera 服务的网络环境，或由 TCIA 提供其他下载入口；这次失败发生在传输连接阶段，不能将它记为数据使用申请被拒。公开链接的访问令牌、口令及签名参数均不写入本研究文档。
