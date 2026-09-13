# 09-12 实验小结：已完成结果、失败尝试与在跑基线

整理日期：2026-09-12（北京时间）。本文记录 09-09 至 09-12 的实际尝试，数值取自本地运行的原始指标与比较文件，没有重新训练或重新评分。队列状态截取于 **2026-09-12 09:56**，后续进度以本地运行目录为准。

**09-13 补充入口：** 下文当时在跑的冻结 VLM 密集任务已于 **09-12 12:14** 完成 55/55 项。最终分割／×4 超分、解剖定位和方向分数，以及实验时间证据，见 [Table 1／2 结果来源与实验时间](0913_table1_table2_results_provenance.md)；本页保留 09:56 的历史快照。

[实验索引](../experiments/README.md) · [当前事项](TODO.md)

## 1. 目前完成了什么

| 实验 | 本次核对的状态 | 主要发现 |
|---|---|---|
| 09-09 CXR＋IV 纵向未来预测 | 训练和评估已完成，09-10 已做新旧同病例复算 | 主模型的报告表达与变化预测退步，报告模板化仍明显 |
| 09-10 旧 Stage 1，8 slots 全任务共享 | joint／冻结 encoder 对照均完成 22,616 步 | joint 分类和报告读出较高，分割和 SR 没有同步改善 |
| 09-11 Stage 1，4＋4 slots／无 slots | 两组均完成 24,000 步，早期验证与最终测试均有匹配比较 | slots 分类更高，疾病列表 macro F1 和 Dice 更低；SR 差距很小 |
| 09-11 六个原始 Qwen／MedGemma 零样本模型 | 六个模型均完成，每个模型 7,382 个请求，汇总报告更新至 09-12 00:15 | 已获得未来预测七项指标、当前分类／报告及派生 QA 结果 |
| 09-12 冻结 VLM＋独立下游 heads | 队列共 55 项；快照为完成 1、运行 5、排队 49、最终失败 0 | 共享 V-JEPA 缓存已完成，五个 VLM 缓存在提取；训练头和方向评测尚无最终分数 |

这里的零样本评测、报告辅助四任务训练和新密集任务适配使用不同输入或任务协议，分别列出。论文扩展主表尚未回填这些分数。

## 2. 已完成的 4＋4 slots 与无 slots 对照

问题是：在相同视觉前端、Qwen3.5-0.8B、任务 heads 和训练样本下，将完整图文 tokens 压缩为 8 个状态 slots，是否改善当前任务读出？分类读前 4 个 slots，疾病列表读全部 8 个，分割／SR 读后 4 个；无 slots 模型直接读完整 tokens。

两组从相同公共参数初始值开始，基础 Qwen／V-JEPA 权重冻结；可训练部分包括适配器、LoRA 和读出。公平性记录核对了 24,000 个 optimizer steps、48,000 个 microbatches 的样本顺序，每任务各更新 6,000 次。相同步数不代表相同 FLOPs 或运行时间。

**最终测试：固定 final checkpoint，24,000 步。** `comparison_final.json` 确认两组评估样本与目标相同。分类 353 张，疾病列表 207 个问题／36 位患者，分割 447 张，SR 536 张。

| 方法 | 分类 AUROC ↑ | 分类 AP ↑ | 疾病列表 macro F1 ↑ | Dice pseudo ↑ | PSNR ×2（dB）↑ | SSIM ×2 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen0.8B，无 slots | 0.9189 | 0.9867 | 0.5209 | 0.6143 | 38.0069 | 0.9742 |
| Ours，4＋4 slots | 0.9663 | 0.9931 | 0.4776 | 0.6079 | 38.0361 | 0.9743 |
| Ours − 无 slots | +0.0475 | +0.0064 | -0.0433 | -0.0064 | +0.0292 | +0.0001 |

第 3,176 步验证也已完成：Ours／无 slots 的 AUROC 为 0.9108／0.8428，疾病列表 macro F1 为 0.2329／0.2464，Dice 为 0.4622／0.4424。分割在早期验证与最终测试中的相对顺序不同，不能用早期预览代替最终比较。

这轮支持“分类读出有所改善”的点估计结论，尚不支持 slots 对所有任务都有收益。疾病列表与分割需要进一步看错误类型；SR 的 +0.0292 dB 很小，未计算本轮差值置信区间或多 seed 结果，不据此宣称稳定优势。

解释这些分数时需保留以下条件：

- 分类和疾病识别允许当前报告进入 encoder，属于报告辅助读出，不能与下节的 image-only 零样本分数直接比较。
- 疾病列表来自本地 Chest ImaGenome 派生阳性问题，不含空答案，未替代官方 MIMIC-CXR-VQA。
- Dice 衡量与 CXAS 三器官伪标签的一致性；SR 使用边长 ×2 的合成退化。
- Ours 的最终 `metrics.json` 中 `step` 字段为空；24,000 步由训练状态及匹配比较文件确认，checkpoint 为 `checkpoint_final.pt`。

设置细节：[4＋4 执行记录](0911_stage1_slot44_run.md)、[无 slots 定义](0911_qwen08_noslots_baseline.md)。

## 3. 六个原始 VLM 的零样本评测

这一轮不加载本项目训练后的权重。Qwen3.5 使用 0.8B、4B、9B 和 27B-FP8；MedGemma 使用 1.5-4B 与 **v1-27B**。27B 的精度、模型代际不同，模型规模不能作为唯一解释变量。

### 未来预测：297 对／94 位患者

输入为源图、截断后的当前报告、源时点 EHR 和请求 horizon；不提供未来图／报告。AP、AUROC、Brier、ECE 使用固定 Yes／No 候选的归一化条件似然。Transition F1 来自生成报告的 CheXbert 标签；GREEN 单独读取 `green_metrics.json`。

| 模型 | AP ↑ | AUROC ↑ | Transition F1 ↑ | RadGraph F1 ↑ | GREEN ↑ | Brier ↓ | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 0.8321 | 0.6838 | 0.0897 | 0.1982 | 0.2925 | 0.3529 | 0.4451 |
| Qwen3.5-4B | 0.8460 | 0.7572 | 0.1487 | 0.2098 | 0.2990 | 0.2062 | 0.2656 |
| Qwen3.5-9B | 0.8555 | 0.8177 | 0.1360 | 0.2044 | 0.3170 | 0.2022 | 0.2790 |
| Qwen3.5-27B-FP8 | 0.8687 | 0.8031 | 0.1071 | 0.2149 | 0.3398 | 0.3122 | 0.3983 |
| MedGemma-1.5-4B | 0.8415 | 0.7225 | 0.0549 | 0.1921 | 0.2949 | 0.5270 | 0.5867 |
| MedGemma-27B（v1） | 0.8496 | 0.8274 | 0.1421 | 0.2068 | 0.2878 | 0.1850 | 0.2048 |

八项计划指标中的 **Direction F1 尚缺该 297 对上的合格方向参考**，此处不计分。下节新增的 82 对方向队列是另一套输入与队列，不能直接补成同一测试集的第八项成绩。

在这些点估计中，Qwen27B 的 AP、RadGraph、GREEN 较高；MedGemma27B 的 AUROC 较高、Brier／ECE 较低；Transition F1 最高的是 Qwen4B。扩大模型没有让所有指标单调改善。高 AP 也不等于变化预测或概率校准良好；需要同时看参考阳性率、逐类支持和不同指标。

### 当前状态：image-only 分类与报告生成

分类 353 张，报告生成 507 张；均不输入同次报告。表中的 CheXbert 为本轮固定实现的 macro-positive F1。

| 模型 | 分类 AUROC ↑ | 分类 AP ↑ | 报告 RadGraph ↑ | 报告 CheXbert ↑ |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | 0.6919 | 0.8525 | 0.1819 | 0.2222 |
| Qwen3.5-4B | 0.7295 | 0.8636 | 0.2238 | 0.4035 |
| Qwen3.5-9B | 0.7154 | 0.8546 | 0.2398 | 0.3797 |
| Qwen3.5-27B-FP8 | 0.7340 | 0.8675 | 0.2300 | 0.4159 |
| MedGemma-1.5-4B | 0.7574 | 0.8607 | 0.2172 | 0.3830 |
| MedGemma-27B（v1） | 0.6827 | 0.8455 | 0.1737 | 0.3606 |

### 派生疾病列表 QA：207 个问题／36 位患者

这一小样本检查只含非空阳性答案，单独报告。解析失败保留在分母中。

| 模型 | Exact match ↑ | Micro F1 ↑ | 解析失败数 |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 0.0145 | 0.2372 | 67 |
| Qwen3.5-4B | 0.0725 | 0.2508 | 75 |
| Qwen3.5-9B | 0.1208 | 0.5019 | 5 |
| Qwen3.5-27B-FP8 | 0.1594 | 0.4621 | 0 |
| MedGemma-1.5-4B | 0.0000 | 0.0248 | 0 |
| MedGemma-27B（v1） | 0.1063 | 0.3654 | 2 |

小 Qwen 的格式失败较多；MedGemma4B 虽然没有解析失败，答案集合匹配仍很低。这说明“输出可解析”和“回答正确”需要分别核对，后续应检查格式服从和疾病词表映射，不能删除失败回答后比较。

官方 VQA 和 MS-CXR 尚未在本地就绪；原生文本 VLM 没有分割／超分输出接口。以上点估计不证明统计显著，也未排除公开权重的预训练数据交叠。完整输入、评分与限制见[原始模型评测协议](0911_raw_model_baseline_sweep.md)。

## 4. 正在尝试的冻结 VLM 密集任务

新实验用于检查：给六个冻结 VLM 加上独立训练的任务头后，能否支持分割、空间 ×4 超分和解剖区域定位。分割／SR 各比较原图、冻结 V-JEPA、冻结 V-JEPA＋新 adapter 三个分支，所有分支都同时读取对应 VLM 的图像 token hidden states。这里训练的是下游 heads／新 adapter，不使用 Ours 的训练后 checkpoint。

同一任务／分支在六个模型上固定初始化、样本顺序和 20 epochs，主结果取 epoch 20。SR 三路共用同一份 LR 缓存；V-JEPA 分支没有像素旁路。协议与真实样本反向传播检查已有记录，但它们不是最终测试成绩。

| 内容 | 已固定的规模／边界 | 09-12 快照 |
|---|---|---|
| 分割伪标签／SR | 4,096 train／249 validation／447 test；CXAS 三器官；空间 ×4 SR | 训练头最终分数待完成 |
| 人工分割测试 | Montgomery 138 张，仅两肺，未参与训练或模型选择 | 待对应头训练完成 |
| 解剖区域定位 | Chest ImaGenome 人工区域框；26 类；患者 400／50／50；查询 20,766／2,597／2,598 | 待训练；不是 MS-CXR 病灶短语定位 |
| 方向预测 | 从 284 对筛为 82 对／154 个 finding-scope 字段；源图＋源报告＋horizon，无 EHR | 六个推理任务排队；单独报告 |
| Bicubic ×4 | 与新 SR 相同的 447 张测试图，无训练 | PSNR **29.7421 dB**，SSIM **0.8998** |

55 项队列由 1 个共享 V-JEPA 缓存、6 个 VLM 缓存、6 个方向推理、42 个独立头训练构成。09:56 时共享缓存完成，Qwen0.8B／4B／9B／27B-FP8 和 MedGemma4B 在提取特征；MedGemma27B 等待资源。Qwen0.8B 特征提取记录为第 2 次尝试，因此“当前最终失败 0”不代表全程没有重试。

已有工程尝试包括：为 Qwen27B 原生 FP8 固定兼容的隔离 kernel 依赖；MedGemma27B 三卡真实样本 HR／LR 特征提取；三种分支的真实数据梯度检查。完整设置与来源见[09-12 密集基线执行记录](0912_frozen_vlm_dense_baselines.md)。原定结果检查时间为 **09-14 08:00，北京时间**；本文不把在跑任务提前记为完成。

## 5. 需要保留的负结果与后续排查

09-09 的 CXR＋IV 纵向训练没有带来整体改善。旧／新测试集规模不同，后来在 267 对／85 位患者的相同病例上复算：主模型 AUPRC 从 0.8149 到 0.8269，但 CheXbert F1 从 0.6151 降到 0.4333，Transition F1 从 0.2842 降到 0.1042。患者 bootstrap 中，后两项下降区间未跨零；AUPRC 差值区间跨零。它比较的是两套流程，不能单独归因于 EHR 或训练时长。

完整新测试集的主模型 297 份输出只有 12 种报告，前五模板占 92.6%；真实未来 state 的报告读出也没有明显恢复。排查重点因此转向 Stage 1 状态与报告 decoder 的训练及接口，尚不能认定唯一原因。复算细节见[09-10 纵向实验复盘](0910_overnight_results_comparison.md)。

09-10 旧 Stage 1 的 joint 分类／报告优于冻结 encoder，但三器官 Dice 为 0.5588／0.5671，SR PSNR 为约 38.09／38.20 dB。这个对照两边都有 slots，不能当作有／无 slots 消融；旧 `diagnosis` 是报告生成，也不能与新疾病列表 F1 混比。详见[旧 Stage 1 指标](0911_stage1_downstream_results.md)。

接下来先完成当前队列和四任务失败样例复盘，再处理官方 VQA、MS-CXR 数据接入与方向标签验收。新评分已在原始 VLM 评测路径接入七项；将来对 Ours／direct 重跑时，需要统一连续分数、参考掩码、输入、checkpoint 和患者统计协议。

## 6. 原始记录与版本入口

这份 Markdown 随 Git 提交，方便在 GitHub 阅读。下面是本地原始文件路径，`runs/` 下的病例、预测、日志、checkpoint 和冻结源码不上传；分数更新时新增带日期记录，不将本页当作实时报告。

| 来源 | 本地路径（相对项目根目录） |
|---|---|
| 4＋4 最终指标／步数 | `code/medworld_stage1/runs/slot44_20260911/joint/evaluation/metrics.json`、`joint/status.json`（同一 run） |
| 匹配比较 | `code/medworld_stage1/runs/qwen08_noslots_20260911/comparison_early.json`、`comparison_final.json` |
| 训练匹配核验 | `code/medworld_stage1/runs/qwen08_noslots_20260911/joint/fairness_audit.json` |
| 六模型指标／GREEN | `code/medworld_baselines/runs/raw_models_20260911/<model>/test/metrics.json`、`green_metrics.json` |
| 零样本协议／运行状态 | `code/medworld_baselines/runs/raw_models_20260911/protocol.json`、`coordinator_status.json`、`<model>/test/status.json` |
| 新队列／Bicubic | `code/medworld_dense_baselines/runs/dense_20260912/status.json`、`queue.json`、`bicubic_metrics.json` |
| 在跑结果预览 | `code/medworld_dense_baselines/runs/dense_20260912/preview/full_tables.md` |

本轮代码提交：`e14b24a`（原始模型零样本评测）、`bd63b71`（冻结 VLM 密集任务队列）。代码入口分别为 [medworld_baselines](../code/medworld_baselines/README.md) 与 [medworld_dense_baselines](../code/medworld_dense_baselines/README.md)。论文位于 `26iclr/` 的独立仓库，论文中的实验计划不等于这里已完成的实验结果。

归档前核验：本页最终匹配比较、六模型三组结果表及 Bicubic 数值逐项与原始 JSON 对齐；六模型完成状态和请求数已检查。原始模型协议 6 项测试、密集任务协议 7 项测试全部通过（包含本地患者划分和 LR 缓存核验）；命令见[开发说明](DEVELOPMENT.md)。
