# 09-11 晚间：原始模型 Table 1 / Table 2 基线

用户要求先运行不同大小的原始模型，不尝试 Ours，希望北京时间 2026-09-12 08:00 左右看到一版结果。原始录音经过带术语提示的本地 ASR 重新识别，并结合权重清单确认 Qwen、MedGemma、vLLM。ASR 仍未正确拼出 MedGemma，名称核对依据是本地 checkpoint，不能称录音识别无歧义。

## 模型与运行

| 模型 | 精度 / 并行 |
|---|---|
| Qwen3.5-0.8B | BF16，单卡 |
| Qwen3.5-4B | BF16，单卡 |
| Qwen3.5-9B | BF16，复用现有本地服务 |
| Qwen3.5-27B-FP8 | 已下载 FP8 checkpoint，TP=2 |
| MedGemma-1.5-4B | BF16，单卡 |
| MedGemma-27B | BF16，TP=4；本地实际名称没有 1.5 |

本地没有发现 Qwen3.5-2B，不新增该行。完整 checkpoint revision、配置哈希及权重清单见 [models.json](../code/medworld_baselines/runs/raw_models_20260911/models.json)。不运行训练或本项目方法，不修改已有实验。物理 GPU 4 按历史异常记录排除，现有训练/服务不被停止。

## 冻结评测边界

- Table 1：既有严格 CXR+IV 连接测试集 297 对 / 94 位患者；源影像、384 个共同 Qwen token 的当前报告、源时点 EHR 和请求 horizon。生成器不读取未来影像/报告、目标标签、精确实际间隔。
- Table 2：沿用最新患者划分与 gold 患者覆盖规则；分类 353 张、报告生成 507 张。输入只有影像，QA 另给问题；同次报告全部 withheld，与早期报告辅助四任务成绩不能直接混比。
- 图像共同预处理为保持比例的 512×512 黑色补边；之后执行模型自身视觉处理器。各模型原生视觉分辨率和计算预算并不相等。
- 贪心生成；报告最多 384 token，QA 最多 192 token。Qwen 关闭 thinking。保留空输出、截断率、报告多样性和 HTTP 失败记录。
- 不做测试集校准或调阈值，不声称已排除公开 checkpoint 的预训练数据交叠。

## 指标与缺口

Table 1 AP/AUROC 使用拥有明确正负参考的固定征象集合；Brier/ECE 使用至少有一个明确参考的固定征象集合。逐类 n/阳性/阴性/概率箱均保存。概率为单 token 完整候选 `Yes` / `No` 的原始 log probability 归一化，不是模型写出的百分比。若某候选不在 top-logprobs 中，强制输出该 token 获取其原始似然；服务必须使用 `raw_logprobs`。Qwen 的真实服务检查已证明强制 token 不会把返回概率改成 1。

Transition 采用当前/目标/生成报告的官方 CheXbert 四状态标签，逐征象 onset/resolution F1，对参考支持项宏平均；另外保存固定 0.5 概率阈值版本，不能把两者混为一个分数。Direction 需要疾病/侧别/比较对象均明确的核验参考，现有原型尚未验收，留空。

RadGraph 复用已验证的 `radgraph==0.1.18` / radgraph-xl / partial F1，并保存逐报告分数。GREEN 使用 [官方实现](https://github.com/Stanford-AIMI/GREEN) commit `29567040fba813e1d62503954ec234e056807d05` 和 [官方模型](https://huggingface.co/StanfordAIMI/GREEN-RadLlama2-7b) revision `832389d56bbd45a36bbd61dfa4f0be76e99f1dad`。固定官方 300-word prompt、chat template、tokenizer、错误解析器；vLLM BF16 贪心批量生成，每条最多 2048 总 token。该执行方式与原包 float16、按 padding 后长度分配生成预算不同，保留 provenance 和逐条分析。主结果采用官方分数，附带严格解析覆盖诊断，失败不缩小分母。

Table 2 官方 VQA 与 MS-CXR phrase grounding 数据尚未在本地就绪。已有 207 个 Chest ImaGenome 派生问题来自 36 位患者的人工标注，只有非空阳性答案，因此另表报告 Exact match / micro F1，不能填作完整官方 VQA 成绩。原始文本输出 VLM 没有分割和超分 decoder，按当前论文零样本行协议记 N/A。

## 持久记录

[实时结果与队列](../code/medworld_baselines/runs/raw_models_20260911/REPORT.md) 每 30 秒更新；分数来自每个模型的 `test/metrics.json`，不在此文复制。新代码入口为 [medworld_baselines](../code/medworld_baselines/README.md)，已登记到实验索引。论文主表尚不回填。

协议检查覆盖参考未知值掩码、无支持类别、概率边界、ECE 最后一个箱、非法概率拒绝、问答解析失败惩罚、源输入对未来目标扰动不变性。真实 Qwen 服务已验证 Yes/No 的原始似然接口；其余模型采用同一接口并保存候选 token ID。
