# 09-15：两表 slots 版本与文本解码核对

## 结论与使用要求

后续主模型应统一使用**四个 JEPA／语言融合深度槽＋四个原生视觉深度槽，`[B,8,1024]`**，并保留报告监督及自由文本生成。当前表内已填的 slots 分数不能算作这一完整新版模型的结果。

| 位置 | 实际来源 | 与新版的关系 |
| --- | --- | --- |
| Table 1 已填的 9B slots／shuffled／full-token 三行 | `medworld_table1/runs/qwen9b_ablation_20260914`；最后语言层八个查询，slots 为 `8×4096` | 旧版预测原型，表下注已标明 legacy |
| Table 2 已填的 9B vision＋slots／shuffled | `frozen_slots_20260913` 与 `frozen_slots_shuffled_remaining_20260913`；冻结视觉塔四个深度汇聚，`4×1024` | 四槽视觉对照，只训练密集任务头；不是完整新版八槽 |
| Table 2 full model 行 | 全部 TBD | 尚无合格的完整六任务结果 |
| 新 Table 1 开发／正式队列 | `medworld_native_forecast`，`runs/qwen9b_native_20260915` | 已采用新版 4＋4；核对时共享 Stage 1 正在运行，Stage 2／评分待执行 |
| 新 Table 2 四任务原型 | `medworld_multitask` | 已采用新版 4＋4，0.8B／9B 真实短测通过，完整预算及正式评测待完成 |

原始表及生成器：[Table 1](../27cvpr/tables/table1_future.tex)、[Table 2](../27cvpr/tables/table2_downstream.tex)、[populate_results.py](../27cvpr/plans/populate_results.py)。本次不改写历史分数的版本归属。新版正式结果须由对应新 checkpoint 完整评测后填写。

两条新版实现共享 4＋4 设计和输出形状，但各自实现、训练，尚未统一成一个已完成六任务和未来预测的 checkpoint。特别是，不能把 Table 2 的 latent 直接交给另一个实验的 Table 1 decoder，仅凭形状相同认为兼容。

## 文本编码和解码实际做到哪一步

编码器可将当前图像与允许的当前报告／EHR 编成状态；后四个视觉槽不读取文本。Table 2 当前报告任务和新 Table 1 Stage 1 的当前图像报告重建均不把待生成的报告送入状态编码器。

已有训练路径：

1. 新 Table 1 Stage 1：`image → slots → state_only_prefix → Qwen → current report`，有独立的 `state_text` token CE，同时训练原生当前图像报告分支。
2. 新 Table 1 Stage 2：`current observation → S_t → world(S_t,h) → predicted slots`，将预测槽接入原生当前图文 decoder，联合未来报告 CE、finding BCE 和 latent loss。
3. 新 Table 2：报告读全部八槽，投影到 Qwen 语言维度后自回归生成；不是只输出疾病列表或分类 logits。

已有 0.8B／9B GPU 验证单独反传未来报告 CE，确认八个查询及 predictor 梯度非零：[9B 证据](../code/medworld_native_forecast/runs/verify_20260915/qwen9b_batch2.json)。因此不是仅靠 latent MSE 训练预测槽。

本次补充 [codec.decode_reports](../code/medworld_native_forecast/codec.py)：直接接收本 checkpoint 的当前／预测 `[B,8,1024]` tensor，用已训练的 state-only prefix 和 decoder 输出报告字符串。调用方无需重新提供图像、源报告或未来监督。支持 CPU tensor 保存重载后解码；拒绝错误形状、非浮点、非有限状态和不合法生成预算。

正式 Table 1 继续使用原生当前图文＋预测槽的 `model.predict` 协议；独立 state-only 生成用于证明 latent 可读出和开展相应消融，不混用两种接口的指标。新增代码不改变正在运行的 `model.py`、`data.py`、`run.py`，三者 SHA256 均与正式队列启动记录一致。

## 本次验证

- CPU：`python -m pytest -q code/medworld_native_forecast/tests`，42 项通过。其中新增 14 项覆盖只传 latent、逐样本 EOS、缓存位置、生成预算和非法输入。
- GPU：新增 [verify_codec.py](../code/medworld_native_forecast/verify_codec.py)，分别读取 0.8B／9B 的已有短测 checkpoint，用两条 validation source observation 生成当前及预测状态；从 pair 输入移除 target，解码阶段禁止调用 observation encoder 或原生图文接口。
- 两个规模、两类状态均产生非空文本（每组两个样本，最多 64 个新 tokens），与 Qwen 原生缓存生成的前 12 tokens 一致；tensor 保存／重载逐元素一致，同一缓存生成设置下报告也一致。
- [0.8B 结果](../code/medworld_native_forecast/runs/codec_audit_20260915/qwen08b.json)；[9B 结果](../code/medworld_native_forecast/runs/codec_audit_20260915/qwen9b_qwen_reference.json)。9B 使用早期 smoke checkpoint，缺少新版执行签名，按已有 runner 的 smoke 兼容规则加载并明确记录；不用于正式初始化或评分。
- 0.8B 的预测状态在 BF16 缓存／完整重算两种模式下出现贪心文本差异；额外核对同一 token 前缀时观察到数值差异。该结果保留为 `cached_uncached_equal=false`，独立入口与 Qwen 原生缓存生成仍一致；不宣称不同计算模式逐字相同。

这些检查确认**接口、自由生成和序列化后的可读出**，不证明报告临床质量。输出文件均标记 `diagnostic_only=true`、`table1_ready=false`、`table2_ready=false`。

## 完成报告能力验收仍需什么

正式训练完成后，在既定 held-out cohort 上评估当前／未来报告的 RadGraph、CheXbert／GREEN 等指标，保留空报告和重复报告统计。还应比较真实 predicted slots、shuffled／去除 slots 的报告变化，判断 decoder 是否有效利用预测状态；梯度非零本身不能证明这一点。

“编解码”在这里指将观察压缩成状态，再从状态生成语义报告。**当前没有纯文本→八槽→原文的无损自编码器**；视觉四槽需要图像，压缩状态也不保证逐字恢复原报告。若额外需要纯文本独立编码，应另定接口和训练目标。
