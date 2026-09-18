# MIMIC-CXR-VQA 抽样零样本评测

在 GPU 0 串行测试 Qwen3.5-0.8B、4B、9B 与 MedGemma-1.5-4B。
默认从官方 test 的 13,793 题中按 semantic type × content type 比例抽取 1,024 题，
固定 seed 20260916；选择过程不读取答案。四模型使用同一图像、问题、公开 110 标签词表。

当前评测目录：`runs/pilot_1024_gpu0_20260916_json/`。
首次自由生成诊断轮保存在 `runs/pilot_1024_gpu0_20260916/`；其 Qwen-0.8B 有 416/1,024
条格式非法或截断输出。最终四模型比较统一使用公开词表 JSON 约束解码，以减轻格式遵循的干扰。
两轮样本、图像、问题、提示词完全一致，旧输出与协议均保留。

```bash
PY=/home/data2/chk/workspace/2026/.venv/bin/python
RUN="$PWD/code/medworld_vqa/runs/pilot_1024_gpu0_20260916_json"
TRAIN="$PWD/code/medworld/runs/qwen35_08b_2gpu_day_gpu67_20260916_161228"

# 仅在新的空目录准备；图像、问答、标签、权重哈希与源码均被冻结。
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 PYTHONPATH=code \
  "$PY" -m medworld_vqa.prepare --run "$RUN" --training-run "$TRAIN"

# 启动或断点恢复：只运行此实验冻结的源码。
PYTHONPATH="$RUN/source" "$PY" -m medworld_vqa.run --run "$RUN" --gpu 0

# 验证采样、集合评分、空答案与无效输出的处理。
PYTHONPATH=code "$PY" -m unittest discover -s code/medworld_vqa/tests -v

# 四模型全部完成后，独立复算并导出不含逐题信息的汇总。
PYTHONPATH=code "$PY" -m medworld_vqa.export --run "$RUN" \
  --out results/mimic_cxr_vqa_pilot_20260916
```

本次已完成：[结果报告](../../results/mimic_cxr_vqa_pilot_20260916/README.md)。
四模型共 4,096 条推理均成功返回；评分已通过独立 scikit-learn 复算。GPU 0 已释放。

推理仅读取 `inputs.jsonl`、PNG 和词表；`references.jsonl` 仅供评分。
512×512 RGB 图像保持宽高比并黑色补边，Qwen 使用 512² 视觉像素预算；
MedGemma 使用原生处理器，因此不宣称视觉 token 数和计算量相等。
所有权重为 BF16；greedy decoding，最多 192 新 tokens，Qwen thinking 关闭；
使用 vLLM `structured_outputs.json` 限定为公开词表中的标签数组，不限制每题的正确答案或候选子集。
无报告、EHR、few-shot 例子或项目微调。

指标是答案集合 exact match 与标签级 micro F1，另报告 Verify、Choose、Query，
7 个内容类型、区域/全图、空/非空答案和患者级 bootstrap 95% CI。
仅接受 JSON 字符串数组（允许外层代码围栏），忽略大小写和多余空白，不做同义词映射。
格式错误、词表外标签、截断、请求失败均计错，不丢弃样本。
预测 `[]` 对空参考的 EM 为 1，但不增加 micro F1 的 TP。

`table1_diagnosis_pilot.csv/.tex` 另列排除 plane/gender 及全部训练重叠患者的临床内容子集。
本次 1,024 题 / 434 患者中，43 患者 / 88 题与当前时序训练重叠；
全部 VQA 保留这些题，Diagnosis 主列使用 884 题 / 382 名无训练重叠患者。
`diagnosis_all_sampled` 和 `vqa_train_disjoint` 另行保存，筛选规则在模型推理前确定。
该子集仍包含器械、技术质量等问题，名称为 **Current-image Diagnosis (CXR-VQA, sampled)**；
它不是当前 Table 1 的未来诊断 AP/AUROC，不能替换原列或当成全量官方成绩。
评测采用本地显式集合评分，尚未声称与官方 evaluator 完全等价。

输出：`REPORT.md`、`status.json`、各模型 `predictions.jsonl`、
`scored_predictions.jsonl`、`metrics.json`、`training_overlap.json`。
只使用 GPU 0，遵守现有 UUID 文件锁；仅停止本队列启动的进程。
运行时复用已有队列的进程内 vLLM 设备发现补丁，不改共享 Python 包。

参考：[数据集](https://physionet.org/content/mimic-ext-mimic-cxr-vqa/1.0.0/)、
[公开答案词表](https://github.com/baeseongsu/mimic-cxr-vqa/blob/master/mimiccxrvqa/dataset/ans2idx.json)。

## 文献对齐的短答复测

[文献、指标与协议说明](https://github.com/MVChem/medical_world_model/blob/31c98ca14173620a3cec67da2defe559d05974da/research_notes/0916_mimic_cxr_vqa_literature.md)。
四模型已完成并通过独立复算：[复测结果与限制](../../results/mimic_cxr_vqa_literature_20260916/README.md)。
新运行目录为 `runs/literature_short_1024_gpu0_20260916`；此前同日的
`runs/literature_1024_gpu0_20260916` 是已停止的原问题长答诊断试跑，不是四模型正式结果。

新协议使用原问题加 LLaVA 官方短答后缀，自由生成，不展示答案词表。
分题型指标沿用 AOR 的 Verify Acc / Choose Acc / Query label micro-F1 定义，
由 `literature_score.py` 做本地保守文本解析；详细边界见研究记录。
不会把此适配写成 AOR 官方 evaluator 或监督训练复现。

准备时 `literature_prepare.py` 复用已冻结 pilot 的相同题目、参考与图片，并校验其哈希。
`--repositories` 指向本地已检出的 CheXagent、AOR、Meissa、LLaVA 公共仓库父目录；
记录 commit 与相关文件哈希。生成后只从运行目录的 `source` 快照启动。

```bash
PYTHONPATH=code /home/data2/chk/workspace/2026/.venv/bin/python \
  -m medworld_vqa.literature_prepare \
  --previous code/medworld_vqa/runs/pilot_1024_gpu0_20260916_json \
  --run code/medworld_vqa/runs/NEW_SHORT_ANSWER_RUN \
  --repositories /tmp/cxrvqa_published_research

PYTHONPATH=code/medworld_vqa/runs/NEW_SHORT_ANSWER_RUN/source \
  /home/data2/chk/workspace/2026/.venv/bin/python -m medworld_vqa.run \
  --run code/medworld_vqa/runs/NEW_SHORT_ANSWER_RUN --gpu 0

PYTHONPATH=code /home/data2/chk/workspace/2026/.venv/bin/python \
  -m medworld_vqa.literature_export \
  --run code/medworld_vqa/runs/NEW_SHORT_ANSWER_RUN \
  --out results/NEW_SHORT_ANSWER_RESULT
```

导出器校验新旧题目/图片一致、4×1,024 响应完整、冻结协议和源码未变，
再用 scikit-learn 独立复算每模型 12 组聚合指标。
解析过程按已冻结规则重放；这不等于人工语义正确性验证。
另执行原样保存的 Meissa 两个评分函数，仅用于说明宽松子串匹配的影响。
结果 CSV 使用 0–1 比例，Markdown/LaTeX 使用百分数。
