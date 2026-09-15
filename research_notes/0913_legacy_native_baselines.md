# CheXagent-8b 与 LLaVA-Med 原生推理适配

入口：`code/medworld_open_baselines/legacy_vlm.py`。两行在Table2计划中标为ZS，
本入口不在测试集上训练；复用 `zero_shot.py prepare` 冻结的原始
`medworld_baselines` cohort、prompt和评分器，推理只读取inputs，不读references。
源图像先经过与原HTTP基线同一函数生成的512px黑边PNG，再进入各方法原生processor。

方法来源与接口：

- [CheXagent-8b官方模型卡](https://huggingface.co/StanfordAIMI/CheXagent-8b)
  snapshot `4934e91451945c8218c267aae9c34929a7677829`。使用模型随附的
  `CheXagentProcessor`/`CheXagentForConditionalGeneration`及官方
  ` USER: <s>... ASSISTANT: <s>`格式；图像张量是`[B,1,3,448,448]`。
  这是旧8b，不能替换成GitHub当前示例中的CheXagent-2-3b。
- [LLaVA-Med官方代码](https://github.com/microsoft/LLaVA-Med)
  commit `30697ca50b5c29a8e955c99330b259776aef27b9`，以及
  [官方v1.5 Mistral模型](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b)
  snapshot `91bb16c122001ddc9cf1fd36ce1dae09448943a2`。使用原生
  `LlavaMistralForCausalLM`、`mistral_instruct`与CLIP336视觉塔。
  单图输入为`[B,3,336,336]`，不重新训练或替换视觉投影。

隔离依赖安装在 `code/medworld_open_baselines/legacy_vendor/`：
transformers4.36.2、tokenizers0.15.2、huggingface-hub0.36.2，
其余依赖使用现有torch2.11及共享环境。未升级或降级共享包。
该隔离由入口在导入Transformers前设置；启动命令无需用户修改PYTHONPATH。

Yes/No在两个tokenizer中均为单token5592/1770。分类和未来疾病概率先直接
`model.forward`取最后位置原始logits，计算全词表logsoftmax保存Yes/No原始log概率，
再求二者归一化概率；不使用受限解码后重归一化的分数。报告384新token、派生QA192，
greedy、无采样。QA继续保留既有派生问答协议，不能把它称为官方VQA基准。
两模型在GPU上以BF16运行，记录实际参数、原生模板、快照和源码指纹。

LLaVA-Med配置中的`tokenizer_model_max_length=2048`会在原生多模态插入后静默裁剪；
入口将推理上限明确设为与既有endpoint匹配的8192，并在超限时报错。
这不改变模型权重，骨干原始max_position_embeddings为32768。
调用者可用 `--max-context` 固定其他上限，并写入plan。

```bash
CUDA_VISIBLE_DEVICES=<空闲GPU> /home/data2/chk/workspace/2026/.venv/bin/python \
  code/medworld_open_baselines/legacy_vlm.py \
  --run <zero_shot.prepare产出的run> \
  --assets code/medworld_open_baselines/runs/comparators_20260913/assets \
  --model chexagent8b --split test --tasks all
```

LLaVA-Med同命令改为`--model llava_med7b`，依赖assets/clip336状态已complete。
输出与旧评分器兼容的`<run>/<model>/test/responses.jsonl`，每条即时保存，
重跑跳过已成功key；截断的最后一条写入可恢复。SIGTERM在当前样本结束后暂停。
短GPU检查加`--limit 1 --tasks table1_report,table1_prob,table2_report,table2_prob`；
完成标记保留smoke标志，不应当作完整评分。

CPU检查已完成：两个真实tokenizer/processor加载、single-token候选、原生图像shape、
LLaVA-Med官方类导入。另以官方类的小尺寸随机配置检查CheXagent图像+文本前向/生成、
LLaVA-Med文本前向/生成；两者第一步generation scores与直接forward原始logits
逐元素一致（max abs error=0）。这些是API兼容检查，不能代表完整预训练模型效果。
检查产物在 `runs/legacy_cpu_inspect_20260913/`。

交接时完整权重仍在下载，因此尚未宣称真实8b/7b GPU推理通过。下载器曾遭遇4.94GB
shard连接中断，需恢复下载；CLIP336官方只有`pytorch_model.bin`而无safetensors，
不能用统一“忽略所有.bin且强制safetensors”规则，已向队列维护者反馈。
