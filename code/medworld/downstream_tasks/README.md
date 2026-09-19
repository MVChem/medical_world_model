# MedWorld downstream tasks

这里集中管理当前状态的下游任务：分类、报告生成、分割和超分。
评估对象是共享 Medical World Encoder 在各任务上的表现；PSNR/SSIM 仅属于超分指标。
Stage 1 四任务训练和 Stage 2 当前任务 replay 使用同一套任务实现。

| 任务 ID | 代码 | 输入与监督 | 评测 |
|---|---|---|---|
| `classification` | [classification.py](classification.py) | 八个 slots；13 类疾病／征象标签，屏蔽未知标签 | macro AUROC/AP、逐类结果 |
| `report` | [report.py](report.py) | 八个 slots；当前报告仅作监督 | 导出预测／参考报告，空报告与重复诊断；临床指标仍需外部 scorer |
| `segmentation` | [segmentation.py](segmentation.py) | 图像＋四个视觉 slots；既有 CXAS 标签或人工肺掩码 | Dice、逐器官结果 |
| `sr` | [super_resolution.py](super_resolution.py) | LR 图像＋四个 LR 视觉 slots；HR 仅作监督 | PSNR、SSIM |
| `vqa` | 尚未接入 | 后续补充正式数据与任务接口 | 待定；不加入当前训练／评测 |
| `grounding` | 尚未接入 | 后续补充 phrase-grounding 数据与接口 | 待定 |

## 文件分工

- [registry.py](registry.py)：当前任务顺序、分类标签及待接入任务说明。
- [training.py](training.py)：四任务 loss 分发；由 `MedWorld.current_loss()` 调用，Stage 2 replay 复用。
- [data.py](data.py)、[pixels.py](pixels.py)：任务数据、原图与原始人工掩码在线读取、LR 在线生成。
  不读取或写入预处理图像／LR／特征缓存；保留既有 CXAS 监督标签。
- [spatial.py](spatial.py)：分割和超分共用的基础 decoder 类、空间 loss 分发。
  两任务各有独立 decoder 实例，参数不混用。
- [featup.py](featup.py)：可选图像引导上采样、固定教师及多视图特征一致性。
- [common.py](common.py)：状态形状检查、位置编码等共享组件。
- [evaluate.py](evaluate.py)：统一评测入口与结果写出；复用四任务的指标函数。

共享编码器、World Model、EMA 和两阶段训练调度仍在上层 `medworld/`；
跨任务患者隔离、时间配对仍在上层 `datasets/`。
分类头和报告 decoder 也由时间预测监督复用。

## 运行与兼容

```bash
# 在仓库根目录评测同一 checkpoint 的四个当前任务。
PYTHONPATH=code python -m medworld.downstream_tasks.evaluate \
  --checkpoint code/medworld/runs/RUN/stage2.pt \
  --task current --split test --gpu auto \
  --out code/medworld/runs/RUN/evaluation_current
```

`--task classification/report/segmentation/sr` 可单独评测；原来的 `all` / `temporal`
选项仍保留。训练继续使用 `medworld.train` 或 `medworld.launch_distributed`。

上层 `decoders.py`、`featup.py`、`evaluate.py` 和 `datasets/current.py`、
`datasets/pixels.py` 仅保留兼容导出，旧脚本和导入路径仍然有效。模型中的模块属性名、
checkpoint 参数键、损失公式、指标定义、数据协议和训练任务顺序均保持原样。

## 后续接入 VQA

目前只记录待接入状态，不迁移或修改独立的 VQA 实验，也不把现有报告任务当作 VQA。
后续可在本目录新增 `vqa.py`，明确问题输入、答案监督和指标，再接入数据适配、
`training.py`、评测分发和配置校验，最后将任务加入 `registry.py` 的活动列表。
不应只加入任务名字就让训练开始调用尚未实现的接口。
