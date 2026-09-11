# Stage 1：四个当前任务

Qwen3.5-0.8B 与冻结 V-JEPA 2.1 ViT-B。分类和疾病识别使用当前图像／报告；分割另读输入图像，SR 另读 LR 图像，HR 只用于监督。

[当前实验索引](../../experiments/README.md) · [4＋4 计划](../../research_notes/0911_stage1_slot_allocation_plan.md) · [数据与训练记录](../../research_notes/0911_stage1_slot44_run.md) · [无 slots 对照](../../research_notes/0911_qwen08_noslots_baseline.md)

## 变体与代码

| 配置识别 | 模型 | diagnosis 的含义 |
|---|---|---|
| `slots=8`，有 `qa_manifest` | `slot44`：分类读前 4 个，分割／SR 读后 4 个，识别读全部 8 个 | 疾病列表问答 |
| `slots=0` | `noslots`：任务 heads 读全部有效图文 tokens | 相同疾病列表问答 |
| `slots=8`，无 `qa_manifest` | `legacy`：旧 8-slot 四任务 | 报告生成，保留历史复现 |

`training.py` 是唯一训练循环，统一处理优化器、任务轮转、累积、验证、保存和恢复。`training_variants.py` 保留各模型的梯度约束。三个旧训练文件仅调用这个循环，显式变体与配置冲突时会报错。

- `networks.py`、`slot44_networks.py`、`noslots_networks.py`：基础 heads 和两种读出。
- `corpus.py`、`slot44_corpus.py`：缓存、划分、标签与确定性采样。
- `prepare.py`、`cache.py`、`slot44_prepare.py`：数据准备、冻结视觉特征、分割伪标签、派生问答。
- `evaluation.py`：公共分类／图像指标及历史评估；`slot44_evaluation.py`：当前四任务评分；`noslots_evaluation.py`：同一评分的无 slots 模型加载入口。
- `launch.py`：当前实验统一启动入口；两个 coordinator 分别负责 Ours 预览、baseline 同步数比较。
- `snapshot.py`：把训练源码与 `medworld_common` 一起冻结到新运行目录。
- `tests/`、`*_check.py`：训练协议测试和真实数据梯度／隔离检查。

## 启动新实验

从项目根目录执行；以下 `NEW_CONFIG.json` 和 `NEW_RUN` 需替换。本轮配置的可追踪副本：[4＋4](configs/slot44_20260911.json)、[无 slots](configs/noslots_20260911.json)。新实验复制后更新本机数据／权重路径及运行截止时间；baseline 同时更新 reference 路径和匹配预览步数。GPU 编号由调用者选择空闲卡。

```bash
/home/data2/chk/workspace/2026/.venv/bin/python code/medworld_stage1/launch.py \
  --config NEW_CONFIG.json --run NEW_RUN --train-gpu 5 --eval-gpu 6
```

长训练可在 tmux 中运行。启动器拒绝覆盖已有运行目录；训练与并行预览使用不同 GPU。启动时保存独立源码快照，后续开发修改不改变该次运行。

直接短检查或历史变体仍可使用统一 trainer：

```bash
CUDA_VISIBLE_DEVICES=5 /home/data2/chk/workspace/2026/.venv/bin/python \
  code/medworld_stage1/train.py --config NEW_CONFIG.json --run NEW_SMOKE \
  --smoke-steps 8 --no-evaluate
```

## 查看结果和恢复

```bash
python scripts/project_status.py
```

当前报告：[Ours](runs/slot44_20260911/REPORT.md) · [无 slots](runs/qwen08_noslots_20260911/REPORT.md)。每个运行保留 `config.json`、`source/`、`source_manifest.json`；训练目录保留 `status.json`、`provenance.json`、`metrics.jsonl`、checkpoint 和评估。

恢复历史实验时使用**该运行自己的 source 和配置**，传入 `--resume .../checkpoint_latest.pt` 与原 `--run`；旧冻结 encoder 对照还需保留 `--freeze-encoder --follow ...`。不要把正在运行的作业切换到开发目录。09-10 的旧 `runner.py` 是固定日期的历史协调器，新实验使用 `launch.py`。

## 指标边界

当前问答是本地 Chest ImaGenome 派生的阳性疾病／征象列表，官方 VQA 数据接入仍待完成。分割指标衡量 CXAS 伪标签一致性；SR 为长宽各 ×2 的合成恢复。分类排除 No Finding，未知／不确定标签不算阴性。两组对照同时保留报告输入；完整 token 和 slots 的计算量可能不同。

早期验证和最终测试分开，正式测试使用统一 final checkpoint。不能将旧报告生成成绩作为新疾病识别 F1，也不能把推理时清零／互换 slots 当作重新训练的无 slots baseline。
