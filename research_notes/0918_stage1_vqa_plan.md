# Stage 1 VQA 接入方案（待实现）

## 已核实的数据与代码

- 数据入口：`code/data/MIMIC_CXR_VQA`，软链接至 `/home/data2/chk/data/MIMIC/mimic-ext-mimic-cxr-vqa-1.0.0`。问答位于 `MIMIC-Ext-MIMIC-CXR-VQA/dataset/{train,valid,test}.json`；图片沿用 `code/data/MIMIC_CXR/files/<image_path>`。
- 2026-09-18 本地 JSON 全量统计如下；每个 split 的首张图片路径已验证存在，尚未做全量图片完整性检查。

| Split | Questions | Images | Patients | Empty answers |
| --- | ---: | ---: | ---: | ---: |
| train | 290031 | 133687 | 52453 | 30463 |
| valid | 73567 | 8610 | 3461 | 10619 |
| test | 13793 | 500 | 500 | 2484 |

原始 train/valid 有 698 名患者重叠，train/test 与 valid/test 均无患者重叠。不能直接宣称三个官方集合患者互斥。

`code/medworld_vqa` 已有四模型抽样零样本评估、集合评分和短答评分。`code/medworld_stage1/slot44_prepare.py` 虽支持 `--vqa-dir`，但只选部分疾病/征象 Query，并要求存在报告及命中已有图像缓存；它不是完整 VQA 适配器。`Slot44Corpus` 的提示词仅允许疾病/征象名称，不能直接覆盖 Verify、Choose 和全部 Query。当前 Stage 1 配置仍使用本地派生问答。

## 建议的任务定义

将 VQA 作为 Stage 1 的监督下游读出任务，第一版用完整 VQA 替换现有 disease-list diagnosis 槽位，保留分类、分割、SR，继续四任务轮转。旧实验与旧 diagnosis 定义保留；新配置显式声明任务和数据版本。

输入当前胸片和问题，输出答案。VQA 分支不能读取当前报告、EHR 或参考答案；报告可作为其他训练任务的监督，但不能进入 VQA 的共享状态或旁路。取消 VQA 对报告存在的要求。

沿用现有结构：图像编码 → 全部 8 个状态 slots → 问题条件化 decoder → 答案。问题在 decoder 端输入，状态仍保持与问题无关，便于检验共享状态是否支持不同问题；后续可单独比较问题条件化的状态编码。VQA 只对答案 token（含 EOS）计算交叉熵，将梯度传回可训练的状态编码部分，保持现有视觉骨干冻结策略。

第一版训练目标使用确定顺序的 JSON 答案数组，涵盖 Verify、Choose、Query 和空集合；改为通用答案提示词，移除疾病词表专用假设。使用已有集合评分比较本项目变体，明确标注受限词表/结构化输出协议。自由短答作为单独协议运行，不能把两种分数混合或称为官方 evaluator 复现。

## 划分、对照与评估

1. 在所有 Stage 1 任务及后续训练阶段统一排除 VQA test 的 500 名患者。若既有 checkpoint 接触过这些患者，它只能用于探索，正式结果需从未接触保留集的初始化重新训练。
2. 内部开发建议从 VQA train 中去掉与 valid 重叠的 698 名患者，并在其他任务同步排除 valid/test 患者。保留原始 split 元数据，报告移除后的覆盖量；称为患者隔离协议。若另做原始官方划分实验，应单列，并明确其 train/valid 患者重叠。
3. VQA 初始占约四分之一任务更新；先按图像采样，再轮换同图问题，避免少数多问题图像主导训练。记录各题型覆盖量，按验证结果决定是否加入题型平衡。
4. 主评估使用完整 test；开发时只看 valid。报告 Verify accuracy、Choose accuracy/集合 EM、Query 标签 micro-F1，并补充整体集合 EM/micro-F1、区域/全图、空答案/非空答案、内容类型及患者 bootstrap CI。预测失败计错，不丢弃样本。
5. 分别比较原始模型零样本、Stage 1 联合训练有 slots、相同训练数据/步数的无 slots。三组保持 VQA 输入和解码协议一致。额外冻结 Stage 1 encoder、只训练相同 VQA decoder 的对照可衡量表示迁移；该设置与联合训练结果分表。
6. 命名为 Current-image VQA，独立于未来诊断/预测任务。已有 1024 题抽样成绩仅作历史参考，不能直接对比新全量成绩。

## 实施顺序

先在 `medworld_vqa` 增加完整数据适配与划分审计，验证全量图片、答案词表、空答案和跨任务患者隔离；将详细审计写入 `runs/<name>_YYYYMMDD/`。随后在 `medworld_stage1` 增加明确的 VQA 配置/变体，扩展 corpus、输入隔离、decoder 提示词与评分入口。完成小批次梯度检查、报告不可达检查与三题型评分检查后，再启动训练。

本次只完成数据软链接、只读数据统计和接入设计；未更改训练行为，未启动实验。启动时按仓库规范登记实验，优先选空闲 GPU 1、2、3。
