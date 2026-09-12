# 0912：六个冻结 VLM 的下游头与方向评测

[09-12 实验小结](0912_recent_experiments_summary.md)记录了本轮启动状态、Bicubic 参考及此前已完成的结果；本页保留完整运行协议。在跑任务的最终分数仍以本地矩阵为准。

运行目录：`code/medworld_dense_baselines/runs/dense_20260912`。
完整矩阵预览：`preview/full_tables.md`。只生成预览，不修改论文 Table 1/2。
用户把时间目标延长到 **2026-09-14（周一）08:00，Asia/Shanghai**。

用户最后的 GPU 安排是使用当时空闲的全部 GPU，取代此前预留三卡的安排。
队列只使用没有计算进程、显存低于 512 MiB、利用率低于 5% 的卡；不终止他人的任务。
原项目空闲的两路 Qwen 9B 服务已释放，记录在 `released_servers.json`。

## 实验矩阵

六个检查点沿用上一轮原生模型扫测的固定模型列表：Qwen3.5 0.8B、4B、9B、
27B-FP8，MedGemma 1.5 4B 和 MedGemma **v1 27B**。没有把 27B 标为 1.5。

每个模型冻结主干，取最后一层语言模型的图像 token hidden states。
在原生二维网格上均匀取 8×8 个位置，再均匀取 1,024 个通道，统一为 `[64,1024]`。
这只统一接口形状，并不声称不同主干的通道语义已经对齐，也不保证保留比例相同。

分割和 ×4 超分各做三种图像输入分支，均同时接收对应 VLM hidden states：

1. 原始图像（超分为 LR）；
2. 冻结 V-JEPA2.1 ViT-B 的原生 `[576,768]` 特征，以无参数的通道取样接头；
3. 同一份 V-JEPA 特征，经新训练的 `768→128→64` adapter 接头。

不加载 Ours 的训练后模型或 adapter。三组分别训练，参数量如实记录。
同一分支在六个 VLM 上保持相同结构、初始化、数据、样本顺序与训练轮数。
VLM 和 V-JEPA 均保持冻结。

所有头统一 20 epochs、AdamW、初始 LR 3e-4、cosine decay、weight decay .01、
有效 batch 8、seed 20260912。保存 epoch 5/10/20，并按 epoch 保存优化器状态。
主结果统一使用第 20 个 epoch；不按测试分数选择模型，也不进行 early stopping。
显存不足时可以减 microbatch，但逐图归一化的损失及有效 batch 不变。

超分的图像分支、VLM 与 V-JEPA 共用**同一份缓存的 LR 图像**，通过原图四倍
抗锯齿 bicubic 下采样并舍入为 uint8 得到。HR 只作为目标。
原图分支使用标准 bicubic LR 残差；V-JEPA 两组完全替换像素输入，没有像素旁路。
另报无训练 Bicubic 参考：447 张测试图 PSNR 29.7421 dB、SSIM 0.8998。

## 数据与指标边界

- 分割伪标签／超分共用：4,096 train、249 validation、447 test。
  来自现有病人级划分和 CXAS 独立质量过滤，所有模型/分支共用名单。
  Dice pseudo 是右肺、左肺、心脏的教师一致性。
- Dice human：Montgomery 138 张外部图，仅两肺，无心脏人工标签。
  不参与训练、调参或选择。NIH 的 leftMask/rightMask 是图像左右，已核对重心并
  映射到患者右肺／左肺，避免左右颠倒。
- 定位先做 Chest ImaGenome 人工解剖区域框，共 26 类，病人划分 400/50/50，
  对应 20,766/2,597/2,598 个查询。报告 mIoU、IoU≥.5 比例。
  **这是解剖区域定位，不能称为 MS-CXR 病灶短语定位。** 后者标注未在本地找到，
  官方账号审核仍限制访问；已向用户询问是否存在可用标注路径。
- 方向标注原有 284 对。按六个 findings、前后明确阳性、方向无冲突、侧别不重复、
  有效源图/报告、6h–30d 时间跨度筛选后，固定为 **82 对 / 154 个 finding-scope 字段**。
  所有模型预测同一个子集。只输入源图、源报告（统一 220 词上限）和预测间隔，
  不输入未来图/报告或 EHR。指标为有支持的 finding/scope/direction 类的 macro F1；
  缺失/无效输出保留为 FN。它是单独注明输入协议的方向预测，不能声称来自原 297 对。

不同任务的头相互独立，不共享训练后的权重，所以解剖定位训练不会更新用于方向、
分割或超分的主干/头。病人互斥按每个任务分别检查。

## 验证和复现

`test_contract.py` 检查空间取样、两条输入的梯度、框几何、padding 不影响损失、
缺失方向不冒充 stable、microbatch 权重等价、病人划分和 LR 像素一致性。
七项测试通过。三种分支、两项密集任务的真实样本 GPU 反向传播均通过，三步损失
正常下降，见 `gpu_head_smoke.json`。这只是连通性验证，不是最终泛化结果。

Qwen 27B 原生 FP8 已成功生成有限的 hidden states。使用隔离的 `kernels==0.12.3`
与 finegrained-fp8 v2；0.16.1 与现有 Transformers 接口不兼容，未升级共享环境。
MedGemma 27B 已在三张 4090 上完成 HR/LR 两次真实样本提取。

固定源代码位于运行目录 `source/`；模型、数据和代码摘要记录在 `protocol.json`。
队列合计 55 项：1 个共享 V-JEPA 缓存、6 个 VLM 缓存、6 个方向推理、42 个训练。
查看 `queue.json` / `status.json` / `logs/` 与自动更新的矩阵。失败不会填为零。

参考：[NIH Montgomery](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/index.html)、
[MS-CXR 数据和访问条件](https://physionet.org/content/ms-cxr/1.1.0/)。
