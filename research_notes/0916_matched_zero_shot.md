# 09-16：按正在训练的融合模型重做 zero-shot

用户要求用一张卡测试未训练的 Qwen0.8B、MedGemma 等本地模型，重点是 Table 1／2，
先不做外部论文对比方法。当前训练 `qwen35_08b_2gpu_day_20260916_095750` 使用 GPU 5、6；
新评测指定空闲 GPU 2，沿用 UUID 文件锁，不改训练进程或其源码快照。

## 为什么旧结果不能直接用

09-11 六个原始 checkpoint 已完成：Qwen3.5 0.8B／4B／9B／27B-FP8，MedGemma 1.5-4B／27B。
但旧 Table 1 提供源 EHR 和时间区间；当前融合模型输入为源影像、源报告和实际时间，且有双向预测。
旧 Table 2 从原图重新做 RGB 预处理，当前训练读取已经缓存的灰度图像转 RGB。
当前 `medworld.evaluate` 默认只生成 64 tokens，旧 baseline 上限为 384，直接比较会混入长度差异。

因此新运行重新冻结所有测试输入和协议，不复用旧预测分数。旧六模型结果仍在
`code/medworld_baselines/runs/raw_models_20260911/REPORT.md`。

## 新协议

- 优先 Qwen3.5-0.8B、MedGemma-1.5-4B，然后 Qwen3.5-4B、9B；均原始 BF16 checkpoint，
  不训练、不加载本项目 LoRA。27B 本轮不满足单卡 BF16，未另做量化版本。
- 从当前训练的配置创建 `UnifiedData`，要求 metadata 与训练 `data_protocol.json` 完全一致。
- Table 1 原始 297 对／94 患者保留，正向和回溯各 297，分别计算指标。
  仅源图像、源报告、实际正负时间间隔；报告截断包含 observation 前缀，共 384 个 Qwen token，
  与 `MedWorld.encode` 所见内容一致。推理不读取目标图像、目标报告、标签或 EHR。
- Table 2 分类 353 张、报告 507 张；导出训练读取的确切 512² 像素，标签和测试 ID 不变，输入不含报告。
- 各模型同样贪心生成最多 384 tokens；Qwen thinking 关闭。后续 Ours 比较入口强制同样的生成长度。
- Qwen 视觉处理 min/max 为训练的 256²；MedGemma 保留原生处理器。共同输入像素一致，但原生视觉
  计算量不同，融合模型还使用 384² JEPA。因此这是测试证据对齐，不宣称等计算量。
- 概率来自原始 Yes/No 似然，真实服务 smoke 验证 unrestricted 和 forced token 的 logprob 一致。
  分类使用训练数据的同一参考标签，不做测试集校准或阈值搜索。Ours 的 trained head 概率来源单列。
- CheXbert、RadGraph partial F1 与 GREEN 使用既有冻结评价器。未来／回溯分开；任何面板缺样本就拒绝计分。
- Direction F1 缺经核验的疾病／侧别真值；官方 VQA 和 MS-CXR 数据未就绪。原生文本 VLM 无分割／SR
  decoder，相应 zero-shot 格子是 N/A。不会把派生正例 QA、解剖框或训练过的 heads 填入这些格子。

## 代码与产物

入口：[code/medworld_zero_shot](../code/medworld_zero_shot/README.md)。
运行目录：`code/medworld_zero_shot/runs/unified_20260916`。

`protocol.json` 保存输入／标签／图像哈希、训练数据 fingerprint 和预算；`models.json` 保存精确路径、
权重文件 SHA256 和处理器哈希；`source/` 冻结执行源码；病例级输入／答案分文件存储。
`REPORT.md`、两个时间方向的 CSV、Table 2 CSV 和 `tables.json` 是汇总入口，均自动更新。
每个模型的 `smoke/` 与正式 `test/` 独立。失败留痕，恢复拒绝变化的协议或源码。

推理、CheXbert／RadGraph 和 GREEN 均串行使用同一张卡。服务只绑定本机，队列只管理它自己启动的进程。
训练结束后运行 `evaluate_ours`，用同一 Stage 2 checkpoint 对两表评测；不从测试结果选择 checkpoint。

## 检查

新增 5 项协议检查和既有 6 项 baseline 检查通过：源证据隔离、正负时间、当前报告 withholding、
完整测试分母、分类概率掩码与融合评测一致、源报告前缀计入截断、未知值／非法概率与校准边界。
真实服务 smoke 和完整运行状态以本轮 `status.json` 及各模型 `smoke/status.json` 为准。

另做独立数据核对：594 条双向样本的输入／目标连接逐条吻合，507 张当前图像与训练缓存
逐像素一致，分类 353／报告 507 的有序 ID 及参考一致。结果在 `independent_cohort_audit.json`，
可通过 `medworld_zero_shot.audit` 复查。

首次 vLLM 启动在生成预测前失败：导入时全机枚举访问了故障 GPU 4，且其 UUID 映射不兼容。
只在评测子进程加入设备发现适配器，保留真实 GPU UUID 和数值计算路径，未修改共享环境。
原始 `source/` 和失败日志保留；修复后的执行快照为 `source_v2/`。适配后已验证物理 GPU 2、
正确 UUID、4090 SM 8.9 以及仅一个 CUDA 可见设备。运行 PID 以 `launcher.json` 为准。

Qwen3.5-0.8B 真实短测已通过 5 个请求：正向／回溯报告、未来概率、当前报告、当前概率；
三个报告均非空并正常停止，两项概率均通过原始似然核验。随后正式推理已开始，首批 96 条
时间任务报告无请求错误。该数字仅是启动检查记录，后续完成度看实时报告。
CheXbert 与 RadGraph 的独立 CPU 短测也通过；相同报告的 RadGraph 三项 F1 均为 1。
