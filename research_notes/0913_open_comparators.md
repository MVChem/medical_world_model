# 09-13：Table 1／Table 2 开源对比方法复现

用户要求在已有 slots／no-slots／shuffled 实验之外，用相同数据和计算资源训练表中的公开对比方法，允许使用全部空闲 GPU。本轮运行目录为 `code/medworld_open_baselines/runs/comparators_20260913`；[自动状态与结果](../code/medworld_open_baselines/runs/comparators_20260913/LIVE.md)。本页记录启动协议，不能代替完整训练结果。

## 方法与预算

| 方法 | 实际实现与任务 | 匹配预算 |
| --- | --- | --- |
| DINOv2-B/14 + heads | 冻结官方编码器，最后层空间特征池化为8×8，再训练独立分类／分割／SR头；不改成4个slots | 分割与SR：4096/249/447，20 epochs、batch8、seed20260913；另有18708训练图的20 epochs独立实验 |
| CheXWorld + heads | 严格加载公开target ViT-B；同一8×8空间接口和同一任务头 | 同DINOv2；分类另用既有C0患者划分13681/160/353、20 epochs，测试标签与旧C0逐项一致 |
| SwinIR (CXR-adapted) | 官方SwinIR-M classical ×4、DF2K预训练；全参数适配，灰度LR复制三通道、RGB输出取均值 | 两套4096／18708训练集分别20 epochs、有效batch8、microbatch1；单RTX4090 |
| BioViL-T + predictor | 官方单图／missing-prior视觉表示冻结，接相同Qwen融合、horizon predictor、report和finding读出 | 16000/230/297对、2400次更新、batch32、seed42 |
| CheXWorld + predictor | 官方冻结视觉表示接同一未来预测训练接口 | 同BioViL-T |
| Qwen2.5-VL-7B、LLaVA-v1.6-Mistral-7B、LLaVA-Med-v1.5-7B、CheXagent-8b | 表中ZS行保持公开checkpoint零样本；相同冻结输入、提示、Yes/No原始概率与官方临床评分 | 既有297对未来测试；当前分类353、报告507；派生QA207单列 |
| Copy Current／Finding-transition prior | 复制当前报告；按训练集拟合horizon×finding×当前标签转移概率，Laplace α=1 | 同16000训练对及297测试对；无需GPU训练 |
| MAIRA-2 | 官方HF仓库需要账户访问授权；当前下载返回401 GatedRepoError | 阻塞于权重访问，未替代为其他模型 |

每个训练任务使用一张同型号RTX4090，优化更新数与数据呈现量显式固定。**这匹配单卡型号、数据、轮数／步数，不是严格相等的GPU小时或FLOPs。** SwinIR完整反向首步约3.87秒，正式训练初段约2.8秒／step，4096版10240步约需数小时；不能宣称其成本与小decoder相等。用户关于预算口径的可选澄清尚未回复时，本轮先按同轮数执行，并单独保存实际秒数、参数量及样本数。

原始公开预训练数据也不同；本地患者互斥不能证明与公开预训练不存在重叠。BioViL-T／CheXWorld未来预测为本项目的 **encoder-swap adaptation**，不是原作者提供的forecast系统。它们迁移相同Stage1共同参数，但重新初始化全部视觉adapter，避免错误复用V-JEPA的特征投影。详见 [适配协议](../code/medworld_open_baselines/biovil_chexworld_protocol.md)。

## 数据与评价

密集任务逐条复用下午固定清单和缓存，扩展版仅增加训练样本。SR所有输入只来自同一uint8 ×4 bicubic LR，HR只作监督目标；同valid ROI PSNR／SSIM。分割沿用CXAS三器官伪标签，以及Montgomery138张外部人工双肺测试。DINOv2／CheXWorld的decoder可训练参数及初始化与下午decoder一致，读取的空间token数不同；分类训练是独立image-only head协议，不把它与旧含当前报告的slots训练混称严格同监督。

未来预测只在训练时使用未来图像与报告作目标；推理输入为源图、源报告、源时点EHR、请求horizon。297对测试集仍没有合格的Direction真值。官方VQA未准备完成，派生疾病问答不填官方VQA列。MAIRA-2病灶grounding也不以解剖框任务代替。

Finding-transition prior的源状态使用已准备的结构化finding标签，是额外注明的输入接口；未知标签不当阴性，未知源标签回退到训练集条件先验。参考概率评分使用与已有基线相同的CheXbert测试标签和固定掩码。Copy Current没有连续分数接口。

## 权重来源与运行检查

官方来源：[DINOv2](https://github.com/facebookresearch/dinov2)、[CheXWorld](https://github.com/LeapLabTHU/CheXWorld)、[BioViL-T](https://huggingface.co/microsoft/BiomedVLP-BioViL-T)、[SwinIR](https://github.com/JingyunLiang/SwinIR)、[Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)、[LLaVA](https://github.com/haotian-liu/LLaVA)、[LLaVA-Med](https://github.com/microsoft/LLaVA-Med)、[CheXagent](https://huggingface.co/StanfordAIMI/CheXagent-8b)、[MAIRA-2](https://huggingface.co/microsoft/maira-2)。LLaVA-v1.6使用Hugging Face维护的格式转换权重，来源单独记录；不替换为不同模型规模。下载文件固定revision，并检查官方LFS SHA-256。CLIP336官方只有`pytorch_model.bin`，下载器明确支持这一例外。

已完成：SwinIR真实LR128→HR512的GPU前向／完整反向／保存恢复；DINOv2和CheXWorld严格公开权重加载、真实HR/LR提取、完整训练接口与断点恢复；同患者划分、输入数组SHA、LR读取HR防护、分类标签对应核验。小型检查目录不计入正式结果。CheXagent／LLaVA-Med已完成原生processor和微型模型API检查，完整公开权重下载完成后仍需实际推理验证。

## 调度、恢复与已有Table1修复

`scheduler_swinir`、`scheduler_trained`、`scheduler_zero_shot`与`scheduler_table1_recovery`分别持久调度，均可使用0–7中无计算进程的空闲卡；与已有队列共用`/tmp/medworld-frozen-slots-{GPU UUID}.lock`。不停止其他运行中的任务。新对比实验按各方法显式预算结束，没有擅自套用旧过夜实验的08:00截止；旧Table1恢复训练配置本身仍保留原06:45截止，因此未完成者必须标为partial。

发现旧未来队列已完成no-slots的2400步，却在后处理时把`.pt`二进制当JSON读取，抛出UnicodeDecodeError并阻塞后续任务。开发版`scripts/overnight_queue.py`现仅解析JSON后缀、对二进制直接计算SHA，新增回归检查通过。活跃旧队列的源码快照未修改。新recovery队列复用原`checkpoint_final.pt`，接续评估，并恢复原slots／shuffled任务；不重新训练已完成的no-slots。

每个新队列保存不可变命令、源码SHA清单、运行日志、成功回执及产物SHA。只有完整预算和期望测试数量均通过才计入完成。当前状态：

```bash
/home/data2/chk/workspace/2026/.venv/bin/python code/medworld_open_baselines/report.py \
  --run code/medworld_open_baselines/runs/comparators_20260913
python scripts/project_status.py
```

协调器退出后，使用对应`scheduler_*/launch.json`里的`argv`恢复；不要用开发源码覆盖原run。下载器使用4MiB小块重试和完整SHA验证，状态在`assets/*/download_status.json`，长连接中断与缺访问授权分别记录。
