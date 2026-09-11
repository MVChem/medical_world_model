# 本地开发与复现

从 [项目 README](../README.md) 进入。所有命令默认在项目根目录执行。

## 环境

当前解释器：`/home/data2/chk/workspace/2026/.venv/bin/python`。训练需要 CUDA；代码从本地加载 Qwen 和冻结视觉权重。共享环境现有版本见 [runtime_20260911.json](runtime_20260911.json)，这是环境记录，不是通用安装锁文件。

可编辑安装仍支持早期数据包和新增公共组件：

```bash
/home/data2/chk/workspace/2026/.venv/bin/python -m pip install --no-deps -e code
```

当前 Stage 1／Table 1 通过脚本入口运行，不依赖重新安装可编辑包。`medworld_common` 是公共实现；各实验保留自己的任务、数据和模型配置。第三方目录、vendor 和权重路径保留现状，换机器需单独配置这些依赖和数据路径。

## 修改和验证

1. 在 `code/` 中修改开发源码，在 `research_notes/` 中记录协议变更。
2. 新运行使用新目录，由启动器冻结源码及公共组件。历史运行的 `source/` 不作为开发目录。
3. 模型／训练修改至少运行对应协议检查；影响任务路由、数据或恢复时再运行真实数据短检查。
4. 检查结果与来源后提交到本地 Git，并更新实验索引。

Stage 1 协议检查和 Table 1 CPU 检查：

```bash
CUDA_VISIBLE_DEVICES='' /home/data2/chk/workspace/2026/.venv/bin/python \
  -m unittest discover -s code/medworld_stage1/tests -v
CUDA_VISIBLE_DEVICES='' /home/data2/chk/workspace/2026/.venv/bin/python \
  -m pytest code/medworld_table1/tests -q
```

GPU 的真实数据检查入口：`code/medworld_stage1/slot44_check.py`、`noslots_check.py`。先用 `--help` 查看参数，并指定空闲 GPU。冻结权重检查、四任务短训练、保存恢复都使用独立检查目录。检查产生的更新不继承到正式模型。

## 恢复与实验来源

恢复时用运行目录自身的源码、原配置、原 run 路径和 checkpoint。已启动的 09-11 实验仍执行整理前的源码快照。整理后的开发代码对新实验生效。

本轮 baseline 的模型／采样核验继续支持旧快照：未移动的任务代码逐文件核对，抽出的 Qwen 函数／类与旧文件逐定义核对语法树。重构不是放宽数据、步数或模型初始化匹配要求。

`metrics.jsonl` 保存逐步训练记录；`provenance.json` 保存输入及源码指纹；`evaluation/` 保留逐样本预测和指标。当前两个 coordinator 的运行报告会自动更新。登记表只存路径，避免文档复制的实时状态过期。

## Git 与 GitHub

根仓库管理自有源码、配置和文档，远端为 [MVChem/medical_world_model](https://github.com/MVChem/medical_world_model)。

```bash
git status --short
git diff
git log --oneline -5
git show before-project-cleanup-20260911:code/medworld_stage1/slot44_train.py
```

整理前有独立提交和标签，整理后的改动另作提交。提交时查看暂存文件，数据、权重、运行产物和第三方仓库由 `.gitignore` 排除。论文在 `26iclr/` 的原有仓库中单独管理；查看它使用 `git -C 26iclr status --short`。

用户已授权将整理后的项目上传此仓库。数据、运行产物和已有论文／第三方仓库继续独立保存；提交前核对暂存清单，避免将它们纳入根仓库。
