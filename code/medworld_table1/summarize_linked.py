"""Refresh the local overnight Table 1, evidence links and training plot."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from common import atomic_json, load_rows


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def render(root, plot=False):
    cfg = read(root/'config.json')
    if not cfg:
        return
    manifest = read(Path(cfg['cache'])/'manifest.json')
    runner = read(root/'runner_status.json') or {}
    lines = ['# 新链接数据：Table 1 与训练进度', '',
        '更新时间：'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S %Z'), '',
        '仅使用 MIMIC-CXR 原始影像、分节报告、官方 CheXpert 弱标签和 MIMIC-IV 原始临床记录。未使用 Qwen-Gate 生成标签筛样本、监督或评价。', '',
        '同住院、6–72h、采集字段已知一致、图像质控通过且当前前已有临床记录。', '',
        '| Split | 配对数 | 患者数 |', '|---|---:|---:|']
    for split,m in manifest['counts'].items():
        lines.append(f"| {split} | {m['pairs']} | {m['patients']} |")
    lines += ['', '训练最多每患者 4 对，确定性抽样，不根据未来变化或标签选择。验证/测试保留该子集全部 pair，三个 split 患者互斥。',
        f"临床值覆盖 {manifest['ehr']['observations_with_values']}/{manifest['observations']} 个图像端点。每条输入记录必须同时满足原生患者/住院键、事件时间和 storetime 截止。", '',
        '## 测试集主表', '',
        '| 方法 | Future R@1 ↑ | Finding AUPRC ↑ | Transition F1 ↑ | RadGraph F1 ↑ | CheXbert F1 ↑ | 报告种数 | 最大重复次数 | 状态 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    methods = [('copy','Copy Current'), ('qwen9b','Qwen3.5-9B zero-shot'),
        ('direct','Qwen3.5-0.8B direct'), ('matched','Stage-1 + matched LWM'), ('ours','MedWorld-JEPA 0.8B')]
    for key,name in methods:
        path = root/key/'evaluation_test'
        metric, diversity = read(path/'metrics.json'), read(path/'diversity.json')
        values = ['待完成']*5
        state = 'pending'
        if metric:
            values = ['—' if metric.get(k) is None else f'{metric[k]:.4f}' for k in
                ('future_r1','finding_auprc','transition_f1','radgraph_f1','chexbert_f1')]
            state = f"n={metric['n']}; 检索 n={metric['retrieval_queries']}; {metric['radgraph_status']}"
        elif read(root/key/'status.json'):
            s = read(root/key/'status.json')
            state = f"{s['state']}; stage {s['stage']}; step {s['stage_step']}; {s['train_hours']:.2f}h"
        unique = str(diversity['unique_reports']) if diversity else '—'
        largest = str(diversity['most_common_count']) if diversity else '—'
        lines.append('| '+' | '.join([name,*values,unique,largest,state])+' |')
    lines += ['', '4 候选检索，患者互异，匹配 horizon/view/当前标签数量及参考覆盖掩码，平分并列分数；随机参考 0.25。缺少连续疾病评分的 Copy/Qwen9B 不填 AUPRC。',
        '未知/不确定真值字段排除，零阳性支持的类别不计入宏平均。每类支持量、稳定/变化分组均在对应 metrics.json。', '',
        '## 验证与诊断入口', '']
    for key,name in methods:
        for folder,label in [('evaluation_validate','完整验证集'),('midpoint_validate','中途验证前 32 对')]:
            path = root/key/folder
            if (path/'metrics.json').exists():
                lines.append(f'- {name} {label}：[指标]({key}/{folder}/metrics.json)，[多样性]({key}/{folder}/diversity.json)。')
    for folder in ['diagnostics_old_final','diagnostics_old_stage1','diagnostics_new_stage1','diagnostics_new_final']:
        if (root/folder/'report.md').exists():
            lines.append(f'- [{folder}]({folder}/report.md)：正确、打乱、均值 state 和真实未来 state 读出。真实未来输入仅是诊断。')
    if (root/'training.png').exists():
        lines += ['', '![训练与验证曲线](training.png)']
    lines += ['', '## 上一版问题与本轮解释边界', '',
        '上一版 1,000 份输出：主模型仅 41 种报告、最大模板重复 572 次；冻结对照仅 12 种。旧主模型 AUPRC 0.7992，Copy Current 的 CheXbert F1 0.5821，高于旧主模型 0.5482。',
        '本轮网络和损失沿用 0.8B 实现。主模型 Stage 1 1.5h，累计最多 6h；冻结对照共享 Stage-1 初始化并跟随相同 Stage-2 数据、更新和学习率。直接模型最多 6h。墙钟最晚 05:45 结束非跟随训练，留出评估时间；实际步数/停止原因见 status.json。',
        '变化包括新配对、临床上下文、每患者训练配额、无放回按 epoch 采样及 384-token 报告预算，因此不能把新旧总分差异单独归因于某项改动。旧测试集与本轮不同；旧分数只作问题背景。',
        'Qwen3.5-9B 为本地预训练模型零样本推理，未在本轮数据训练；其原生图像输入为 512px，0.8B direct 为 256px，JEPA 特征为 384px，不宣称计算量一致。',
        '当前报告发布时刻未知，本轮仍是回顾性“当前报告已可用”假设。storetime 是记录可用性的代理，未证明记录版本不可变。验证患者仅 62、测试患者仅 94；单次 seed 的小样本结果不等于正式论文结论。', '',
        '## 运行状态', '', '```json', json.dumps(runner, ensure_ascii=False, indent=2), '```', '']
    temp = root/'REPORT.md.tmp'
    temp.write_text('\n'.join(lines))
    temp.replace(root/'REPORT.md')
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1,2,figsize=(11,4))
        any_data = False
        for key in ('ours','direct','matched'):
            if (root/key/'metrics.jsonl').exists():
                rows = load_rows(root/key/'metrics.jsonl')
                if rows:
                    any_data = True
                    stride = max(1,len(rows)//160)
                    bins = [rows[i:i+stride] for i in range(0,len(rows),stride)]
                    axes[0].plot([sum(r['train_hours'] for r in b)/len(b) for b in bins],
                        [sum(r['text'] for r in b)/len(b) for b in bins], label=key)
            if (root/key/'validation_history.jsonl').exists():
                rows = load_rows(root/key/'validation_history.jsonl')
                axes[1].plot([r['step'] for r in rows],[r['metrics']['text'] for r in rows],'.-',label=key)
        axes[0].set(xlabel='Training hours',ylabel='Report CE',title='Training (smoothed by disjoint bins)')
        axes[1].set(xlabel='Optimizer step',ylabel='Report CE',title='Validation (first 64 fixed pairs)')
        for ax in axes:
            ax.grid(alpha=.2)
            if ax.get_lines(): ax.legend()
        if any_data:
            fig.tight_layout()
            fig.savefig(root/'training.png',dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--plot',action='store_true')
    a = p.parse_args()
    render(a.run,a.plot)
