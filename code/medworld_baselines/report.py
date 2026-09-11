"""Render an aggregate-only status report without exposing patient records."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from base import *

def render(run):
    protocol=read(run/'protocol.json');inventory=read(run/'models.json')
    result={}
    for m in inventory:
        p=run/m['id']/'test'/'metrics.json'
        result[m['id']]=read(p) if p.exists() else {}
    def f(v):return '—' if v is None else f'{v:.4f}'
    lines=['# 原始模型 Table 1 / Table 2 基线评测', '',
        '更新时间：'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S CST'),'',
        '六个本地公开 checkpoint，零样本推理，无本项目训练。目标：2026-09-12 08:00 前形成可检查结果。',
        '以下是初步研究评测。空值表示未完成或缺少合格接口/参考，不能读作 0。', '',
        '## 运行进度','', '| 模型 | 状态 | 已完成请求 / 总请求 | 当前任务 |','|---|---|---:|---|']
    for m in inventory:
        p=run/m['id']/'test/status.json';s=read(p) if p.exists() else {}
        jobs=s.get('tasks',{});done=sum(x['completed'] for x in jobs.values())
        c=protocol['counts']['test'];total=c['table1']*7+c['classification']*13+c['report_generation']+c['derived_qa']
        lines.append(f"| {m['label']} | {s.get('status','queued')} | {done} / {total} | {s.get('current_task','—')} |")
    lines+=['','## Table 1：未来预测','',
        f"固定测试集 {protocol['counts']['test']['table1']} 对 / {protocol['counts']['test']['table1_patients']} 位患者。仅源时点证据。",'',
        '| 模型 | AP ↑ | AUROC ↑ | Transition F1 ↑ | Direction F1 ↑ | RadGraph F1 ↑ | GREEN ↑ | Brier ↓ | ECE ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in inventory:
        d=result[m['id']].get('table1',{})
        gp=run/m['id']/'test/green_metrics.json'
        if gp.exists():d=dict(d,green=read(gp).get('mean'))
        lines.append('| '+m['label']+' | '+' | '.join(f(d.get(k)) for k in ['ap','auroc','transition_f1','direction_f1','radgraph_f1','green','brier','ece'])+' |')
    lines+=['','Yes/No 的归一化条件似然用于 AP、AUROC、Brier、ECE；十个等宽概率箱，未在测试集调阈值或拟合校准器。Transition 使用实际生成报告的 CheXbert 标签；概率阈值版本另存 metrics.json。',
        'Direction F1 缺少经核验的疾病/侧别方向真值，暂不计分。GREEN 采用冻结的官方评价器，逐条评分与错误分析单独保存。','',
        '## Table 2：当前状态','',
        f"分类 {protocol['counts']['test']['classification']} 张；报告生成 {protocol['counts']['test']['report_generation']} 张。所有输入均不含同次报告。",'',
        '| 模型 | 分类 AUROC ↑ | 分类 AP ↑ | VQA Acc. | VQA μF1 | 报告 RadGraph ↑ | 报告 CheXbert ↑ | Grounding mIoU / A@.5 | Dice pseudo / human | SR ×4 PSNR / SSIM |',
        '|---|---:|---:|---:|---:|---:|---:|---|---|---|']
    for m in inventory:
        d=result[m['id']];c=d.get('table2_classification',{});r=d.get('table2_report',{})
        lines.append(f"| {m['label']} | {f(c.get('auroc'))} | {f(c.get('ap'))} | — | — | {f(r.get('radgraph_f1'))} | {f(r.get('chexbert_f1'))} | — / — | N/A / N/A | N/A / N/A |")
    lines+=['','官方 MIMIC-CXR-VQA 与 MS-CXR 定位数据在本地未就绪。原生文本输出 VLM 没有分割/超分输出接口，按论文的零样本行协议记 N/A。',
        '预训练数据交叠未知；本地按患者划分不能排除公开模型预训练见过部分 MIMIC 数据。','',
        '## 补充：Chest ImaGenome 派生疾病列表问答','',
        '207 个问题，36 张图像 / 36 位患者，人工标注来源，只有非空阳性答案；这是单列的 pilot，不是官方 VQA benchmark。','',
        '| 模型 | Exact match ↑ | Micro F1 ↑ | 解析失败 |','|---|---:|---:|---:|']
    for m in inventory:
        q=result[m['id']].get('derived_qa',{})
        lines.append(f"| {m['label']} | {f(q.get('exact_match'))} | {f(q.get('micro_f1'))} | {q.get('invalid_outputs','—')} |")
    lines+=['','## 可复核文件','',
        '- [protocol.json](protocol.json)：冻结协议、数据指纹和样本规模。',
        '- [models.json](models.json)：精确 checkpoint 路径、revision 和权重清单。',
        '- 每个模型的 `test/responses.jsonl` 保存原始回答、似然与失败信息；`test/metrics.json` 保存参考覆盖和逐类分数。',
        '- `source/` 保存本轮源码；`coordinator.log` 与各模型日志记录调度与错误。',
        '- 病例级数据仅保存在本地受限目录，论文主表未自动填写。','']
    tmp=run/'REPORT.md.tmp';tmp.write_text('\n'.join(lines));tmp.replace(run/'REPORT.md')
    atomic(run/'aggregate.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);render(p.parse_args().run.resolve())
