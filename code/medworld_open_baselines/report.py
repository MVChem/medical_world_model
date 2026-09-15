"""Summarize open comparator jobs without confusing running/partial with final."""
import argparse
import datetime as dt
import json
from pathlib import Path
import time

def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default

def render(run,output='REPORT.md'):
    stamp=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()
    lines=['# 开源对比方法复现', '', f'更新时间：{stamp}', '',
        '复用既有患者划分、目标和评分；训练预算、官方预训练来源与实际GPU小时分别记录。',
        '4096与18708训练图的结果分开；Table1使用16000/230/297对。公开模型的ZS行保持零样本。', '',
        '以下状态来自任务回执；未完成训练不填写最终指标。', '',
        '| 任务 | 状态 | GPU | 日志 |', '| --- | --- | --- | --- |']
    counts={};jobs=[]
    for q in sorted(run.glob('scheduler*/queue.json')):
        if (q.parent/'superseded.json').exists():continue
        for row in read(q,[]):
            status=row['status'];counts[status]=counts.get(status,0)+1
            log=row.get('log');link=f'[log]({log})' if log else ''
            lines.append(f'| {row["id"]} | {status} | {row.get("gpus",[])} | {link} |')
            jobs.append(row)
    lines+=['','| 权重 | 下载状态 |', '| --- | --- |']
    assets=[]
    for p in sorted((run/'assets').glob('*/download_status.json')):
        d=read(p,{});assets.append(d)
        lines.append(f'| {d.get("repo",p.parent.name)} | {d.get("status")} {d.get("error_type", "")} |')
    lines+=['', 'MAIRA-2官方权重当前需要HF账户访问授权，未授权状态不计为完成。', '',
            '| 已完成指标文件 | 主要指标 |','| --- | --- |']
    accepted=[]
    for j in jobs:
        if j['status']!='complete':continue
        for path in j['spec'].get('artifacts',[]):
            p=Path(path)
            if p.name not in ['metrics.json','green_metrics.json']:continue
            d=read(p,{})
            vals={}
            for key,value in d.items():
                if key in ['ap','auroc','brier','ece','transition_f1','radgraph_f1','chexbert_f1','mean'] and isinstance(value,(int,float)):
                    vals[key]=round(value,5)
            for group in ['table1','table2_classification','table2_report']:
                for k,v in d.get(group,{}).items():
                    if k in ['ap','auroc','brier','ece','transition_f1','radgraph_f1','chexbert_f1'] and isinstance(v,(int,float)):
                        vals[group+'.'+k]=round(v,5)
            for group in ['test','human_test']:
                for k,v in d.get('metrics',{}).get(group,{}).items():
                    if k in ['dice','psnr','ssim','ap','auroc'] and isinstance(v,(int,float)):
                        vals[group+'.'+k]=round(v,5)
            lines.append(f'| [{j["id"]}]({p}) | {json.dumps(vals,ensure_ascii=False)} |')
            accepted.append(dict(job=j['id'],path=str(p),summary=vals))
    prior=run/'simple_future/finding_prior/test/metrics.json'
    if prior.exists():
        d=read(prior,{})
        vals={k:round(v,5) for k,v in d.get('table1',{}).items() if k in ['ap','auroc','brier','ece']}
        lines.append(f'| [Finding-transition prior]({prior}) | {json.dumps(vals)} |')
    lines+=['', 'Finding-transition prior使用训练集拟合，源状态来自已准备的结构化finding标签；该输入接口单独注明。',
            'Direction在297对测试集缺合格真值；官方VQA仍缺本轮正式数据。留空不等于零分。', '',
            '运行的源码、命令、预算、checkpoint、逐样本预测与数据指纹保存在各任务目录。']
    (run/output).write_text('\n'.join(lines)+'\n')
    (run/'status.json').write_text(json.dumps(dict(updated=stamp,counts=counts,
        phase='running' if any(k in counts for k in ['queued','running']) else 'finished' if jobs else 'preparing',
        jobs=len(jobs),downloads={a.get('model'):a.get('status') for a in assets}),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',default='REPORT.md');p.add_argument('--watch',action='store_true')
    args=p.parse_args()
    while True:
        render(args.run.resolve(),args.output)
        if not args.watch:break
        time.sleep(30)
