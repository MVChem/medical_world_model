"""Persistent single-run coordinator and fixed-time validation preview."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from bootstrap import ROOT, PILOT, PROJECT, atomic_json, digest


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def curves(run):
    path=run/'joint/metrics.jsonl'
    if not path.exists():return
    rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    if not rows:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(2,2,figsize=(10,6))
    for ax,task in zip(axes.flat,['classification','diagnosis','segmentation','sr']):
        rr=[r for r in rows if r['task']==task]
        xs=[r['step'] for r in rr];ys=np.array([r['loss'] for r in rr])
        ax.plot(xs,ys,alpha=.2,lw=.6)
        if len(ys)>=20:ax.plot(xs[19:],np.convolve(ys,np.ones(20)/20,'valid'),lw=1.2)
        ax.set_title('Disease recognition' if task=='diagnosis' else task.capitalize())
        ax.set_xlabel('Optimizer step');ax.set_ylabel('Unweighted task loss');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(run/'training.png',dpi=150);fig.savefig(run/'training.pdf');plt.close(fig)


def render(run,phase):
    cfg=read(run/'config.json');selection=read(Path(cfg['selection']))
    status=read(run/'joint/status.json');early=read(run/'early/evaluation/metrics.json')
    final=read(run/'joint/evaluation/metrics.json')
    lines=['# Stage 1：0.8B，4＋4 slots','',f'更新时间：{datetime.now().astimezone().isoformat(timespec="seconds")}',
           f'协调器：{phase}。训练：{status.get("phase","尚未开始")}；optimizer step：{status.get("step",0)} / {cfg["max_steps"]}。','',
           '分类读 S1–4；疾病列表读 S1–8；分割与 SR 读 S5–8。SR 包含两层空间 attention。',
           'Qwen3.5-0.8B 与 V-JEPA 基础权重冻结；重新初始化可训练参数，从头训练本轮适配器、slots 与 decoder。','',
           f'问答来源：**{selection.get("qa_source")}**。',
           f'所有任务训练池排除 gold 测试患者；移除原训练图像 {selection.get("removed_from_original_train_images")} 张。','',
           '| 问答划分 | 图像 | 问题 | 整图 | 区域 |','|---|---:|---:|---:|---:|']
    for s,c in selection.get('qa_coverage',{}).items():
        lines.append(f'| {s} | {c["images"]} | {c["questions"]} | {c["whole"]} | {c["region"]} |')
    if selection.get('derived_positive_only'):
        lines+=['','本地构造候选只使用有明确阳性标注的问题；未标注项未改成阴性。本版不评估空答案能力，不属于官方 VQA benchmark。']
    lines+=['',f'早期 checkpoint 预定时间：{cfg["early_checkpoint_time"]}；随后在另一张 GPU 做验证集评估，训练继续。',
            '预览使用验证集；最终测试使用统一 final checkpoint，不按测试结果选择模型。','',
            '[训练日志](joint.log) · [训练状态](joint/status.json) · [配置](config.json) · [数据划分](../../data/'+Path(cfg['selection']).parent.name+'/selection.json)','',
            '![四任务训练 loss](training.png)','']
    if status.get('last'):
        last=status['last'];lines +=[f'最近更新：{last["task"]} loss={last["loss"]:.6g}；共享 encoder 梯度范数={last.get("encoder_grad_norm",0):.6g}。','']
    for name,m,link in [('早期验证',early,'early/evaluation'),('最终测试',final,'joint/evaluation')]:
        if not m:lines +=[f'{name}：等待评估。',''];continue
        c=m.get('classification',{});d=m.get('disease_recognition',{}).get('all',{});g=m.get('segmentation',{});s=m.get('sr',{})
        def val(x):return f'{x:.4f}' if isinstance(x,(int,float)) else '待完成'
        lines +=[f'**{name}**：split={m.get("split")}；step={m.get("step")}。','',
                  '| AUC | AP | F1 | Dice | PSNR | SSIM |','|---:|---:|---:|---:|---:|---:|',
                  '| '+' | '.join(val(x) for x in [c.get('macro_auroc'),c.get('macro_auprc'),d.get('macro',{}).get('f1'),g.get('mean_dice'),s.get('psnr'),s.get('ssim')])+' |','',
                  f'[完整指标]({link}/metrics.json) · [诊断问答与预测]({link}/disease_recognition.jsonl) · [分割样例]({link}/seg_00.png) · [超分样例]({link}/sr_00.png)','']
        if d:lines +=[f'疾病集合 micro F1={val(d["micro"]["f1"])}；无法解析输出 {d["invalid_outputs"]}/{d["n"]}。宏平均使用固定疾病／征象词表。','']
    lines+=['分类与疾病识别是当前报告辅助的多模态读出；分割是 CXAS 伪标签一致性；SR 是合成 ×2。','']
    if 'error' in status:lines +=[f'训练错误：{status["error"]}','']
    temp=run/'REPORT.tmp.md';temp.write_text('\n'.join(lines));temp.replace(run/'REPORT.md')


def main(args):
    os.umask(0o077)
    run=args.run.resolve();run.mkdir(parents=True,exist_ok=True)
    if (run/'runner_status.json').exists():raise FileExistsError('Use a fresh coordinator directory')
    cfg=json.loads(args.config.read_text())
    atomic_json(run/'config.json',cfg)
    source=run/'source';source.mkdir(exist_ok=True)
    for p in ROOT.glob('*.py'):shutil.copy2(p,source/p.name)
    for name in ['model.py','common.py']:shutil.copy2(PILOT/name,source/name)
    bootstrap=(source/'bootstrap.py').read_text()
    bootstrap=bootstrap.replace('PROJECT = ROOT.parent.parent',f'PROJECT = Path({str(PROJECT)!r})')
    bootstrap=bootstrap.replace("PILOT = ROOT.parent / 'medworld_table1'",f'PILOT = Path({str(PILOT)!r})')
    bootstrap=bootstrap.replace('from common import',"sys.path.insert(0, str(ROOT))\nfrom common import")
    (source/'bootstrap.py').write_text(bootstrap)
    shutil.copy2(args.config,source/'config.json')
    atomic_json(run/'source_manifest.json',{p.name:digest(p) for p in source.glob('*.py')})
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    for row in memory.strip().splitlines():
        index,used=map(int,row.split(','))
        if index in [args.train_gpu,args.eval_gpu] and used>1000:
            raise RuntimeError(f'GPU {index} is occupied ({used} MiB)')
    def launch(name,gpu,command):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
        with (run/f'{name}.log').open('a') as log:
            p=subprocess.Popen([sys.executable,*map(str,command)],stdout=log,stderr=subprocess.STDOUT,
                              env=env,cwd=source,start_new_session=True)
        atomic_json(run/f'{name}_process.json',dict(pid=p.pid,gpu=gpu,started=time.time(),command=list(map(str,command))))
        print('launched',name,p.pid,'gpu',gpu,flush=True)
        return p
    train=launch('joint',args.train_gpu,[source/'slot44_train.py','--config',run/'config.json','--run',run/'joint'])
    preview=None;last_curve=0;phase='training'
    while True:
        status=read(run/'joint/status.json')
        if time.time()-last_curve>120:
            curves(run);last_curve=time.time()
        if preview is None:
            ckpt=run/'joint/checkpoint_early.pt'
            if not ckpt.exists() and status.get('phase') in ['training_complete','evaluating','complete']:
                ckpt=run/'joint/checkpoint_final.pt'
            if ckpt.exists():
                preview=launch('early',args.eval_gpu,[source/'slot44_evaluation.py','--checkpoint',ckpt,
                    '--out',run/'early','--split','validate','--limit','0','--qa-limit','128'])
        if preview is not None and preview.poll() is not None:
            phase='training_with_preview_ready' if preview.returncode==0 else 'training_preview_failed'
        if train.poll() is not None:
            if train.returncode!=0:phase='training_failed'
            elif preview is not None and preview.poll() is not None:phase='complete'
            else:phase='training_complete_waiting_preview'
        atomic_json(run/'runner_status.json',dict(phase=phase,pid=os.getpid(),updated=time.time(),
                    training_pid=train.pid,preview_pid=preview.pid if preview else None))
        render(run,phase)
        if phase in ['complete','training_failed']:
            curves(run);break
        time.sleep(15)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--train-gpu',type=int,default=0);p.add_argument('--eval-gpu',type=int,default=1)
    main(p.parse_args())
