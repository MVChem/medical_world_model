"""Durable no-slot baseline and automatic comparisons at identical steps."""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from bootstrap import atomic_json, digest
from snapshot import create_snapshot, source_hashes


def read(p):
    return json.loads(p.read_text()) if p.exists() else {}


def scores(m):
    return [m.get('classification',{}).get('macro_auroc'),m.get('classification',{}).get('macro_auprc'),
            m.get('disease_recognition',{}).get('all',{}).get('macro',{}).get('f1'),
            m.get('segmentation',{}).get('mean_dice'),m.get('sr',{}).get('psnr'),m.get('sr',{}).get('ssim')]


def record_comparison(run, name, reference, baseline, refstep, basestep):
    assert refstep==basestep, 'Refuse comparison between different optimizer steps'
    r=read(reference/'metrics.json');b=read(baseline/'metrics.json')
    assert r['split']==b['split'] and r['qa_source']==b['qa_source']
    manifest={}
    for file,keys in [('classification.jsonl',['image_id','labels']),('segmentation.jsonl',['image_id']),
                      ('sr.jsonl',['image_id','hr_size','lr_size']),
                      ('disease_recognition.jsonl',['id','image_id','question','answer'])]:
        def identities(path):
            return [{k:q[k] for k in keys} for q in [json.loads(s) for s in path.read_text().splitlines()]]
        assert identities(reference/file)==identities(baseline/file), f'Evaluation examples differ: {file}'
        manifest[file]={'reference_predictions_sha256':digest(reference/file),'baseline_predictions_sha256':digest(baseline/file)}
    names=['auc','ap','macro_f1','dice','psnr','ssim']
    rs=dict(zip(names,scores(r)));bs=dict(zip(names,scores(b)))
    result=dict(step=refstep,split=r['split'],same_evaluation_examples_and_targets=True,
                ours=rs,qwen_no_slots=bs,ours_minus_baseline={k:rs[k]-bs[k] for k in names},
                prediction_files=manifest,comparison='Matched optimizer steps, sample stream, inputs, heads and common initial parameters; different context lengths')
    atomic_json(run/f'comparison_{name}.json',result)
    return result


def render(run,phase):
    cfg=read(run/'config.json');refpath=Path(cfg['reference_run']);reference=read(refpath/'status.json');baseline=read(run/'joint/status.json')
    lines=['# Qwen3.5-0.8B：无状态 slots 对照','',f'更新时间：{datetime.now().astimezone().isoformat(timespec="seconds")}',
           f'协调器：{phase}。Ours：{reference.get("step",0)} 步；无 slots baseline：{baseline.get("step",0)} 步。','',
           '相同 V-JEPA 视觉前端、Qwen0.8B 骨干与独立 decoder、LoRA、四任务 heads。Baseline 直接读取全部图像／报告 tokens，没有 8 个可学习状态 slots。',
           '共用参数从 Ours 的 step=0 checkpoint 精确复制；未使用任何训练后的 Ours 权重。',
           '两者采用相同患者划分、每一步样本、13类标签屏蔽、问答词表、优化器及训练步数。完整 token 上下文的计算量与运行时间不保证相同。','',
           '计划各 24,000 次 optimizer 更新，每任务 6,000 次。Baseline 跟随主实验实际步数；不会因为晚启动而按同一时刻提前截断。',
           f'早期比较固定在 {cfg["matched_preview_step"]} 步；最终比较使用相同步数的 final checkpoint。','',
           '[训练状态](joint/status.json) · [训练日志](joint.log) · [公平性核验](joint/fairness_audit.json) · [配置](config.json)','']
    for label,name in [(f'第{cfg["matched_preview_step"]}步验证','early'),('最终测试','final')]:
        m=read(run/f'comparison_{name}.json')
        if not m:lines +=[f'{label}：等待相同步数的两份结果。',''];continue
        lines +=[f'**{label}**，step={m["step"]}，split={m["split"]}。','',
                  '| Method | AUC | AP | F1 | Dice | PSNR | SSIM |','|---|---:|---:|---:|---:|---:|---:|']
        for title,key in [('Qwen0.8B, no slots','qwen_no_slots'),('Ours, 4+4 slots','ours'),('Ours − baseline','ours_minus_baseline')]:
            lines.append('| '+title+' | '+' | '.join(f'{m[key][k]:.4f}' for k in ['auc','ap','macro_f1','dice','psnr','ssim'])+' |')
        lines +=['',f'[比较与来源核验](comparison_{name}.json)','']
    lines +=['本轮是当前报告辅助读出；问答采用相同的本地 Chest ImaGenome 派生阳性问题；分割为 CXAS 伪标签一致性，SR 为合成 ×2。','']
    if baseline.get('error'):lines +=[f'Baseline 错误：{baseline["error"]}','']
    tmp=run/'REPORT.tmp.md';tmp.write_text('\n'.join(lines));tmp.replace(run/'REPORT.md')


def main(args):
    os.umask(0o077)
    run=args.run.resolve();source=run/'source';cfg=read(run/'config.json')
    if (run/'runner_status.json').exists():raise FileExistsError('Coordinator already has state')
    if not source.exists():
        create_snapshot(run)
    atomic_json(run/'source_manifest.json',source_hashes(source))
    def available(preferred):
        result=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        usage={int(s.split(',')[0]):int(s.split(',')[1]) for s in result.strip().splitlines()}
        return next((g for g in preferred if usage.get(g,999999)<1000),None)
    def launch(name,gpu,command):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
        with (run/f'{name}.log').open('a') as log:
            p=subprocess.Popen([sys.executable,*map(str,command)],stdout=log,stderr=subprocess.STDOUT,
                               env=env,cwd=source,start_new_session=True)
        atomic_json(run/f'{name}_process.json',dict(pid=p.pid,gpu=gpu,started=time.time(),command=list(map(str,command))))
        print('launched',name,p.pid,'gpu',gpu,flush=True)
        return p
    if available([args.train_gpu]) is None:raise RuntimeError(f'GPU {args.train_gpu} occupied')
    training=launch('joint',args.train_gpu,[source/'noslots_train.py','--config',run/'config.json','--run',run/'joint'])
    preview=None;phase='training';refpath=Path(cfg['reference_run']);alignment_error=None
    while True:
        if preview is None and (run/'joint/checkpoint_matched_early.pt').exists():
            candidates = [cfg['preview_gpu']] if 'preview_gpu' in cfg else [5,6,7]
            gpu=available([g for g in candidates if g!=args.train_gpu])
            if gpu is not None:
                preview=launch('early',gpu,[source/'noslots_evaluation.py','--checkpoint',run/'joint/checkpoint_matched_early.pt',
                    '--out',run/'early','--split','validate','--limit','0','--qa-limit','128'])
        if preview is not None and preview.poll() is not None:
            phase='training_with_matched_preview' if preview.returncode==0 else 'training_preview_failed'
            if preview.returncode==0 and not (run/'comparison_early.json').exists():
                r=read(refpath.parent/'early/evaluation/metrics.json');b=read(run/'early/evaluation/metrics.json')
                if r.get('completed') and b.get('completed'):
                    try:record_comparison(run,'early',refpath.parent/'early/evaluation',run/'early/evaluation',r['step'],b['step'])
                    except Exception as e:alignment_error=str(e);phase='comparison_alignment_error'
        if training.poll() is not None:
            if training.returncode!=0:phase='training_failed'
            else:
                r=read(refpath/'evaluation/metrics.json');b=read(run/'joint/evaluation/metrics.json')
                rs=read(refpath/'status.json');bs=read(run/'joint/status.json')
                if r.get('completed') and b.get('completed'):
                    try:
                        assert rs['task_examples']==bs['task_examples'], 'Final task example counts differ'
                        record_comparison(run,'final',refpath/'evaluation',run/'joint/evaluation',rs['step'],bs['step'])
                        phase='complete'
                    except Exception as e:alignment_error=str(e);phase='comparison_alignment_error'
                else:phase='waiting_for_reference_final_evaluation'
        atomic_json(run/'runner_status.json',dict(phase=phase,pid=os.getpid(),updated=time.time(),
            training_pid=training.pid,preview_pid=preview.pid if preview else None,alignment_error=alignment_error))
        render(run,phase)
        if phase in ['complete','training_failed']:break
        time.sleep(15)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--train-gpu',type=int,default=1)
    main(p.parse_args())
