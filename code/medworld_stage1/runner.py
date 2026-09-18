"""Durable coordinator: readiness, real-data smoke, training, and aggregate report."""
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import bootstrap
from bootstrap import *


RUN=ROOT/'runs/overnight_20260910'


def launch(name,gpu,args):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
    with (RUN/f'{name}.log').open('a') as log:
        p=subprocess.Popen([sys.executable,str(ROOT/'train.py'),*args],stdout=log,stderr=subprocess.STDOUT,
                           env=env,cwd=ROOT,start_new_session=True)
    print('launched',name,'pid',p.pid,'gpu',gpu,flush=True)
    return p


def render(state):
    cfg=json.loads((ROOT/'config.json').read_text())
    manifest_path=Path(cfg['cache'])/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    lines=['# 四任务 Stage 1 夜间实验','',f'更新时间：{datetime.now().astimezone().isoformat(timespec="seconds")}',
           '',f'协调器状态：**{state}**。Qwen3.5-0.8B，8×1024 多模态 slots。',
           f'训练截止：{cfg["train_deadline"]}。超分线性倍率 {cfg["scale"]}，像素数倍率 {cfg["scale"]**2}。','',
           '[完整计划](https://github.com/MVChem/medical_world_model/blob/31c98ca14173620a3cec67da2defe559d05974da/research_notes/0910_stage1_four_task_plan.md)','']
    if manifest:
        lines+=['| 划分 | 图像 | 患者 | 分类 | diagnosis | 分割候选 | SR |','|---|---:|---:|---:|---:|---:|---:|']
        for split,c in manifest['coverage'].items():
            t=c['tasks'];lines.append(f'| {split} | {c["images"]} | {c["patients"]} | {t["classification"]} | {t["diagnosis"]} | {t["segmentation"]} | {t["sr"]} |')
        lines+=['','分割有效数以 CXAS 伪标签质控后的 data_usage.json 为准；候选来自全体 CXR，今晚训练固定子集。','']
    for name in ['joint','frozen_control']:
        folder=RUN/name
        sp=folder/'status.json';mp=folder/'evaluation/metrics.json'
        if not sp.exists():lines +=[f'**{name}**：尚未启动。',''];continue
        s=json.loads(sp.read_text())
        lines +=[f'**{name}**：{s["phase"]}，step={s["step"]}，训练计算时间={s.get("training_seconds",0)/3600:.2f} h。',
                 f'实际任务样本呈现次数：`{json.dumps(s.get("task_examples",{}))}`。',
                 f'[日志]({name}.log) · [详细状态]({name}/status.json)','']
        if 'error' in s:lines +=[f'错误：{s["error"]}','']
        if mp.exists():
            m=json.loads(mp.read_text())
            if m.get('classification',{}).get('test'):
                c=m['classification']['test']['actual'];lines +=[f'Classification：macro AUPRC={c["macro_auprc"]}，AUROC={c["macro_auroc"]}。','']
            if m.get('segmentation',{}).get('actual'):
                g=m['segmentation'];lines +=[f'Segmentation teacher-agreement Dice：{g["actual"]["mean_dice"]:.4f}；null={g["null"]["mean_dice"]:.4f}；shuffle={g["shuffled"]["mean_dice"]:.4f}。','']
            if m.get('sr',{}).get('actual'):
                a=m['sr']['actual'];b=m['sr']['bicubic'];lines +=[f'SR：PSNR={a["psnr"]:.4f} dB，SSIM={a["ssim"]:.4f}；bicubic={b["psnr"]:.4f} dB / {b["ssim"]:.4f}。','']
            d=m.get('diagnosis',{})
            if 'n' in d:
                lines +=[f'Diagnosis：{d["n"]} 个生成样例，唯一文本比例={d["unique_fraction"]:.3f}。','']
                clinical=d.get('clinical',{})
                if 'chexbert' in clinical:lines +=[f'CheXbert F1：{clinical["chexbert"]["macro_positive_f1"]}。','']
                if 'radgraph' in clinical:lines +=[f'RadGraph partial F1：{clinical["radgraph"]["partial_f1"]}。','']
                for k in ['clinical_error']:
                    if k in d:lines +=[d[k],'']
                for k in ['chexbert_error','radgraph_error']:
                    if k in clinical:lines +=[clinical[k],'']
            if 'frozen_probe' in m:lines +=[f'冻结 slots 后重新训练统一线性 probe，test AUPRC={m["frozen_probe"]["test"]["macro_auprc"]}。','']
            lines +=[f'[完整指标]({name}/evaluation/metrics.json) · [报告样例]({name}/evaluation/diagnosis.jsonl)','']
    lines +=['限定：报告已经进入 slots，分类/diagnosis 是多模态读出；分割数值衡量模仿 CXAS，不代表人工 GT 准确率；SR 是保留长宽比、HR 长边不超过 512 的合成降采样恢复。',
              'null/shuffle 是推理干预；frozen_control 是同初始化、同数据、同更新数的冻结共享 encoder 训练对照。最终测试使用 final checkpoint，best checkpoint 另存，避免按测试选模型。','']
    temp=RUN/'REPORT.tmp.md';temp.write_text('\n'.join(lines));temp.replace(RUN/'REPORT.md')


def main():
    RUN.mkdir(parents=True,exist_ok=True)
    cache=ROOT/'data/overnight_20260910'
    cfg=json.loads((ROOT/'config.json').read_text())
    state='waiting_for_caches'
    while not all((cache/n).exists() for n in ['manifest.json','features.json','segmentation.json']):
        atomic_json(RUN/'runner_status.json',dict(phase=state,pid=os.getpid(),updated=time.time()))
        render(state)
        if datetime.now().astimezone().isoformat()[:10]>'2026-09-10' and time.time()>datetime.fromisoformat('2026-09-11T02:00:00+08:00').timestamp():
            raise TimeoutError('Preprocessing did not complete by 02:00; see preparation logs')
        time.sleep(15)
    from quality import quality
    quality(cache)
    # Preserve the exact reviewed implementation and relevant imported model code.
    snapshot=RUN/'source';snapshot.mkdir(exist_ok=True)
    for p in ROOT.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    shutil.copy2(ROOT/'config.json',snapshot/'config.json')
    shutil.copy2(PILOT/'model.py',snapshot/'imported_pilot_model.py')
    shutil.copy2(PILOT/'common.py',snapshot/'imported_pilot_common.py')
    gpu_memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    for line in gpu_memory.strip().splitlines():
        index,mem=map(int,line.split(','))
        if index in [1,5] and mem>1000:raise RuntimeError(f'GPU {index} no longer free ({mem} MiB); refusing to overlap')
    smoke=dict(cfg,validation_count=4,diagnosis_eval_count=4,probe_train_count=16,probe_steps=4,evaluation_limit=8,evaluation_split='validate')
    atomic_json(RUN/'smoke_config.json',smoke)
    p=launch('real_smoke',1,['--config',str(RUN/'smoke_config.json'),'--run',str(RUN/'real_smoke'),'--smoke-steps','4'])
    while p.poll() is None:
        render('real_data_smoke');time.sleep(15)
    if p.returncode:raise RuntimeError('Real-data smoke failed; see real_smoke.log')
    checks=json.loads((RUN/'real_smoke/status.json').read_text())
    assert checks['phase']=='complete'
    primary=launch('joint',1,['--run',str(RUN/'joint')])
    control=launch('frozen_control',5,['--run',str(RUN/'frozen_control'),'--freeze-encoder','--follow',str(RUN/'joint')])
    while primary.poll() is None or control.poll() is None:
        atomic_json(RUN/'runner_status.json',dict(phase='training_or_evaluation',pid=os.getpid(),
            primary_pid=primary.pid,control_pid=control.pid,primary_exit=primary.poll(),control_exit=control.poll(),updated=time.time()))
        render('training_or_evaluation');time.sleep(30)
    state='complete' if primary.returncode==0 and control.returncode==0 else 'failed'
    atomic_json(RUN/'runner_status.json',dict(phase=state,primary_exit=primary.returncode,control_exit=control.returncode,updated=time.time()))
    render(state)


if __name__=='__main__':
    try:main()
    except Exception as e:
        atomic_json(RUN/'runner_status.json',dict(phase='failed',error=f'{type(e).__name__}: {e}',updated=time.time()))
        render(f'failed: {e}')
        raise
