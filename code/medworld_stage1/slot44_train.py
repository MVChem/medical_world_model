"""Persistent four-task training with task-aware sampling, restart, and matched control."""
import argparse
import json
import math
import os
import signal
import time
from datetime import datetime
from pathlib import Path

import bootstrap
from bootstrap import *
import torch
from slot44_corpus import Slot44Corpus as Corpus
from slot44_networks import Slot44Model as FourTaskModel

STOP=False


def signal_stop(*args):
    global STOP
    STOP=True


def optimizer_for(model,cfg):
    groups=[]
    for lora,lr in [(True,cfg['lora_learning_rate']),(False,cfg['learning_rate'])]:
        params=[p for n,p in model.named_parameters() if p.requires_grad and ('lora_' in n)==lora]
        if params:groups.append(dict(params=params,lr=lr,base_lr=lr))
    return torch.optim.AdamW(groups,betas=(.9,.95),weight_decay=.01)


@torch.no_grad()
def validate(model,corpus,count):
    model.eval()
    result={}
    weights=corpus.pos_weight.cuda()
    for task in model.cfg['tasks']:
        rows=corpus.pools['validate',task][:count]
        total=0.
        bs=model.cfg['batch_sizes'][task]
        for start in range(0,len(rows),bs):
            ii=rows[start:start+bs]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=model.loss(corpus.batch(ii,task),task,weights)
            total+=float(loss)*len(ii)
        result[task]=total/len(rows) if rows else None
    model.train()
    return result


def train(args,cfg):
    os.umask(0o077)
    seed_all(cfg['seed'])
    run=args.run.resolve()
    run.mkdir(parents=True,exist_ok=True)
    if (run/'status.json').exists() and not args.resume:
        raise FileExistsError('Existing run; require --resume')
    atomic_json(run/'config.json',dict(**cfg,freeze_encoder=args.freeze_encoder))
    signal.signal(signal.SIGTERM,signal_stop)
    signal.signal(signal.SIGINT,signal_stop)
    checkpoint_dir=Path(cfg['cache'])
    signatures={n:digest(checkpoint_dir/n) for n in ['manifest.json','observations.jsonl','features.json','segmentation.json','segmentation_qc.json']}
    signatures['selection']=digest(Path(cfg['selection']))
    signatures['qa_manifest']=digest(Path(cfg['qa_manifest']))
    signatures['answer_vocabulary']=digest(Path(cfg['answer_vocabulary']))
    signatures['qwen_config']=digest(Path(cfg['qwen'])/'config.json')
    signatures['qwen_weights']=digest(Path(cfg['qwen'])/'model.safetensors-00001-of-00001.safetensors')
    source_hashes={p.name:digest(p) for p in ROOT.glob('*.py')}
    atomic_json(run/'provenance.json',dict(inputs=signatures,source=source_hashes,torch=torch.__version__,
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    model=FourTaskModel(cfg,freeze_encoder=args.freeze_encoder).cuda()
    corpus=Corpus(cfg,model.tokenizer)
    atomic_json(run/'data_usage.json',dict(truncation=corpus.truncation,
        pools={f'{s}/{t}':len(p) for (s,t),p in corpus.pools.items()},pos_weight=corpus.pos_weight.tolist(),
        classification_label_coverage=corpus.label_coverage,missing_report_by_task=corpus.missing_report))
    optimizer=optimizer_for(model,cfg)
    weights=corpus.pos_weight.cuda()
    step=0
    best=float('inf')
    best_tasks={t:float('inf') for t in cfg['tasks']}
    initial=None
    elapsed=0.
    task_examples={t:0 for t in cfg['tasks']}
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=False)
        assert saved['signatures']==signatures and saved['cfg']==cfg
        assert saved['freeze_encoder']==args.freeze_encoder
        model.load_compact(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        step=saved['step']; best=saved['best']; initial=saved['initial']; elapsed=saved['elapsed']
        task_examples=saved['task_examples']
        best_tasks=saved.get('best_tasks',best_tasks)
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    started=time.time()
    deadline=datetime.fromisoformat(cfg['train_deadline']).timestamp()
    last_save=time.monotonic()
    def status(phase,**extra):
        atomic_json(run/'status.json',dict(phase=phase,pid=os.getpid(),step=step,
            task_examples=task_examples,
            training_seconds=elapsed,updated=time.time(),**extra))
    def save(name):
        atomic_torch(run/name,dict(model=model.compact_state(),optimizer=optimizer.state_dict(),step=step,
            best=best,best_tasks=best_tasks,initial=initial,elapsed=elapsed,cfg=cfg,signatures=signatures,source_hashes=source_hashes,
            freeze_encoder=args.freeze_encoder,task_examples=task_examples,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all()))
        print('saved',name,'step',step,flush=True)
    try:
        status('initial_validation')
        if initial is None:
            initial=validate(model,corpus,cfg['validation_count'])
            atomic_json(run/'validation_initial.json',initial)
            save('checkpoint_initial.pt')
        model.train()
        print('initial',initial,'trainable',sum(p.numel() for p in model.parameters() if p.requires_grad),flush=True)
        while not STOP:
            if args.follow:
                path=args.follow/'status.json'
                if not path.exists():time.sleep(5);continue
                ref=json.loads(path.read_text())
                if step>=ref['step']:
                    if ref['phase'] in ['training_complete','evaluating','complete','failed','interrupted']:
                        break
                    time.sleep(2)
                    continue
            elif time.time()>=deadline or elapsed>=cfg['max_train_hours']*3600 or step>=cfg['max_steps']:
                break
            task=cfg['tasks'][step%len(cfg['tasks'])]
            task_step=step//len(cfg['tasks'])
            tick=time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            for g in optimizer.param_groups:
                g['lr']=g['base_lr']*min(1.,(step+1)/cfg['warmup_steps'])
            raw=0.
            for micro in range(cfg['gradient_accumulation']):
                indices=corpus.sample(task,task_step,micro)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=model.loss(corpus.batch(indices,task),task,weights)
                if not torch.isfinite(loss):raise FloatingPointError(f'{task} loss not finite')
                (loss*cfg['loss_weights'][task]/cfg['gradient_accumulation']).backward()
                raw+=float(loss.detach())/cfg['gradient_accumulation']
                task_examples[task]+=len(indices)
            slot_grad=model.encoder.slots.grad
            slot_norm=float(slot_grad.float().norm()) if slot_grad is not None else 0.
            state_grad=model.last_state.grad
            route_norms=state_grad.float().norm(dim=(0,2)).tolist() if state_grad is not None else []
            if len(route_norms)!=8 or not all(math.isfinite(x) for x in route_norms):
                raise RuntimeError(f'{task}: invalid gradient at state routing boundary')
            expected=range(4) if task=='classification' else range(4,8) if task in ('segmentation','sr') else range(8)
            if step<4 and not all(route_norms[i]>0 for i in expected):
                raise RuntimeError(f'{task}: disconnected assigned slot')
            inactive=range(4,8) if task=='classification' else range(4) if task in ('segmentation','sr') else []
            if any(route_norms[i]!=0 for i in inactive):
                raise RuntimeError(f'{task}: decoder read the wrong state group')
            encoder_grad=math.sqrt(sum(float(p.grad.float().square().sum()) for p in model.encoder.parameters()
                                      if p.requires_grad and p.grad is not None))
            if not args.freeze_encoder and step<4 and (not math.isfinite(slot_norm) or slot_norm==0):
                raise RuntimeError(f'{task}: no finite nonzero gradient to slots')
            grad_norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True))
            optimizer.step()
            torch.cuda.synchronize()
            duration=time.monotonic()-tick
            elapsed+=duration
            step+=1
            record=dict(step=step,task=task,loss=raw,weighted_loss=raw*cfg['loss_weights'][task],
                slot_grad_norm=slot_norm,state_route_grad_norms=route_norms,encoder_grad_norm=encoder_grad,
                grad_norm=grad_norm,seconds=duration,time=time.time())
            with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            status('training',last=record)
            if step<=4 or step%20==0:print(json.dumps(record),flush=True)
            if step%cfg['validation_every']==0 or (args.smoke_steps and step==args.smoke_steps):
                val=validate(model,corpus,cfg['validation_count'])
                score=sum(val[t]/max(initial[t],1e-8) for t in cfg['tasks'])/len(cfg['tasks'])
                vr=dict(step=step,losses=val,normalized_score=score,time=time.time())
                atomic_json(run/'validation_latest.json',vr)
                with (run/'validation_history.jsonl').open('a') as f:f.write(json.dumps(vr)+'\n')
                if score<best:
                    best=score
                    save('checkpoint_best.pt')
                for t in cfg['tasks']:
                    if val[t]<best_tasks[t]:
                        best_tasks[t]=val[t]
                        save(f'checkpoint_best_{t}.pt')
                save('checkpoint_latest.pt')
                print('validation',vr,flush=True)
            if time.monotonic()-last_save>=cfg['checkpoint_minutes']*60 or step==4:
                save('checkpoint_latest.pt')
                last_save=time.monotonic()
            early_time=datetime.fromisoformat(cfg['early_checkpoint_time']).timestamp()
            if time.time()>=early_time and step%len(cfg['tasks'])==0 and not (run/'checkpoint_early.pt').exists():
                save('checkpoint_early.pt')
                atomic_json(run/'early_checkpoint.json',dict(step=step,time=time.time(),selection='fixed wall-clock deadline',
                            evaluation_split='validate'))
            if args.smoke_steps and step>=args.smoke_steps:break
        save('checkpoint_interrupted.pt' if STOP else 'checkpoint_final.pt')
        status('interrupted' if STOP else 'training_complete',wall_seconds=time.time()-started)
        if not STOP and not args.no_evaluate:
            optimizer.zero_grad(set_to_none=True)
            del optimizer
            torch.cuda.empty_cache()
            status('evaluating')
            from slot44_evaluation import evaluate
            evaluate(model,corpus,run)
            status('complete',wall_seconds=time.time()-started)
    except Exception as e:
        status('failed',error=f'{type(e).__name__}: {e}')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=ROOT/'config.json')
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--freeze-encoder',action='store_true')
    p.add_argument('--follow',type=Path)
    p.add_argument('--resume',type=Path)
    p.add_argument('--smoke-steps',type=int,default=0)
    p.add_argument('--no-evaluate',action='store_true')
    a=p.parse_args()
    train(a,json.loads(a.config.read_text()))
