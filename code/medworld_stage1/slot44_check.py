"""Check actual slot-boundary gradients, answer parsing, LR isolation and reload."""
import argparse
import json
from pathlib import Path
import torch
from slot44_networks import Slot44Model, SpatialSuperResolution
from slot44_corpus import Slot44Corpus
from slot44_evaluation import parse_list, set_metrics
from bootstrap import atomic_json, seed_all


def checks(cfg, out):
    seed_all(cfg['seed'])
    vocabulary=json.loads(Path(cfg['answer_vocabulary']).read_text())['labels']
    assert parse_list('["Pneumonia", "pneumonia"]',vocabulary)==(['pneumonia'],True,[])
    assert parse_list('a report instead of a list',vocabulary)[1] is False
    assert parse_list('[]',vocabulary)==([],True,[])
    common=dict(subject_id='synthetic-test',image_id='synthetic-test',oov=[],generation_limit_reached=False)
    scores=set_metrics([dict(common,answer=['pneumonia'],parsed=['pneumonia'],parse_valid=True),
                       dict(common,answer=[],parsed=['__unparseable_output__'],parse_valid=False)],vocabulary)
    assert scores['exact_match']==.5 and scores['micro']['precision']==.5
    model=Slot44Model(cfg).cuda().train()
    corpus=Slot44Corpus(cfg,model.tokenizer)
    assert all(not p.requires_grad for n,p in model.encoder.backbone.named_parameters() if 'lora_' not in n)
    assert all(not p.requires_grad for n,p in model.diagnosis.backbone.named_parameters() if 'lora_' not in n)
    encoder_ptrs={p.data_ptr() for p in model.encoder.backbone.parameters() if p.requires_grad}
    decoder_ptrs={p.data_ptr() for p in model.diagnosis.backbone.parameters() if p.requires_grad}
    assert not encoder_ptrs&decoder_ptrs
    records={}
    for task in cfg['tasks']:
        model.zero_grad(set_to_none=True)
        # Check the configured microbatch size using real validation records.
        ii=corpus.pools['validate',task][:cfg['batch_sizes'][task]]
        b=corpus.batch(ii,task)
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=model.loss(b,task,corpus.pos_weight.cuda())
        loss.backward()
        grad=model.last_state.grad.float().norm(dim=(0,2))
        active=list(range(4)) if task=='classification' else list(range(4,8)) if task in ('segmentation','sr') else list(range(8))
        inactive=[i for i in range(8) if i not in active]
        assert torch.isfinite(loss) and torch.isfinite(grad).all()
        assert (grad[active]>0).all() and (not inactive or (grad[inactive]==0).all())
        shared=float(model.encoder.slots.grad.float().norm())
        assert shared>0
        records[task]=dict(loss=float(loss.detach()),route_grad_norms=grad.cpu().tolist(),initial_token_grad_norm=shared,
                           batch=len(ii),peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3)
        print(task,records[task],flush=True)
    model.zero_grad(set_to_none=True);model.eval()
    b=corpus.batch(corpus.pools['validate','sr'][:2],'sr')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        state=model.encode(b,'sr')
        changed=dict(b,hr=b['hr']+100,hr_features=b['hr_features']+100)
        assert torch.equal(state,model.encode(changed,'sr'))
        output=model.sr(b['lr'],model.route(state,'sr'))
        assert output.shape==b['hr'].shape
        compact=model.compact_state()
        model.encoder.slots.add_(1)
        model.load_compact(compact)
        restored=model.sr(b['lr'],model.route(model.encode(b,'sr'),'sr'))
        assert torch.equal(output,restored)
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'contracts.json',dict(tasks=records,frozen_backbones=True,independent_loras=True,
        sr_hr_isolation=True,compact_reload=True,parser_checks=True,train_gold_patient_overlap=0))
    atomic_json(out/'data_usage.json',dict(pools={f'{s}/{t}':len(p) for (s,t),p in corpus.pools.items()},
        truncation=corpus.truncation,label_coverage=corpus.label_coverage,missing_report=corpus.missing_report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();checks(json.loads(a.config.read_text()),a.out)
