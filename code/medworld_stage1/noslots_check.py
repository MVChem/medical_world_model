"""Real-data no-slot checks, exact shared initialization, and sampler audit."""
import argparse
import copy
import json
from pathlib import Path
import torch
from bootstrap import atomic_json, atomic_torch, seed_all, ROOT
from slot44_corpus import Slot44Corpus
from noslots_networks import NoSlotsModel, TokenMemory
from noslots_audit import audit


def main(cfg,out):
    seed_all(cfg['seed'])
    model=NoSlotsModel(cfg)
    initialization=model.initialize_matched()
    assert initialization['trainable_parameters']==9255365-8192
    assert not hasattr(model.encoder,'slots')
    assert all(not p.requires_grad for n,p in model.encoder.backbone.named_parameters() if 'lora_' not in n)
    assert all(not p.requires_grad for n,p in model.diagnosis.backbone.named_parameters() if 'lora_' not in n)
    model=model.cuda().train()
    corpus=Slot44Corpus(cfg,model.tokenizer)
    fairness=audit(cfg,corpus,ROOT)
    checks={}
    for task in cfg['tasks']:
        model.zero_grad(set_to_none=True)
        ii=corpus.pools['validate',task][:cfg['batch_sizes'][task]]
        b=corpus.batch(ii,task)
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=model.loss(b,task,corpus.pos_weight.cuda())
        loss.backward()
        memory=model.last_memory
        assert memory.values.shape[1]==64+b['ids'].shape[1]
        assert torch.equal(memory.mask[:,64:],b['text_mask'])
        grad=memory.values.grad.float()
        assert torch.isfinite(loss) and torch.isfinite(grad).all() and grad.norm()>0
        padding=float((grad*(~memory.mask.bool())[:,:,None]).norm())
        assert padding==0, f'{task} padded positions affect decoder'
        encoder_grad=sum(float(p.grad.float().square().sum()) for p in model.encoder.parameters() if p.requires_grad and p.grad is not None)**.5
        assert encoder_grad>0
        checks[task]=dict(loss=float(loss.detach()),memory_tokens=memory.mask.sum(-1).tolist(),
            encoder_gradient_norm=encoder_grad,padding_gradient_norm=padding,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
        print(task,checks[task],flush=True)
    model.zero_grad(set_to_none=True);model.eval()
    b=corpus.batch(corpus.pools['validate','sr'][:2],'sr')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        a=model.encode(b,'sr')
        changed=dict(b,hr_features=b['hr_features']+100,hr=b['hr']+100)
        z=model.encode(changed,'sr')
        assert torch.equal(a.values,z.values) and torch.equal(a.mask,z.mask)
        output=model.sr(b['lr'],a)
        assert output.shape==b['hr'].shape
        corrupt=TokenMemory(a.values+(~a.mask.bool())[:,:,None]*100,a.mask)
        assert torch.equal(output,model.sr(b['lr'],corrupt))
        out.mkdir(parents=True,exist_ok=True)
        atomic_torch(out/'compact_reload.pt',model.compact_state())
        model.encoder.visual_position.add_(1)
        model.load_compact(torch.load(out/'compact_reload.pt',map_location='cpu',weights_only=True))
        restored=model.sr(b['lr'],model.encode(b,'sr'))
        assert torch.equal(output,restored)
    atomic_json(out/'contracts.json',dict(initialization=initialization,tasks=checks,no_learned_state_slots=True,
                full_token_readout=True,sr_hr_input_isolation=True,padding_content_invariance=True,checkpoint_file_reload=True))
    atomic_json(out/'fairness_audit.json',fairness)
    print('all no-slot contracts and 24000-step sample audit passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(json.loads(a.config.read_text()),a.out)
