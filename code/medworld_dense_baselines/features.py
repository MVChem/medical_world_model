"""Frozen native multimodal hidden states; resumable and fail-closed caches."""
import argparse
import json
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from common import *

sys.path.insert(0,str(PROJECT/'code/medworld_dense_baselines/vendor'))
sys.path.insert(0,str(PROJECT/'code/medworld_dense_baselines/vendor_compat'))

PROMPT = 'Represent the anatomy and spatial details of this chest radiograph.'

def sample_indices(n, k, device=None):
    if n < k:
        raise ValueError(f'cannot downsample {n} values to {k}')
    # Equal-width bins, take the center of each bin, preserve order, no padding.
    return torch.floor((torch.arange(k,device=device)+.5)*n/k).long()

def align_hidden(hidden):
    if hidden.ndim != 2 or not torch.isfinite(hidden).all():
        raise ValueError('invalid native image hidden state')
    side=int(len(hidden)**.5)
    if side*side!=len(hidden):
        raise ValueError('square image canvas must produce a square native token grid')
    idx=sample_indices(side,8,hidden.device)
    grid=hidden.reshape(side,side,-1).index_select(0,idx).index_select(1,idx).reshape(64,-1)
    return grid.index_select(1,sample_indices(hidden.shape[1],1024,hidden.device))

def downsample(images):
    return F.interpolate(images,scale_factor=.25,mode='bicubic',align_corners=False,antialias=True).clamp(0,1)

def load_vlm(spec):
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration, Gemma3ForConditionalGeneration
    cls = Qwen3_5ForConditionalGeneration if spec['family']=='qwen' else Gemma3ForConditionalGeneration
    proc = AutoProcessor.from_pretrained(spec['path'],local_files_only=True)
    if spec['family']=='qwen':
        proc.image_processor.size = {'longest_edge':512*512,'shortest_edge':256*256}
    kwargs = dict(local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',
        device_map='auto',max_memory={i:'21GiB' for i in range(torch.cuda.device_count())})
    model = cls.from_pretrained(spec['path'],**kwargs).eval().requires_grad_(False)
    if any(str(v) in ('cpu','disk') for v in getattr(model,'hf_device_map',{}).values()):
        raise RuntimeError('CPU/disk offload is excluded; wait for enough GPUs')
    return model,proc

def model_inputs(proc,image,device):
    messages=[{'role':'user','content':[{'type':'text','text':PROMPT},{'type':'image','image':image}]}]
    return proc.apply_chat_template(messages,tokenize=True,return_dict=True,return_tensors='pt',
        add_generation_prompt=False,enable_thinking=False).to(device)

def extract(run,mid,limit=None):
    torch.set_num_threads(4)
    spec=load_model_spec(mid)
    data=run/'data'; out=run/mid; out.mkdir(parents=True,exist_ok=True)
    rows=read_rows(data/'observations.jsonl')
    images=np.load(data/'images.npy',mmap_mode='r')
    lr_images=np.load(data/'lr_images.npy',mmap_mode='r')
    n=len(rows)
    contract={'cohort_sha256':digest(data/'observations.jsonl'),'shape':[n,64,1024],
              'image_sha256':json.loads((data/'manifest.json').read_text())['image_sha256'],
              'lr_image_sha256':json.loads((data/'manifest.json').read_text())['lr_image_sha256'],
              'config_sha256':spec['config_sha256'],'prompt':PROMPT,'layer':'final language model image tokens',
              'sampling':'8x8 spatial equal-width bin centers, then 1024 channel bin centers; no learned alignment',
              'sr_input':'bicubic-antialiased LR x4, uint8 round shared with head and V-JEPA',
              'code_sha256':digest(Path(__file__))}
    meta=out/'feature_contract.json'
    if meta.exists():
        if json.loads(meta.read_text())!=contract:
            prior=out/'features_done.npy'
            assert not prior.exists() or not np.load(prior).any(), 'nonempty feature contract changed'
    atomic(meta,contract)
    done=np.load(out/'features_done.npy') if (out/'features_done.npy').exists() else np.zeros((n,2),bool)
    required=np.array([[True,'sr' in r['tasks']] for r in rows])
    if np.all(done[required]):
        print('features already complete',flush=True);return
    arrays={branch:np.lib.format.open_memmap(out/f'{branch}_hidden.npy',mode='r+' if (out/f'{branch}_hidden.npy').exists() else 'w+',
        dtype=np.float16,shape=(n,64,1024)) for branch in ['hr','lr']}
    model,proc=load_vlm(spec)
    device=model.get_input_embeddings().weight.device
    token_id=getattr(model.config,'image_token_id',None) or getattr(model.config,'image_token_index',None)
    assert token_id is not None
    started=time.time(); completed=0; shapes=set()
    mapping=getattr(model,'hf_device_map',{'':str(device)})
    atomic(out/'model_loaded.json',dict(device_map={k:str(v) for k,v in mapping.items()},
        physical_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),image_token_id=token_id,
        quantization=str(getattr(model,'quantization_method',None))))
    with torch.inference_mode():
        for i,r in enumerate(rows):
            for j,branch in enumerate(['hr','lr']):
                if done[i,j] or not required[i,j]:continue
                arr=np.array(lr_images[i] if branch=='lr' else images[i],copy=True)
                image=Image.fromarray(arr).convert('RGB')
                inputs=model_inputs(proc,image,device)
                output=model.model(**inputs,use_cache=False,return_dict=True)
                h=output.last_hidden_state[0]
                mask=inputs['input_ids'][0].to(h.device)==token_id
                native=h[mask]
                shapes.add(tuple(native.shape))
                aligned=align_hidden(native).float().cpu().numpy().astype(np.float16)
                if not np.isfinite(aligned).all():raise ValueError('float16 overflow')
                arrays[branch][i]=aligned
                arrays[branch].flush()
                done[i,j]=True
                np.save(out/'features_done.tmp.npy',done)
                os.replace(out/'features_done.tmp.npy',out/'features_done.npy')
                completed+=1
                if completed%20==0 or completed==1:
                    atomic(out/'features_progress.json',dict(done=int(done[required].sum()),total=int(required.sum()),
                        seconds=time.time()-started,native_shapes=sorted(shapes)))
                    print('features',mid,int(done[required].sum()),'/',int(required.sum()),'seconds',round(time.time()-started),flush=True)
                if limit and completed>=limit:return
    assert np.all(done[required])
    atomic(out/'features_complete.json',dict(**contract,native_shapes=sorted(shapes),complete=True,
        actual_forward_calls=completed,seconds=time.time()-started))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,default=DEFAULT_RUN)
    p.add_argument('--model',required=True)
    p.add_argument('--limit',type=int)
    a=p.parse_args();extract(a.run,a.model,a.limit)
