"""Frozen official GREEN prompts/tokenizer/parser with local batched vLLM inference."""
import argparse
import ast
import os
import re
import subprocess
import sys
import time
import types
from base import *

GREEN_REV='832389d56bbd45a36bbd61dfa4f0be76e99f1dad'
GREEN_PATH=CACHE/'models--StanfordAIMI--GREEN-RadLlama2-7b/snapshots'/GREEN_REV

def official_functions(run):
    root=Path(__file__).resolve().parent/'green_official'
    if not root.exists():root=PROJECT/'code/medworld_baselines/vendor/GREEN/green_score'
    namespace={'re':re}
    tree=ast.parse((root/'utils.py').read_text())
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['make_prompt','clean_responses']]
    tree2=ast.parse((root/'green.py').read_text())
    cls=next(n for n in tree2.body if isinstance(n,ast.ClassDef) and n.name=='GREEN')
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['parse_error_counts','compute_green','compute_error_count']]
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    parser=types.SimpleNamespace()
    for node in init.body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Attribute) and node.targets[0].attr in ['categories','sub_categories']:
            setattr(parser,node.targets[0].attr,ast.literal_eval(node.value))
    exec(compile(ast.Module(body=funcs+methods,type_ignores=[]),str(root),'exec'),namespace)
    for m in methods:setattr(parser,m.name,types.MethodType(namespace[m.name],parser))
    parser.clean_response=namespace['clean_responses']
    return namespace['make_prompt'],parser,{p.name:digest(p) for p in [root/'utils.py',root/'green.py']}

def prepare_jobs(args):
    # Preserve the exact upstream slow tokenizer on its compatible Transformers 4 API.
    sys.path.insert(0,str(T1/'metric_vendor'))
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(GREEN_PATH,trust_remote_code=True,local_files_only=True,
        add_eos_token=True,use_fast=True,padding_side='left')
    tokenizer.pad_token=tokenizer.eos_token
    make_prompt,parser,hashes=official_functions(args.run)
    out=args.run/args.model/'test'
    done={r['key']:r for r in rows(out/'responses.jsonl') if r.get('ok')}
    jobs=[]
    for ref in rows(args.run/'cohort/table1_references_test.jsonl'):
        pred=done['table1_report|'+ref['id']+'|']
        prompt=make_prompt(ref['target_report'],pred['text'])
        chat=[{'from':'human','value':prompt},{'from':'gpt','value':''}]
        text=tokenizer.apply_chat_template(chat,tokenize=False,add_generation_prompt=True)
        ids=tokenizer(text,add_special_tokens=True)['input_ids']
        jobs.append(dict(id=ref['id'],prompt_token_ids=ids,input_length=len(ids)))
    write_rows(out/'green_jobs.jsonl',jobs)
    atomic(out/'green_protocol.json',dict(checkpoint=str(GREEN_PATH),revision=GREEN_REV,source_sha256=hashes,
        tokenizer_source_sha256=digest(GREEN_PATH/'tokenization_chexagent.py'),
        prompt='official make_prompt, 300 words per report; official two-message chat template; BOS and EOS as upstream',
        decoding='vLLM BF16 greedy, maximum 2048 total tokens per record; no input truncation; overlong or unscorable outputs receive zero with failure count',
        scoring='Unmodified official compute_green and parse_error_counts; all 297 records retained in denominator',
        engine_difference='Batched vLLM execution and BF16; original package loads float16 and budgets output by padded batch length'))

def main(args):
    os.umask(0o077);run=args.run.resolve()
    if args.prepare:return prepare_jobs(args)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),VLLM_USE_FLASHINFER_SAMPLER='0',OMP_NUM_THREADS='4')
    deadline=time.monotonic()+7200
    index=read(GREEN_PATH/'model.safetensors.index.json')
    while not all((GREEN_PATH/f).exists() for f in set(index['weight_map'].values())):
        if time.monotonic()>deadline:
            atomic(run/'green_status.json',dict(status='unavailable',reason='model download did not complete'));return
        time.sleep(30)
    from vllm import LLM,SamplingParams
    import sentencepiece as spm
    engine=LLM(model=str(GREEN_PATH),skip_tokenizer_init=True,dtype='bfloat16',max_model_len=2048,
        max_num_seqs=16,max_num_batched_tokens=4096,gpu_memory_utilization=.88,enforce_eager=True,
        disable_log_stats=True)
    decoder=spm.SentencePieceProcessor(model_file=str(GREEN_PATH/'tokenizer.model'))
    _,parser,_=official_functions(run)
    pending={m['id'] for m in read(run/'models.json')}
    while pending:
        progress=False
        for name in list(pending):
            out=run/name/'test'
            if not (run/name/'inference_finished.json').exists():continue
            if not (out/'responses.jsonl').exists():pending.remove(name);continue
            rc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--run',str(run),'--model',name,'--prepare'],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=300).returncode
            if rc:
                atomic(out/'green_metrics.json',dict(mean=None,status='tokenization_failed',returncode=rc));pending.remove(name);continue
            jobs=rows(out/'green_jobs.jsonl')
            response_path=out/'green_responses.jsonl'
            previous={r['id']:r for r in rows(response_path)} if response_path.exists() else {}
            for start in range(0,len(jobs),16):
                chunk=[r for r in jobs[start:start+16] if r['id'] not in previous]
                valid=[r for r in chunk if r['input_length']<2048]
                for r in chunk:
                    if r not in valid:previous[r['id']]=dict(id=r['id'],score=0.,valid=False,error='input over total token budget')
                if valid:
                    outputs=engine.generate([dict(prompt_token_ids=r['prompt_token_ids']) for r in valid],
                        [SamplingParams(temperature=0,max_tokens=2048-r['input_length'],detokenize=False,stop_token_ids=[2]) for r in valid],use_tqdm=False)
                    for row,output in zip(valid,outputs):
                        answer=output.outputs[0];text=parser.clean_response(decoder.decode([i for i in answer.token_ids if i not in [0,1,2]]))
                        score=parser.compute_green(text)
                        parsed=all('['+c+']:' in text for c in ['Clinically Significant Errors','Matched Findings'])
                        # Keep the unmodified official score; stricter parsing coverage is diagnostic.
                        valid_output=parsed and answer.finish_reason!='length'
                        previous[row['id']]=dict(id=row['id'],score=float(score or 0.),strict_score=float(score or 0.) if valid_output else 0.,
                            official_score=score,valid=valid_output,analysis=text,error_counts=parser.compute_error_count(text),
                            finish_reason=answer.finish_reason,input_tokens=row['input_length'],output_tokens=len(answer.token_ids))
                ordered=[previous[r['id']] for r in jobs if r['id'] in previous]
                write_rows(response_path,ordered)
                atomic(out/'green_metrics.json',dict(mean=sum(r['score'] for r in ordered)/len(jobs) if len(ordered)==len(jobs) else None,
                    strict_mean=sum(r.get('strict_score',0.) for r in ordered)/len(jobs) if len(ordered)==len(jobs) else None,
                    n=len(jobs),completed=len(ordered),invalid_outputs=sum(not r['valid'] for r in ordered),
                    status='complete' if len(ordered)==len(jobs) else 'running',checkpoint_revision=GREEN_REV))
                print(name,'GREEN',len(ordered),'/',len(jobs),flush=True)
            pending.remove(name);progress=True
        if not progress and pending:time.sleep(20)
    atomic(run/'green_status.json',dict(status='complete',finished=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--gpu',type=int,default=6)
    p.add_argument('--prepare',action='store_true');p.add_argument('--model');main(p.parse_args())
