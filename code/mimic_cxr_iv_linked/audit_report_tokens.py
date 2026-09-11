"""Audit the OLD pilot's token budget; never truncate the linked text itself."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

from common import ROOT,dump_json,dump_jsonl,read_jsonl


def main(args):
    os.umask(0o077)
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    pairs=list(read_jsonl(args.out/'pairs.jsonl'))
    wanted={(r['subject_id'],r[k]) for r in pairs for k in ['source_study','target_study']}
    records=[r for r in read_jsonl(args.out/'studies.jsonl') if (r['subject_id'],r['study_id']) in wanted]
    results=[];start=time.monotonic()
    for i in range(0,len(records),256):
        batch=records[i:i+256]
        tokens=tokenizer([r['report']['text'] for r in batch],add_special_tokens=False,truncation=False)['input_ids']
        for r,t in zip(batch,tokens):
            text=r['report']['text'];offset=text.find('IMPRESSION:')
            prefix_count=len(tokenizer(text[:offset],add_special_tokens=False)['input_ids']) if offset>=0 else None
            results.append(dict(subject_id=r['subject_id'],study_id=r['study_id'],split=r['split'],
                token_count=len(t),would_exceed_256_tokens=len(t)>256,
                impression_starts_after_256_tokens=prefix_count is not None and prefix_count>=256))
    dump_jsonl(args.out/'report_token_audit.jsonl',results)
    lookup={(r['subject_id'],r['study_id']):r for r in results}
    stats={}
    for split in ['all','train','validate','test']:
        selected=[r for r in pairs if split=='all' or r['split']==split]
        stats[split]=dict(pairs=len(selected),**{side+'_reports_over_256_tokens':sum(lookup[(r['subject_id'],r[side+'_study'])]['would_exceed_256_tokens'] for r in selected) for side in ['source','target']})
    dump_json(args.out/'report_token_summary.json',dict(tokenizer=str(args.tokenizer.resolve()),
        purpose='Audit old pilot token cap on new full cleaned reports; text in this dataset remains uncapped.',
        observations=len(results),over_256_tokens=sum(r['would_exceed_256_tokens'] for r in results),
        impression_starts_after_256_tokens=sum(r['impression_starts_after_256_tokens'] for r in results),splits=stats,elapsed_seconds=time.monotonic()-start))
    print(f'[report audit] {len(results):,} observations; {sum(r["would_exceed_256_tokens"] for r in results):,} exceed old 256-token limit',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,default=ROOT.parent/'medworld_table1/weights/Qwen3.5-0.8B')
    main(p.parse_args())
