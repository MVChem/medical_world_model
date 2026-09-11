"""Build a local, auditable VLM manifest; preserve the original strict subset."""
import argparse
from collections import Counter,defaultdict
import hashlib
from itertools import pairwise
from pathlib import Path
from common import read_jsonl,dump_json,dump_jsonl,timestamp,file_sha256,link_status


def run(root,out):
    out.mkdir(parents=True,exist_ok=True)
    original={r['pair_id']:r for r in read_jsonl(root/'linked_pairs.jsonl')}
    ims={r['dicom_id']:r for r in read_jsonl(root/'images.jsonl') if r['view'] in ('AP','PA')}
    timelines=defaultdict(list)
    for r in read_jsonl(root/'studies.jsonl'):
        timelines[r['subject_id']].append({k:r[k] for k in ('subject_id','study_id','timestamp','latest_timestamp','selected_frontal','split','report')})
    counts=Counter()
    def records():
        for subject,rows in sorted(timelines.items()):
            rows.sort(key=lambda r:(r['timestamp'],r['study_id']))
            ties=Counter(r['timestamp'] for r in rows)
            for a,b in pairwise(rows):
                if (ties[a['timestamp']]>1 or ties[b['timestamp']]>1 or a['latest_timestamp']>=b['timestamp']
                    or not a['split'] or a['split']!=b['split']):
                    counts['order_or_split_excluded']+=1
                    continue
                va,vb=a['selected_frontal'],b['selected_frontal']
                if not va or not vb:
                    counts['no_frontal_at_one_or_both_ends']+=1
                    continue
                if not a['report']['valid'] or not b['report']['valid']:
                    counts['report_sections_unparsed_review_needed']+=1
                    continue
                common=next((v for v in ('PA','AP') if v in va and v in vb),None)
                ia=ims[va[common or next(v for v in ('PA','AP') if v in va)]]
                ib=ims[vb[common or next(v for v in ('PA','AP') if v in vb)]]
                gap=(timestamp(ib['timestamp'])-timestamp(ia['timestamp'])).total_seconds()/3600
                if gap<=0 or not (ia['exists'] and ib['exists']):
                    counts['nonpositive_gap_or_missing_image']+=1
                    continue
                key=':'.join([subject,a['study_id'],b['study_id'],ia['dicom_id'],ib['dicom_id']])
                pid=hashlib.sha256(key.encode()).hexdigest()[:24]
                old=original.get(pid)
                counts['expanded_pairs']+=1
                counts['original_pairs']+=bool(old)
                status=link_status(ia['hadm_ids'],ib['hadm_ids'])
                hadm=ia['hadm_ids'][0] if status=='same_admission' else None
                yield dict(pair_id=pid,subject_id=subject,split=a['split'],
                    source_study=a['study_id'],target_study=b['study_id'],
                    source_image=ia['dicom_id'],target_image=ib['dicom_id'],
                    source_path=ia['path'],target_path=ib['path'],
                    source_time=ia['timestamp'],target_time=ib['timestamp'],
                    source_view=ia['view'],target_view=ib['view'],same_view=bool(common),
                    realized_gap_hours=gap,horizon_bin=('0-6h' if gap<6 else '6-24h' if gap<=24 else '>24-72h' if gap<=72 else '>72-168h' if gap<=168 else '>168-720h' if gap<=720 else '>30d'),
                    source_report=a['report']['text'],target_report=b['report']['text'],
                    source_report_sha256=a['report']['clean_sha256'],target_report_sha256=b['report']['clean_sha256'],
                    original_strict_subset=bool(old),tiers=old['tiers'] if old else {},
                    link_status=status,hadm_id=hadm,
                    source_hadm_ids=ia['hadm_ids'],target_hadm_ids=ib['hadm_ids'],
                    source_stay_ids=ia['stay_ids'],target_stay_ids=ib['stay_ids'],
                    report_text_definition='Untruncated original Findings/Impression sections; comparative language may reference an examination other than the supplied earlier image.')
    dump_jsonl(out/'manifest_expanded.jsonl',records())
    assert counts['original_pairs']==len(original),(counts,len(original))
    dump_json(out/'manifest_provenance.json',dict(counts=dict(counts),
        source_hashes={p.name:file_sha256(p) for p in [root/'studies.jsonl',root/'images.jsonl',root/'linked_pairs.jsonl']},
        manifest_sha256=file_sha256(out/'manifest_expanded.jsonl'),
        rules='Strictly adjacent examinations in the full patient timeline; unique nonoverlapping order and official split; both have AP/PA and parsed report sections; all positive gaps; prefer a common PA then AP, otherwise choose PA then AP at each endpoint. Not all combinations. Original strict subset retained exactly.'))
    print(dict(counts),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    run(a.run,a.out)
