"""Independent CXR observations, deterministic sampling and explicit task masks."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import bootstrap
from bootstrap import *
import numpy as np
from PIL import Image


def rank(key):
    return hashlib.sha256(('stage1-20260910-' + key).encode()).hexdigest()


def pixels(row, size, scale):
    try:
        with Image.open(row['image']) as im:
            im = im.convert('L')
            w0, h0 = im.size
            ratio = min(1., size / max(w0, h0))
            w = max(scale, int(w0 * ratio) // scale * scale)
            h = max(scale, int(h0 * ratio) // scale * scale)
            resized = np.asarray(im.resize((w, h), Image.Resampling.BICUBIC))
        if min(w0, h0) < 64 or resized.std() < 2:
            raise ValueError('too small or nearly constant')
        # Align top/left padding with the downsampling grid.
        x = ((size-w)//2)//scale*scale
        y = ((size-h)//2)//scale*scale
        canvas = np.zeros((size, size), np.uint8)
        canvas[y:y+h, x:x+w] = resized
        return canvas, [y, x, h, w], [h0, w0], None
    except Exception as e:
        return None, None, None, str(e)


def prepare(cache, size=512, scale=2, train_frontal=20000, train_other=4000, eval_frontal=256, eval_other=64):
    os.umask(0o077)
    cache.mkdir(parents=True, exist_ok=True)
    if (cache/'manifest.json').exists():
        print('manifest already complete', flush=True)
        return
    source = PROJECT/'code/mimic_cxr_iv_linked/runs/full_20260909'
    studies = {}
    for r in load_rows(source/'studies.jsonl'):
        studies[r['study_id']] = dict(report=r['report']['text'], report_valid=r['report']['valid'],
            report_path=r['report']['path'], report_sha256=r['report']['raw_sha256'],
            labels=[(r['labels'] or {}).get(k, -2) for k in FINDINGS], report_flags=r['report']['flags'])
    candidates = collections.defaultdict(list)
    counts = collections.Counter()
    patient_splits = {}
    with (cache/'candidates.jsonl').open('w') as out:
        for im in load_rows(source/'images.jsonl'):
            split, pid = im['split'], im['subject_id']
            if split not in ('train', 'validate', 'test'):
                counts['no_official_split'] += 1
                continue
            if pid in patient_splits:
                assert patient_splits[pid] == split
            patient_splits[pid] = split
            st = studies[im['study_id']]
            frontal = im['view'] in ('AP','PA')
            flags = dict(classification=frontal and st['report_valid'] and any(x in (0,1) for x in st['labels']),
                         diagnosis=frontal and st['report_valid'] and len(st['report'].split()) >= 5,
                         segmentation=frontal, sr=True)
            counts['all_images'] += 1
            if not im['exists'] or min(int(im['rows'] or 0), int(im['columns'] or 0)) < 64:
                counts['missing_or_small_metadata'] += 1
                continue
            rec = dict(id=im['dicom_id'], subject_id=pid, study_id=im['study_id'], split=split,
                       image=im['path'], view=im['view'], tasks=flags)
            out.write(json.dumps(rec)+'\n')
            candidates[split, frontal].append(rec)
            counts[f'candidate_{split}_{"frontal" if frontal else "other"}'] += 1
    selected = []
    for split in ('train','validate','test'):
        patients, used_studies = collections.Counter(), set()
        for frontal, limit in [(True, train_frontal if split=='train' else eval_frontal),
                               (False, train_other if split=='train' else eval_other)]:
            added = 0
            for r in sorted(candidates[split,frontal], key=lambda x: rank(x['id'])):
                if r['study_id'] in used_studies or patients[r['subject_id']] >= (4 if split=='train' else 1):
                    continue
                r.update(studies[r['study_id']])
                selected.append(r)
                patients[r['subject_id']] += 1
                used_studies.add(r['study_id'])
                added += 1
                if added >= limit:
                    break
    print('selected before decode', len(selected), dict(counts), flush=True)
    images = np.lib.format.open_memmap(cache/'images.tmp.npy', mode='w+', dtype=np.uint8,
                                      shape=(len(selected),size,size))
    rows, rejected = [], []
    with ThreadPoolExecutor(max_workers=12) as pool:
        for r, (canvas, box, native, error) in zip(selected, pool.map(lambda x: pixels(x,size,scale), selected)):
            if error:
                rejected.append(dict(id=r['id'],error=error))
                continue
            i = len(rows)
            images[i] = canvas
            r.update(index=i, box=box, native_size=native, hr_size=box[2:], lr_size=[x//scale for x in box[2:]])
            rows.append(r)
            if i % 1000 == 0:
                print('decoded',i,flush=True)
    images.flush()
    del images
    # Extra unused tail, if any, is explicitly excluded by observations/valid_count.
    os.replace(cache/'images.tmp.npy',cache/'images.npy')
    write_rows(cache/'observations.jsonl',rows)
    write_rows(cache/'rejected.jsonl',rejected)
    coverage = {}
    for split in ('train','validate','test'):
        rr = [r for r in rows if r['split']==split]
        labels = np.asarray([r['labels'] for r in rr])
        coverage[split] = dict(images=len(rr),patients=len({r['subject_id'] for r in rr}),
            views=dict(collections.Counter(r['view'] for r in rr)),
            tasks={t:sum(r['tasks'][t] for r in rr) for t in ('classification','diagnosis','segmentation','sr')},
            labels={f:{str(v):int((labels[:,i]==v).sum()) for v in [-2,-1,0,1]} for i,f in enumerate(FINDINGS)})
    sets = [{r['subject_id'] for r in rows if r['split']==s} for s in ('train','validate','test')]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    atomic_json(cache/'manifest.json',dict(seed=20260910,scale=scale,hr_canvas=size,valid_count=len(rows),
        counts=dict(counts),coverage=coverage,rejected=len(rejected),findings=FINDINGS,organs=ORGANS,
        source_hashes={f:digest(source/f) for f in ['images.jsonl','studies.jsonl']},
        observations_sha256=digest(cache/'observations.jsonl'),
        patient_disjoint=True,iv_filter=False,qwen_label_filter=False))
    print('prepared',coverage,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,default=ROOT/'data/overnight_20260910')
    p.add_argument('--scale',type=int,choices=[2,4],default=2)
    p.add_argument('--train-frontal',type=int,default=20000)
    p.add_argument('--train-other',type=int,default=4000)
    p.add_argument('--eval-frontal',type=int,default=256)
    p.add_argument('--eval-other',type=int,default=64)
    a=p.parse_args()
    prepare(a.cache,scale=a.scale,train_frontal=a.train_frontal,train_other=a.train_other,
            eval_frontal=a.eval_frontal,eval_other=a.eval_other)
