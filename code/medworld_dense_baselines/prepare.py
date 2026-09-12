"""Freeze per-metric cohorts before any model sees targets."""
import argparse
import ast
import collections
import csv
import json
import numpy as np
from PIL import Image
from common import *

def canvas(image, box=None):
    image = image.convert('L')
    w, h = image.size
    if box is None:
        ratio = min(512 / w, 512 / h)
        ww, hh = max(4, int(w * ratio) // 4 * 4), max(4, int(h * ratio) // 4 * 4)
        box = [((512-hh)//2)//4*4, ((512-ww)//2)//4*4, hh, ww]
    y, x, hh, ww = box
    arr = np.zeros((512, 512), np.uint8)
    arr[y:y+hh, x:x+ww] = np.asarray(image.resize((ww, hh), Image.Resampling.BICUBIC))
    return arr, box

def prepare(run, train_count=4096):
    os.umask(0o077)
    data = run / 'data'
    if (data / 'manifest.json').exists():
        print('cohort already frozen', flush=True)
        return
    selection = json.loads((PROJECT/'code/medworld_stage1/data/slot44_20260911_derived_v2/selection.json').read_text())
    overrides = selection['patient_split_overrides']
    old_rows = read_rows(OLD/'observations.jsonl')
    valid = np.load(OLD/'seg_valid_qc.npy')
    chosen = []
    # Preserve the existing patient holdout, including its gold-patient overrides.
    for split, limit in [('train', train_count), ('validate', 256), ('test', 100000)]:
        candidates = [r for r in old_rows if overrides.get(r['subject_id'], r['split']) == split
                      and r['view'] in ('AP', 'PA') and valid[r['index']]]
        candidates.sort(key=lambda r: rank(r['id']))
        for r in candidates[:limit]:
            chosen.append(dict(id=r['id'], subject_id=r['subject_id'], image=r['image'],
                old_index=r['index'], box=r['box'], split=split, kind='mimic',
                tasks=['segmentation', 'sr']))
    # Anatomical localization is a separate, explicitly named task, not MS-CXR lesion grounding.
    boxfile = GOLD/'gold_bbox_coordinate_annotations_1000images.csv'
    bbox_rows = list(csv.DictReader(boxfile.open()))
    needed = {r['image_id'].removesuffix('.dcm') for r in bbox_rows}
    image_meta = {}
    with (LINKED/'images.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            if r['dicom_id'] in needed and r['exists']:
                image_meta[r['dicom_id']] = r
    # These heads are trained independently. No localization training can change the
    # frozen VLM or the separate segmentation/SR heads.
    patients = sorted({r['subject_id'] for r in image_meta.values()}, key=rank)
    n = len(patients)
    psplit = {p: ('train' if i < int(.8*n) else 'validate' if i < int(.9*n) else 'test')
              for i, p in enumerate(patients)}
    index = {r['id']: i for i, r in enumerate(chosen)}
    for iid, m in sorted(image_meta.items()):
        if iid not in index:
            index[iid] = len(chosen)
            chosen.append(dict(id=iid, subject_id=m['subject_id'], image=m['path'],
                split='grounding_only', kind='mimic', tasks=[]))
    queries = []
    seen = set()
    for r in bbox_rows:
        iid = r['image_id'].removesuffix('.dcm')
        if iid not in image_meta:
            continue
        key = (iid, r['bbox_name'])
        if key in seen:
            raise ValueError(f'duplicate image-region annotation: {key}')
        seen.add(key)
        m = image_meta[iid]
        xy = ast.literal_eval(r['coord_original'])
        w, h = int(m['columns']), int(m['rows'])
        xy = np.clip(np.array(xy) / [w,h,w,h], 0, 1).tolist()
        if xy[2] <= xy[0] or xy[3] <= xy[1]:
            continue
        queries.append(dict(index=index[iid], id=iid, subject_id=m['subject_id'],
            split=psplit[m['subject_id']], phrase=r['bbox_name'], xyxy_native_normalized=xy))
    human = data/'montgomery'
    assert (human/'manifest.json').exists(), 'human download must complete before cohort freeze'
    for path in sorted((human/'CXR_png').glob('*.png')):
        iid = 'montgomery:' + path.stem
        index[iid] = len(chosen)
        chosen.append(dict(id=iid, subject_id=iid, image=str(path), split='human_test',
            kind='montgomery', tasks=['segmentation'],
            # NIH folder names refer to image sides: image-left is the patient's right lung.
            masks=[str(human/'ManualMask'/side/path.name) for side in ['leftMask','rightMask']]))
    images = np.lib.format.open_memmap(data/'images.npy', mode='w+', dtype=np.uint8,
                                      shape=(len(chosen),512,512))
    old_images = np.load(OLD/'images.npy', mmap_mode='r')
    human_masks = np.lib.format.open_memmap(data/'human_masks.npy', mode='w+', dtype=np.uint8,
                                          shape=(138,2,256,256))
    hi = 0
    for i, r in enumerate(chosen):
        r['index'] = i
        if 'old_index' in r:
            images[i] = old_images[r['old_index']]
        else:
            with Image.open(r['image']) as im:
                images[i], r['box'] = canvas(im)
        if r['kind'] == 'montgomery':
            y,x,h,w = [v//2 for v in r['box']]
            masks = np.zeros((2,256,256),np.uint8)
            for j, path in enumerate(r['masks']):
                with Image.open(path) as im:
                    masks[j,y:y+h,x:x+w] = (np.asarray(im.convert('L').resize((w,h),Image.Resampling.NEAREST)) > 0)
            human_masks[hi] = masks
            r['human_index'] = hi
            hi += 1
    images.flush()
    human_masks.flush()
    cache_lr(data)
    for q in queries:
        y,x,h,w = chosen[q['index']]['box']
        a,b,c,d = q.pop('xyxy_native_normalized')
        q['xyxy'] = [(x+a*w)/512,(y+b*h)/512,(x+c*w)/512,(y+d*h)/512]
    write_rows(data/'observations.jsonl', chosen)
    write_rows(data/'grounding.jsonl', queries)
    vocab = sorted({q['phrase'] for q in queries})
    atomic(data/'query_vocab.json', vocab)
    for task_rows in [[r for r in chosen if 'sr' in r['tasks']], queries]:
        sets = [{r['subject_id'] for r in task_rows if r['split']==s} for s in ['train','validate','test']]
        assert all(not sets[a]&sets[b] for a,b in [(0,1),(0,2),(1,2)])
    coverage = dict(images=dict(collections.Counter(r['split'] for r in chosen)),
        grounding_queries=dict(collections.Counter(r['split'] for r in queries)),
        grounding_patients=dict(collections.Counter(psplit.values())),
        segmentation_pseudo_organs=['right lung','left lung','heart'],
        segmentation_human_organs=['right lung','left lung'])
    atomic(data/'manifest.json', dict(coverage=coverage, images=len(chosen), seed=SEED,
        patient_disjoint_within_each_task=True, tasks_trained_independently=True,
        sources={str(f):digest(f) for f in [OLD/'observations.jsonl',OLD/'seg_valid_qc.npy',boxfile]},
        cohort_sha256=digest(data/'observations.jsonl'), grounding_sha256=digest(data/'grounding.jsonl'),
        image_sha256=digest(data/'images.npy'), human_mask_sha256=digest(data/'human_masks.npy'),
        lr_image_sha256=digest(data/'lr_images.npy'),
        localization_scope='Chest ImaGenome human anatomical region boxes; NOT MS-CXR lesion boxes',
        sr='4x antialiased bicubic reduction from 512 canvas. LR only for all input encoders.',
        pseudo_scope='CXAS teacher agreement; QC cohort fixed independently of predictions',
        human_scope='Montgomery external test, both lungs only, no heart annotation',
        human_mask_mapping={'right lung':'leftMask (image-left)','left lung':'rightMask (image-right)'}))
    print(json.dumps(coverage, indent=2), flush=True)

def cache_lr(data):
    import torch
    from features import downsample
    torch.set_num_threads(4)
    images=np.load(data/'images.npy',mmap_mode='r')
    lr=np.lib.format.open_memmap(data/'lr_images.npy',mode='w+',dtype=np.uint8,shape=(len(images),128,128))
    for start in range(0,len(images),32):
        x=torch.from_numpy(np.array(images[start:start+32],copy=True))[:,None].float()/255
        lr[start:start+32]=(downsample(x)[:,0]*255).round().byte().numpy()
    lr.flush()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run',type=Path,default=DEFAULT_RUN)
    p.add_argument('--train-count',type=int,default=4096)
    a = p.parse_args()
    prepare(a.run,a.train_count)
