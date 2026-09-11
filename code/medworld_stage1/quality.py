"""Reject grossly fragmented teacher masks; preserve original probabilities unchanged."""
import collections
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import bootstrap
from bootstrap import *
import numpy as np
from scipy.ndimage import label


def quality(cache):
    if (cache/'segmentation_qc.json').exists():return
    rows=load_rows(cache/'observations.jsonl')
    probs=np.load(cache/'seg_probs.npy',mmap_mode='r')
    valid=np.load(cache/'seg_valid.npy')
    def inspect(i):
        fractions=[]
        for mask in probs[i]>.5:
            components,n=label(mask,structure=np.ones((3,3)))
            counts=np.bincount(components.ravel())[1:]
            fractions.append(float(counts.max()/max(counts.sum(),1)) if len(counts) else 0.)
        return i,fractions
    flags=[];counts=collections.Counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,fractions in pool.map(inspect,np.flatnonzero(valid).tolist()):
            if min(fractions)<.90:
                valid[i]=False
                flags.append(dict(index=i,id=rows[i]['id'],view=rows[i]['view'],largest_component_fractions=fractions))
                counts[rows[i]['view']]+=1
    np.save(cache/'seg_valid_qc.npy',valid)
    write_rows(cache/'seg_fragmentation_rejected.jsonl',flags)
    atomic_json(cache/'segmentation_qc.json',dict(valid=int(valid.sum()),fragmentation_rejected=len(flags),
        rejection_by_view=dict(counts),rule='all three organs: largest 8-connected component >= 90% of binary foreground',
        valid_by_split={s:int(sum(valid[i] for i,r in enumerate(rows) if r['split']==s)) for s in ['train','validate','test']},
        teacher_probabilities_modified=False,limitations='geometry heuristic only; severe pathology may be disproportionately excluded',
        source_segmentation_sha256=digest(cache/'segmentation.json'),valid_mask_sha256=digest(cache/'seg_valid_qc.npy')))
    print('segmentation QC',int(valid.sum()),'accepted;',len(flags),'fragmented masks rejected',flush=True)


if __name__=='__main__':quality(ROOT/'data/overnight_20260910')
