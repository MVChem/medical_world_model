"""Decode and hash every distinct longitudinal endpoint; near matches are flags."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from common import dump_json, fingerprint, read_jsonl


def inspect_image(row):
    result = dict(dicom_id=row['dicom_id'], path=row['path'], valid=False)
    try:
        path = Path(row['path'])
        content = path.read_bytes()
        result.update(file=fingerprint(path), byte_sha256=hashlib.sha256(content).hexdigest())
        with Image.open(io.BytesIO(content)) as image:
            image.load()  # Decode full pixels, detecting truncated/corrupt files.
            result.update(width=image.width,height=image.height,mode=image.mode,
                pixel_sha256=hashlib.sha256(str((image.mode,image.size)).encode()+image.tobytes()).hexdigest())
            thumb = np.asarray(image.convert('L').resize((9,8),Image.Resampling.LANCZOS))
            bits = (thumb[:,1:]>thumb[:,:-1]).ravel()
            result['dhash64'] = f'{sum(int(v)<<i for i,v in enumerate(bits)):016x}'
            result.update(valid=True, thumbnail_std=float(thumb.std()), almost_constant=bool(thumb.std()<1),
                          dimensions_match_metadata=all(not row.get(k) or int(row[k])==v for k,v in [('rows',image.height),('columns',image.width)]))
    except (OSError,ValueError,Image.DecompressionBombError) as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
    return result


def main(args):
    os.umask(0o077)
    pairs=list(read_jsonl(args.out/'pairs.jsonl'))
    wanted={r[k] for r in pairs for k in ['source_image','target_image']}
    rows=[r for r in read_jsonl(args.out/'images.jsonl') if r['dicom_id'] in wanted]
    path=args.out/'image_qc.jsonl'
    cached={r['dicom_id']:r for r in read_jsonl(path)} if path.exists() else {}
    remaining=[]
    for r in rows:
        old=cached.get(r['dicom_id'])
        if old and old.get('file') == fingerprint(r['path']):
            continue
        remaining.append(r)
    start=time.monotonic()
    last=start
    print(f'[image QC] {len(rows):,} distinct pair endpoints; {len(remaining):,} require scanning',flush=True)
    with path.open('a') as output, ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i,result in enumerate(executor.map(inspect_image,remaining,chunksize=8)):
            output.write(json.dumps(result)+'\n')
            cached[result['dicom_id']]=result
            if time.monotonic()-last>25:
                output.flush()
                print(f'[image QC] {i+1:,}/{len(remaining):,} new images; {(i+1)/(time.monotonic()-start):.1f} images/s',flush=True)
                last=time.monotonic()
    scoped=[cached[r['dicom_id']] for r in rows]
    dump_json(args.out/'image_qc_summary.json',dict(scope='Every distinct image selected as a longitudinal pair endpoint; other inventory images retain content-QC=not_scanned',
        images=len(rows),valid=sum(r['valid'] for r in scoped),invalid=sum(not r['valid'] for r in scoped),
        almost_constant=sum(r.get('almost_constant',False) for r in scoped),
        metadata_dimension_mismatch=sum(r.get('dimensions_match_metadata') is False for r in scoped),
        hashes='SHA256 original file + decoded full pixels; 64-bit difference hash for within-pair review flags only',
        near_duplicate_rule='dHash Hamming <=2 flags review; never auto-delete clinically stable follow-ups',
        elapsed_seconds=time.monotonic()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16)
    main(p.parse_args())
