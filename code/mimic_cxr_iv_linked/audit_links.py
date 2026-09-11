"""Verify native IV patient/admission/stay foreign keys after full extraction."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('POLARS_MAX_THREADS','8')
import polars as pl

from common import dump_json,read_jsonl


def main(args):
    out=args.out
    summaries=json.loads((out/'iv_summary.json').read_text())['tables']
    subjects=[r['subject_id'] for r in read_jsonl(out/'patients.jsonl')]
    admissions=pl.DataFrame([{'subject_id':r['subject_id'],'hadm_id':r['hadm_id']} for r in read_jsonl(out/'admissions.jsonl')]).with_columns(pl.lit(True).alias('_exists'))
    stays=pl.DataFrame([{'subject_id':r['subject_id'],'hadm_id':r['hadm_id'],'stay_id':r['stay_id']} for r in read_jsonl(out/'icustays.jsonl')]).with_columns(pl.lit(True).alias('_stay_exists'))
    results={}
    for table in summaries:
        columns=table['columns'];name=table['table']
        if 'subject_id' not in columns:
            continue
        keys=[k for k in ['subject_id','hadm_id','stay_id'] if k in columns]
        grouped=(pl.scan_parquet(out/'iv'/(name+'.parquet')).group_by(keys).agg(pl.len().alias('records')).collect(engine='streaming'))
        audit=dict(retained_rows=table['retained_rows'],native_key_groups=len(grouped),
            rows_with_patient_missing_from_iv_patients=int(grouped.filter(~pl.col('subject_id').is_in(subjects))['records'].sum()))
        assert grouped['records'].sum()==table['retained_rows']
        if 'hadm_id' in keys:
            nonempty=grouped.filter(pl.col('hadm_id')!='')
            joined=nonempty.join(admissions,on=['subject_id','hadm_id'],how='left')
            audit['rows_without_native_hadm_id']=int(grouped.filter(pl.col('hadm_id')=='')['records'].sum())
            audit['rows_with_hadm_patient_key_not_found']=int(joined.filter(pl.col('_exists').is_null())['records'].sum())
        if 'stay_id' in keys:
            nonempty=grouped.filter(pl.col('stay_id')!='')
            joined=nonempty.join(stays,on=['subject_id','hadm_id','stay_id'],how='left')
            audit['rows_with_stay_patient_admission_key_not_found']=int(joined.filter(pl.col('_stay_exists').is_null())['records'].sum())
        results[name]=audit
        print(f'[key audit] {name}: {audit}',flush=True)
    dump_json(out/'native_key_audit.json',dict(tables=results,
        policy='Counts native patient/admission/ICU key consistency; empty hadm remains unassigned and raw records retained.',
        scope='Patient/hadm/stay foreign keys; emar/poe detail event-key completeness is not separately adjudicated.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
