"""Count clinical records available by each CXR cutoff, separately from linkage."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('POLARS_MAX_THREADS','8')
import polars as pl

from common import dump_json, dump_jsonl, read_jsonl

# ICD codes/date-only records and tables without a recording timestamp are audit
# context only. Never promote them to current evidence using a guessed timestamp.
EVENT_TABLES = {
    'labs': ('hosp/labevents', 'charttime'),
    'medication_administration': ('hosp/emar', 'charttime'),
    'icu_chart': ('icu/chartevents', 'charttime'),
    'icu_input': ('icu/inputevents', 'starttime'),
    'icu_procedure': ('icu/procedureevents', 'starttime'),
    'icu_output': ('icu/outputevents', 'charttime'),
}


def dt(column):
    return pl.col(column).str.to_datetime('%Y-%m-%d %H:%M:%S%.f',strict=False)


def event_times(path, event_column):
    return pl.scan_parquet(path).select('subject_id','hadm_id',
        dt(event_column).alias('event_time'),dt('storetime').alias('recorded_time')).with_columns(
        pl.when(pl.col('event_time').is_not_null() & pl.col('recorded_time').is_not_null())
        .then(pl.max_horizontal('event_time','recorded_time')).otherwise(None).alias('available_time'))


def historical_counts(observations, events):
    """Both frames use subject_id/hadm_id; events have available_time datetimes."""
    counts=(events.filter(pl.col('available_time').is_not_null())
        .group_by('subject_id','hadm_id','available_time').agg(pl.len().alias('events_at_time'))
        .sort('available_time').with_columns(
            pl.col('events_at_time').cum_sum().over('subject_id','hadm_id').alias('history_count'))
        .select('subject_id','hadm_id','available_time','history_count'))
    return observations.sort('cutoff').join_asof(counts,
        left_on='cutoff',right_on='available_time',by=['subject_id','hadm_id'],strategy='backward',check_sortedness=False)


def main(args):
    os.umask(0o077)
    pairs=list(read_jsonl(args.out/'pairs.jsonl'))
    wanted={r[k] for r in pairs for k in ['source_image','target_image']}
    observations=[]
    for r in read_jsonl(args.out/'images.jsonl'):
        if r['dicom_id'] in wanted and r['patient_in_iv'] and len(r['hadm_ids'])==1:
            observations.append(dict(dicom_id=r['dicom_id'],subject_id=r['subject_id'],hadm_id=r['hadm_ids'][0],cutoff=r['timestamp']))
    frame=pl.DataFrame(observations).with_columns(dt('cutoff'))
    assert frame['cutoff'].null_count()==0,'Invalid CXR cutoff timestamp'
    hadms=frame['hadm_id'].unique().to_list()
    results={r['dicom_id']:dict(dicom_id=r['dicom_id'],subject_id=r['subject_id'],hadm_id=r['hadm_id'],
        cutoff=r['cutoff'],history_counts={},latest_record_availability={}) for r in observations}
    audits={}
    for name,(table,event_column) in EVENT_TABLES.items():
        path=args.out/'iv'/(table+'.parquet')
        if not path.exists() or not path.with_suffix('.json').exists():
            raise FileNotFoundError(f'Incomplete IV table: {path}')
        events=event_times(path,event_column).filter(pl.col('hadm_id').is_in(hadms)).collect(engine='streaming')
        valid=events['available_time'].is_not_null()
        audits[name]=dict(scoped_records=len(events),with_both_timestamps=int(valid.sum()),
            without_usable_timestamps=int((~valid).sum()),table=table,
            selected_endpoint_admissions=len(hadms),
            null_hadm_rows='Preserved in linked Parquet; excluded from admission-specific availability counts.')
        joined=historical_counts(frame,events)
        for r in joined.iter_rows(named=True):
            value=results[r['dicom_id']]
            value['history_counts'][name]=r['history_count'] or 0
            value['latest_record_availability'][name]=str(r['available_time']) if r['available_time'] is not None else None
            if r['available_time'] is not None:
                assert r['available_time']<=r['cutoff']
        print(f'[availability] {name}: {len(events):,} linked-admission records; {(joined["history_count"].fill_null(0)>0).sum():,} endpoints with prior records',flush=True)
        del events,joined
    for r in results.values():
        r['has_recorded_clinical_history']=any(r['history_counts'].values())
        r['has_labs_and_icu_chart']=r['history_counts']['labs']>0 and r['history_counts']['icu_chart']>0
        r['report_available_time']=None
        r['availability_definition']='Same subject and native hadm_id; max(event occurrence,storetime)<=selected CXR timestamp; counts over available admission history.'
        r['interpretation']='Recorded-event availability proxy; not proof of completeness, immutable record versions, or clinical correctness.'
    dump_jsonl(args.out/'clinical_availability.jsonl',results.values())
    dump_json(args.out/'availability_summary.json',dict(endpoints=len(results),tables=audits,
        with_any_prior_record=sum(r['has_recorded_clinical_history'] for r in results.values()),
        report_cutoff_verified=False, excludes=['missing native hadm_id','missing event or storetime','ICD discharge codes','date-only events','future-recorded rows']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
