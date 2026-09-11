"""Stream every local IV table to Parquet for the full CXR patient universe.

Native identifiers and fields are preserved. This is a retrospective linked
database, NOT a model input. Availability filtering belongs to a separate view.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import gzip
import json
import os
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from common import DEFAULT_IV, VERSION, dump_json, file_sha256, fingerprint


def export_table(source, target, subjects, cohort_sha):
    started = time.monotonic()
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    sidecar = target.with_suffix('.json')
    signature = dict(version=VERSION, source=fingerprint(source), cxr_subjects_sha256=cohort_sha)
    if sidecar.exists() and target.exists():
        old = json.loads(sidecar.read_text())
        if old['signature'] != signature:
            raise ValueError(f'Changed source/cohort for {source}; use a new output directory.')
        print(f'[IV reuse] {source.parent.name}/{source.name}: {old["retained_rows"]:,}', flush=True)
        return old
    with gzip.open(source, 'rt', newline='') as f:
        fields = next(csv.reader(f))
    options = pacsv.ConvertOptions(column_types={k:pa.string() for k in fields}, strings_can_be_null=False)
    reader = pacsv.open_csv(str(source), read_options=pacsv.ReadOptions(block_size=32 << 20),
        parse_options=pacsv.ParseOptions(newlines_in_values=True), convert_options=options)
    temp = target.with_suffix('.partial.parquet')
    wanted = pa.array(subjects, type=pa.string())
    scanned = kept = 0
    matched_subjects = set()
    last_log = started
    writer = None
    try:
        for batch in reader:
            table = pa.Table.from_batches([batch])
            # Record ordinal counts parsed CSV records, not physical lines.
            table = table.append_column('_source_record', pa.array(range(scanned+1, scanned+len(table)+1), type=pa.int64()))
            scanned += len(table)
            if 'subject_id' in fields:
                table = table.filter(pc.is_in(table['subject_id'], value_set=wanted))
                matched_subjects.update(pc.unique(table['subject_id']).to_pylist())
            if writer is None:
                metadata = {b'source_table':str(source).encode(), b'role':b'retrospective_linked_records; not forecasting inputs'}
                writer = pq.ParquetWriter(temp, table.schema.with_metadata(metadata), compression='zstd', compression_level=3)
            if len(table):
                writer.write_table(table)
            kept += len(table)
            if time.monotonic()-last_log > 25:
                print(f'[IV] {source.parent.name}/{source.name}: scanned {scanned:,}, retained {kept:,}', flush=True)
                last_log = time.monotonic()
        if writer is None:
            schema = reader.schema.append(pa.field('_source_record',pa.int64()))
            writer = pq.ParquetWriter(temp,schema,compression='zstd')
    finally:
        if writer is not None:
            writer.close()
    temp.replace(target)
    result = dict(signature=signature, table=source.parent.name+'/'+source.name.removesuffix('.csv.gz'),
        output=str(target.resolve()), scanned_rows=scanned, retained_rows=kept,
        linked_patients=len(matched_subjects), filter='exact subject_id in full CXR universe' if 'subject_id' in fields else 'full dictionary/provider table',
        columns=fields, native_identifiers_preserved=True, source_record='1-based parsed data-record ordinal; excludes header',
        elapsed_seconds=time.monotonic()-started)
    dump_json(sidecar,result)
    print(f'[IV complete] {result["table"]}: {kept:,}/{scanned:,} rows ({result["elapsed_seconds"]:.0f}s)',flush=True)
    return result


def main(args):
    os.umask(0o077)
    subjects = json.loads((args.out/'cxr_subjects.json').read_text())
    digest = file_sha256(args.out/'cxr_subjects.json')
    files = sorted(args.iv.glob('*/*.csv.gz'),key=lambda x:x.stat().st_size,reverse=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = {executor.submit(export_table,p,args.out/'iv'/p.parent.name/(p.name.removesuffix('.csv.gz')+'.parquet'),subjects,digest):p for p in files}
        for job in as_completed(jobs):
            results.append(job.result())
            dump_json(args.out/'iv_progress.json',dict(completed_tables=len(results),total_tables=len(files),tables=results))
    dump_json(args.out/'iv_summary.json',dict(version=VERSION,scope='ALL local hosp/icu CSV tables; CXR patients across their available IV history',
        tables=sorted(results,key=lambda x:x['table']),total_tables=len(results),retained_rows=sum(x['retained_rows'] for x in results),
        rows_without_hadm_id='Retained at native patient/event keys; never guessed into an admission.',
        dictionaries='Complete public codebooks/provider key tables retained to interpret joins.'))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--iv',type=Path,default=DEFAULT_IV)
    p.add_argument('--workers',type=int,default=3)
    main(p.parse_args())
