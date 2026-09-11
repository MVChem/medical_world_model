"""Read back final artifact contracts and record reproducibility provenance."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import platform
import shutil

import pyarrow.parquet as pq

from common import ROOT,VERSION,dump_json,file_sha256,read_jsonl,timestamp


def validate(args):
    os.umask(0o077)
    out=args.out
    summary=json.loads((out/'summary.json').read_text())
    pairs=list(read_jsonl(out/'linked_pairs.jsonl'))
    ids={r['pair_id'] for r in pairs}
    assert len(ids)==len(pairs),'Duplicate pair IDs'
    splits=defaultdict(set)
    for r in pairs:
        splits[r['split']].add(r['subject_id'])
        delta=(timestamp(r['target_time'])-timestamp(r['source_time'])).total_seconds()/3600
        assert abs(delta-r['realized_gap_hours'])<1e-8
        assert r['horizon_bin'] in ['6-24h','>24-72h','>72-168h','>168-720h']
        if r['tiers']['same_admission_6h_72h_image_qc_prior_ehr']:
            assert r['link_status']=='same_admission'
            assert r['image_qc']['passed_automatic_image_qc']
            current=r['current_clinical_availability']
            assert current['has_recorded_clinical_history']
            assert all(timestamp(x)<=timestamp(r['source_time']) for x in current['latest_record_availability'].values() if x)
    values=list(splits.values())
    assert all(not a&b for i,a in enumerate(values) for b in values[i+1:])
    for tier,counts in summary['tiers'].items():
        for split in ['train','validate','test']:
            members=list(read_jsonl(out/'cohorts'/tier/(split+'.jsonl')))
            assert len(members)==counts[split]['pairs']
            assert all(r['tiers'][tier] and r['split']==split for r in members)
    allowed={'pair_id','current_image','current_cutoff','horizon_bin','clinical_history_counts','clinical_query','contract','current_report','report_policy'}
    contracts={}
    for split in ['train','validate','test']:
        input_rows=list(read_jsonl(out/'forecast_views'/(split+'_inputs_image_ehr.jsonl')))
        report_rows=list(read_jsonl(out/'forecast_views'/(split+'_inputs_report_assumed.jsonl')))
        target_rows=list(read_jsonl(out/'forecast_views'/(split+'_targets.jsonl')))
        assert [r['pair_id'] for r in input_rows]==[r['pair_id'] for r in target_rows]==[r['pair_id'] for r in report_rows]
        assert len(input_rows)==summary['tiers'][summary['primary_candidate']][split]['pairs']
        for r in input_rows+report_rows:
            assert set(r)<=allowed
            assert set(r['clinical_query'])=={'subject_id','hadm_id','cutoff','policy'}
            assert Path(r['current_image']).is_file()
        assert all(r['current_report'] is None for r in input_rows)
        assert all(r['current_report'] and 'retrospective' in r['report_policy'] for r in report_rows)
        assert all(Path(r['future_image']).is_file() and r['future_report'] for r in target_rows)
        contracts[split]=len(input_rows)
    clinical=json.loads((out/'review_clinical_context.json').read_text())
    byid={r['pair_id']:r for r in pairs}
    checked_rows=0
    for pair_id,groups in clinical.items():
        cutoff=timestamp(byid[pair_id]['source_time'])
        for table,content in groups['before_current'].items():
            for r in content['recent_rows']:
                event=timestamp(r.get('charttime') or r.get('starttime'))
                assert event<=cutoff and timestamp(r['storetime'])<=cutoff
                assert not {'endtime','amount','totalamount','dischtime'} & set(r)
                checked_rows+=1
    tables=summary['clinical_tables']['tables']
    for r in tables:
        assert pq.ParquetFile(out/'iv'/(r['table']+'.parquet')).metadata.num_rows==r['retained_rows']
    anchors=json.loads((out/'example_regression.json').read_text())
    assert all(r['found_in_6h_30d_pairs'] and r['link_status']=='same_admission' for r in anchors[:4])
    source=out/'source';source.mkdir(exist_ok=True)
    hashes={}
    for path in sorted(ROOT.glob('*.py')):
        shutil.copy2(path,source/path.name)
        hashes[path.name]=file_sha256(path)
    shutil.copy2(ROOT/'README.md',source/'README.md')
    import PIL,polars,pyarrow
    dump_json(out/'provenance.json',dict(version=VERSION,python=platform.python_version(),
        software=dict(pillow=PIL.__version__,polars=polars.__version__,pyarrow=pyarrow.__version__),code_sha256=hashes,
        upstream_metadata_parser_sha256=file_sha256(ROOT.parent/'MIMIC_example/build_mimic_transitions.py'),
        original_inputs='inventory_summary.signature and per-table iv/*.json source path/size/mtime; image SHA256 in image_qc',
        derived_manifests_sha256={name:file_sha256(out/name) for name in ['images.jsonl','studies.jsonl','pairs.jsonl','linked_pairs.jsonl','clinical_availability.jsonl','image_qc.jsonl']},
        clinical_model_used=False))
    dump_json(out/'validation.json',dict(status='passed',pairs=len(pairs),patient_split_disjoint=True,
        forecast_contracts=contracts,parquet_tables_checked=len(tables),review_current_rows_time_checked=checked_rows,
        original_four_examples_recovered=True,current_report_availability_verified=False))
    print(f'[validation] passed; {len(pairs):,} pair IDs, {len(tables)} Parquet tables, input/target views, split and example checks',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    validate(p.parse_args())
