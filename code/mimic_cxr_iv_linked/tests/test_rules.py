import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import available_time, clean_report, containing_ids, link_status, timestamp
from link_iv import export_table
import pyarrow.parquet as pq


def test_combined_report_not_duplicated_and_no_truncation():
    result=clean_report('FINAL REPORT\nFINDINGS AND IMPRESSION: '+('No edema. '*400))
    assert len(result['sections'])==1
    assert result['text'].count('No edema.')==400
    assert result['flags']['combined_heading']
    assert not result['token_truncated']


def test_repeated_different_sections_and_uncertainty_are_preserved():
    result=clean_report('FINDINGS: Possible small effusion.\nIMPRESSION: Cannot exclude effusion.\nIMPRESSION: No pneumothorax.')
    assert len(result['sections'])==3
    assert result['flags']['repeated_heading_different_content']
    assert 'Cannot exclude' in result['text']
    assert not clean_report('Unsectioned example')['valid']


def test_ambiguous_endpoint_not_resolved_by_common_intersection():
    assert link_status(['a','b'],['a'])=='ambiguous_endpoint'
    assert link_status(['a'],['b'])=='different_admissions'
    assert link_status(['a'],['a'])=='same_admission'
    d=timestamp('2100-01-01 12:00:00')
    assert containing_ids(d,[('a',d,d),('b',None,d)])==['a']


def test_delayed_or_unknown_recording_is_not_current_evidence():
    cutoff=timestamp('2100-01-01 12:00:00')
    assert available_time('2100-01-01 11:00:00','2100-01-01 13:00:00')>cutoff
    assert available_time('2100-01-01 11:00:00','') is None
    assert available_time('2100-01-01 13:00:00','2100-01-01 11:00:00')>cutoff


def test_streamed_patient_filter_preserves_multiline_and_native_keys(tmp_path):
    source=tmp_path/'hosp'/'events.csv.gz'
    source.parent.mkdir()
    with gzip.open(source,'wt') as f:
        f.write('subject_id,hadm_id,value\n1,,"line one\nline two"\n2,22,unwanted\n1,11,retained\n')
    target=tmp_path/'linked.parquet'
    result=export_table(source,target,['1'],'fixture')
    rows=pq.read_table(target).to_pylist()
    assert result['scanned_rows']==3 and result['retained_rows']==2
    assert rows[0]['value']=='line one\nline two'
    assert rows[0]['hadm_id']==''
    assert [r['_source_record'] for r in rows]==[1,3]
    assert export_table(source,target,['1'],'fixture')['retained_rows']==2
