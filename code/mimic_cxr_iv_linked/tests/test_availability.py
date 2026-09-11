from datetime import datetime
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import polars as pl
from availability import historical_counts,dt


def test_fractional_cxr_cutoffs_are_preserved():
    x=pl.DataFrame({'cutoff':['2100-01-01 12:00:00','2100-01-01 12:00:00.500000']}).with_columns(dt('cutoff'))
    assert x['cutoff'].null_count()==0
    assert (x['cutoff'][1]-x['cutoff'][0]).total_seconds()==0.5


def test_counts_respect_patient_admission_cutoff_and_missing_time():
    t=datetime(2100,1,1,12)
    obs=pl.DataFrame({'dicom_id':['a','b','c'],'subject_id':['1','1','2'],
        'hadm_id':['11','12','11'],'cutoff':[t,t,t]})
    ev=pl.DataFrame({'subject_id':['1','1','1','1','2'],'hadm_id':['11','11','11','12','11'],
        'available_time':[datetime(2100,1,1,11),datetime(2100,1,1,13),None,datetime(2100,1,1,9),datetime(2100,1,1,14)]})
    result={r['dicom_id']:r['history_count'] or 0 for r in historical_counts(obs,ev).iter_rows(named=True)}
    assert result=={'a':1,'b':1,'c':0}
