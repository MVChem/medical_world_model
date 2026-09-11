"""Check clinical cutoff isolation and preservation of original report targets."""
from datetime import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import polars as pl
from common import write_rows
from prepare_linked import cutoff_rows
from data import Corpus


def test_ehr_requires_both_timestamps_patient_and_admission():
    points = pl.DataFrame([dict(id='a', subject_id='p', hadm_id='h', cutoff=datetime(2020, 1, 3, 12))])
    rows = []
    for i, (item, patient, hadm, event, recorded) in enumerate([
        ('valid', 'p', 'h', '2020-01-03 10:00:00', '2020-01-03 11:00:00'),
        ('late', 'p', 'h', '2020-01-03 10:00:00', '2020-01-03 13:00:00'),
        ('future', 'p', 'h', '2020-01-03 13:00:00', '2020-01-03 11:00:00'),
        ('missing', 'p', 'h', '2020-01-03 10:00:00', None),
        ('patient', 'other', 'h', '2020-01-03 10:00:00', '2020-01-03 11:00:00'),
        ('admission', 'p', 'other', '2020-01-03 10:00:00', '2020-01-03 11:00:00'),
        ('old', 'p', 'h', '2019-12-01 10:00:00', '2020-01-03 11:00:00'),
        ('boundary', 'p', 'h', '2020-01-03 12:00:00', '2020-01-03 12:00:00'),
    ]):
        rows.append(dict(itemid=item, subject_id=patient, hadm_id=hadm,
            charttime=event, storetime=recorded, _source_record=i, value='original'))
    result = cutoff_rows(points, pl.DataFrame(rows), [r['itemid'] for r in rows])
    assert set(result['itemid']) == {'valid', 'boundary'}


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1
    def encode(self, s, **kwargs):
        return [ord(c)+2 for c in s]
    def __call__(self, rows, max_length, **kwargs):
        return {'input_ids': [self.encode(r)[:max_length] for r in rows]}


def test_context_never_replaces_report_targets_and_source_does_not_use_future(tmp_path):
    obs = [dict(id=str(i), split='train', report=f'report{i}', ehr_text=f'clinical{i}',
        image='unused', labels=[0]*6) for i in range(5)]
    write_rows(tmp_path/'observations.jsonl', obs)
    pairs = [dict(id=str(i), source=str(i), target=str((i+1)%5), horizon=0) for i in range(5)]
    for split in ('train', 'validate', 'test'):
        write_rows(tmp_path/(split+'.jsonl'), pairs)
    np.save(tmp_path/'vjepa_features.npy', np.zeros((5, 64, 768), dtype=np.float16))
    cfg = dict(cache=str(tmp_path), report_tokens=100, ehr_tokens=100, use_ehr=True,
        seed=42, batch_size=1, gradient_accumulation=1, sampling='permutation')
    corpus = Corpus(cfg, Tokenizer())
    batch = corpus.batch(pairs[:1], device='cpu')
    original = batch['source_ids'].clone()
    assert batch['source_target_ids'][0].tolist() == Tokenizer().encode('report0')+[1]
    assert batch['target_target_ids'][0].tolist() == Tokenizer().encode('report1')+[1]
    corpus.context_tokens[1] = Tokenizer().encode('altered future clinical text')
    assert (corpus.batch(pairs[:1], device='cpu')['source_ids'] == original).all()
    # Check the real training sampler with CPU batches, including an epoch boundary.
    corpus.batch = lambda rows, stage1=False: rows
    first = [corpus.training_batch(2, step, 0)[0]['id'] for step in range(5)]
    assert len(set(first)) == 5
    assert [corpus.training_batch(2, step, 0)[0]['id'] for step in range(5)] == first
    assert len([corpus.training_batch(2, step, 0) for step in range(5, 7)]) == 2
