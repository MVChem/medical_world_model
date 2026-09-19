"""Verify reuse of completed raw-Qwen predictions against source-only test inputs.

Only final predictions, references and provenance are reused. Pixel caches are
never opened: source images are rebuilt in memory and checked against PNG hashes.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

from ..config import load_config
from ..datasets import UnifiedData
from .data import _rows, _sha256, FINDINGS
from .classification import classification_metrics


def audit(config, baseline, out, report_rescore=None):
    cfg = load_config(config)
    data = UnifiedData(cfg)
    baseline = Path(baseline).resolve()
    protocol = json.loads((baseline / 'protocol.json').read_text())
    inventory = json.loads((baseline / 'models.json').read_text())
    model = next(m for m in inventory if m['id'] == 'qwen08b')
    for artifact in model['weight_files']:
        if _sha256(Path(cfg['qwen']) / artifact['name']) != artifact['sha256']:
            raise ValueError('Baseline Qwen weights differ')
    original_cfg = json.loads((baseline / 'training_config.json').read_text())
    if cfg['vision_pixels'] != original_cfg['vision_pixels']:
        raise ValueError('Native visual preprocessing resolution differs')
    names=['cohort/table2_inputs_test.jsonl','cohort/table2_references_test.jsonl',
           'cohort/table1_inputs_test.jsonl','cohort/table1_references_test.jsonl']
    for name in names:
        if _sha256(baseline / name) != protocol['file_sha256'][name]:
            raise ValueError('Baseline cohort provenance changed')
    inputs = _rows(baseline / names[0]); refs={r['id']:r for r in _rows(baseline / names[1])}
    test=baseline/'qwen08b/test'
    old_predictions = {task:_rows(test/(task+'.jsonl')) for task in ('classification','report','temporal')}
    checked=set()
    def pixels_match(image, old_path):
        key=str(Path(old_path).relative_to(baseline))
        buf=io.BytesIO(); image.convert('RGB').save(buf,format='PNG')
        if hashlib.sha256(buf.getvalue()).hexdigest()!=protocol['file_sha256'][key]:
            raise ValueError(f'Source pixels differ from baseline input: {key}')
    for task,flag in [('classification','classification'),('report','report_generation')]:
        cohort=data.rows(task,'test'); old=[r for r in inputs if r[flag]]; pred=old_predictions[task]
        if [r['id'] for r in cohort]!=[r['id'] for r in old] or [r['id'] for r in cohort]!=[r['id'] for r in pred]:
            raise ValueError(f'{task} test membership/order differs')
        for row,inp,prediction in zip(cohort,old,pred):
            ref=refs[row['id']]
            target=row['labels'] if task=='classification' else row['report_target']
            if ref['labels' if task=='classification' else 'report']!=target:
                raise ValueError('Baseline references differ')
            if prediction['labels' if task=='classification' else 'reference']!=target:
                raise ValueError('Scored prediction references differ')
            if str(row['subject_id'])!=inp['patient']:
                raise ValueError('Patient identity differs')
            if row['id'] not in checked:
                pixels_match(data.current._example(task,'test',row)['image'],inp['image']);checked.add(row['id'])
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(cfg['qwen'],local_files_only=True)
    temporal_inputs=_rows(baseline/names[2]); temporal_refs=_rows(baseline/names[3]); temporal=data.rows('temporal','test')
    if [r['id'] for r in temporal]!=[r['id'] for r in temporal_inputs] or [r['id'] for r in temporal]!=[r['id'] for r in old_predictions['temporal']]:
        raise ValueError('Temporal cohort differs')
    checked_temporal=set()
    for pair,inp,ref,pred in zip(temporal,temporal_inputs,temporal_refs,old_predictions['temporal']):
        source,target=[data.temporal.lookup[pair[k]] for k in ('source','target')]
        text='Chest radiograph observation.\nReport:\n'+source['report']
        ids=tokenizer(text,truncation=True,max_length=cfg['context_tokens'])['input_ids']
        if inp['source_text_token_ids']!=ids or inp['delta_hours']!=pair['delta_hours'] or inp['direction']!=pair['direction']:
            raise ValueError('Temporal context/time differs')
        if ref['target_report']!=target['report'] or ref['labels']!=target['labels'] or pred['reference']!=target['report'] or pred['labels']!=target['labels']:
            raise ValueError('Temporal targets differ')
        if pair['source'] not in checked_temporal:
            pixels_match(data.temporal._observation(pair['source'])[0],inp['image']);checked_temporal.add(pair['source'])
    rows=old_predictions['classification']
    metrics=classification_metrics([r['labels'] for r in rows],[r['probabilities'] for r in rows],FINDINGS)
    stored=json.loads((test/'metrics.json').read_text())
    for new,old in [('macro_auroc','auroc'),('macro_ap','ap')]:
        if abs(metrics[new]-stored['table2_classification'][old])>1e-10:
            raise ValueError('Recomputed baseline metric differs')
    result={'status':'passed','baseline':str(baseline),'model':'Qwen3.5-0.8B native, no project training/adapters',
        'data_fingerprint':data.fingerprint,'protocol_sha256':_sha256(baseline/'protocol.json'),
        'classification_n':len(rows),'report_n':len(old_predictions['report']),'temporal_n':len(temporal),
        'current_exact_pixel_hashes':len(checked),'temporal_exact_pixel_hashes':len(checked_temporal),
        'generation_tokens':protocol['generation']['max_new_tokens'],
        'classification':metrics,'clinical_baseline_metrics':stored,
        'result_sha256':{p.name:_sha256(p) for p in [test/'metrics.json',test/'classification.jsonl',test/'report.jsonl',test/'temporal.jsonl',test/'table2_report_chexbert.json']},
        'segmentation':None,'sr':None,'dense_baseline_status':'Native Qwen has no pixel output head; N/A. Bicubic SR is a separate reference.'}
    if report_rescore is not None:
        rescore=json.loads(Path(report_rescore).read_text())
        if rescore['n']!=len(old_predictions['report']):raise ValueError('Incomplete native report re-score')
        if Path(rescore['input']).resolve()!=(test/'report.jsonl').resolve():raise ValueError('Re-score input is not the native report predictions')
        result['native_report_rescore']=rescore
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('classification','clinical_baseline_metrics','result_sha256')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--baseline',required=True);p.add_argument('--out',required=True)
    p.add_argument('--report-rescore',help='Native report metrics produced by the current clinical_report scorer')
    a=p.parse_args();audit(a.config,a.baseline,a.out,a.report_rescore)
