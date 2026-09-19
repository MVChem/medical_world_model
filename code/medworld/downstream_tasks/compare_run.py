"""Summarize final Stage 1/2 downstream tests against audited native Qwen 0.8B."""
import argparse
import json
from pathlib import Path


def compare(run):
    run=Path(run);audit=json.loads((run/'baseline_audit.json').read_text())
    if audit['status']!='passed':raise ValueError('Baseline audit did not pass')
    from .data import _sha256
    for name,digest in audit['result_sha256'].items():
        if _sha256(Path(audit['baseline'])/'qwen08b/test'/name)!=digest:
            raise ValueError('Audited baseline results changed during training')
    rows=[{'model':'Qwen3.5-0.8B native (no project training)',
           'classification_auroc':audit['classification']['macro_auroc'],
           'classification_ap':audit['classification']['macro_ap'],
           'report_chexbert_f1':audit['native_report_rescore']['chexbert_f1'],
           'segmentation_dice':None,'human_dice':None,'sr_psnr':None,'sr_ssim':None}]
    for stage in ('stage1','stage2'):
        root=run/'evaluation'/stage
        metrics={task:json.loads((root/task/'summary.json').read_text())['tasks'][task]
                 for task in ('classification','report','segmentation','sr')}
        for task in metrics:
            summary=json.loads((root/task/'summary.json').read_text())
            if summary['data_fingerprint']!=audit['data_fingerprint'] or summary['limit'] is not None:
                raise ValueError('Test comparison requires complete matching cohorts')
        if metrics['classification']['n']!=audit['classification_n'] or metrics['report']['n']!=audit['report_n']:
            raise ValueError('Classification/report denominator differs from baseline')
        clinical=json.loads((root/'report/clinical.json').read_text())
        original_provenance=audit['native_report_rescore']['provenance']
        if clinical['provenance']!=original_provenance:
            raise ValueError('Clinical scorer provenance differs from native baseline')
        human=json.loads((root/'segmentation_human/summary.json').read_text())['tasks']['segmentation']
        rows.append({'model':stage,'classification_auroc':metrics['classification']['macro_auroc'],
                     'classification_ap':metrics['classification']['macro_ap'],
                     'report_chexbert_f1':clinical['chexbert_f1'],
                     'segmentation_dice':metrics['segmentation']['mean_dice'],'human_dice':human['mean_dice'],
                     'sr_psnr':metrics['sr']['psnr'],'sr_ssim':metrics['sr']['ssim']})
    fields=['classification_auroc','classification_ap','report_chexbert_f1','segmentation_dice','human_dice','sr_psnr','sr_ssim']
    result={'rows':rows,'deltas_vs_native':{row['model']:{k:row[k]-rows[0][k] for k in fields if row[k] is not None and rows[0][k] is not None} for row in rows[1:]},
            'deltas_stage2_vs_stage1':{k:rows[2][k]-rows[1][k] for k in fields},
            'baseline_audit':'baseline_audit.json','generation_tokens':384,'dense_baseline':'Native Qwen: N/A (no pixel decoder).',
            'test_policy':'Fixed held-out test; final stage checkpoints, no test-based selection.'}
    result['bicubic_reference']=json.loads((run/'bicubic_reference.json').read_text())
    from .classification import classification_metrics
    from .data import FINDINGS, _rows
    from ..model import TEMPORAL_COLUMNS
    temporal_root=run/'evaluation/stage2/temporal'
    temporal_summary=json.loads((temporal_root/'summary.json').read_text())
    temporal_rows=_rows(temporal_root/'temporal.jsonl')
    native_rows=_rows(Path(audit['baseline'])/'qwen08b/test/temporal.jsonl')
    if (temporal_summary['data_fingerprint']!=audit['data_fingerprint'] or
        temporal_summary['limit'] is not None or len(temporal_rows)!=audit['temporal_n'] or
        [(r['id'],r['labels']) for r in temporal_rows]!=[(r['id'],r['labels']) for r in native_rows]):
        raise ValueError('Temporal comparison requires matching complete cohorts and labels')
    names=[FINDINGS[i] for i in TEMPORAL_COLUMNS]
    native_temporal=classification_metrics([r['labels'] for r in native_rows],
                                           [r['probabilities'] for r in native_rows],names)
    trained_temporal=temporal_summary['tasks']['temporal']
    result['temporal_classification']={
        'n':len(temporal_rows),'native':native_temporal,'stage2':trained_temporal,
        'delta':{k:trained_temporal[k]-native_temporal[k] for k in ('macro_auroc','macro_ap')}}
    (run/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Downstream comparison','', 'Fixed test cohorts; raw 0–1 scores except PSNR (dB). Reports use greedy generation up to 384 tokens.', '',
           '| Model | Classification AUROC | Classification AP | Report CheXbert F1 | Dice (pseudo) | Dice (human) | SR PSNR | SR SSIM |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| '+row['model']+' | '+' | '.join('N/A' if row[k] is None else f'{row[k]:.5f}' for k in fields)+' |')
    bicubic=result['bicubic_reference']
    lines.append(f"| Bicubic (separate SR reference) | N/A | N/A | N/A | N/A | N/A | {bicubic['psnr']:.5f} | {bicubic['ssim']:.5f} |")
    lines+=['','Native Qwen uses original Yes/No likelihoods; adapted models use trained classification heads.',
            'Native Qwen has no segmentation/SR decoder. These cells are N/A, not scores from random heads.',
            'Stage 2 temporal evaluation is stored separately in `evaluation/stage2/temporal/`.','',
            '| Temporal prediction | AUROC | AP |','|---|---:|---:|',
            f"| Native Qwen | {native_temporal['macro_auroc']:.5f} | {native_temporal['macro_ap']:.5f} |",
            f"| Stage 2 | {trained_temporal['macro_auroc']:.5f} | {trained_temporal['macro_ap']:.5f} |"]
    (run/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);compare(p.parse_args().run)
