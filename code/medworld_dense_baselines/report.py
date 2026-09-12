"""Live preview only; never write the paper's Table 1 or Table 2."""
import argparse
import csv
import html
import json
from common import *

def report(run):
    models=json.loads(MODEL_FILE.read_text())
    matrix=[]
    cols=['model','input_variant','epochs','Dir. F1','anatomy mIoU','anatomy A@0.5','Dice pseudo','Dice human (lungs)','PSNR x4','SSIM x4']
    def fmt(v):return '' if v is None else f'{v:.4f}'
    for m in models:
        mid=m['id'];direction=run/mid/'direction_metrics.json'
        dd=json.loads(direction.read_text()) if direction.exists() else {}
        for variant in ['image','vjepa','vjepa_adapter']:
            row={'model':m['label']+(' (v1)' if mid=='medgemma27b' else ''),'input_variant':variant,'epochs':20}
            if variant=='image':row['Dir. F1']=fmt(dd.get('macro_f1')) if dd.get('complete') else ''
            for task in ['grounding','segmentation','sr']:
                f=run/mid/f'{task}_{variant}'/'metrics.json'
                if not f.exists():continue
                result=json.loads(f.read_text());rr=result['metrics'];test=rr['test']
                if task=='grounding':row.update({'anatomy mIoU':fmt(test['iou']),'anatomy A@0.5':fmt(test['acc50'])})
                elif task=='segmentation':row.update({'Dice pseudo':fmt(test['dice']),'Dice human (lungs)':fmt(rr['human_test']['dice'])})
                else:row.update({'PSNR x4':fmt(test['psnr']),'SSIM x4':fmt(test['ssim'])})
            matrix.append(row)
    out=run/'preview';out.mkdir(exist_ok=True)
    with (out/'dense_matrix.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader();writer.writerows(matrix)
    manifest=json.loads((run/'data/manifest.json').read_text())
    direct=json.loads((run/'direction/manifest.json').read_text())
    queue=json.loads((run/'queue.json').read_text()) if (run/'queue.json').exists() else []
    text=['# Frozen VLM dense probes — live preview','',
        'Target: Monday 2026-09-14 08:00 Asia/Shanghai. Empty cells are pending/unmeasured, not zero.',
        '','All six VLMs are frozen. Each downstream head is independently initialized and trained for 20 epochs on the same per-task patient splits. No world-model checkpoint is used.',
        '', 'Image variants: raw image; native frozen V-JEPA features with parameter-free channel selection; native V-JEPA features with a newly trained adapter. Every variant also receives the same cached VLM hidden states for that model.',
        '', '**SR leakage control:** both VLM and V-JEPA states are extracted from one shared cached rounded 4x low-resolution image. The raw-image variant uses a standard bicubic LR residual; feature variants fully replace pixels and have no original-image or bicubic skip path.',
        '', '**Metric scope:** localization uses human Chest ImaGenome anatomical region boxes and a closed region vocabulary, not MS-CXR lesion-phrase grounding. Pseudo Dice is CXAS teacher agreement (three organs); human Dice is external Montgomery testing (two lungs only). These modified cohorts must be footnoted if later transferred to a paper table.',
        '',f'Direction F1: {direct["eligible_pairs"]} eligible pairs / {direct["fields"]} persistent-positive finding/scope fields, selected from {direct["original_gold_pairs"]} gold pairs before inference. This is a separate source-image/report-only forecasting protocol; no future image/report is supplied.',
        '', 'Cohort counts: `'+json.dumps(manifest['coverage'])+'`','',
        '| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    text+=['| '+' | '.join(str(row.get(k,'')) for k in cols)+' |' for row in matrix]
    text+=['','## Queue','', '| Job | State | GPUs |','| --- | --- | --- |']
    text+=['| '+j['id']+' | '+j['status']+' | '+str(j.get('assigned_gpus',''))+' |' for j in queue]
    text+=['','The original Table 1/2 rows and prior zero-shot values remain in [the prior complete matrix](../../../../../results/baseline_matrix_preview_20260912/preview.md). The paper files have not been modified.']
    (out/'preview.md').write_text('\n'.join(text)+'\n')
    rows=''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in cols)+'</tr>' for r in matrix)
    (out/'preview.html').write_text('<!doctype html><meta charset="utf-8"><title>Dense baseline progress</title><style>body{font:14px system-ui;margin:30px}td,th{padding:8px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style><h1>Dense baseline progress</h1><p>20 matched epochs; blank = pending. Anatomy grounding; pseudo three-organ Dice; human two-lung Dice. See preview.md for full protocol.</p><table><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in cols)+'</tr>'+rows+'</table>')
    atomic(out/'aggregate.json',dict(columns=cols,rows=matrix,cohort=manifest,direction_protocol=direct))
    # Preserve every original paper row and every completed zero-shot row in a full preview.
    old=PROJECT/'results/baseline_matrix_preview_20260912'
    full=['# Full Table 1 / Table 2 preview','',
          'Original model rows and zero-shot measurements are retained. Blank cells are unmeasured. '+
          'New dense rows are frozen VLM + trained head results at 20 epochs, not zero-shot results.',
          '',f'Direction uses a separate {direct["eligible_pairs"]}-pair source-image/report-only cohort. '+
          'New grounding entries are human anatomical-region localization with a closed query vocabulary, not MS-CXR lesion grounding. '+
          'New human Dice entries use Montgomery two-lung external evaluation. See preview.md for the complete protocol.','']
    for tab in ['table1','table2']:
        with (old/f'{tab}.csv').open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f);headers=list(reader.fieldnames);fullrows=list(reader)
        if tab=='table1':
            for r in fullrows:
                for m in models:
                    if r['Method'].startswith(m['label']+' (') and 'ZS' in r['Method']:
                        dp=run/m['id']/'direction_metrics.json'
                        if dp.exists():
                            d=json.loads(dp.read_text())
                            if d.get('complete'):r['Dir. F1 ↑']=fmt(d['macro_f1'])
        else:
            for row in matrix:
                r={h:'' for h in headers}
                r['Method']=row['model']+' + trained heads / '+row['input_variant']
                mapping={'定位 mIoU ↑':'anatomy mIoU','定位 A@.5 ↑':'anatomy A@0.5',
                    'Dice pseudo ↑':'Dice pseudo','Dice human ↑':'Dice human (lungs)',
                    'SR ×4 PSNR ↑':'PSNR x4','SR ×4 SSIM ↑':'SSIM x4'}
                for k,v in mapping.items():r[k]=row.get(v,'')
                fullrows.append(r)
            if (run/'bicubic_metrics.json').exists():
                b=json.loads((run/'bicubic_metrics.json').read_text())
                r={h:'' for h in headers};r.update({'Method':'Bicubic x4 (no training)',
                    'SR ×4 PSNR ↑':fmt(b['psnr']),'SR ×4 SSIM ↑':fmt(b['ssim'])})
                fullrows.append(r)
        with (out/f'{tab}_full.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=headers);writer.writeheader();writer.writerows(fullrows)
        full += [f'## {tab}','', '| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
        full += ['| '+' | '.join(r[h] for h in headers)+' |' for r in fullrows]
        full += ['']
    (out/'full_tables.md').write_text('\n'.join(full)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT_RUN)
    a=p.parse_args();report(a.run)
