"""Full classification, segmentation and VQA evaluation for one checkpoint."""
import argparse
import json
from pathlib import Path
from ..gpu import acquire_gpu
from ..downstream_tasks.registry import TASKS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--task', choices=('all', *TASKS), default='all')
    p.add_argument('--split', choices=('validate', 'test', 'human_test'), default='test')
    p.add_argument('--limit', type=int)
    p.add_argument('--vqa-per-type', type=int, default=0)
    p.add_argument('--vqa-seed', type=int, default=42)
    p.add_argument('--gpu', default='auto')
    a = p.parse_args()
    if a.limit is not None and a.limit <= 0:
        p.error('limit must be positive')
    if a.split == 'human_test' and a.task != 'segmentation':
        p.error('human_test is for segmentation only')
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        p.error('Choose a new output folder')
    lock, device = acquire_gpu(a.gpu)
    try:
        import torch
        from ..datasets import UnifiedData
        from ..runtime import load_model, atomic_json
        from ..datasets.protocol import _sha256
        from ..downstream_tasks.registry import FINDINGS
        from ..downstream_tasks.classification import classification_metrics
        from ..downstream_tasks.segmentation import segmentation_metrics
        from ..downstream_tasks.segmentation.metrics import aggregate_segmentation
        from ..downstream_tasks.text.metrics import vqa_metrics
        model, saved = load_model(a.checkpoint, device)
        data = UnifiedData(model.cfg)
        if data.fingerprint != saved['data_fingerprint']:
            raise ValueError('Evaluation data differs from training protocol')
        out.mkdir(parents=True)
        summary = {'checkpoint_sha256': _sha256(Path(a.checkpoint)), 'data_fingerprint': data.fingerprint,
                   'predictions_sha256': {}, 'slot_conditioning': model.cfg['slot_conditioning'], 'split': a.split, 'limit': a.limit, 'tasks': {}}
        with torch.no_grad():
            for task in TASKS if a.task == 'all' else (a.task,):
                from .selection import select_vqa
                indices = list(range(len(data.rows(task, a.split))))
                if task == 'vqa':
                    indices, selection = select_vqa(data.rows(task, a.split), a.vqa_per_type, a.vqa_seed)
                    summary['vqa_selection'] = selection
                    atomic_json(out / 'vqa_selection.json', selection)
                count = len(indices)
                count = min(count, a.limit) if a.limit else count
                if not count:
                    raise ValueError(f'Empty evaluation cohort: {task}/{a.split}')
                records = []
                with (out / f'{task}.jsonl').open('w', buffering=1) as journal:
                    for index in range(count):
                        batch = data.batch(task, a.split, [indices[index]])
                        prediction = model.predict(task, batch)
                        row = {'id': batch['ids'][0], 'patient': batch['subject_ids'][0]}
                        if task == 'classification':
                            row.update(labels=batch['labels'][0].tolist(), probabilities=prediction[0].float().cpu().tolist())
                        elif task == 'segmentation':
                            row.update(segmentation_metrics(prediction.cpu(), batch['targets'], batch['mask']))
                            for key, batch_key in [('dataset', 'segmentation_dataset'), ('volume_id', 'volume_id'),
                                                   ('target_names', 'target_names')]:
                                row[key] = batch[batch_key][0]
                        else:
                            row.update(question=batch['questions'][0], answer=batch['answers'][0],
                                       semantic_type=batch['semantic_types'][0], prediction=prediction[0])
                        records.append(row)
                        journal.write(json.dumps(row, ensure_ascii=False) + '\n')
                        if (index + 1) % 25 == 0:
                            print(f'{task}: {index + 1}/{count}', flush=True)
                if task == 'classification':
                    metrics = classification_metrics([r['labels'] for r in records], [r['probabilities'] for r in records], FINDINGS)
                elif task == 'segmentation':
                    metrics = aggregate_segmentation(records)
                    metrics['target_kind'] = ('human Montgomery lung masks' if a.split == 'human_test'
                                              else 'human-reviewed CXR and MRI masks')
                else:
                    metrics = vqa_metrics(records)
                summary['predictions_sha256'][task] = _sha256(out / f'{task}.jsonl')
                summary['tasks'][task] = {'n': count, **metrics}
                atomic_json(out / 'summary.json', summary)
    finally:
        if lock is not None:
            lock.close()


if __name__ == '__main__':
    main()
