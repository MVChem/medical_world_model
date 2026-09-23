"""Native Qwen3.5 baseline on MedWorld's matched classification/VQA test data."""
import argparse
import json
import time
from pathlib import Path


def yes_no_ids(tokenizer):
    ids = [tokenizer.encode(word, add_special_tokens=False) for word in ('Yes', 'No')]
    if any(len(row) != 1 for row in ids) or ids[0] == ids[1]:
        raise ValueError('Classification requires distinct single-token Yes/No answers')
    return [row[0] for row in ids]


def positive_probability(logits, ids):
    return logits[..., ids].float().softmax(-1)[..., 0]


def native_inputs(processor, image, prompt, pixels, device, family="qwen"):
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    options = {"min_pixels": pixels**2, "max_pixels": pixels**2} if family == "qwen" else {}
    inputs = processor(text=[text], images=[image], return_tensors="pt", **options)
    return inputs.to(device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--gpu', default='auto')
    p.add_argument('--model', choices=('qwen08b', 'qwen4b', 'qwen9b', 'medgemma4b'), default='qwen08b')
    p.add_argument('--tasks', nargs='+', choices=('classification', 'vqa'), default=['classification', 'vqa'])
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    if a.limit is not None and a.limit <= 0:
        p.error('--limit must be positive')
    if len(set(a.tasks)) != len(a.tasks):
        p.error('Tasks must be distinct')
    from medworld.config import load_config
    from medworld.gpu import acquire_gpu
    cfg = load_config(a.config)
    from .models import models
    spec = next(s for s in models() if s['id'] == a.model)
    out = a.out.resolve()
    if out.exists() and any(out.iterdir()):
        p.error('Choose an empty output directory')
    lock, device = acquire_gpu(a.gpu)
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from medworld.datasets import UnifiedData
        from medworld.datasets.vqa import INSTRUCTION
        from medworld.downstream_tasks.registry import FINDINGS
        from medworld.downstream_tasks.classification.metrics import classification_metrics
        from medworld.downstream_tasks.text.metrics import vqa_metrics
        from medworld.evaluation.protocol import metric_protocol
        from medworld.evaluation.selection import reference_manifest
        from medworld.runtime import atomic_json, seed_all
        from medworld.datasets.protocol import _sha256
        seed_all(cfg['seed'])
        startup_started = time.monotonic()
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out / 'status.json', {'status': 'loading'})
        data = UnifiedData(cfg)
        processor = AutoProcessor.from_pretrained(spec['path'], local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(
            spec['path'], local_files_only=True, dtype=torch.bfloat16,
            attn_implementation='sdpa').to(device).eval().requires_grad_(False)
        ids = yes_no_ids(processor.tokenizer)
        root = Path(spec['path'])
        summary = {'metric_protocol': metric_protocol('table2'), 'references': {},
                   'baseline': 'native_model_no_project_training', 'model_id': a.model, 'model_label': spec['label'], 'model': str(root),
                   'split': 'test', 'limit': a.limit, 'partial': a.limit is not None,
                   'data_fingerprint': data.fingerprint, 'tasks': {},
                   'timing': {'startup_seconds': time.monotonic() - startup_started, 'tasks': {}},
                   'classification_scoring': 'softmax over raw next-token Yes/No logits',
                   'vqa_scoring': 'MedWorld strict JSON label-set EM/micro-F1; unconstrained greedy decoding',
                   'unsupported_tasks': {'segmentation': 'N/A: native model has no segmentation head'},
                   'weights_sha256': {f.name: _sha256(f) for f in sorted(root.glob('*.safetensors'))},
                   'source_sha256': _sha256(Path(__file__))}
        atomic_json(out / 'config.json', cfg)
        atomic_json(out / 'data_protocol.json', data.metadata)
        atomic_json(out / 'summary.json', summary)
        with torch.inference_mode():
            for task in a.tasks:
                from medworld.evaluation.selection import select_vqa
                indices = list(range(len(data.rows(task, 'test'))))
                if task == 'vqa':
                    indices, selection = select_vqa(data.rows(task, 'test'), cfg['testing']['vqa_per_type'], cfg['testing']['vqa_seed'])
                    summary['vqa_selection'] = selection
                    atomic_json(out / 'vqa_selection.json', selection)
                count = len(indices)
                count = min(count, a.limit) if a.limit else count
                if not count:
                    raise ValueError(f'Empty test cohort: {task}')
                records = []
                inference_seconds = 0.
                with (out / f'{task}.jsonl').open('w') as journal:
                    for index in range(count):
                        batch = data.batch(task, 'test', [indices[index]])
                        image = batch['images'][0]
                        row = {'id': batch['ids'][0], 'patient': batch['subject_ids'][0]}
                        inference_started = time.monotonic()
                        if task == 'classification':
                            probabilities = []
                            for finding in FINDINGS:
                                prompt = f'Does this chest radiograph show {finding}? Answer only Yes or No.'
                                inputs = native_inputs(processor, image, prompt, cfg['vision_pixels'], device, spec['family'])
                                logits = model(**inputs, logits_to_keep=1, use_cache=False).logits[0, -1]
                                probabilities.append(float(positive_probability(logits, ids)))
                            row.update(probabilities=probabilities, labels=batch['labels'][0].tolist())
                        else:
                            inputs = native_inputs(processor, image, INSTRUCTION + batch['questions'][0],
                                                   cfg['vision_pixels'], device, spec['family'])
                            generated = model.generate(**inputs, do_sample=False,
                                                       max_new_tokens=cfg['generation_tokens'])
                            answer = processor.tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:],
                                                                skip_special_tokens=True)
                            row.update(question=batch['questions'][0], prediction=answer,
                                       answer=batch['answers'][0], semantic_type=batch['semantic_types'][0])
                        inference_seconds += time.monotonic() - inference_started
                        journal.write(json.dumps(row, ensure_ascii=False) + '\n')
                        journal.flush()
                        records.append(row)
                        atomic_json(out / 'status.json', {'status': 'running', 'task': task,
                                                        'completed': index + 1, 'total': count})
                scoring_started = time.monotonic()
                metrics = (classification_metrics([r['labels'] for r in records],
                           [r['probabilities'] for r in records], FINDINGS)
                           if task == 'classification' else vqa_metrics(records))
                summary['tasks'][task] = {'n': count, **metrics}
                summary['timing']['tasks'][task] = {'n': count, 'inference_seconds': inference_seconds,
                    'scoring_seconds': time.monotonic() - scoring_started,
                    'inference_seconds_per_example': inference_seconds / count}
                summary['references'][task] = reference_manifest(task, records)
                summary.setdefault('predictions_sha256', {})[task] = _sha256(out / f'{task}.jsonl')
                atomic_json(out / 'summary.json', summary)
        atomic_json(out / 'status.json', {'status': 'complete', 'partial': a.limit is not None})
    except BaseException as error:
        if out.exists():
            (out / 'status.json').write_text(json.dumps({'status': 'failed', 'error': repr(error)}) + '\n')
        raise
    finally:
        if lock is not None:
            lock.close()


if __name__ == '__main__':
    main()
