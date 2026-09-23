"""Native Qwen9B Table 1 predictions from source image/report evidence only."""
import argparse
import json
import math
import time
from pathlib import Path

from .evaluate import native_inputs, positive_probability, yes_no_ids


LOS_DAYS = (0., .5, 1., 2., 3., 5., 7., 10., 14., 21., 30., 60., 90., 180., 365.)
DIRECTIONS = ('improved', 'stable', 'worsened')
# Fixed without reference labels; each code is one unique token in local Qwen9B.
# Qwen tokenizes multi-digit numbers separately, so 0..109 cannot be used here.
OPTION_CODES = tuple(('A B C D E F G H I J K L M N O P Q R S T U V W X Y Z '
                      'AA AB AC AD AE AF AG AH AI AJ AK AL AM AN AO AP AQ AR AS AT AU AV AW AX AY AZ '
                      'BA BB BC BD BE BF BG BH BI BJ BK BL BM BN BO BP BR BS BT BU BV BW BX BY '
                      'CA CB CC CD CE CF CG CH CI CK CL CM CN CO CP CR CS CT CU CV CW CX CY '
                      'DA DB DC DD DE DF DG DH DI DJ DK').split())


def option_ids(tokenizer, count):
    """Use fixed alphabetic answer codes, requiring one distinct token per option."""
    if type(count) is not int or not 1 <= count <= len(OPTION_CODES):
        raise ValueError('The option count must be an integer between 1 and 110')
    encoded = [tokenizer.encode(code, add_special_tokens=False) for code in OPTION_CODES[:count]]
    if any(len(ids) != 1 for ids in encoded) or len({tuple(ids) for ids in encoded}) != count:
        raise ValueError('Native future options require distinct single-token alphabetic codes')
    return [ids[0] for ids in encoded]


def option_probabilities(logits, ids):
    import torch
    selected = logits[..., ids].float()
    if not torch.isfinite(selected).all():
        raise ValueError('Native future option logits must be finite')
    return selected.softmax(-1)


def _truncate(text, limit):
    return text.encode('utf-8')[:limit].decode('utf-8', errors='ignore')


def future_prompt(task, batch, cfg):
    """Read only explicitly allowed source fields, even when targets are present."""
    from medworld.datasets.vqa import VOCABULARY
    if (len(batch['images']) != 1 or len(batch['reports']) != 1
            or len(batch['delta_hours']) != 1):
        raise ValueError('Native future evaluation expects one source example')
    report = batch['reports'][0]
    hours = float(batch['delta_hours'][0])
    if not isinstance(report, str) or not report.strip() or not math.isfinite(hours) or hours <= 0:
        raise ValueError('A nonempty source report and finite positive requested horizon are required')
    if task in ('mortality_30d', 'remaining_los') and hours != (720 if task == 'mortality_30d' else 24):
        raise ValueError('Outcome requests must use the fixed horizon, never the realized outcome')
    report = _truncate(report, cfg['future_source_bytes'])
    prompt = ('Use only the source chest radiograph and the source report below to make a forecast.\n'
              f'Source report:\n{report}\nRequested forecast horizon: {hours:g} hours.\n')
    options = None
    if task in ('future_vqa', 'progression'):
        questions = batch['questions']
        if len(questions) != 1 or not isinstance(questions[0], str) or not questions[0].strip():
            raise ValueError('Future VQA/progression requires a source-anchored task question')
        question = _truncate(questions[0], cfg['context_tokens'] - 1)
        prompt += f'Question about the requested future examination: {question}\n'
        options = VOCABULARY if task == 'future_vqa' else DIRECTIONS
    elif task == 'future_report':
        prompt += ('Generate the full report for the future chest radiograph. '
                   'Return only the predicted report.\n')
    elif task == 'mortality_30d':
        prompt += 'Will this patient die within 30 days after the source examination? Answer only Yes or No.\n'
    elif task == 'remaining_los':
        prompt += ('Estimate the number of days from the source examination until discharge from '
                   'the current hospital admission. The forecast request is fixed at 24 hours; '
                   'it is not the observed discharge interval.\n')
        options = LOS_DAYS
    else:
        raise ValueError(f'Unsupported native future task: {task}')
    if options is not None:
        prompt += 'Options:\n' + '\n'.join(f'{code}: {value}' for code, value in zip(OPTION_CODES, options))
        prompt += '\nReturn only the alphabetic option code.'
    return prompt, options


def predict_future(model, processor, task, batch, cfg, device):
    """One next-token forward for categorical/outcome tasks; greedy report text."""
    prompt, options = future_prompt(task, batch, cfg)
    inputs = native_inputs(processor, batch['images'][0], prompt, cfg['vision_pixels'], device)
    if task == 'future_report':
        from transformers import StoppingCriteria, StoppingCriteriaList
        prefix_length = inputs['input_ids'].shape[1]
        budget = cfg['future_report_tokens'] - 1
        tokenizer = processor.tokenizer

        class ByteBudget(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                output = tokenizer.decode(input_ids[0, prefix_length:], skip_special_tokens=True)
                return len(output.encode('utf-8')) >= budget

        generated = model.generate(**inputs, do_sample=False, max_new_tokens=cfg['future_report_tokens'],
                                   stopping_criteria=StoppingCriteriaList([ByteBudget()]))
        answer = tokenizer.decode(generated[0, prefix_length:], skip_special_tokens=True)
        return _truncate(answer, budget)
    logits = model(**inputs, logits_to_keep=1, use_cache=False).logits[0, -1]
    if task == 'mortality_30d':
        probability = float(positive_probability(logits, yes_no_ids(processor.tokenizer)))
        if not math.isfinite(probability):
            raise ValueError('Native mortality probability is not finite')
        return probability
    probabilities = option_probabilities(logits, option_ids(processor.tokenizer, len(options)))
    if task == 'remaining_los':
        return sum(float(p) * days for p, days in zip(probabilities, LOS_DAYS))
    return options[int(probabilities.argmax())]


def native_readout_protocol(cfg):
    from medworld.datasets.vqa import VOCABULARY
    return {'source_report_bytes': cfg['future_source_bytes'],
            'question_bytes': cfg['context_tokens'] - 1,
            'option_codes': list(OPTION_CODES),
            'future_vqa': {'method': 'one next-token forward; argmax over fixed option-code logits',
                           'options': list(VOCABULARY)},
            'progression': {'method': 'one next-token forward; argmax over fixed option-code logits',
                            'options': list(DIRECTIONS)},
            'mortality_30d': {'method': 'next-token Yes/No softmax positive probability', 'horizon_hours': 720},
            'remaining_los': {'method': 'expected days under fixed option-code softmax',
                              'options_days': list(LOS_DAYS), 'horizon_hours': 24},
            'future_report': {'method': 'greedy decoding with a UTF-8 byte stopping budget',
                              'max_output_bytes': cfg['future_report_tokens'] - 1}}


def main():
    from medworld.evaluation.protocol import FUTURE_TASKS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--gpu', default='auto')
    parser.add_argument('--model', choices=('qwen9b',), default='qwen9b')
    parser.add_argument('--tasks', nargs='+', choices=FUTURE_TASKS, default=list(FUTURE_TASKS))
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    if len(args.tasks) != len(set(args.tasks)):
        parser.error('Tasks must be distinct')
    from medworld.config import load_config
    from medworld.gpu import acquire_gpu
    from .models import models
    cfg = load_config(args.config)
    spec = next(row for row in models() if row['id'] == args.model)
    if not cfg['future_enabled']:
        parser.error('Native Table 1 evaluation requires future_enabled=true')
    if Path(cfg['qwen']).resolve() != Path(spec['path']).resolve():
        parser.error('Native Table 1 uses the same local Qwen3.5-9B backbone as the slots arm')
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Choose an empty output directory')
    lock, device = acquire_gpu(args.gpu)
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from medworld.datasets import UnifiedData
        from medworld.datasets.protocol import _sha256
        from medworld.datasets.vqa import VOCABULARY
        from medworld.runtime import atomic_json, seed_all
        from medworld.evaluation.protocol import metric_protocol
        from medworld.evaluation.future_metrics import reference_fingerprint, score_future_task
        from medworld.evaluation.future_common import reference_for_row, scoring_protocol, load_radgraph
        seed_all(cfg['seed'])
        startup_started = time.monotonic()
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out / 'status.json', {'status': 'loading'})
        data = UnifiedData(cfg)
        future = data.future
        if future is None:
            raise ValueError('The configured dataset has no future-task references')
        processor = AutoProcessor.from_pretrained(spec['path'], local_files_only=True)
        # Check the whole fixed answer vocabulary before loading model weights.
        option_ids(processor.tokenizer, len(VOCABULARY))
        option_ids(processor.tokenizer, len(LOS_DAYS))
        yes_no_ids(processor.tokenizer)
        radgraph = load_radgraph(cfg) if 'future_report' in args.tasks else None
        model = AutoModelForImageTextToText.from_pretrained(
            spec['path'], local_files_only=True, dtype=torch.bfloat16,
            attn_implementation='sdpa').to(device).eval().requires_grad_(False)
        model_root = Path(spec['path'])
        summary = {'metric_protocol': metric_protocol('table1'),
                   'readout_protocol': native_readout_protocol(cfg),
                   'baseline': 'native_model_no_project_training', 'model_id': args.model,
                   'model_label': spec['label'], 'model': str(model_root),
                   'split': 'test', 'limit': args.limit, 'partial': args.limit is not None,
                   'data_fingerprint': data.fingerprint, 'future_data_fingerprint': future.fingerprint,
                   'tasks': {}, 'references': {}, 'predictions_sha256': {},
                   'timing': {'startup_seconds': time.monotonic() - startup_started, 'tasks': {}},
                   'weights_sha256': {file.name: _sha256(file) for file in sorted(model_root.glob('*.safetensors'))},
                   'source_sha256': _sha256(Path(__file__))}
        atomic_json(out / 'config.json', cfg)
        atomic_json(out / 'data_protocol.json', data.metadata)
        atomic_json(out / 'summary.json', summary)
        with torch.inference_mode():
            for task in args.tasks:
                rows = future.rows(task, 'test')
                count = min(len(rows), args.limit) if args.limit else len(rows)
                if not count:
                    raise ValueError(f'Empty future test cohort: {task}')
                references, predictions = [], {}
                inference_seconds = 0.
                with (out / f'{task}.jsonl').open('w', buffering=1) as journal:
                    for index in range(count):
                        # No label, target report or target image enters inference.
                        batch = future.batch(task, 'test', [index], source_only=True)
                        inference_started = time.monotonic()
                        prediction = predict_future(model, processor, task, batch, cfg, device)
                        inference_seconds += time.monotonic() - inference_started
                        reference = reference_for_row(task, rows[index], future)
                        references.append(reference)
                        predictions[reference['id']] = prediction
                        journal.write(json.dumps({**reference, 'prediction': prediction}, ensure_ascii=False) + '\n')
                        atomic_json(out / 'status.json', {'status': 'running', 'task': task,
                                                        'completed': index + 1, 'total': count})
                protocol = scoring_protocol(task, references, cfg, radgraph_scorer=radgraph)
                scoring_started = time.monotonic()
                summary['tasks'][task] = score_future_task(task, references, predictions,
                                                          protocol=protocol, radgraph_scorer=radgraph)
                summary['timing']['tasks'][task] = {'n': count, 'inference_seconds': inference_seconds,
                    'scoring_seconds': time.monotonic() - scoring_started,
                    'inference_seconds_per_example': inference_seconds / count}
                summary['references'][task] = {'n': count, 'references_sha256': reference_fingerprint(references)}
                summary['predictions_sha256'][task] = _sha256(out / f'{task}.jsonl')
                atomic_json(out / f'{task}_protocol.json', protocol)
                atomic_json(out / 'summary.json', summary)
        complete = all(summary['tasks'][task]['complete'] for task in args.tasks)
        atomic_json(out / 'status.json', {'status': 'complete' if complete or args.limit is not None else 'incomplete',
                                        'partial': args.limit is not None, 'scoring_complete': complete})
        if not complete and args.limit is None:
            raise ValueError('Native Table 1 scoring is incomplete; inspect per-task support and scorer status')
    except BaseException as error:
        if out.exists():
            from medworld.runtime import atomic_json
            atomic_json(out / 'status.json', {'status': 'failed', 'error': repr(error)})
        raise
    finally:
        if lock is not None:
            lock.close()


if __name__ == '__main__':
    main()
