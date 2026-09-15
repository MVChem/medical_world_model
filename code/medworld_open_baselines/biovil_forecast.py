"""Matched forecast adaptation using frozen public BioViL-T/CheXWorld features.

Uses the existing Table-1 trainer and evaluator without modifying either.
The new visual adapter is reset; all other Stage-1 tensors are transferred.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
TABLE1 = ROOT.parent / 'medworld_table1'
sys.path.insert(0, str(TABLE1))
import torch
from torch import nn
from common import atomic_json, digest, read_config
import ablation_model


class RepresentationForecast(ablation_model.AblationMedWorld):
    def __init__(self, cfg):
        super().__init__(cfg)
        width = self.encoder.visual_position.shape[-1]
        dimension = cfg['visual_feature_dim']
        self.encoder.adapter = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, width),
                                             nn.GELU(), nn.Linear(width, width))
        self.register_buffer('adapted_visual_encoder', torch.tensor(1))

    def load_compact(self, state):
        if 'adapted_visual_encoder' in state:
            return super().load_compact(state)
        # Only an original Stage-1 initialization may omit the adaptation marker.
        if any(k.startswith('target_encoder.') for k in state):
            raise ValueError('Cannot silently adapt an existing Stage-2 checkpoint')
        omitted = [k for k in state if k.startswith('encoder.adapter.')]
        state = {k: v for k, v in state.items() if k not in omitted}
        missing, unexpected = self.load_state_dict(state, strict=False)
        bad = [k for k in missing if not k.startswith('encoder.adapter.') and
               k != 'adapted_visual_encoder' and ('lora_' in k or '.backbone.' not in k)]
        if bad or unexpected:
            raise ValueError((bad, unexpected))


def prepare(args):
    cfg = read_config(args.base_config)
    source = Path(cfg['cache'])
    output = Path(args.output).resolve()
    cache = output / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    for name in ('observations.jsonl', 'train.jsonl', 'validate.jsonl', 'test.jsonl', 'manifest.json'):
        destination = cache / name
        if destination.is_symlink():
            if destination.resolve() != (source / name).resolve():
                raise ValueError(f'Existing cohort link differs: {destination}')
        elif destination.exists():
            raise FileExistsError(destination)
        else:
            destination.symlink_to(source / name)
    cfg.update(cache=str(cache), representation_encoder=args.encoder,
               visual_feature_dim=512 if args.encoder == 'biovil_t' else 768,
               state_condition='slots', max_stage2_steps=args.steps,
               batch_size=args.batch_size, gradient_accumulation=1,
               train_deadline=None, total_hours=1e6)
    cfg.pop('feature_reuse_cache', None)
    atomic_json(output / 'config.json', cfg)
    atomic_json(output / 'adaptation_protocol.json', dict(
        method=args.encoder + ' + matched fusion/predictor/readouts (adapted)',
        source_cohort=str(source), split_sha256={s:digest(source / (s+'.jsonl')) for s in ('train','validate','test')},
        pretrained_visual_encoder_frozen=True, native_feature_dimension=cfg['visual_feature_dim'],
        stage1_checkpoint=cfg.get('initialize_stage1'),
        visual_adapter='all parameters reinitialized; native input width; remaining transferred parameters identical',
        optimizer_updates=args.steps, batch_size=args.batch_size, gradient_accumulation=1,
        training_example_presentations=args.steps*args.batch_size,
        compute_matching='optimizer updates and effective batch matched; wall time/GPU hours measured separately',
        initial_training='shared V-JEPA-derived Stage-1 warm start; no additional encoder-specific Stage-1 updates',
        source_input='current image, current report, source-time EHR, horizon; no future evidence',
        label='encoder-swap adaptation, not an original published forecasting method'))
    print(output / 'config.json')


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--base-config', default=str(TABLE1/'configs/overnight_20260913_slots.json'))
    prep.add_argument('--encoder', choices=['biovil_t','chexworld'], required=True)
    prep.add_argument('--output', required=True)
    prep.add_argument('--steps', type=int, default=2400)
    prep.add_argument('--batch-size', type=int, default=32)
    train = sub.add_parser('train')
    train.add_argument('--config', required=True)
    train.add_argument('--run', required=True)
    train.add_argument('--resume')
    train.add_argument('--smoke-steps', type=int, default=0)
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('--config', required=True)
    evaluate.add_argument('--checkpoint', required=True)
    evaluate.add_argument('--out', required=True)
    evaluate.add_argument('--split', choices=['validate','test'], default='test')
    evaluate.add_argument('--limit', type=int, default=0)
    evaluate.add_argument('--score-only', action='store_true')
    evaluate.add_argument('--skip-radgraph', action='store_true')
    green = sub.add_parser('green')
    green.add_argument('--config', required=True)
    green.add_argument('--predictions', required=True)
    green.add_argument('--out', required=True)
    args = p.parse_args()
    if args.command == 'prepare':
        return prepare(args)
    cfg = read_config(args.config)
    if args.command == 'green':
        import os
        from types import SimpleNamespace
        from common import load_rows, write_rows
        output = Path(args.out).resolve()
        name = cfg['representation_encoder'] + '_adapted'
        observations = {r['id']:r for r in load_rows(Path(cfg['cache'])/'observations.jsonl')}
        pairs = load_rows(Path(cfg['cache'])/'test.jsonl')
        predictions = load_rows(args.predictions)
        if [r['id'] for r in predictions] != [r['id'] for r in pairs]:
            raise ValueError('GREEN prediction IDs/order differ from fixed Table-1 cohort')
        (output/'cohort').mkdir(parents=True, exist_ok=True)
        (output/name/'test').mkdir(parents=True, exist_ok=True)
        write_rows(output/'cohort/table1_references_test.jsonl',
                   [dict(id=r['id'], target_report=observations[r['target']]['report']) for r in pairs])
        write_rows(output/name/'test/responses.jsonl',
                   [dict(key='table1_report|'+r['id']+'|',ok=True,text=r['report']) for r in predictions])
        atomic_json(output/name/'inference_finished.json',dict(status='complete'))
        atomic_json(output/'models.json',[dict(id=name)])
        sys.path.insert(0, str(ROOT.parent/'medworld_baselines'))
        import green_eval
        return green_eval.main(SimpleNamespace(run=output,gpu=os.environ['CUDA_VISIBLE_DEVICES'],prepare=False))
    metadata = json.loads((Path(cfg['cache'])/'features.json').read_text())
    if metadata['encoder'] != cfg['representation_encoder'] or metadata['shape'][-1] != cfg['visual_feature_dim'] or metadata.get('partial'):
        raise ValueError('Incomplete or wrong pretrained feature cache')
    ablation_model.build_model = RepresentationForecast
    torch.set_num_threads(4)
    if args.command == 'train':
        import train as shared_train
        def build_and_record(configuration):
            model = RepresentationForecast(configuration)
            atomic_json(Path(args.run)/'model_size.json', dict(
                representation_encoder=configuration['representation_encoder'],
                visual_feature_dim=configuration['visual_feature_dim'],
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                trainable_by_module={name:sum(p.numel() for p in module.parameters() if p.requires_grad)
                                     for name,module in model.named_children()},
                frozen_visual_encoder_parameters=27349928 if configuration['representation_encoder']=='biovil_t' else 85797120,
                visual_encoder_included_in_model_object=False))
            return model
        def record_initialization(path, value):
            if Path(path).name == 'initialization.json':
                value = dict(value, reset_parameters=['encoder.adapter.*'],
                             representation_encoder=cfg['representation_encoder'],
                             visual_checkpoint_sha256=metadata['checkpoint_sha256'],
                             transfer_policy='Shared Stage-1 tensors except all visual adapter parameters')
            return atomic_json(path, value)
        shared_train.build_model = build_and_record
        shared_train.atomic_json = record_initialization
        args.mode, args.follow = cfg['representation_encoder'] + '_adapted', None
        return shared_train.train(args, cfg)
    import evaluate as shared_evaluate
    args.mode = cfg['representation_encoder'] + '_adapted'
    # predict() is also called directly here, outside shared_evaluate's CLI,
    # so this adapter must perform the CLI's output-directory initialization.
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if not args.score_only:
        shared_evaluate.predict(args, cfg)
    # RadGraph's legacy AllenNLP/tokenizer code needs Transformers 4; Qwen 3.5
    # generation has already imported Transformers 5 in this process.
    command = [sys.executable, str(ROOT/'biovil_score.py'), '--config', args.config,
               '--out', args.out, '--mode', args.mode, '--split', args.split,
               '--limit', str(args.limit)]
    if args.skip_radgraph:
        command.append('--skip-radgraph')
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
