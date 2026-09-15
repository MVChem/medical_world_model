"""Diagnostic GPU check of observation -> slots -> text and predicted slots -> text.

Uses two validation source observations and a read-only checkpoint. Outputs are
local research artifacts, never Table 1/2 metrics or proof of report accuracy.
"""
import argparse
import io
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from codec import decode_reports
from data import Corpus
from launch_queue import ALLOWED_GPUS, acquire_idle
from model import NativeForecast
from run import atomic_json, autocast, check_runtime, digest, memory, runtime_signature


@torch.no_grad()
def uncached_reference(model, state, limit):
    """Independent full-sequence greedy decoding of the training prefix."""
    sequence = model.state_only_prefix(state)['embeds']
    ended = torch.zeros(len(state), dtype=torch.bool, device=model.device)
    tokens = []
    for _ in range(limit):
        mask = torch.ones(sequence.shape[:2], device=model.device, dtype=torch.long)
        positions = torch.arange(sequence.shape[1], device=model.device)[None, None].expand(3, len(state), -1)
        out = model.decoder.model.language_model(inputs_embeds=sequence, attention_mask=mask,
                    position_ids=positions, use_cache=False, return_dict=True)
        weight = model.decoder.lm_head.weight
        token = F.linear(out.last_hidden_state[:, -1].to(weight.dtype), weight).argmax(-1)
        token = torch.where(ended, model.tokenizer.pad_token_id, token)
        tokens.append(token)
        ended |= token == model.tokenizer.eos_token_id
        if ended.all():
            break
        sequence = torch.cat([sequence, model.decoder.model.get_input_embeddings()(token[:, None])], 1)
    return model.tokenizer.batch_decode(torch.stack(tokens, 1), skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--gpus', nargs='+', type=int, default=[3, 4, 7])
    parser.add_argument('--max-new-tokens', type=int, default=64)
    args = parser.parse_args()
    if not set(args.gpus) <= ALLOWED_GPUS or args.max_new_tokens < 1:
        parser.error('Use eligible GPUs and a positive token budget')
    os.umask(0o077)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    allocation = acquire_idle(args.gpus)
    if allocation is None:
        raise RuntimeError('No eligible idle GPU for the diagnostic')
    gpu, uuid, lock = allocation
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        torch.set_num_threads(4)
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        cfg = checkpoint['config']
        compatibility_note = check_runtime(checkpoint, runtime_signature(cfg), smoke=bool(checkpoint['smoke_only']))
        model = NativeForecast(cfg, checkpoint['condition'])
        model.load_compact(checkpoint['model'])
        model.eval()
        corpus = Corpus(cfg, model.tokenizer)
        rows = [{k: v for k, v in row.items() if k != 'target'} for row in corpus.pairs['validate'][:2]]
        batch = corpus.batch(rows, source_only=True)
        with torch.no_grad(), autocast('cuda'):
            current = model.state(batch, image_only=True)
            future = model.world(model.state(batch), batch['horizon'])
        # From this point the decoder receives only tensors. Forbid accidental
        # re-encoding or native-image evidence on either report readout path.
        def forbidden(*args, **kwargs):
            raise AssertionError('State-only decoding attempted to read observations')
        model.state = forbidden
        model.source_inputs = forbidden
        model.native_prefix = forbidden
        results = {}
        with torch.inference_mode(), autocast('cuda'):
            for name, state in [('current', current), ('predicted_future', future)]:
                reference_limit = min(12, args.max_new_tokens)
                short = decode_reports(model, state, max_new_tokens=reference_limit)
                reference = uncached_reference(model, state, reference_limit)
                # BF16 cached and full recomputation may choose different tokens
                # near a tie. The supported reference is Qwen's cached generate.
                prefix = model.state_only_prefix(state)
                model.decoder.model.rope_deltas = None
                official_ids = model.decoder.generate(inputs_embeds=prefix['embeds'],
                    attention_mask=prefix['mask'], max_new_tokens=reference_limit,
                    do_sample=False, use_cache=True, eos_token_id=model.tokenizer.eos_token_id,
                    pad_token_id=model.tokenizer.pad_token_id)
                official = model.tokenizer.batch_decode(official_ids, skip_special_tokens=True)
                if short != official:
                    raise AssertionError(f'{name}: state decoder differs from Qwen.generate')
                # Raw tensor serialization is lossless; text generation is not.
                buffer = io.BytesIO()
                torch.save(state.cpu(), buffer)
                buffer.seek(0)
                restored = torch.load(buffer, weights_only=True)
                torch.testing.assert_close(restored, state.cpu(), rtol=0, atol=0)
                reports = decode_reports(model, restored, max_new_tokens=args.max_new_tokens)
                repeat = decode_reports(model, state, max_new_tokens=args.max_new_tokens)
                if reports != repeat:
                    raise AssertionError(f'{name}: reports differ after tensor save/reload')
                results[name] = dict(shape=list(state.shape), reports=reports,
                    empty_reports=sum(not report.strip() for report in reports),
                    qwen_generate_equal=True, cached_uncached_equal=short == reference,
                    tensor_roundtrip_exact=True,
                    reports_equal_after_tensor_reload=True)
        result = dict(passed=True, diagnostic_only=True, table1_ready=False, table2_ready=False,
            checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=digest(args.checkpoint),
            checkpoint_stage=checkpoint['stage'], checkpoint_smoke_only=checkpoint['smoke_only'],
            runtime_compatibility_note=compatibility_note,
            qwen=cfg['qwen'], gpu=gpu, gpu_uuid=uuid, decoder_reads_observations=False,
            max_new_tokens=args.max_new_tokens, teacher_forcing=False, clinical_quality_evaluated=False,
            results=results, memory=memory('cuda'))
        out.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(out, result)
        print(json.dumps(dict(passed=True, out=str(out), empty_reports={k: v['empty_reports'] for k, v in results.items()})))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
