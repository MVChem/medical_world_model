"""GPU correctness/stress checks, never clinical evaluation or initialization."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch

from data import Corpus, source_view
from model import NativeForecast
from run import atomic_json, autocast, memory, optimizer_for, seed_all, target_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    cfg = json.loads(Path(args.config).read_text())
    cfg['generation_tokens'] = 8
    seed_all(cfg['seed'])
    model = NativeForecast(cfg, 'slots')
    corpus = Corpus(cfg, model.tokenizer)
    batch = corpus.batch(corpus.pairs['train'][:args.batch_size])
    model.eval()
    # At no slots, manually assembled inputs must retain the exact native VLM
    # computation, including image features and all multimodal position indices.
    with torch.no_grad(), autocast('cuda'):
        inputs = model.source_inputs(source_view(batch))
        expected = model.decoder.model(**inputs, use_cache=False).last_hidden_state
        prefix = model.native_prefix(inputs)
        actual = model.decoder.model.language_model(inputs_embeds=prefix['embeds'], attention_mask=prefix['mask'],
                 position_ids=prefix['positions'], use_cache=False, return_dict=True).last_hidden_state
        difference = float((expected - actual).float().abs().max())
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        tokens = model.decoder.generate(**inputs, max_new_tokens=8, do_sample=False, use_cache=True,
                        eos_token_id=model.tokenizer.eos_token_id, pad_token_id=model.tokenizer.pad_token_id)
        reference = model.tokenizer.batch_decode(tokens[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        encoder = model.encoder
        model.encoder = None  # exercise the identical native control generation path
        native_reports, _ = model.predict(source_view(batch))
        model.encoder = encoder
        if native_reports != reference:
            raise AssertionError('native greedy generation differs from Qwen.generate')
    del expected, actual, prefix, tokens, inputs
    gc.collect()
    torch.cuda.empty_cache()
    model.begin_stage2()
    target_initial = target_digest(model)
    model.train()
    # Independently prove report CE reaches both slot groups and predictor:
    # a latent loss alone would not establish decoder use of future slots.
    with autocast('cuda'):
        _, parts = model.losses(batch, 2)
    parts['text'].backward()
    gradients = model.gradient_audit(2)
    slot_gradient = model.encoder.slot_queries.grad.detach().float().norm(dim=-1).tolist()
    assert all(value > 0 for value in slot_gradient)
    model.zero_grad(set_to_none=True)
    del parts
    gc.collect()
    torch.cuda.empty_cache()
    # Keep actual pixels/features. Repeat benign existing text tokens to exercise
    # the complete allowed budgets; these synthetic lengths are not scored data.
    source_max = cfg['report_tokens'] + cfg['ehr_tokens'] + 32
    for side in ('source', 'target'):
        ids = batch[side+'_ids']
        repeats = (source_max + ids.shape[1]-1) // ids.shape[1]
        batch[side+'_ids'] = ids.repeat(1, repeats)[:, :source_max]
        batch[side+'_mask'] = torch.ones_like(batch[side+'_ids'])
        ids = batch[side+'_target_ids']
        maximum = cfg['report_tokens'] + 1
        repeats = (maximum + ids.shape[1]-1) // ids.shape[1]
        batch[side+'_target_ids'] = ids.repeat(1, repeats)[:, :maximum]
        batch[side+'_target_ids'][:, -1] = model.tokenizer.eos_token_id
        batch[side+'_target_mask'] = torch.ones_like(batch[side+'_target_ids'])
    optimizer = optimizer_for(model, cfg)
    torch.cuda.reset_peak_memory_stats()
    timings = []
    # Include Adam states plus gradient accumulation and both training phases.
    for stage in (1, 2):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(2):
            with autocast('cuda'):
                loss, parts = model.losses(batch, stage)
            (loss/2).backward()
        model.gradient_audit(stage)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        timings.append(dict(stage=stage, microbatches=2, seconds=time.monotonic()-started,
                            loss=float(loss.detach()), memory=memory('cuda')))
        del loss, parts
    assert target_digest(model) == target_initial
    result = dict(passed=True, smoke_only=True, table1_ready=False,
                  native_hidden_exact=True, native_hidden_max_error=difference,
                  native_greedy_generation_equal=True, report_only_gradient_norms=gradients,
                  report_only_eight_query_gradient_norms=slot_gradient,
                  current_and_future_context_tokens=source_max, report_target_tokens=cfg['report_tokens']+1,
                  native_image_pixels=cfg.get('native_image_pixels',512), microbatch=args.batch_size,
                  stress_uses_repeated_text_only=True, fixed_target_unchanged=True, stages=timings)
    atomic_json(out, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
