"""Decode this checkpoint's current or predicted 4+4 state into report text.

The input is a learned state, not token IDs. Use states and the decoder from the
same NativeForecast checkpoint: tensor dimensions alone do not define a shared
latent space across experiments. This is the state-only Stage-1 readout; Table 1
uses NativeForecast.predict with the additional native current-evidence path.
"""
from numbers import Integral

import torch
import torch.nn.functional as F


@torch.no_grad()
def decode_reports(model, state, *, max_new_tokens=None):
    """Return one report per [B, 8, 1024] state, without images or target text.

    Call ``model.eval()`` first. On CUDA, use the same autocast context as the
    forecasting runner. Floating-point CPU states are accepted and moved to the
    decoder device. Generation uses the exact state-only prefix used in training.
    No claim of lossless text reconstruction or clinical accuracy is implied.
    """
    if model.training:
        raise ValueError('Call model.eval() before decoding states')
    if getattr(model, 'encoder', None) is None or not hasattr(model, 'slot_projection'):
        raise ValueError('State decoding requires a new 4+4 slots checkpoint')
    if not isinstance(state, torch.Tensor) or state.ndim != 3 or state.shape[0] < 1 or tuple(state.shape[1:]) != (8, 1024):
        raise ValueError('Expected a nonempty [B, 8, 1024] state tensor')
    if not state.is_floating_point() or not torch.isfinite(state).all():
        raise ValueError('State values must be finite floating-point numbers')
    limit = model.cfg['generation_tokens'] if max_new_tokens is None else max_new_tokens
    if isinstance(limit, bool) or not isinstance(limit, Integral) or limit <= 0:
        raise ValueError('max_new_tokens must be a positive integer')
    eos, pad = model.tokenizer.eos_token_id, model.tokenizer.pad_token_id
    if eos is None or pad is None:
        raise ValueError('Report tokenizer must define EOS and padding tokens')
    prefix = model.state_only_prefix(state.to(model.device))
    language, weight = model.decoder.model.language_model, model.decoder.lm_head.weight
    output = language(inputs_embeds=prefix['embeds'], attention_mask=prefix['mask'],
                      position_ids=prefix['positions'], use_cache=True, return_dict=True)
    mask, cache = prefix['mask'], output.past_key_values
    ended = torch.zeros(len(state), dtype=torch.bool, device=model.device)
    generated = []
    for index in range(limit):
        token = F.linear(output.last_hidden_state[:, -1].to(weight.dtype), weight).argmax(-1)
        token = torch.where(ended, pad, token)
        generated.append(token)
        ended |= token == eos
        if ended.all() or index + 1 == limit:
            break
        mask = torch.cat([mask, mask.new_ones((len(mask), 1))], 1)
        position = mask.sum(-1, keepdim=True) - 1 + prefix['delta']
        output = language(input_ids=token[:, None], attention_mask=mask,
                          position_ids=position[None].expand(3, -1, -1), past_key_values=cache,
                          use_cache=True, return_dict=True)
        cache = output.past_key_values
    return model.tokenizer.batch_decode(torch.stack(generated, 1), skip_special_tokens=True)
