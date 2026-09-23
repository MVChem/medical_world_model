"""Incremental inference for the existing byte Transformer, with identical weights.

Only inference changes: each layer caches causal self-attention keys/values and
the projected source memory. This avoids recomputing every report prefix while
retaining the TextDecoder training path and checkpoint parameter names.
"""

import torch
import torch.nn.functional as F

from ..common.decoder import sequence_positions


def _heads(value, attention):
    batch, length, width = value.shape
    return value.reshape(batch, length, attention.num_heads, width // attention.num_heads).transpose(1, 2)


def _projection(attention, value, index):
    if not attention._qkv_same_embed_dim:
        raise ValueError("Cached byte decoder requires equal query/key/value dimensions")
    width = attention.embed_dim
    weight = attention.in_proj_weight[index * width:(index + 1) * width]
    bias = None if attention.in_proj_bias is None else attention.in_proj_bias[index * width:(index + 1) * width]
    return _heads(F.linear(value, weight, bias), attention)


def _attend(attention, query, keys, values, valid):
    query = _projection(attention, query, 0)
    value = F.scaled_dot_product_attention(query, keys, values,
                                           attn_mask=valid[:, None, None, :], dropout_p=0)
    value = value.transpose(1, 2).reshape(query.shape[0], query.shape[2], attention.embed_dim)
    return attention.out_proj(value)


class ByteDecoderCache:
    """One generated sequence batch; call ``step`` once for each next input byte."""

    def __init__(self, decoder, features, questions, max_length):
        if decoder.training or type(max_length) is not int or max_length < 1:
            raise ValueError("Cached generation requires eval mode and a positive token budget")
        self.decoder = decoder
        memory, padding = decoder.prefix(features, questions)
        self.memory_valid = ~padding
        self.positions = sequence_positions(max_length, decoder.embedding.embedding_dim,
                                             decoder.embedding.weight)
        self.layers = []
        for layer in decoder.transformer.layers:
            if not layer.norm_first:
                raise ValueError("Cached byte decoding requires the configured pre-norm Transformer")
            self.layers.append({"memory_k": _projection(layer.multihead_attn, memory, 1),
                                "memory_v": _projection(layer.multihead_attn, memory, 2),
                                "self_k": None, "self_v": None})
        self.valid = torch.empty((len(features), 0), device=features.device, dtype=torch.bool)
        self.position = 0

    def step(self, ids):
        if ids.shape != (self.valid.shape[0], 1) or self.position >= self.positions.shape[1]:
            raise ValueError("Cache expects one input byte per row within its position budget")
        decoder = self.decoder
        values = decoder.embedding(ids) + self.positions[:, self.position:self.position + 1] + decoder.answer_type
        self.valid = torch.cat((self.valid, ids.ne(decoder.tokenizer.pad_token_id)), 1)
        for layer, state in zip(decoder.transformer.layers, self.layers):
            normalized = layer.norm1(values)
            keys = _projection(layer.self_attn, normalized, 1)
            vals = _projection(layer.self_attn, normalized, 2)
            state["self_k"] = keys if state["self_k"] is None else torch.cat((state["self_k"], keys), 2)
            state["self_v"] = vals if state["self_v"] is None else torch.cat((state["self_v"], vals), 2)
            values = values + _attend(layer.self_attn, normalized, state["self_k"], state["self_v"], self.valid)
            values = values + _attend(layer.multihead_attn, layer.norm2(values),
                                       state["memory_k"], state["memory_v"], self.memory_valid)
            values = values + layer.linear2(layer.activation(layer.linear1(layer.norm3(values))))
        self.position += 1
        if decoder.transformer.norm is not None:
            values = decoder.transformer.norm(values)
        return decoder.output_head(values)


@torch.no_grad()
def generate_cached(decoder, features, questions, max_new_tokens=None):
    max_new_tokens = decoder.generation_tokens if max_new_tokens is None else max_new_tokens
    cache = ByteDecoderCache(decoder, features, questions, max_new_tokens)
    ids = torch.full((len(features), 1), decoder.tokenizer.bos_token_id,
                     device=features.device, dtype=torch.long)
    ended = torch.zeros(len(features), device=features.device, dtype=torch.bool)
    generated = []
    for _ in range(max_new_tokens):
        logits = cache.step(ids)[:, -1].clone()
        logits[:, [decoder.tokenizer.pad_token_id, decoder.tokenizer.bos_token_id]] = -torch.inf
        token = logits.argmax(-1).masked_fill(ended, decoder.tokenizer.pad_token_id)
        generated.append(token)
        ended |= token.eq(decoder.tokenizer.eos_token_id)
        if ended.all():
            break
        ids = token[:, None]
    return decoder.tokenizer.batch_decode(torch.stack(generated, 1).cpu())
