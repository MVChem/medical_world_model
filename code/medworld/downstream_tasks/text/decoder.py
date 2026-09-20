"""One image-conditioned text decoder for VQA."""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from ... import STATE_WIDTH
from ..common import validate_state


def chunked_ce(states, targets, weight, chunk=32):
    valid = targets != -100
    states, targets = states[valid], targets[valid]
    if not len(targets):
        return states.sum() * 0
    def loss_fn(h, y):
        return F.cross_entropy(F.linear(h.to(weight.dtype), weight).float(), y, reduction="sum")
    total = states.new_zeros((), dtype=torch.float32)
    for start in range(0, len(targets), chunk):
        args = states[start:start + chunk], targets[start:start + chunk]
        total = total + (checkpoint(loss_fn, *args, use_reentrant=False) if torch.is_grad_enabled() else loss_fn(*args))
    return total / len(targets)


class TextDecoder(nn.Module):
    def __init__(self, language, output_head, tokenizer, cfg):
        super().__init__()
        self.language, self.output_head, self.tokenizer = language, output_head, tokenizer
        self.answer_tokens = cfg['answer_tokens']
        self.ce_chunk_tokens = cfg['ce_chunk_tokens']
        self.context_tokens = cfg['context_tokens']
        hidden = language.get_input_embeddings().weight.shape[1]
        self.image_projection = nn.Linear(cfg['decoder_width'], hidden)

    def prefix(self, images, questions):
        if images is None or images.ndim != 3 or len(images) != len(questions):
            raise ValueError('Text decoding requires image features and one question per image')
        embedding = self.language.get_input_embeddings()
        prompts = [self.tokenizer.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False,
                   add_generation_prompt=True, enable_thinking=False) for q in questions]
        tokens = self.tokenizer(prompts, padding=True, truncation=True, max_length=self.context_tokens,
                                return_tensors='pt', add_special_tokens=False)
        ids, mask = tokens['input_ids'].to(images.device), tokens['attention_mask'].to(images.device)
        visual = self.image_projection(images.float()).to(embedding.weight.dtype)
        return torch.cat([visual, embedding(ids)], 1), torch.cat([mask.new_ones(visual.shape[:2]), mask], 1)

    def targets(self, answers, device):
        rows = [self.tokenizer.encode(answer, add_special_tokens=False)[:self.answer_tokens - 1]
                + [self.tokenizer.eos_token_id] for answer in answers]
        ids = torch.full((len(rows), max(map(len, rows))), self.tokenizer.pad_token_id, device=device, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            ids[i, :len(row)] = torch.tensor(row, device=device)
            mask[i, :len(row)] = 1
        return ids, mask

    def loss(self, images, questions, answers):
        if len(answers) != len(images) or any(not a.strip() for a in answers):
            raise ValueError('One nonempty serialized answer is required per image')
        prefix, mask = self.prefix(images, questions)
        ids, valid = self.targets(answers, prefix.device)
        sequence = torch.cat([prefix, self.language.get_input_embeddings()(ids)], 1)
        mask = torch.cat([mask, valid], 1)
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        hidden = self.language(inputs_embeds=sequence, attention_mask=mask, position_ids=positions,
                               use_cache=False, return_dict=True).last_hidden_state
        return chunked_ce(hidden[:, prefix.shape[1] - 1:-1], ids.masked_fill(~valid.bool(), -100),
                          self.output_head.weight, self.ce_chunk_tokens)

    @torch.no_grad()
    def generate(self, images, questions, max_new_tokens=64):
        if self.training or type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError('Generation requires eval mode and a positive token budget')
        prefix, mask = self.prefix(images, questions)
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        output = self.language(inputs_embeds=prefix, attention_mask=mask, position_ids=positions,
                               use_cache=True, return_dict=True)
        ended = torch.zeros(len(prefix), device=prefix.device, dtype=torch.bool)
        generated = []
        for index in range(max_new_tokens):
            weight = self.output_head.weight
            token = F.linear(output.last_hidden_state[:, -1].to(weight.dtype), weight).argmax(-1)
            token = torch.where(ended, self.tokenizer.pad_token_id, token)
            generated.append(token)
            ended |= token == self.tokenizer.eos_token_id
            if ended.all() or index + 1 == max_new_tokens:
                break
            mask = torch.cat([mask, mask.new_ones((len(mask), 1))], 1)
            output = self.language(input_ids=token[:, None], attention_mask=mask,
                                   position_ids=(mask.sum(-1) - 1)[:, None],
                                   past_key_values=output.past_key_values, use_cache=True, return_dict=True)
        return self.tokenizer.batch_decode(torch.stack(generated, 1).cpu(), skip_special_tokens=True)
