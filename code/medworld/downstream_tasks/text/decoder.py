"""Autoregressive task-owned UTF-8 answer decoder, without a pretrained VLM.

Question and answer limits count byte tokens (including EOS), not Qwen tokens.
``generation_tokens`` counts new tokens including EOS. A byte budget may retain
fewer characters for non-ASCII languages. Answers preserve their serialization.
"""
import torch
from torch import nn
import torch.nn.functional as F

from ..common.decoder import ByteTokenizer, sequence_positions


class TextDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width = cfg["decoder_width"]
        self.context_tokens = cfg["context_tokens"]
        self.answer_tokens = cfg["answer_tokens"]
        self.generation_tokens = cfg["generation_tokens"]
        self.tokenizer = ByteTokenizer()
        self.embedding = nn.Embedding(self.tokenizer.vocab_size, width,
                                      padding_idx=self.tokenizer.pad_token_id)
        self.feature_projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
        self.question_type = nn.Parameter(torch.zeros(1, 1, width))
        self.answer_type = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerDecoderLayer(width, 8, width * 4, dropout=0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerDecoder(layer, cfg["decoder_depth"], norm=nn.LayerNorm(width))
        self.output_head = nn.Linear(width, self.tokenizer.vocab_size)

    def prefix(self, features, questions):
        if (not isinstance(features, torch.Tensor) or features.ndim != 3
                or features.shape[0] < 1 or features.shape[1] < 1
                or isinstance(questions, str) or len(features) != len(questions)
                or any(not isinstance(question, str) for question in questions)):
            raise ValueError("Text decoding requires task features and one question string per example")
        ids, valid = self.tokenizer.batch(questions, self.context_tokens, features.device)
        questions = self.embedding(ids)
        questions = questions + sequence_positions(questions.shape[1], questions.shape[2], questions) + self.question_type
        images = self.feature_projection(features.to(self.embedding.weight.dtype))
        memory = torch.cat((images, questions), 1)
        padding = torch.cat((valid.new_zeros(images.shape[:2]), ~valid), 1)
        return memory, padding

    def targets(self, answers, device):
        if not answers or any(not isinstance(answer, str) or not answer.strip() for answer in answers):
            raise ValueError("One nonempty serialized answer is required per example")
        return self.tokenizer.batch(answers, self.answer_tokens, device)

    def decode_logits(self, memory, memory_padding, decoder_ids):
        """Predict next bytes; position i cannot see any decoder input after i."""
        embedded = self.embedding(decoder_ids)
        embedded = embedded + sequence_positions(embedded.shape[1], embedded.shape[2], embedded) + self.answer_type
        causal = torch.ones((decoder_ids.shape[1], decoder_ids.shape[1]),
                            device=decoder_ids.device, dtype=torch.bool).triu(1)
        decoded = self.transformer(embedded, memory, tgt_mask=causal,
                                   tgt_key_padding_mask=decoder_ids.eq(self.tokenizer.pad_token_id),
                                   memory_key_padding_mask=memory_padding)
        return self.output_head(decoded)

    def logits(self, features, questions, decoder_ids):
        memory, padding = self.prefix(features, questions)
        return self.decode_logits(memory, padding, decoder_ids)

    def loss(self, features, questions, serialized_answers):
        memory, padding = self.prefix(features, questions)
        if len(serialized_answers) != len(features):
            raise ValueError("Provide one serialized answer per example")
        ids, valid = self.targets(serialized_answers, memory.device)
        start = ids.new_full((len(ids), 1), self.tokenizer.bos_token_id)
        inputs = torch.cat((start, ids[:, :-1]), 1)
        logits = self.decode_logits(memory, padding, inputs)
        return F.cross_entropy(logits.float().flatten(0, 1), ids.masked_fill(~valid, -100).flatten())

    @torch.no_grad()
    def generate(self, features, questions, max_new_tokens=None):
        max_new_tokens = self.generation_tokens if max_new_tokens is None else max_new_tokens
        if self.training or type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("Generation requires eval mode and a positive token budget")
        memory, padding = self.prefix(features, questions)
        inputs = torch.full((len(features), 1), self.tokenizer.bos_token_id,
                            device=features.device, dtype=torch.long)
        ended = torch.zeros(len(features), device=features.device, dtype=torch.bool)
        generated = []
        for _ in range(max_new_tokens):
            logits = self.decode_logits(memory, padding, inputs)[:, -1].clone()
            logits[:, [self.tokenizer.pad_token_id, self.tokenizer.bos_token_id]] = -torch.inf
            token = logits.argmax(-1).masked_fill(ended, self.tokenizer.pad_token_id)
            generated.append(token)
            ended |= token.eq(self.tokenizer.eos_token_id)
            if ended.all():
                break
            inputs = torch.cat((inputs, token[:, None]), 1)
        return self.tokenizer.batch_decode(torch.stack(generated, 1).cpu())
