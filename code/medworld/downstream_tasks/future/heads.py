"""Identical future-task heads for raw-input and slot-conditioned arms.

Time is a requested horizon, never a future outcome. Question bytes preserve
the region/finding wording of progression questions. Reports use a separate
task-owned byte decoder; no pretrained language decoder is shared or required.
"""

import torch
from torch import nn
import torch.nn.functional as F

from ..common.decoder import ByteTokenizer, sequence_positions
from ..text import TextDecoder
from ...datasets.vqa import VOCABULARY
from ...predictor import time_features
from .generation import generate_cached


FUTURE_TASKS = ("future_vqa", "progression", "future_report", "mortality_30d", "remaining_los")
REPORT_QUESTION = "Generate the full report for the future chest radiograph."


class FutureHeads(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width = cfg["decoder_width"]
        self.context_tokens = cfg["context_tokens"]
        self.report_tokens = cfg.get("future_report_tokens", 2048)
        self.vocabulary = VOCABULARY
        self.answer_to_id = {answer: i for i, answer in enumerate(self.vocabulary)}
        self.tokenizer = ByteTokenizer()
        self.time = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        self.question_embedding = nn.Embedding(self.tokenizer.vocab_size, width,
                                               padding_idx=self.tokenizer.pad_token_id)
        self.question_type = nn.Parameter(torch.zeros(1, 1, width))
        self.queries = nn.Parameter(torch.randn(4, 1, width) * .02)
        layer = nn.TransformerEncoderLayer(width, 8, width * 4, dropout=0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.readout = nn.TransformerEncoder(layer, 1, norm=nn.LayerNorm(width),
                                             enable_nested_tensor=False)
        self.vqa = nn.Linear(width, len(self.vocabulary))
        self.direction = nn.Linear(width, 3)
        self.mortality = nn.Linear(width, 1)
        self.los = nn.Linear(width, 1)
        self.report = TextDecoder({**cfg, "answer_tokens": self.report_tokens,
                                  "generation_tokens": self.report_tokens})

    def context(self, features, hours):
        if not isinstance(features, torch.Tensor) or features.ndim != 3 or not len(features):
            raise ValueError("Future heads require nonempty [B,N,D] task features")
        if not isinstance(hours, torch.Tensor) or hours.shape != (len(features),):
            raise ValueError("One requested future horizon is required per source")
        if not hours.is_floating_point() or not torch.isfinite(hours).all() or (hours <= 0).any():
            raise ValueError("Future horizons must be finite positive hours")
        horizon = self.time(time_features(hours.to(features.device)))[:, None]
        return torch.cat((features, horizon.to(features.dtype)), 1)

    def read(self, task, features, hours, questions=None):
        indices = {"future_vqa": 0, "progression": 1, "mortality_30d": 2, "remaining_los": 3}
        if task not in indices:
            raise ValueError(f"Unsupported categorical/outcome future task: {task}")
        memory = self.context(features, hours)
        query = self.queries[indices[task]][None].expand(len(features), -1, -1)
        memory = torch.cat((query.to(memory.dtype), memory), 1)
        padding = torch.zeros(memory.shape[:2], device=memory.device, dtype=torch.bool)
        if task in ("future_vqa", "progression"):
            if (not isinstance(questions, (list, tuple)) or len(questions) != len(features)
                    or any(not isinstance(question, str) or not question.strip() for question in questions)):
                raise ValueError("Future VQA/progression requires one nonempty question per source")
            ids, valid = self.tokenizer.batch(questions, self.context_tokens, memory.device)
            text = self.question_embedding(ids)
            text = text + sequence_positions(text.shape[1], text.shape[2], text) + self.question_type
            memory = torch.cat((memory, text.to(memory.dtype)), 1)
            padding = torch.cat((padding, ~valid), 1)
        return self.readout(memory, src_key_padding_mask=padding)[:, 0]

    def logits(self, task, features, hours, questions=None):
        value = self.read(task, features, hours, questions)
        if task == "future_vqa":
            return self.vqa(value)
        if task == "progression":
            return self.direction(value)
        if task == "mortality_30d":
            return self.mortality(value).squeeze(-1)
        return self.los(value).squeeze(-1)

    def loss(self, task, features, batch):
        hours = batch["delta_hours"]
        if task == "future_report":
            return self.report.loss(self.context(features, hours),
                                    [REPORT_QUESTION] * len(features), batch["answers"])
        logits = self.logits(task, features, hours, batch.get("questions"))
        if task == "future_vqa":
            answers = batch["answers"]
            if (not isinstance(answers, (list, tuple)) or len(answers) != len(features)
                    or any(not isinstance(answer, str) or answer not in self.answer_to_id for answer in answers)):
                raise ValueError("Future VQA requires one official categorical answer label per source")
            labels = torch.tensor([self.answer_to_id[answer] for answer in answers], device=logits.device)
            return F.cross_entropy(logits.float(), labels)
        labels = batch["labels"]
        if (not isinstance(labels, torch.Tensor) or labels.shape != (len(features),)
                or not torch.isfinite(labels).all()):
            raise ValueError("Future labels must be a finite [B] tensor")
        labels = labels.to(logits.device)
        if task == "progression":
            if labels.is_floating_point() or labels.dtype == torch.bool or not ((labels >= 0) & (labels <= 2)).all():
                raise ValueError("Progression labels must be integer improved/stable/worsened IDs 0/1/2")
            return F.cross_entropy(logits.float(), labels.long())
        if task == "mortality_30d":
            if not ((labels == 0) | (labels == 1)).all():
                raise ValueError("30-day mortality labels must be binary")
            return F.binary_cross_entropy_with_logits(logits.float(), labels.float())
        if (labels < 0).any():
            raise ValueError("Remaining LOS targets must be nonnegative days")
        days = F.softplus(logits.float())
        return F.smooth_l1_loss(torch.log1p(days), torch.log1p(labels.float()))

    @torch.no_grad()
    def predict(self, task, features, batch):
        if task == "future_report":
            reports = generate_cached(self.report, self.context(features, batch["delta_hours"]),
                                      [REPORT_QUESTION] * len(features), self.report_tokens)
            # Reserve the same EOS slot as training and native Qwen. Decoding
            # invalid byte sequences can expand them into UTF-8 replacements.
            return [report.encode("utf-8")[:self.report_tokens - 1].decode("utf-8", errors="ignore")
                    for report in reports]
        logits = self.logits(task, features, batch["delta_hours"], batch.get("questions"))
        if task == "future_vqa":
            return [self.vocabulary[index] for index in logits.argmax(-1).tolist()]
        if task == "progression":
            return logits.argmax(-1)
        if task == "mortality_30d":
            return logits.float().sigmoid()
        return F.softplus(logits.float())
