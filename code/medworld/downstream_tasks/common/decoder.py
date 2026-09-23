"""Task-owned pixel/report input path, independent of the slot-producing VLM.

Text uses UTF-8 bytes, not a pretrained vocabulary. ``context_tokens`` caps the
number of byte tokens including a final EOS; padding is excluded from attention.
Pixels are RGB in [0, 1], aspect-preserving resized and centered on a black square.
"""
import math

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn

from . import positional_encoding, validate_state
from ... import STATE_SLOTS, STATE_WIDTH


class ByteTokenizer:
    """Small reversible tokenizer with no weights, files, or external assets."""

    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2
    vocab_size = 259

    def encode(self, text, max_bytes=None):
        if not isinstance(text, str):
            raise ValueError("Byte tokenization requires text")
        raw = text.encode("utf-8")
        if max_bytes is not None:
            if type(max_bytes) is not int or max_bytes < 0:
                raise ValueError("max_bytes must be a nonnegative integer")
            # Never create an incomplete trailing UTF-8 code point in a target.
            raw = raw[:max_bytes].decode("utf-8", errors="ignore").encode("utf-8")
        return [byte + 3 for byte in raw]

    def batch(self, texts, max_length, device=None):
        if type(max_length) is not int or max_length < 1 or not texts:
            raise ValueError("A nonempty text batch and positive length are required")
        rows = [self.encode(text, max_length - 1) + [self.eos_token_id] for text in texts]
        ids = torch.full((len(rows), max(map(len, rows))), self.pad_token_id,
                         device=device, dtype=torch.long)
        valid = torch.zeros_like(ids, dtype=torch.bool)
        for index, row in enumerate(rows):
            ids[index, :len(row)] = torch.tensor(row, device=device)
            valid[index, :len(row)] = True
        return ids, valid

    def decode(self, ids):
        raw = []
        for token in ids:
            token = int(token)
            if token == self.eos_token_id:
                break
            if token in (self.pad_token_id, self.bos_token_id):
                continue
            if not 3 <= token < self.vocab_size:
                raise ValueError("Token outside the byte vocabulary")
            raw.append(token - 3)
        # A freshly trained generator may emit invalid UTF-8; retain its errors.
        return bytes(raw).decode("utf-8", errors="replace")

    def batch_decode(self, ids, **_):
        return [self.decode(row) for row in ids]


def sequence_positions(length, width, reference):
    """Deterministic positions without a learned maximum sequence length."""
    frequency = torch.exp(-math.log(10000) * torch.arange(0, width, 2,
                           device=reference.device, dtype=torch.float32) / width)
    phase = torch.arange(length, device=reference.device, dtype=torch.float32)[:, None] * frequency
    positions = torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)
    return positions[None].to(reference.dtype)


class TaskDecoder(nn.Module):
    """Decode raw pixels/reports with optional slots as additional conditions.

    Both arms construct exactly these parameters. The slot projection is unused
    for baseline forwards; the baseline does not construct a slot producer.
    Image forwards return the fixed square patch grid. Report-only forwards
    return one learned readout token, suitable for classification and text.
    """

    def __init__(self, cfg):
        super().__init__()
        width = cfg["decoder_width"]
        self.vision_pixels = cfg["vision_pixels"]
        self.patch_size = cfg.get("task_patch_size", 16)
        self.context_tokens = cfg["context_tokens"]
        if (type(self.patch_size) is not int or self.patch_size < 1
                or self.vision_pixels % self.patch_size or width % 8):
            raise ValueError("Pixels must divide into patches and decoder width must be divisible by eight")
        self.tokenizer = ByteTokenizer()
        self.patch_projection = nn.Conv2d(3, width, self.patch_size, stride=self.patch_size)
        self.image_norm = nn.LayerNorm(width)
        self.report_embedding = nn.Embedding(self.tokenizer.vocab_size, width,
                                             padding_idx=self.tokenizer.pad_token_id)
        self.image_type = nn.Parameter(torch.zeros(1, 1, width))
        self.report_type = nn.Parameter(torch.zeros(1, 1, width))
        self.readout = nn.Parameter(torch.zeros(1, 1, width))
        self.slot_projection = nn.Sequential(nn.LayerNorm(STATE_WIDTH), nn.Linear(STATE_WIDTH, width))
        self.slot_type = nn.Parameter(torch.zeros(1, STATE_SLOTS, width))
        layer = nn.TransformerEncoderLayer(width, 8, width * 4, dropout=0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, cfg["decoder_depth"],
                                                 norm=nn.LayerNorm(width), enable_nested_tensor=False)

    def prepare_images(self, images):
        """Return a CPU float [B,3,S,S] batch; no model or device access needed."""
        if not images:
            raise ValueError("At least one PIL image is required")
        pixels = []
        for image in images:
            if not isinstance(image, Image.Image):
                raise ValueError("prepare_images requires PIL images")
            image = ImageOps.pad(image.convert("RGB"), (self.vision_pixels, self.vision_pixels),
                                 method=Image.Resampling.BICUBIC, color=0, centering=(.5, .5))
            pixels.append(torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1))
        return torch.stack(pixels).float().div_(255)

    def forward(self, images=None, reports=None, slots=None, prepared=None):
        pixels = prepared
        if pixels is None and images is not None:
            pixels = images if isinstance(images, torch.Tensor) else self.prepare_images(images)
        if pixels is None and reports is None:
            raise ValueError("Raw task decoding requires images, reports, or both")
        if pixels is not None:
            expected = (3, self.vision_pixels, self.vision_pixels)
            if not isinstance(pixels, torch.Tensor) or pixels.ndim != 4 or tuple(pixels.shape[1:]) != expected:
                raise ValueError(f"Prepared pixels must have shape [B,{expected[0]},{expected[1]},{expected[2]}]")
            batch_size = pixels.shape[0]
        else:
            batch_size = len(reports)
        if batch_size < 1:
            raise ValueError("Raw task decoding requires a nonempty batch")
        if reports is not None and (isinstance(reports, str) or len(reports) != batch_size
                                    or any(not isinstance(report, str) for report in reports)):
            raise ValueError("Provide one report string per example")

        reference = self.patch_projection.weight
        if pixels is not None:
            was_uint8 = pixels.dtype == torch.uint8
            if not was_uint8 and not pixels.is_floating_point():
                raise ValueError("Pixels must be floating [0,1] or uint8 [0,255]")
            pixels = pixels.to(device=reference.device, dtype=reference.dtype)
            if was_uint8:
                pixels = pixels / 255
            image_tokens = self.image_norm(self.patch_projection(pixels * 2 - 1).flatten(2).transpose(1, 2))
            size = self.vision_pixels // self.patch_size
            tokens = image_tokens + positional_encoding(size, size, image_tokens.shape[-1]).to(image_tokens)
            tokens = tokens + self.image_type
        else:
            tokens = self.readout.expand(batch_size, -1, -1)
        output_length = tokens.shape[1]
        padding = torch.zeros(tokens.shape[:2], device=tokens.device, dtype=torch.bool)
        if reports is not None:
            ids, valid = self.tokenizer.batch(reports, self.context_tokens, tokens.device)
            text = self.report_embedding(ids)
            text = text + sequence_positions(text.shape[1], text.shape[2], text) + self.report_type
            tokens = torch.cat((tokens, text), 1)
            padding = torch.cat((padding, ~valid), 1)
        if slots is not None:
            validate_state(slots)
            if len(slots) != batch_size:
                raise ValueError("Provide one slot state per raw input")
            conditions = self.slot_projection(slots.to(reference)) + self.slot_type
            tokens = torch.cat((tokens, conditions), 1)
            padding = torch.cat((padding, padding.new_zeros(batch_size, STATE_SLOTS)), 1)
        return self.transformer(tokens, src_key_padding_mask=padding)[:, :output_length]
