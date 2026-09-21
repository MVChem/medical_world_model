"""Batch equal-length Qwen vision sequences without cross-image attention."""
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5VisionAttention, apply_rotary_pos_emb_vision,
)


class BatchedVisionAttention(Qwen3_5VisionAttention):
    def forward(self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        if not lengths or len(set(lengths)) != 1 or position_embeddings is None:
            return super().forward(hidden_states, cu_seqlens, position_embeddings, **kwargs)
        batch, length = len(lengths), lengths[0]
        total = hidden_states.shape[0]
        if batch * length != total:
            raise ValueError("Vision sequence lengths do not match the packed input")
        q, k, v = self.qkv(hidden_states).reshape(total, 3, self.num_heads, -1).unbind(1)
        q, k = apply_rotary_pos_emb_vision(q, k, *position_embeddings)
        q, k, v = [x.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
                   for x in (q, k, v)]
        output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scaling, is_causal=False)
        return self.proj(output.transpose(1, 2).reshape(total, -1).contiguous())


def batch_vision_attention(vision):
    """Change only the local module's forward; keep all weights and state keys."""
    for block in vision.blocks:
        if type(block.attn) is not Qwen3_5VisionAttention:
            raise TypeError("Unexpected Qwen vision attention implementation")
        block.attn.__class__ = BatchedVisionAttention
