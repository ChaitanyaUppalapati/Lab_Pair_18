"""GPT-style decoder with attention and causal masking implemented manually."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ManualMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, context_length: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(context_length, context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, context_length, context_length), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        shape = (batch, length, self.n_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.causal_mask[:, :, :length, :length], float("-inf"))
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        attended = (weights @ v).transpose(1, 2).contiguous().view(batch, length, width)
        return self.resid_dropout(self.out_proj(attended))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, context_length: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = ManualMultiHeadSelfAttention(d_model, n_heads, dropout, context_length)
        self.ln2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        return x + self.feed_forward(self.ln2(x))


class CharacterGPT(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, model_config: dict):
        super().__init__()
        d_model = int(model_config["d_model"])
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)
        self.dropout = nn.Dropout(float(model_config["dropout"]))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    int(model_config["n_heads"]),
                    int(model_config["d_ff"]),
                    float(model_config["dropout"]),
                    context_length,
                )
                for _ in range(int(model_config["n_layers"]))
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, length = token_ids.shape
        if length > self.context_length:
            raise ValueError(f"Sequence length {length} exceeds context {self.context_length}")
        positions = torch.arange(length, device=token_ids.device)
        x = self.dropout(self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :])
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, max_new_tokens: int, temperature: float | None, eos_id: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits = self(ids[:, -self.context_length :])[:, -1, :]
            if temperature is None or temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                probabilities = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probabilities, num_samples=1)
            ids = torch.cat((ids, next_id), dim=1)
            if int(next_id.item()) == eos_id:
                break
        return ids


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
