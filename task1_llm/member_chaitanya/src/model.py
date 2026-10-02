"""Task 1.2: GPT-style character language model written from scratch.

No prebuilt Transformer or attention modules are used (no nn.Transformer*,
nn.MultiheadAttention or F.scaled_dot_product_attention). Attention, the causal
mask, LayerNorm and the block wiring are implemented below with plain tensor ops;
only nn.Linear, nn.Embedding and nn.Dropout are used as building blocks.

Block layout (Pre-LN):
    x = x + Dropout(MultiHeadSelfAttention(LayerNorm(x)))
    x = x + Dropout(FeedForward(LayerNorm(x)))
followed by a final LayerNorm and a linear LM head over the vocabulary.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """y = (x - mean) / sqrt(var + eps) * gamma + beta over the last dimension."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, block_size: int, dropout: float, bias: bool):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        # causal mask: position t may attend to positions <= t only
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, block_size, block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
        weights = self.attn_dropout(F.softmax(scores.float(), dim=-1).to(q.dtype))
        out = (weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation: str, bias: bool):
        super().__init__()
        self.fc_in = nn.Linear(d_model, d_ff, bias=bias)
        self.fc_out = nn.Linear(d_ff, d_model, bias=bias)
        acts = {"gelu": nn.GELU(), "relu": nn.ReLU()}
        if activation not in acts:
            raise ValueError(f"model.activation must be one of {sorted(acts)}, got {activation!r}")
        self.act = acts[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(self.act(self.fc_in(x)))


class Block(nn.Module):
    def __init__(self, cfg: dict, block_size: int):
        super().__init__()
        d, p = cfg["d_model"], cfg["dropout"]
        self.ln_attn = LayerNorm(d)
        self.attn = MultiHeadSelfAttention(d, cfg["n_heads"], block_size, p, cfg["bias"])
        self.ln_ffn = LayerNorm(d)
        self.ffn = FeedForward(d, cfg["d_ff"], cfg["activation"], cfg["bias"])
        self.resid_dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.resid_dropout(self.attn(self.ln_attn(x)))
        x = x + self.resid_dropout(self.ffn(self.ln_ffn(x)))
        return x


class CharGPT(nn.Module):
    def __init__(self, model_cfg: dict, vocab_size: int, block_size: int):
        super().__init__()
        d = model_cfg["d_model"]
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(block_size, d)
        self.emb_dropout = nn.Dropout(model_cfg["dropout"])
        self.blocks = nn.ModuleList(Block(model_cfg, block_size) for _ in range(model_cfg["n_layers"]))
        self.ln_final = LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)
        if model_cfg["tie_weights"]:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.block_size}")
        pos = torch.arange(T, device=idx.device)
        x = self.emb_dropout(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_final(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float, eos_id: int | None = None,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        """temperature == 0 -> greedy decoding; otherwise sample from softmax(logits / temperature)."""
        for _ in range(max_new_tokens):
            context = idx[:, -self.block_size :]
            logits, _ = self(context)
            logits = logits[:, -1, :].float()
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, 1, generator=generator)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and bool((next_id == eos_id).all()):
                break
        return idx


def optimizer_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Weight decay on 2-D weight matrices (linear and embedding weights) only."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
