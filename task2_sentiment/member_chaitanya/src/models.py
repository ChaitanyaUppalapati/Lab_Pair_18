"""The three Task 2 models (architectures specified by the user). All share the same word-embedding layer
(vocabulary, dimension 128, random N(0, 0.1) init, <pad> row zero, embedding dropout) so architecture is the
only variable. Every model returns one logit per review (P(positive) = sigmoid(logit)).

Attention and the Transformer block are written by hand (no nn.MultiheadAttention / nn.Transformer*).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_embedding(vocab_size: int, dim: int) -> nn.Embedding:
    emb = nn.Embedding(vocab_size, dim, padding_idx=0)
    nn.init.normal_(emb.weight, std=0.1)
    with torch.no_grad():
        emb.weight[0].zero_()
    return emb


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x [B, T, D], mask [B, T] bool -> [B, D]."""
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


# --------------------------------------------------------------------------- baseline
class FastTextBigram(nn.Module):
    """fastText-style: mean of unigram embeddings and hashed-bigram embeddings -> linear output (no hidden layer).

    Bigrams are consecutive token pairs of the cleaned sequence, hashed into `bigram_buckets` rows of a separate
    embedding table: bucket = (a * 1_000_003 + b) mod buckets (deterministic, unlike Python's hash()).
    """

    def __init__(self, vocab_size, embed_dim=128, bigram_buckets=131072, embed_dropout=0.3):
        super().__init__()
        self.embed = make_embedding(vocab_size, embed_dim)
        self.bigram = nn.Embedding(bigram_buckets, embed_dim)
        nn.init.normal_(self.bigram.weight, std=0.1)
        self.buckets = bigram_buckets
        self.drop = nn.Dropout(embed_dropout)
        self.out = nn.Linear(embed_dim, 1)

    def forward(self, ids, **_):
        mask = ids != 0
        uni = self.embed(ids)
        a, b = ids[:, :-1], ids[:, 1:]
        bmask = mask[:, :-1] & mask[:, 1:]
        bi = self.bigram((a * 1_000_003 + b) % self.buckets)
        x = torch.cat([uni, bi], 1)
        m = torch.cat([mask, bmask], 1)
        return self.out(masked_mean(self.drop(x), m)).squeeze(-1)


# --------------------------------------------------------------------------- HAN
class AttentionPool(nn.Module):
    """Additive attention pooling of Yang et al. (2016): u = tanh(W h + b), a = softmax(u . c)."""

    def __init__(self, dim, attn_dim):
        super().__init__()
        self.proj = nn.Linear(dim, attn_dim)
        self.context = nn.Parameter(torch.randn(attn_dim) * 0.1)

    def forward(self, h, mask):
        scores = torch.tanh(self.proj(h)) @ self.context            # [B, T]
        scores = scores.masked_fill(~mask, float("-inf"))
        all_masked = ~mask.any(1, keepdim=True)
        scores = scores.masked_fill(all_masked, 0.0)                  # avoid NaN for empty rows
        w = torch.softmax(scores, dim=1)
        return (w.unsqueeze(-1) * h).sum(1), w


class HAN(nn.Module):
    """Hierarchical attention network: word BiLSTM + attention -> sentence vectors -> sentence BiLSTM + attention
    -> document vector -> linear output. Input ids [B, S, W]."""

    def __init__(self, vocab_size, embed_dim=128, hidden=64, attn_dim=128, embed_dropout=0.3, dropout=0.3):
        super().__init__()
        self.embed = make_embedding(vocab_size, embed_dim)
        self.edrop = nn.Dropout(embed_dropout)
        self.word_rnn = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.word_attn = AttentionPool(2 * hidden, attn_dim)
        self.sent_rnn = nn.LSTM(2 * hidden, hidden, batch_first=True, bidirectional=True)
        self.sent_attn = AttentionPool(2 * hidden, attn_dim)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, 1)

    @staticmethod
    def _run_rnn(rnn, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.clamp(min=1).cpu(), batch_first=True,
                                                   enforce_sorted=False)
        out, _ = rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        return out

    def forward(self, ids, return_attention=False, **_):
        B, S, W = ids.shape
        word_mask = ids != 0                                          # [B, S, W]
        sent_mask = word_mask.any(-1)                                 # [B, S]
        flat_ids = ids.view(B * S, W)
        flat_mask = word_mask.view(B * S, W)
        keep = sent_mask.view(-1)                                     # run the word LSTM on real sentences only
        sent_vecs = torch.zeros(B * S, self.out.in_features, device=ids.device)
        word_w = torch.zeros(B * S, W, device=ids.device)
        if keep.any():
            x = self.edrop(self.embed(flat_ids[keep]))
            h = self._run_rnn(self.word_rnn, x, flat_mask[keep].sum(1))
            v, w = self.word_attn(h, flat_mask[keep])
            sent_vecs[keep] = v
            word_w[keep] = w
        sent_vecs = self.drop(sent_vecs.view(B, S, -1))
        h = self._run_rnn(self.sent_rnn, sent_vecs, sent_mask.sum(1))
        doc, sent_w = self.sent_attn(h, sent_mask)
        logit = self.out(self.drop(doc)).squeeze(-1)
        if return_attention:
            return logit, word_w.view(B, S, W), sent_w
        return logit


# --------------------------------------------------------------------------- Transformer
class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_dropout)

    def forward(self, x, key_mask):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)   # each [B, h, T, dk]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)                          # [B, h, T, T]
        scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        att = self.attn_drop(torch.softmax(scores, dim=-1))
        out = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class Block(nn.Module):
    """Pre-LayerNorm encoder block: x + Attn(LN(x)); x + FFN(LN(x))."""

    def __init__(self, d_model, n_heads, d_ff, attn_dropout, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, attn_dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = x + self.drop(self.attn(self.ln1(x), mask))
        return x + self.drop(self.ff(self.ln2(x)))


class TransformerClassifier(nn.Module):
    """2-layer, 4-head, width-128 pre-LN encoder with learned positions and masked mean pooling."""

    def __init__(self, vocab_size, embed_dim=128, n_layers=2, n_heads=4, d_ff=512, max_len=256,
                 embed_dropout=0.3, attn_dropout=0.1, dropout=0.2):
        super().__init__()
        self.embed = make_embedding(vocab_size, embed_dim)
        self.pos = nn.Parameter(torch.randn(max_len, embed_dim) * 0.02)
        self.edrop = nn.Dropout(embed_dropout)
        self.blocks = nn.ModuleList([Block(embed_dim, n_heads, d_ff, attn_dropout, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(embed_dim, 1)

    def forward(self, ids, **_):
        mask = ids != 0
        T = ids.size(1)
        x = self.edrop(self.embed(ids)) + self.pos[:T]
        for blk in self.blocks:
            x = blk(x, mask)
        return self.out(self.drop(masked_mean(self.ln_f(x), mask))).squeeze(-1)


def build_model(cfg: dict, vocab_size: int) -> nn.Module:
    m = dict(cfg["model"])
    kind = m.pop("type")
    cls = {"fasttext_bigram": FastTextBigram, "han": HAN, "transformer": TransformerClassifier}[kind]
    return cls(vocab_size=vocab_size, **m)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
