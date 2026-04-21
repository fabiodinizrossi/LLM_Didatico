# transformer_block.py
# Bloco Transformer com multi-head attention, LayerNorm e FFN

import torch
import torch.nn as nn
from attention import CausalSelfAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, ctx_len, n_heads=4, dropout=0.0):
        super().__init__()

        if emb_dim % n_heads != 0:
            raise ValueError("emb_dim deve ser divisível por n_heads")

        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads

        self.heads = nn.ModuleList([
            CausalSelfAttention(self.head_dim, ctx_len, dropout)
            for _ in range(n_heads)
        ])

        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        # divide embedding entre heads
        x_split = x.view(B, T, self.n_heads, self.head_dim)

        head_outputs = []
        for i, head in enumerate(self.heads):
            h = x_split[:, :, i, :]   # (B, T, head_dim)
            head_outputs.append(head(h))

        # concatena heads
        out = torch.cat(head_outputs, dim=-1)  # (B, T, emb_dim)
        out = self.out_proj(out)
        out = self.dropout(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, emb_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, ctx_len, n_heads=4, dropout=0.0):
        super().__init__()

        self.ln1 = nn.LayerNorm(emb_dim)
        self.attn = MultiHeadAttention(
            emb_dim=emb_dim,
            ctx_len=ctx_len,
            n_heads=n_heads,
            dropout=dropout
        )

        self.ln2 = nn.LayerNorm(emb_dim)
        self.ff = FeedForward(emb_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
