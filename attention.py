# attention.py
import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, head_dim, ctx_len, dropout=0.0):
        super().__init__()

        self.q = nn.Linear(head_dim, head_dim, bias=False)
        self.k = nn.Linear(head_dim, head_dim, bias=False)
        self.v = nn.Linear(head_dim, head_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(ctx_len, ctx_len))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (C ** 0.5))
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))

        w = torch.softmax(att, dim=-1)
        w = self.dropout(w)

        out = w @ v
        return out
