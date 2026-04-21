# model_gptmini.py
# GPT-mini causal LM

import torch
import torch.nn as nn
from transformer_block import TransformerBlock


class GPTMini(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, n_heads=4, n_layers=2, ctx_len=64, dropout=0.0):
        super().__init__()

        self.ctx_len = ctx_len

        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(ctx_len, emb_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                emb_dim=emb_dim,
                ctx_len=ctx_len,
                n_heads=n_heads,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        """
        x: (B, T)
        retorna logits: (B, T, vocab_size)
        """
        B, T = x.shape

        if T > self.ctx_len:
            raise ValueError(f"seq len {T} > ctx_len {self.ctx_len}")

        pos = torch.arange(T, device=x.device)  # (T,)

        h = self.tok_emb(x) + self.pos_emb(pos)  # (B, T, C)

        for blk in self.blocks:
            h = blk(h)

        h = self.ln_f(h)
        logits = self.head(h)

        return logits


if __name__ == "__main__":
    vocab_size = 50
    model = GPTMini(vocab_size=vocab_size, emb_dim=32, n_heads=4, n_layers=2, ctx_len=16)

    x = torch.randint(0, vocab_size, (2, 8))
    logits = model(x)

    print("Entrada:", x.shape)
    print("Saída:", logits.shape)
