# 05_model_minilm.py
# Modelo mínimo: Embedding -> Linear

import torch
import torch.nn as nn


class MiniLM(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        """
        x: tensor de shape (B, T)
        retorna logits de shape (B, T, V)
        """
        emb = self.tok_emb(x)      # (B, T, E)
        logits = self.head(emb)    # (B, T, V)
        return logits


if __name__ == "__main__":
    vocab_size = 30
    emb_dim = 16
    model = MiniLM(vocab_size, emb_dim)

    x = torch.randint(0, vocab_size, (2, 5))  # batch=2, seq_len=5
    logits = model(x)

    print("Entrada:", x.shape)
    print("Saída:", logits.shape)
