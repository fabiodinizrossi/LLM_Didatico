# eval_gptmini.py
# Avaliação do GPTMini no corpus: loss e perplexidade

from pathlib import Path
import math
import torch
import torch.nn as nn

from tokenizer import BPETokenizer, MERGE_SET, STOI, ITOS
from model_gptmini import GPTMini

DATA = Path("data")
CORPUS = DATA / "corpus.txt"
MODEL_PATH = DATA / "gptmini.pt"

CTX = 32
EMB_DIM = 128
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.0

if not CORPUS.exists():
    raise FileNotFoundError(f"Corpus não encontrado: {CORPUS}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Checkpoint não encontrado: {MODEL_PATH}")

text = CORPUS.read_text(encoding="utf-8")

tk = BPETokenizer(STOI, ITOS, MERGE_SET)

ids = []
for line in text.splitlines():
    line = line.strip()
    if line:
        ids.extend(tk.encode(line))

if len(ids) < 2:
    raise ValueError("Corpus pequeno demais para avaliação.")

vocab_size = len(ITOS)

model = GPTMini(
    vocab_size=vocab_size,
    emb_dim=EMB_DIM,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    ctx_len=CTX,
    dropout=DROPOUT
)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

loss_fn = nn.CrossEntropyLoss(reduction="mean")

total_loss = 0.0
total_tokens = 0

with torch.no_grad():
    # Avaliação em janelas para respeitar o ctx_len do modelo
    for start in range(0, len(ids) - 1, CTX):
        chunk = ids[start:start + CTX + 1]

        if len(chunk) < 2:
            continue

        x = torch.tensor(chunk[:-1], dtype=torch.long).unsqueeze(0)  # (1, T)
        y = torch.tensor(chunk[1:], dtype=torch.long).unsqueeze(0)   # (1, T)

        logits = model(x)

        loss = loss_fn(
            logits.reshape(-1, vocab_size),
            y.reshape(-1)
        )

        n_tokens = y.numel()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

if total_tokens == 0:
    raise ValueError("Não foi possível calcular a avaliação: total_tokens = 0.")

mean_loss = total_loss / total_tokens
pp = math.exp(mean_loss)

print(f"Loss: {mean_loss:.4f} | Perplexidade: {pp:.2f}")
