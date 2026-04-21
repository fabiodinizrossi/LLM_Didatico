# eval.py
# Avaliação do MiniLM no corpus: loss e perplexidade

from pathlib import Path
import math
import torch
import torch.nn as nn

from tokenizer import BPETokenizer, MERGE_SET, STOI, ITOS
from model_minilm import MiniLM

DATA = Path('data')
CORPUS = DATA / 'corpus.txt'
MODEL_PATH = DATA / 'minilm.pt'

if not CORPUS.exists():
    raise FileNotFoundError(f"Corpus não encontrado: {CORPUS}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Checkpoint não encontrado: {MODEL_PATH}")

# --- leitura do corpus ---
text = CORPUS.read_text(encoding='utf-8')

# --- tokenizer ---
tk = BPETokenizer(STOI, ITOS, MERGE_SET)
ids = tk.encode(text, add_bos_eos=True)

if len(ids) < 2:
    raise ValueError("Corpus pequeno demais para avaliação.")

# --- modelo ---
vocab_size = len(ITOS)
model = MiniLM(vocab_size=vocab_size, emb_dim=128)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# --- loss ---
loss_fn = nn.CrossEntropyLoss(reduction='mean')

with torch.no_grad():
    x = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0)  # (1, T)
    y = torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0)   # (1, T)

    logits = model(x)  # (1, T, V)

    loss = loss_fn(
        logits.reshape(-1, vocab_size),
        y.reshape(-1)
    )

    pp = math.exp(loss.item())

print(f"Loss: {loss.item():.4f} | Perplexidade: {pp:.2f}")
