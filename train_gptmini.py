# train_gptmini.py
# Treino didático do GPTMini

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_gptmini import GPTMini
from tokenizer import BPETokenizer, MERGE_SET, STOI, ITOS
from dataset import CausalLMDataset

DATA = Path("data")

BATCH = 16
CTX = 32
EPOCHS = 80
LR = 3e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- leitura do corpus ---
text = (DATA / "corpus.txt").read_text(encoding="utf-8")

tk = BPETokenizer(STOI, ITOS, MERGE_SET)

# Melhor: preservar linhas e concatenar
ids = []
for line in text.splitlines():
    line = line.strip()
    if line:
        ids.extend(tk.encode(line))

# --- dataset ---
train_ds = CausalLMDataset(ids, ctx_len=CTX)

if len(train_ds) == 0:
    raise ValueError(
        f"Corpus pequeno demais para ctx_len={CTX}. "
        "Aumente o corpus ou reduza CTX."
    )

train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

# --- modelo ---
vocab_size = len(ITOS)
model = GPTMini(
    vocab_size=vocab_size,
    emb_dim=128,
    n_heads=4,
    n_layers=2,
    ctx_len=CTX,
    dropout=0.0
).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# --- treino ---
for epoch in range(EPOCHS):
    model.train()
    pbar = tqdm(train_ld, desc=f"Epoch {epoch+1}/{EPOCHS}")
    running = 0.0

    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        loss = loss_fn(
            logits.reshape(-1, vocab_size),
            y.reshape(-1)
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        running += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.3f}")

    print("loss médio:", running / len(train_ld))

# --- salvar checkpoint ---
ckpt_path = DATA / "gptmini.pt"
torch.save(model.state_dict(), ckpt_path)
print("Checkpoint salvo em:", ckpt_path)
