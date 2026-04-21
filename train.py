# 06_train.py

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_minilm import MiniLM
from tokenizer import BPETokenizer, MERGE_SET, STOI, ITOS
from dataset import CausalLMDataset

DATA = Path('data')

BATCH = 16
CTX = 8
EPOCHS = 60
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- leitura ---
text = (DATA / 'corpus.txt').read_text(encoding='utf-8')

tk = BPETokenizer(STOI, ITOS, MERGE_SET)

# Melhor: preservar linhas
ids = []
for line in text.splitlines():
    ids.extend(tk.encode(line))

# --- dataset ---
dataset = CausalLMDataset(ids, ctx_len=CTX)

if len(dataset) == 0:
    raise ValueError("Corpus pequeno demais para o ctx_len definido.")

loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

# --- modelo ---
vocab_size = len(ITOS)
model = MiniLM(vocab_size, emb_dim=128).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# --- treino ---
for epoch in range(EPOCHS):
    model.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    total = 0
    for x, y in pbar:
        x, y = x.to(device), y.to(device)

        logits = model(x)

        loss = loss_fn(
            logits.reshape(-1, vocab_size),
            y.reshape(-1)
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.3f}")

    print("epoch loss:", total / len(loader))

# --- salvar ---
torch.save(model.state_dict(), DATA / 'minilm.pt')
print("Pesos salvos em data/minilm.pt")
