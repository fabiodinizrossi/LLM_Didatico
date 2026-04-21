# generate.py
# Geração autoregressiva apenas com MiniLM (versão simplificada do pipeline)

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from tokenizer import BPETokenizer, MERGE_SET, STOI, ITOS
from model_minilm import MiniLM

# --- argumentos ---
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='data/minilm.pt', help='checkpoint do MiniLM')
parser.add_argument('--prompt', type=str, default='o gato')
parser.add_argument('--max_new_tokens', type=int, default=40)
parser.add_argument('--top_k', type=int, default=0)
parser.add_argument('--top_p', type=float, default=0.0)
args = parser.parse_args()

# --- device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- carregar modelo ---
vocab_size = len(ITOS)
model = MiniLM(vocab_size=vocab_size, emb_dim=128)

model.load_state_dict(torch.load(args.model, map_location=device))
model.to(device)
model.eval()

# --- tokenizer ---
tk = BPETokenizer(STOI, ITOS, MERGE_SET)

BOS = STOI['<bos>']
EOS = STOI['<eos>']

# prompt sem <eos>, apenas <bos> no início
ids = [BOS] + tk.encode(args.prompt, add_bos_eos=False)

# --- função de amostragem ---
@torch.no_grad()
def sample_next(logits, top_k=0, top_p=0.0):
    probs = F.softmax(logits, dim=-1)

    # top-k
    if top_k > 0:
        k = min(top_k, probs.size(-1))
        values, indices = torch.topk(probs, k)
        filtered = torch.zeros_like(probs)
        filtered.scatter_(0, indices, values)
        total = filtered.sum()
        if total > 0:
            probs = filtered / total

    # top-p (nucleus sampling)
    if top_p > 0.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=0)

        mask = cum_probs > top_p
        if mask.any():
            mask[1:] = mask[:-1].clone()
            mask[0] = False

        sorted_probs[mask] = 0.0

        filtered = torch.zeros_like(probs)
        filtered.scatter_(0, sorted_idx, sorted_probs)
        total = filtered.sum()
        if total > 0:
            probs = filtered / total

    # fallback de segurança
    if probs.sum() <= 0:
        probs = F.softmax(logits, dim=-1)

    return torch.multinomial(probs, num_samples=1).item()

# --- geração ---
with torch.no_grad():
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(args.max_new_tokens):
        logits = model(x)[:, -1, :]
        nxt = sample_next(
            logits.squeeze(0),
            top_k=args.top_k,
            top_p=args.top_p
        )

        x = torch.cat(
            [x, torch.tensor([[nxt]], dtype=torch.long, device=device)],
            dim=1
        )

        if nxt == EOS:
            break

# --- saída ---
print("\n=== TEXTO GERADO ===")
print(tk.decode(x[0].tolist()))
