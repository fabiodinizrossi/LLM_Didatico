# generate_gptmini.py
# Geração autoregressiva com GPTMini

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from tokenizer import BPETokenizer, MERGE_SET, STOI, ITOS
from model_gptmini import GPTMini

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='data/gptmini.pt', help='checkpoint do GPTMini')
parser.add_argument('--prompt', type=str, default='o gato')
parser.add_argument('--max_new_tokens', type=int, default=40)
parser.add_argument('--top_k', type=int, default=0)
parser.add_argument('--top_p', type=float, default=0.0)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BOS = STOI['<bos>']
EOS = STOI['<eos>']

ctx_len = 32
vocab_size = len(ITOS)

model_path = Path(args.model)
if not model_path.exists():
    raise FileNotFoundError(f"Checkpoint não encontrado: {model_path}")

model = GPTMini(
    vocab_size=vocab_size,
    emb_dim=128,
    n_heads=4,
    n_layers=2,
    ctx_len=ctx_len,
    dropout=0.0
)

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

tk = BPETokenizer(STOI, ITOS, MERGE_SET)

ids = [BOS] + tk.encode(args.prompt, add_bos_eos=False)

@torch.no_grad()
def sample_next(logits, top_k=0, top_p=0.0):
    probs = F.softmax(logits, dim=-1)

    if top_k > 0:
        k = min(top_k, probs.size(-1))
        values, indices = torch.topk(probs, k)
        filtered = torch.zeros_like(probs)
        filtered.scatter_(0, indices, values)
        total = filtered.sum()
        if total > 0:
            probs = filtered / total

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

    if probs.sum() <= 0:
        probs = F.softmax(logits, dim=-1)

    return torch.multinomial(probs, num_samples=1).item()

with torch.no_grad():
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(args.max_new_tokens):
        x_cond = x[:, -ctx_len:]
        logits = model(x_cond)[:, -1, :]
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

print("\n=== TEXTO GERADO (GPTMini) ===")
print(tk.decode(x[0].tolist()))
