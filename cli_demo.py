# cli_demo.py
# REPL simples para geração interativa com GPTMini

from pathlib import Path
import torch
import torch.nn.functional as F

from tokenizer import BPETokenizer, MERGE_SET, STOI, ITOS
from model_gptmini import GPTMini

MODEL_PATH = Path("data/gptmini.pt")

CTX = 32
EMB_DIM = 128
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.0
MAX_NEW_TOKENS = 40
DEFAULT_TOP_K = 3
DEFAULT_TOP_P = 0.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Checkpoint não encontrado: {MODEL_PATH}")

BOS = STOI["<bos>"]
EOS = STOI["<eos>"]

tk = BPETokenizer(STOI, ITOS, MERGE_SET)

model = GPTMini(
    vocab_size=len(ITOS),
    emb_dim=EMB_DIM,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    ctx_len=CTX,
    dropout=DROPOUT
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


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


@torch.no_grad()
def generate(prompt, max_new_tokens=MAX_NEW_TOKENS, top_k=DEFAULT_TOP_K, top_p=DEFAULT_TOP_P):
    ids = [BOS] + tk.encode(prompt, add_bos_eos=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        x_cond = x[:, -CTX:]
        logits = model(x_cond)[:, -1, :]
        nxt = sample_next(logits.squeeze(0), top_k=top_k, top_p=top_p)

        x = torch.cat(
            [x, torch.tensor([[nxt]], dtype=torch.long, device=device)],
            dim=1
        )

        if nxt == EOS:
            break

    return tk.decode(x[0].tolist())


print("CLI do GPTMini")
print("Digite um prompt e pressione Enter.")
print("Comandos:")
print("  /sair                encerra")
print("  /config              mostra configuração atual")
print("  /topk N              altera top_k")
print("  /topp X              altera top_p")
print("  /max N               altera max_new_tokens")
print()

top_k = DEFAULT_TOP_K
top_p = DEFAULT_TOP_P
max_new_tokens = MAX_NEW_TOKENS

while True:
    try:
        s = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nEncerrado.")
        break

    if not s:
        continue

    if s == "/sair":
        print("Encerrado.")
        break

    if s == "/config":
        print(f"top_k={top_k} | top_p={top_p} | max_new_tokens={max_new_tokens}")
        continue

    if s.startswith("/topk "):
        try:
            top_k = int(s.split(maxsplit=1)[1])
            print(f"top_k atualizado para {top_k}")
        except ValueError:
            print("Valor inválido para top_k.")
        continue

    if s.startswith("/topp "):
        try:
            top_p = float(s.split(maxsplit=1)[1])
            print(f"top_p atualizado para {top_p}")
        except ValueError:
            print("Valor inválido para top_p.")
        continue

    if s.startswith("/max "):
        try:
            max_new_tokens = int(s.split(maxsplit=1)[1])
            print(f"max_new_tokens atualizado para {max_new_tokens}")
        except ValueError:
            print("Valor inválido para max_new_tokens.")
        continue

    out = generate(
        prompt=s,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p
    )
    print(out)
    print()
