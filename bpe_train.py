# 02_bpe_train.py
# Treina um BPE didático e salva merges/vocab de forma consistente

from pathlib import Path
from collections import Counter
import json
import re

DATA = Path("data")
DATA.mkdir(exist_ok=True)

CORPUS = DATA / "corpus.txt"
BPE_DIR = DATA / "bpe"
BPE_DIR.mkdir(exist_ok=True)

MERGES_PATH = BPE_DIR / "merges.json"
VOCAB_PATH = BPE_DIR / "vocab.json"

DEFAULT = """o gato sentou no tapete
o cachorro correu para o parque
o gato correu atrás do rato
"""

if not CORPUS.exists():
    print("[AVISO] corpus.txt não encontrado. Criando corpus padrão...")
    CORPUS.write_text(DEFAULT, encoding="utf-8")
else:
    print(f"[OK] Lendo corpus existente: {CORPUS}")

# Leitura e normalização simples
text = CORPUS.read_text(encoding="utf-8").lower()
text = re.sub(r"[ \t]+", " ", text)
text = text.strip()

words = text.split()

# Cada palavra começa como sequência de caracteres + </w>
# Ex.: "gato" -> ('g', 'a', 't', 'o', '</w>')
vocab = Counter(tuple(list(word) + ["</w>"]) for word in words)

def get_stats(current_vocab):
    """
    Conta frequência de pares adjacentes de símbolos no vocabulário.
    """
    pairs = Counter()
    for symbols, freq in current_vocab.items():
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pairs[pair] += freq
    return pairs

def merge_vocab(best_pair, current_vocab):
    """
    Faz merge do melhor par no vocabulário inteiro.
    Ex.: ('a', 't') -> 'at'
    """
    merged_token = "".join(best_pair)
    new_vocab = Counter()

    for symbols, freq in current_vocab.items():
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                new_symbols.append(merged_token)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_vocab[tuple(new_symbols)] += freq

    return new_vocab

# Para esse corpus pequeno, 3 já é suficiente
num_merges = 3
merges = []

for step in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break

    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    merges.append(list(best))

    print(f"Merge {step + 1:02d}: {best} -> {''.join(best)} (freq={pairs[best]})")

# Extrai símbolos/tokens finais corretamente
symbols = set()
for tokenized_word in vocab:
    for token in tokenized_word:
        symbols.add(token)

specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
itos = specials + sorted(symbols)
stoi = {token: idx for idx, token in enumerate(itos)}

MERGES_PATH.write_text(
    json.dumps(merges, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

VOCAB_PATH.write_text(
    json.dumps({"itos": itos, "stoi": stoi}, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\nBPE treinado com sucesso.")
print("Salvo em:", BPE_DIR)
print("Quantidade de merges:", len(merges))
print("Tamanho do vocabulário:", len(itos))
print("Alguns tokens do vocab:", itos[:20])
