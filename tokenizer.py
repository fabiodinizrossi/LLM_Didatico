# 03_tokenizer.py
# Tokenizer BPE didático (encode/decode) compatível com o treino

from pathlib import Path
import json

DATA = Path("data")
BPE_DIR = DATA / "bpe"

MERGES_PATH = BPE_DIR / "merges.json"
VOCAB_PATH = BPE_DIR / "vocab.json"

if not MERGES_PATH.exists() or not VOCAB_PATH.exists():
    raise FileNotFoundError(
        "Arquivos do BPE não encontrados. Rode primeiro o 02_bpe_train.py"
    )

MERGES = json.loads(MERGES_PATH.read_text(encoding="utf-8"))
VOC = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))

ITOS = VOC["itos"]
STOI = VOC["stoi"]

MERGE_SET = {tuple(pair) for pair in MERGES}

PAD = STOI["<pad>"]
BOS = STOI["<bos>"]
EOS = STOI["<eos>"]
UNK = STOI["<unk>"]

class BPETokenizer:
    """
    Tokenizer BPE didático:
    - quebra palavra em caracteres + </w>
    - aplica merges aprendidos
    - converte tokens em IDs
    - reconstrói texto no decode
    """

    def __init__(self, stoi, itos, merge_set):
        self.stoi = stoi
        self.itos = itos
        self.merge_set = merge_set

    def _bpe_tokenize_word(self, word):
        tokens = list(word) + ["</w>"]

        changed = True
        while changed:
            changed = False
            i = 0
            new_tokens = []

            while i < len(tokens):
                if i < len(tokens) - 1:
                    pair = (tokens[i], tokens[i + 1])
                    if pair in self.merge_set:
                        new_tokens.append(tokens[i] + tokens[i + 1])
                        i += 2
                        changed = True
                        continue

                new_tokens.append(tokens[i])
                i += 1

            tokens = new_tokens

        return tokens

    def tokenize(self, text):
        all_tokens = []
        for word in text.lower().split():
            all_tokens.extend(self._bpe_tokenize_word(word))
        return all_tokens

    def encode(self, text, add_bos_eos=True):
        ids = []

        if add_bos_eos:
            ids.append(BOS)

        tokens = self.tokenize(text)
        for tok in tokens:
            ids.append(self.stoi.get(tok, UNK))

        if add_bos_eos:
            ids.append(EOS)

        return ids

    def decode(self, ids, skip_specials=True):
        tokens = []

        special_tokens = {"<pad>", "<bos>", "<eos>"}
        if skip_specials:
            special_tokens.add("<unk>")

        for idx in ids:
            if not (0 <= idx < len(self.itos)):
                continue

            tok = self.itos[idx]
            if skip_specials and tok in special_tokens:
                continue

            tokens.append(tok)

        # Reconstrói palavras usando </w>
        words = []
        current_word = ""

        for tok in tokens:
            if tok.endswith("</w>"):
                current_word += tok[:-4]  # remove </w>
                words.append(current_word)
                current_word = ""
            else:
                current_word += tok

        if current_word:
            words.append(current_word)

        return " ".join(words).strip()


if __name__ == "__main__":
    tk = BPETokenizer(STOI, ITOS, MERGE_SET)

    s = "o gato correu"
    print("Texto original:", s)

    tokens = tk.tokenize(s)
    print("Tokens:", tokens)

    ids = tk.encode(s)
    print("IDs:", ids)

    decoded = tk.decode(ids)
    print("Decode:", decoded)
