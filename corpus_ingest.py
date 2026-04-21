# 01_corpus_ingest.py
# Lê corpus, normaliza e mostra estatísticas (versão corrigida e didática)

from pathlib import Path
import re

# Diretório e arquivo
DATA = Path('data')
DATA.mkdir(exist_ok=True)
CORPUS = DATA / 'corpus.txt'

# Texto padrão (usado apenas se o arquivo não existir)
DEFAULT = """o gato sentou no tapete
o cachorro correu para o parque
o gato correu atrás do rato
"""

# --- GARANTIA DE EXISTÊNCIA DO ARQUIVO ---
if not CORPUS.exists():
    print("[AVISO] corpus.txt não encontrado. Criando com conteúdo padrão...")
    CORPUS.write_text(DEFAULT, encoding='utf-8')
else:
    print("[OK] Lendo corpus existente:", CORPUS)

# --- LEITURA ---
raw_text = CORPUS.read_text(encoding='utf-8')

# --- NORMALIZAÇÃO ---
text = raw_text.lower()

# Mantém quebras de linha (importante para análise)
text = re.sub(r"[ \t]+", " ", text)  # limpa espaços extras, mas preserva \n
text = text.strip()

# --- ESTATÍSTICAS ---
num_chars = len(text)
num_lines = text.count('\n') + 1 if text else 0
num_words = len(text.split())

# --- SAÍDA ---
print("\n--- PRÉVIA DO CORPUS ---")
print(text[:200], "...\n")

print("--- ESTATÍSTICAS ---")
print("Caracteres:", num_chars)
print("Linhas:", num_lines)
print("Palavras:", num_words)
