# LLM Didático: Do Zero ao GPT Mini

Este projeto implementa, de forma didática e incremental, um pipeline completo de construção de um modelo de linguagem (LLM), desde o texto bruto até um modelo estilo GPT com mecanismo de atenção.

## Visão Geral

- MiniLM → baseline simples
- GPTMini → modelo com self-attention

## Estrutura do Projeto

corpus_ingest.py      → entrada de dados  
bpe_train.py          → tokenização (BPE)  
tokenizer.py          → encode/decode  
dataset.py            → pares (x, y)  

model_minilm.py       → baseline  
train.py              → treino MiniLM  
generate.py           → geração MiniLM  
eval.py               → avaliação MiniLM  

attention.py          → self-attention  
transformer_block.py  → bloco Transformer  
model_gptmini.py      → GPTMini  
train_gptmini.py      → treino GPT  
generate_gptmini.py   → geração GPT  
eval_gptmini.py       → avaliação GPT  

io_checkpoints.py     → salvar/carregar modelos  
cli_demo.py           → interface interativa  

## Instalação

pip install -r requirements.txt

## Pipeline

1. python corpus_ingest.py  
2. python bpe_train.py
3. python tokenizer.py
4. python dataset.py
5. python model_minilm.py
6. python train.py  
7. python generate.py  
8. python eval.py
9. python attention.py
10. python transformer_block.py
11. python model_gptmini.py
12. python train_gptmini.py  
13. python generate_gptmini.py  
14. python eval_gptmini.py
15. python io_checkpoints.py 
16. python cli_demo.py

## Resultados

MiniLM → pior geração  
GPTMini → melhor coerência com atenção  

## Uso

> gato
> cachorro


## Conceitos

- BPE  
- Self-Attention  
- Transformer  
- Perplexidade  
- Geração autoregressiva  

## Objetivo

Mostrar como LLMs funcionam internamente de forma simples e educacional.
