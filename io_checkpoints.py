# io_checkpoints.py
# Utilidades simples de checkpoint (versão melhorada)

import torch
from pathlib import Path


def save_ckpt(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
    }, path)

    print(f"[OK] Checkpoint salvo em: {path}")


def load_ckpt(model, path, strict=True):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {path}")

    checkpoint = torch.load(path, map_location='cpu')

    # Compatível com versão antiga (state_dict direto)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)

    print(f"[OK] Checkpoint carregado de: {path}")

    return model
