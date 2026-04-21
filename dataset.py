# 04_dataset.py
# Dataset causal para Language Modeling

from torch.utils.data import Dataset
import torch


class CausalLMDataset(Dataset):
    def __init__(self, token_ids, ctx_len=32):
        if len(token_ids) < 2:
            raise ValueError("token_ids precisa ter pelo menos 2 tokens.")

        self.ids = token_ids
        self.ctx_len = ctx_len

    def __len__(self):
        return max(0, len(self.ids) - self.ctx_len)

    def __getitem__(self, idx):
        x_ids = self.ids[idx : idx + self.ctx_len]
        y_ids = self.ids[idx + 1 : idx + self.ctx_len + 1]

        x = torch.tensor(x_ids, dtype=torch.long)
        y = torch.tensor(y_ids, dtype=torch.long)

        return x, y
