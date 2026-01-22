# model_module.py
# Readable nanoTabPFN model

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class FeatureEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.lin = nn.Linear(1, embedding_size)

    def forward(self, x: torch.Tensor, split: int) -> torch.Tensor:
        # x: (B, R, C)
        x = x.unsqueeze(-1)  # (B, R, C, 1)
        mean = x[:, :split].mean(dim=1, keepdim=True)
        std = x[:, :split].std(dim=1, keepdim=True) + 1e-20
        x = (x - mean) / std
        x = torch.clip(x, -100, 100)
        return self.lin(x)  # (B, R, C, E)


class TargetEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.lin = nn.Linear(1, embedding_size)

    def forward(self, y_train: torch.Tensor, num_rows: int) -> torch.Tensor:
        # y_train: (B, split) or (B, split, 1)
        if y_train.ndim == 2:
            y_train = y_train.unsqueeze(-1)  # (B, split, 1)

        mean = y_train.mean(dim=1, keepdim=True)  # (B,1,1)
        pad = mean.repeat(1, num_rows - y_train.shape[1], 1)
        y = torch.cat([y_train, pad], dim=1)  # (B, R, 1)
        y = y.unsqueeze(-1)                   # (B, R, 1, 1)
        return self.lin(y)                    # (B, R, 1, E)


class Block(nn.Module):
    """One nanoTabPFN transformer block: feature-attn then datapoint-attn then MLP."""

    def __init__(self, embedding_size: int, nhead: int, mlp_hidden: int):
        super().__init__()
        self.attn_feat = nn.MultiheadAttention(embedding_size, nhead, batch_first=True)
        self.attn_row = nn.MultiheadAttention(embedding_size, nhead, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embedding_size),
        )
        self.n1 = nn.LayerNorm(embedding_size)
        self.n2 = nn.LayerNorm(embedding_size)
        self.n3 = nn.LayerNorm(embedding_size)

    def forward(self, table: torch.Tensor, split: int) -> torch.Tensor:
        # table: (B, R, C, E)
        B, R, C, E = table.shape

        # attention across features (per row)
        t = table.reshape(B * R, C, E)
        t = self.attn_feat(t, t, t)[0] + t
        table = self.n1(t.reshape(B, R, C, E))

        # attention across rows (per column), with leakage prevention
        t = table.transpose(1, 2).reshape(B * C, R, E)  # (B*C, R, E)
        left = self.attn_row(t[:, :split], t[:, :split], t[:, :split])[0]
        right = self.attn_row(t[:, split:], t[:, :split], t[:, :split])[0]
        t2 = torch.cat([left, right], dim=1) + t
        table = self.n2(t2.reshape(B, C, R, E).transpose(1, 2))  # back to (B,R,C,E)

        # MLP
        table = self.n3(self.mlp(table) + table)
        return table


class NanoTabPFNModel(nn.Module):
    def __init__(self, embedding_size: int, num_attention_heads: int, mlp_hidden_size: int, num_layers: int, num_outputs: int):
        super().__init__()
        self.feat_enc = FeatureEncoder(embedding_size)
        self.tgt_enc = TargetEncoder(embedding_size)
        self.blocks = nn.ModuleList(
            [Block(embedding_size, num_attention_heads, mlp_hidden_size) for _ in range(num_layers)]
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_size, mlp_hidden_size),
            nn.GELU(),
            nn.Linear(mlp_hidden_size, num_outputs),
        )

    def forward(self, src, train_test_split_index: int):
        x, y_train = src  # x: (B,R,C), y_train: (B,split)
        x_emb = self.feat_enc(x, train_test_split_index)     # (B,R,C,E)
        y_emb = self.tgt_enc(y_train, x_emb.shape[1])        # (B,R,1,E)
        table = torch.cat([x_emb, y_emb], dim=2)             # (B,R,C+1,E)

        for blk in self.blocks:
            table = blk(table, train_test_split_index)

        z = table[:, train_test_split_index:, -1, :]         # (B,R_test,E)
        return self.decoder(z)                               # (B,R_test,K)


class NanoTabPFNClassifier:
    """Simple wrapper to use nanoTabPFN like a classifier."""

    def __init__(self, model: NanoTabPFNModel, device):
        self.model = model.to(device)
        self.device = device

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.X_train = X_train
        self.y_train = y_train.astype(int)
        self.num_classes = int(np.unique(self.y_train).size)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        x = np.concatenate([self.X_train, X_test], axis=0)
        y = self.y_train

        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
            y_t = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.model((x_t, y_t), train_test_split_index=len(self.X_train)).squeeze(0)
            logits = logits[:, : self.num_classes]
            return F.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.predict_proba(X_test).argmax(axis=1)
