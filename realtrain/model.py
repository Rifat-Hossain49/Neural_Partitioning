from __future__ import annotations

import torch
from torch import nn


class StateQNet(nn.Module):
    def __init__(self, input_dim: int, action_count: int, hidden: int = 512, dropout: float = 0.10):
        super().__init__()
        self.input_dim = int(input_dim)
        self.action_count = int(action_count)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, action_count),
        )

    def forward(self, x):
        return self.net(x)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model
