# models/w.py
from __future__ import annotations

import torch
import torch.nn as nn


class W(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError(f"x must have shape (N,2), got {tuple(x.shape)}")
        return self.net(x)
