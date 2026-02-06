#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn
from lyapnn.geometry.r12 import rho, to_y


class WNet(nn.Module):
    def __init__(self, hidden: int = 64, depth: int = 3):
        super().__init__()
        layers, in_dim = [], 2
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.Tanh()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)


class HomogV(nn.Module):
    """V(x) = rho(x)^mu * (softplus(Wraw(y)) + eps)."""

    def __init__(self, mu: float = 2.0, eps: float = 1e-3, hidden: int = 64, depth: int = 3):
        super().__init__()
        self.mu = float(mu)
        self.eps = float(eps)
        self.Wraw = WNet(hidden=hidden, depth=depth)
        self.sp = nn.Softplus(beta=2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = rho(x).unsqueeze(1)
        y = to_y(x)
        W = self.sp(self.Wraw(y)) + self.eps
        Vx = (r ** self.mu) * W
        return torch.where(r < 1e-8, torch.zeros_like(Vx), Vx)
