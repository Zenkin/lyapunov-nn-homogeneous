#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class Params:
    a1: float = 1.0
    a2: float = 2.0
    a3: float = 1.0
    c1: float = 1.0
    c2: float = 2.0
    Fc: float = 0.2
    vs: float = 0.05


def smooth_abs(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(x * x + eps)


def phi(v: torch.Tensor, Fc: float, vs: float) -> torch.Tensor:
    return Fc * torch.tanh(v / vs)


def f_inf(x_t: torch.Tensor, p: Params) -> torch.Tensor:
    """Infinity approximation dynamics in shifted coordinates x_t = (x1_tilde, x2)."""
    x1, x2 = x_t[:, 0], x_t[:, 1]
    drag = p.a2 * smooth_abs(x2) * x2
    x2dot = -drag - p.a3 * (x1 ** 3)
    return torch.stack([x2, x2dot], dim=1)


def f_full(x: torch.Tensor, p: Params) -> torch.Tensor:
    """Original dynamics in original coordinates x = (x1, x2)."""
    x1, x2 = x[:, 0], x[:, 1]
    drag = p.a2 * smooth_abs(x2) * x2
    fric = phi(x2, p.Fc, p.vs)
    x2dot = -fric + p.a1 * (x1 - p.c1) - drag - p.a3 * ((x1 - p.c2) ** 3)
    return torch.stack([x2, x2dot], dim=1)


def equilibrium_x1(p: Params) -> float:
    """Return x1 equilibrium (x2_eq = 0) for the full system (closest real root)."""
    a3, a1, d = float(p.a3), float(p.a1), float(p.c2 - p.c1)
    roots = np.roots([a3, 0.0, -a1, -a1 * d])
    rr = [r.real for r in roots if abs(r.imag) < 1e-10]
    z = float(rr[0]) if rr else float(min(roots, key=lambda r: abs(r.imag)).real)
    return z + float(p.c2)
