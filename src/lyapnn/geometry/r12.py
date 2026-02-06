#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import torch
from lyapnn.systems.duffing_friction import smooth_abs


def rho(x: torch.Tensor) -> torch.Tensor:
    """Weighted radius for r=(1,2): rho = sqrt(x1^2 + |x2|)."""
    x1, x2 = x[:, 0], x[:, 1]
    return torch.sqrt(x1 * x1 + smooth_abs(x2) + 1e-12)


def to_y(x: torch.Tensor, r_min: float = 1e-8) -> torch.Tensor:
    """Normalization to the r-sphere: y1 = x1/r, y2 = x2/r^2."""
    r = torch.clamp(rho(x), min=r_min)
    return torch.stack([x[:, 0] / r, x[:, 1] / (r * r)], dim=1)


def sample_Sr1(n: int, seed: int = 0) -> np.ndarray:
    """Sample points on S_r(1) for r=(1,2): |y1|^2 + |y2| = 1."""
    rng = np.random.default_rng(seed)
    y1 = rng.uniform(-1.0, 1.0, size=n)
    y2 = (1.0 - y1 ** 2) * rng.choice([-1.0, 1.0], size=n)
    return np.column_stack([y1, y2]).astype(np.float32)


def sample_box(
    n: int,
    x1_min: float,
    x1_max: float,
    x2_min: float,
    x2_max: float,
    seed: int = 0,
) -> np.ndarray:
    """Uniformly sample points in a 2D box."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(float(x1_min), float(x1_max), size=n)
    x2 = rng.uniform(float(x2_min), float(x2_max), size=n)
    return np.column_stack([x1, x2]).astype(np.float32)
