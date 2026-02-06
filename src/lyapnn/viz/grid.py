#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import torch


def make_grid(x1_lim: Tuple[float, float], x2_lim: Tuple[float, float], n: int, device: str):
    dev = torch.device(device)
    x1 = torch.linspace(x1_lim[0], x1_lim[1], n, device=dev)
    x2 = torch.linspace(x2_lim[0], x2_lim[1], n, device=dev)
    X1t, X2t = torch.meshgrid(x1, x2, indexing="ij")
    X = torch.stack([X1t.reshape(-1), X2t.reshape(-1)], dim=1)
    return X1t.detach().cpu().numpy(), X2t.detach().cpu().numpy(), X


def bbox_from_mask(X1: np.ndarray, X2: np.ndarray, mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    if not np.any(mask):
        return None
    i, j = np.where(mask)
    xs = X1[i, j]
    ys = X2[i, j]
    return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())


def scaled_bbox(
    bbox: Tuple[float, float, float, float],
    scale: float,
    x1_lim: Tuple[float, float],
    x2_lim: Tuple[float, float],
) -> Tuple[float, float, float, float]:
    xmin, xmax, ymin, ymax = bbox
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    hx, hy = 0.5 * (xmax - xmin), 0.5 * (ymax - ymin)
    hx2, hy2 = max(1e-12, scale * hx), max(1e-12, scale * hy)
    xmin2, xmax2 = cx - hx2, cx + hx2
    ymin2, ymax2 = cy - hy2, cy + hy2
    xmin2 = max(x1_lim[0], xmin2); xmax2 = min(x1_lim[1], xmax2)
    ymin2 = max(x2_lim[0], ymin2); ymax2 = min(x2_lim[1], ymax2)
    return xmin2, xmax2, ymin2, ymax2
