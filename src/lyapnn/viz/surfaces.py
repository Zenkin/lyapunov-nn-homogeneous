#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _sym_clip_q(Z: np.ndarray, q: float = 0.99) -> float:
    L = float(np.quantile(np.abs(Z), q))
    if not np.isfinite(L) or L <= 0:
        L = float(np.max(np.abs(Z)) + 1e-12)
    return L


def plot_surface3d(
    title: str,
    X1: np.ndarray,
    X2: np.ndarray,
    Z: np.ndarray,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    save_path: Optional[str] = None,
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    sym_clip_q: Optional[float] = None,
    add_colorbar: bool = True,
) -> None:
    """
    Simple interactive 3D surface plot (matplotlib) that you can rotate with the mouse.

    If sym_clip_q is provided, vmin/vmax are set to +/- clip(abs(Z), q),
    which is useful for signed fields like Vdot.
    """
    Zp = np.asarray(Z)

    if sym_clip_q is not None:
        L = _sym_clip_q(Zp, float(sym_clip_q))
        vmin, vmax = -L, +L

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    surf = ax.plot_surface(
        X1, X2, Zp,
        linewidth=0,
        antialiased=True,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    if add_colorbar:
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"[plot] {save_path}")
    plt.show()
