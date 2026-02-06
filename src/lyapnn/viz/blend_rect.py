#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import os

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Rect:
    """Axis-aligned rectangle in SHIFTED coordinates."""
    x1_min: float
    x1_max: float
    x2_min: float
    x2_max: float


def _smoothstep(t: np.ndarray) -> np.ndarray:
    # t in [0,1] -> smooth 0..1
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def blend_weight_rect(x1: np.ndarray, x2: np.ndarray, outer: Rect, inner: Rect) -> np.ndarray:
    """
    Returns s(x) in [0,1]:
      - s=0 outside OUTER (use V_inf only)
      - s=1 inside INNER (use W only)
      - in between: smooth transition from outer boundary to inner boundary
    Works for arbitrary nested rectangles (inner inside outer).
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    in_outer = (x1 >= outer.x1_min) & (x1 <= outer.x1_max) & (x2 >= outer.x2_min) & (x2 <= outer.x2_max)
    in_inner = (x1 >= inner.x1_min) & (x1 <= inner.x1_max) & (x2 >= inner.x2_min) & (x2 <= inner.x2_max)

    s = np.zeros_like(x1, dtype=float)
    s[in_inner] = 1.0

    # For points between inner and outer, compute a normalized "radius" to outer boundary in L_inf-like manner.
    mid = in_outer & (~in_inner)
    if not np.any(mid):
        return s

    # Compute per-axis normalized distance from inner boundary towards outer boundary.
    # For each axis:
    #  - If inside inner slab => 0
    #  - If left of inner_min => (inner_min - x)/(inner_min - outer_min) in (0..1]
    #  - If right of inner_max => (x - inner_max)/(outer_max - inner_max) in (0..1]
    def axis_u(x, omin, omax, imin, imax):
        u = np.zeros_like(x, dtype=float)
        left = x < imin
        right = x > imax
        # avoid div by zero if user gives degenerate nested boxes
        den_l = max(1e-12, (imin - omin))
        den_r = max(1e-12, (omax - imax))
        u[left] = (imin - x[left]) / den_l
        u[right] = (x[right] - imax) / den_r
        return np.clip(u, 0.0, 1.0)

    u1 = axis_u(x1[mid], outer.x1_min, outer.x1_max, inner.x1_min, inner.x1_max)
    u2 = axis_u(x2[mid], outer.x2_min, outer.x2_max, inner.x2_min, inner.x2_max)
    u = np.maximum(u1, u2)  # 0 at inner boundary, 1 at outer boundary

    # We want s=1 at inner boundary, s=0 at outer boundary
    s_mid = 1.0 - _smoothstep(u)
    s[mid] = s_mid
    return s


def plot_blend_maps(
    X1: np.ndarray,
    X2: np.ndarray,
    V_inf: np.ndarray,
    W: np.ndarray,
    V_blend: np.ndarray,
    margin: np.ndarray,
    outer: Rect,
    inner: Rect,
    outpath: Optional[str] = None,
    show: bool = True,
) -> Dict[str, float]:
    """
    2x2 diagnostic figure:
      (1) V_blend heatmap + outer/inner rectangles
      (2) margin heatmap + contour margin=0
      (3) bad mask (margin>0)
      (4) V_inf and W comparison (optional quick view)
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    def draw_rect(ax, r: Rect, label: str):
        xs = [r.x1_min, r.x1_max, r.x1_max, r.x1_min, r.x1_min]
        ys = [r.x2_min, r.x2_min, r.x2_max, r.x2_max, r.x2_min]
        ax.plot(xs, ys, linewidth=2.0, label=label)

    # (1) V_blend
    ax = axs[0, 0]
    im = ax.imshow(V_blend, origin="lower",
                   extent=[X1.min(), X1.max(), X2.min(), X2.max()],
                   aspect="auto")
    fig.colorbar(im, ax=ax)
    draw_rect(ax, outer, "outer (mix starts)")
    draw_rect(ax, inner, "inner (W only)")
    ax.set_title("V_blend (heatmap)")
    ax.set_xlabel("x1_tilde")
    ax.set_ylabel("x2")
    ax.legend(loc="best")

    # (2) margin + contour 0
    ax = axs[0, 1]
    im = ax.imshow(margin, origin="lower",
                   extent=[X1.min(), X1.max(), X2.min(), X2.max()],
                   aspect="auto")
    fig.colorbar(im, ax=ax)
    try:
        cs = ax.contour(X1, X2, margin, levels=[0.0], linewidths=1.5)
        ax.clabel(cs, inline=True, fontsize=9, fmt="margin=0")
    except Exception:
        pass
    draw_rect(ax, outer, "outer")
    draw_rect(ax, inner, "inner")
    ax.set_title("margin = Vdot + alpha*V (heatmap)")
    ax.set_xlabel("x1_tilde")
    ax.set_ylabel("x2")

    # (3) bad mask
    ax = axs[1, 0]
    bad = (margin > 0.0).astype(float)
    im = ax.imshow(bad, origin="lower",
                   extent=[X1.min(), X1.max(), X2.min(), X2.max()],
                   aspect="auto")
    fig.colorbar(im, ax=ax)
    draw_rect(ax, outer, "outer")
    draw_rect(ax, inner, "inner")
    ax.set_title("bad regions (margin > 0)")
    ax.set_xlabel("x1_tilde")
    ax.set_ylabel("x2")

    # (4) quick compare V_inf vs W
    ax = axs[1, 1]
    im = ax.imshow(V_inf - W, origin="lower",
                   extent=[X1.min(), X1.max(), X2.min(), X2.max()],
                   aspect="auto")
    fig.colorbar(im, ax=ax)
    draw_rect(ax, outer, "outer")
    draw_rect(ax, inner, "inner")
    ax.set_title("V_inf - W_scaled (positive means W inside)")
    ax.set_xlabel("x1_tilde")
    ax.set_ylabel("x2")

    if outpath:
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        fig.savefig(outpath, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)

    # Basic diagnostics
    return {
        "Vblend_min": float(np.nanmin(V_blend)),
        "Vblend_max": float(np.nanmax(V_blend)),
        "margin_max": float(np.nanmax(margin)),
        "bad_frac_%": float(np.mean(margin > 0.0) * 100.0),
    }
