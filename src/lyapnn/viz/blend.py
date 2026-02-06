# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


@dataclass
class BlendVizCfg:
    title: str = "V_blend"
    xlabel: str = "x1"
    ylabel: str = "x2"

    # Overlay options
    draw_rect: bool = True
    rect_color: str = "k"
    rect_ls: str = "--"
    rect_lw: float = 1.5

    draw_contours: bool = True
    contour_color: str = "k"
    contour_lw: float = 1.5

    # Heatmap options
    clip_q: float = 0.995

    # Save path
    save_path: Optional[str] = None


def _sym_clip(Z: np.ndarray, q: float) -> float:
    q = float(q)
    q = 0.5 if q <= 0 else (0.999 if q >= 1 else q)
    L = float(np.quantile(np.abs(Z), q))
    if not np.isfinite(L) or L <= 0:
        L = float(np.max(np.abs(Z)) + 1e-12)
    return L


def plot_blend_maps(
    X1: np.ndarray,
    X2: np.ndarray,
    V_inf: np.ndarray,
    W_sc: np.ndarray,
    V_blend: np.ndarray,
    margin: np.ndarray,
    c1: float,
    c2: float,
    rect_original: Optional[Tuple[float, float, float, float]],
    cfg: BlendVizCfg,
) -> Dict[str, float]:
    """
    Produce a compact 2x2 figure:
      (1) V_blend heatmap + contours of V_inf=c1,c2
      (2) margin heatmap (Vdot + alpha V)
      (3) bad regions mask (margin>=0)
      (4) compare contours: V_inf=c1 and W_sc=c1 (optional sanity)
    """
    ext = (float(X1.min()), float(X1.max()), float(X2.min()), float(X2.max()))
    vL = _sym_clip(margin, cfg.clip_q)

    fig, axs = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    fig.suptitle(cfg.title)

    # --- (1) V_blend ---
    ax = axs[0, 0]
    im = ax.imshow(V_blend.T, origin="lower", extent=ext, aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title("V_blend")
    ax.set_xlabel(cfg.xlabel)
    ax.set_ylabel(cfg.ylabel)

    if cfg.draw_contours:
        cs = ax.contour(X1, X2, V_inf, levels=[float(c1), float(c2)],
                        colors=cfg.contour_color, linewidths=cfg.contour_lw)
        ax.clabel(cs, inline=True, fontsize=9, fmt="V=%.3g")

    if cfg.draw_rect and rect_original is not None:
        x1a, x1b, x2a, x2b = rect_original
        ax.add_patch(Rectangle((x1a, x2a), x1b - x1a, x2b - x2a,
                               fill=False, ec=cfg.rect_color, ls=cfg.rect_ls, lw=cfg.rect_lw))

    # --- (2) margin ---
    ax = axs[0, 1]
    im = ax.imshow(margin.T, origin="lower", extent=ext, aspect="auto", vmin=-vL, vmax=vL, cmap="RdBu_r")
    plt.colorbar(im, ax=ax)
    ax.set_title("margin = Vdot + alpha*V (blend)")
    ax.set_xlabel(cfg.xlabel)
    ax.set_ylabel(cfg.ylabel)
    ax.contour(X1, X2, margin, levels=[0.0], colors="k", linewidths=1.5)

    # --- (3) bad mask ---
    ax = axs[1, 0]
    bad = (margin >= 0.0).astype(np.float32)
    im = ax.imshow(bad.T, origin="lower", extent=ext, aspect="auto", vmin=0, vmax=1, cmap="gray_r")
    plt.colorbar(im, ax=ax)
    ax.set_title("bad regions (margin >= 0)")
    ax.set_xlabel(cfg.xlabel)
    ax.set_ylabel(cfg.ylabel)

    # --- (4) contour sanity ---
    ax = axs[1, 1]
    im = ax.imshow(V_inf.T, origin="lower", extent=ext, aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title("Contours sanity: V_inf and W_sc")
    ax.set_xlabel(cfg.xlabel)
    ax.set_ylabel(cfg.ylabel)
    ax.contour(X1, X2, V_inf, levels=[float(c1)], colors="k", linewidths=2.0)
    ax.contour(X1, X2, W_sc, levels=[float(c1)], colors="tab:orange", linewidths=2.0)

    if cfg.save_path:
        os.makedirs(os.path.dirname(cfg.save_path) or ".", exist_ok=True)
        fig.savefig(cfg.save_path, dpi=200)
    plt.show()

    # diagnostics
    return {
        "margin_max": float(np.max(margin)),
        "bad_frac_%": float(np.mean(margin >= 0.0) * 100.0),
    }
