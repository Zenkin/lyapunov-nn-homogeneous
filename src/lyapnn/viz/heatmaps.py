# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from contextlib import contextmanager

from lyapnn.viz.grid import bbox_from_mask, scaled_bbox
from lyapnn.viz.legend import add_hatch_legend_boxes
from lyapnn.viz.patches import add_hatched_ring_patch


def _sym_clip_q(A: np.ndarray, q: float = 0.99) -> float:
    L = float(np.quantile(np.abs(A), q))
    if not np.isfinite(L) or L <= 0:
        L = float(np.max(np.abs(A)) + 1e-12)
    return L


def _levels_sym(L: float, n: int = 11) -> np.ndarray:
    n = max(3, int(n))
    if n % 2 == 0:
        n += 1
    return np.linspace(-float(L), float(L), n)


def _extent(X1: np.ndarray, X2: np.ndarray) -> Tuple[float, float, float, float]:
    return float(X1.min()), float(X1.max()), float(X2.min()), float(X2.max())


def _imshow(ax, Z: np.ndarray, ext, title: str, cmap=None, vmin=None, vmax=None):
    im = ax.imshow(Z.T, origin="lower", extent=ext, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    return im


def _save_and_show(fig, save_path: Optional[str]) -> None:
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"[plot] {save_path}")
    plt.show()


@contextmanager
def _hatch_lw(lw: float):
    old = plt.rcParams.get("hatch.linewidth", 1.0)
    plt.rcParams["hatch.linewidth"] = float(lw)
    try:
        yield
    finally:
        plt.rcParams["hatch.linewidth"] = old


def plot4_full_with_X_and_glue(
    title: str,
    X1: np.ndarray,
    X2: np.ndarray,
    Vx: np.ndarray,
    Vdx: np.ndarray,
    alpha_for_margin: float,
    xlabel: str,
    ylabel: str,
    x1_lim: Tuple[float, float],
    x2_lim: Tuple[float, float],
    inner_scale: float = 0.85,
    outer_scale: float = 1.15,
    bad_bbox_scale: float = 1.20,
    glue_labels: Tuple[str, str] = ("A", "B"),
    glue_color: str = "tab:purple",
    draw_bad_bbox: bool = False,
    save_path: Optional[str] = None,
) -> None:
    ext = _extent(X1, X2)
    x1_min, x1_max, x2_min, x2_max = ext

    X_mask = (Vdx < 0.0)
    bad_bbox0 = bbox_from_mask(X1, X2, ~X_mask)
    bad_bbox = None if bad_bbox0 is None else scaled_bbox(bad_bbox0, bad_bbox_scale, x1_lim, x2_lim)

    # DEBUG:
    #  Collect and print all bbox coordinates.
    #  Remove or redirect to logger after glue-region design is finalized.
    boxes = {}  # name -> (xmin, xmax, ymin, ymax)
    if bad_bbox is not None:
        boxes["bad_bbox"] = tuple(map(float, bad_bbox))
    # =========================================

    margin = Vdx + float(alpha_for_margin) * Vx
    signV = np.sign(Vdx)

    Ld = _sym_clip_q(Vdx, 0.99)
    Lm = _sym_clip_q(margin, 0.99)
    lev_d = _levels_sym(Ld, 11)
    lev_m = _levels_sym(Lm, 11)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=12)

    _imshow(axs[0, 0], Vx, ext, "V")

    _imshow(
        axs[0, 1], Vdx, ext,
        "Vdot (RdBu_r, symmetric clip) | contours incl. 0",
        cmap="RdBu_r", vmin=-Ld, vmax=+Ld
    )
    cs_d = axs[0, 1].contour(X1, X2, Vdx, levels=lev_d, colors="k", linewidths=0.8, alpha=0.55)
    axs[0, 1].clabel(cs_d, inline=True, fontsize=8, fmt="%.2g")
    axs[0, 1].contour(X1, X2, Vdx, levels=[0.0], colors="red", linewidths=2.6)
    axs[0, 1].text(
        x1_min + 0.02 * (x1_max - x1_min),
        x2_max - 0.06 * (x2_max - x2_min),
        "X: {Vdot < 0}",
        color="black",
        fontsize=10,
    )

    if bad_bbox is not None:
        xmin, xmax, ymin, ymax = bad_bbox

        if draw_bad_bbox:
            axs[0, 1].add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, lw=2.2, ec="black"))

        if inner_scale > 0 and outer_scale > 0:
            inner_bbox = scaled_bbox(bad_bbox, inner_scale, x1_lim, x2_lim)
            outer_bbox = scaled_bbox(bad_bbox, outer_scale, x1_lim, x2_lim)

            # DEBUG:
            #  Collect and print all bbox coordinates.
            #  Remove or redirect to logger after glue-region design is finalized.
            boxes["glue_inner_bbox"] = tuple(map(float, inner_bbox))
            boxes["glue_outer_bbox"] = tuple(map(float, outer_bbox))
            # =========================================

            for bb, lab in ((inner_bbox, glue_labels[0]), (outer_bbox, glue_labels[1])):
                xmn, xmx, ymn, ymx = bb
                axs[0, 1].add_patch(Rectangle((xmn, ymn), xmx - xmn, ymx - ymn, fill=False, lw=2.0, ls="--", ec=glue_color))
                axs[0, 1].text(xmn, ymx, f" {lab}", va="bottom", ha="left", fontsize=9, color="black")

    _imshow(axs[1, 0], signV, ext, "sign(Vdot)", cmap="RdBu_r", vmin=-1, vmax=1)

    _imshow(
        axs[1, 1], margin, ext,
        "margin = Vdot + alpha*V (RdBu_r, symmetric clip)",
        cmap="RdBu_r", vmin=-Lm, vmax=+Lm
    )
    cs_m = axs[1, 1].contour(X1, X2, margin, levels=lev_m, colors="k", linewidths=0.8, alpha=0.55)
    axs[1, 1].clabel(cs_m, inline=True, fontsize=8, fmt="%.2g")
    axs[1, 1].contour(X1, X2, margin, levels=[0.0], colors="red", linewidths=2.2)

    for ax in axs.flat:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # DEBUG:
    #  Collect and print all bbox coordinates.
    #  Remove or redirect to logger after glue-region design is finalized.
    print("[boxes] plot4_full_with_X_and_glue:")
    for k, (xmin, xmax, ymin, ymax) in boxes.items():
        print(f"  {k}: xmin={xmin:.6g}, xmax={xmax:.6g}, ymin={ymin:.6g}, ymax={ymax:.6g}")
    # =========================================

    _save_and_show(fig, save_path)


def plot_vdot_bad_region_hatched(
    title: str,
    X1: np.ndarray,
    X2: np.ndarray,
    Vdx: np.ndarray,
    xlabel: str,
    ylabel: str,
    x1_lim: Tuple[float, float],
    x2_lim: Tuple[float, float],
    bad_contour_lw: float = 2.6,
    bad_hatch_lw: float = 0.6,
    bad_hatch: str = "////",
    blue_bbox_lw: float = 2.6,
    outer_pad: float = 2.0,
    blue_hatch_lw: float = 0.6,
    blue_hatch: str = "////",
    red_bbox_lw: float = 2.6,
    red: str = "tab:red",
    blue: str = "tab:blue",
    green_hatch: str = "\\\\",
    green_hatch_lw: float = 0.6,
    green: str = "tab:green",
    legend_loc: str = "upper left",
    legend_box_size: float = 0.04,
    legend_pad: float = 0.012,
    legend_rounding: float = 0.01,
    save_path: Optional[str] = None,
) -> None:
    bad = (Vdx >= 0.0)
    ext = _extent(X1, X2)
    Ld = _sym_clip_q(Vdx, 0.99)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_title(title)

    _imshow(ax, Vdx, ext, "", cmap="RdBu_r", vmin=-Ld, vmax=+Ld)
    ax.contour(X1, X2, Vdx, levels=[0.0], colors="black", linewidths=bad_contour_lw)

    with _hatch_lw(bad_hatch_lw):
        ax.contourf(
            X1, X2, bad.astype(np.float32),
            levels=[0.5, 1.5],
            colors="none",
            hatches=[bad_hatch],
            alpha=0.0,
        )

    bad_bbox0 = bbox_from_mask(X1, X2, bad)
    inner_bbox = None if bad_bbox0 is None else scaled_bbox(bad_bbox0, 1.0, x1_lim, x2_lim)

    # DEBUG:
    #  Collect and print all bbox coordinates.
    #  Remove or redirect to logger after glue-region design is finalized.
    boxes = {}
    if inner_bbox is not None:
        boxes["inner_bbox"] = tuple(map(float, inner_bbox))
    # =========================================

    if inner_bbox is None:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        _save_and_show(fig, save_path)
        return

    xmin, xmax, ymin, ymax = inner_bbox
    pad = float(max(0.0, outer_pad))
    oxmin = max(x1_lim[0], xmin - pad); oxmax = min(x1_lim[1], xmax + pad)
    oymin = max(x2_lim[0], ymin - pad); oymax = min(x2_lim[1], ymax + pad)

    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, lw=blue_bbox_lw, ec=blue))
    ax.add_patch(Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin, fill=False, lw=blue_bbox_lw, ec=blue))

    rxmin = 0.5 * (xmin + oxmin); rxmax = 0.5 * (xmax + oxmax)
    rymin = 0.5 * (ymin + oymin); rymax = 0.5 * (ymax + oymax)
    ax.add_patch(Rectangle((rxmin, rymin), rxmax - rxmin, rymax - rymin, fill=False, lw=red_bbox_lw, ec=red))

    outer_bbox = (oxmin, oxmax, oymin, oymax)
    red_bbox = (rxmin, rxmax, rymin, rymax)

    # DEBUG:
    #  Collect and print all bbox coordinates.
    #  Remove or redirect to logger after glue-region design is finalized.
    boxes["outer_bbox"] = tuple(map(float, outer_bbox))
    boxes["red_bbox"] = tuple(map(float, red_bbox))
    # =========================================

    add_hatched_ring_patch(ax, outer_bbox=outer_bbox, inner_bbox=red_bbox,
                           color=green, hatch=green_hatch, hatch_lw=green_hatch_lw, zorder=2)
    add_hatched_ring_patch(ax, outer_bbox=outer_bbox, inner_bbox=inner_bbox,
                           color=blue, hatch=blue_hatch, hatch_lw=blue_hatch_lw, zorder=1)

    add_hatch_legend_boxes(
        ax,
        items=[
            {"color": blue, "hatch": blue_hatch, "hatch_lw": blue_hatch_lw},
            {"color": "black", "hatch": bad_hatch, "hatch_lw": bad_hatch_lw},
            {"color": green, "hatch": green_hatch, "hatch_lw": green_hatch_lw},
        ],
        loc=legend_loc,
        box_size=float(legend_box_size),
        pad=float(legend_pad),
        rounding=float(legend_rounding),
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # DEBUG:
    #  Collect and print all bbox coordinates.
    #  Remove or redirect to logger after glue-region design is finalized.
    print("[boxes] plot_vdot_bad_region_hatched:")
    for k, (xmin, xmax, ymin, ymax) in boxes.items():
        print(f"  {k}: xmin={xmin:.6g}, xmax={xmax:.6g}, ymin={ymin:.6g}, ymax={ymax:.6g}")
    # =========================================

    _save_and_show(fig, save_path)
