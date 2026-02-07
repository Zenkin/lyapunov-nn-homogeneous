from __future__ import annotations

from typing import Optional, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def _extent(X1: np.ndarray, X2: np.ndarray) -> tuple[float, float, float, float]:
    return float(X1.min()), float(X1.max()), float(X2.min()), float(X2.max())


def _bad_mask(V: np.ndarray, Vdot: np.ndarray) -> np.ndarray:
    return (V <= 0.0) | (Vdot >= 0.0)


def _draw_boxes(
    ax: plt.Axes,
    w_box: Optional[Tuple[float, float, float, float]],
    x_box: Optional[Tuple[float, float, float, float]],
) -> None:
    if w_box is not None:
        x1_min, x1_max, x2_min, x2_max = w_box
        ax.add_patch(
            Rectangle(
                (x1_min, x2_min),
                x1_max - x1_min,
                x2_max - x2_min,
                fill=False,
                lw=2.0,
                ec="tab:purple",
                ls="--",
                label="W box",
            )
        )
    if x_box is not None:
        x1_min, x1_max, x2_min, x2_max = x_box
        ax.add_patch(
            Rectangle(
                (x1_min, x2_min),
                x1_max - x1_min,
                x2_max - x2_min,
                fill=False,
                lw=2.0,
                ec="tab:orange",
                ls="-.",
                label="X box",
            )
        )


def _maybe_hatch_between(
    ax: plt.Axes,
    X1: np.ndarray,
    X2: np.ndarray,
    w_box: Optional[Tuple[float, float, float, float]],
    x_box: Optional[Tuple[float, float, float, float]],
) -> None:
    if w_box is None or x_box is None:
        return
    x1_min, x1_max, x2_min, x2_max = w_box
    xi_min, xi_max, yi_min, yi_max = x_box
    w_mask = (X1 >= x1_min) & (X1 <= x1_max) & (X2 >= x2_min) & (X2 <= x2_max)
    x_mask = (X1 >= xi_min) & (X1 <= xi_max) & (X2 >= yi_min) & (X2 <= yi_max)
    between = w_mask & (~x_mask)
    ax.contourf(
        X1,
        X2,
        between.astype(float),
        levels=[0.5, 1.5],
        colors="none",
        hatches=["////"],
        alpha=0.0,
    )


def plot_heatmap_pair(
    *,
    X1: np.ndarray,
    X2: np.ndarray,
    V: np.ndarray,
    Vdot: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Optional[str],
    show: bool,
    w_box: Optional[Tuple[float, float, float, float]] = None,
    x_box: Optional[Tuple[float, float, float, float]] = None,
    hatch_between: bool = False,
) -> None:
    ext = _extent(X1, X2)
    bad = _bad_mask(V, Vdot)
    bad_mask = np.ma.masked_where(~bad, bad.astype(float))

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle(title)

    im0 = axs[0].imshow(V.T, origin="lower", extent=ext, aspect="auto")
    axs[0].imshow(bad_mask.T, origin="lower", extent=ext, aspect="auto", cmap="Reds", alpha=0.35)
    cs0 = axs[0].contour(X1, X2, V, colors="white", linewidths=0.8)
    axs[0].clabel(cs0, inline=True, fontsize=8, fmt="%.2g")
    _draw_boxes(axs[0], w_box, x_box)
    if hatch_between:
        _maybe_hatch_between(axs[0], X1, X2, w_box, x_box)
    axs[0].set_title("V")
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel)
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(Vdot.T, origin="lower", extent=ext, aspect="auto", cmap="RdBu_r")
    axs[1].imshow(bad_mask.T, origin="lower", extent=ext, aspect="auto", cmap="Reds", alpha=0.35)
    cs1 = axs[1].contour(X1, X2, Vdot, colors="white", linewidths=0.8)
    axs[1].clabel(cs1, inline=True, fontsize=8, fmt="%.2g")
    _draw_boxes(axs[1], w_box, x_box)
    if hatch_between:
        _maybe_hatch_between(axs[1], X1, X2, w_box, x_box)
    axs[1].set_title("dV")
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(ylabel)
    fig.colorbar(im1, ax=axs[1])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"[plot] {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_surface_3d(
    *,
    X1: np.ndarray,
    X2: np.ndarray,
    Z: np.ndarray,
    title: str,
    save_path: Optional[str],
    show: bool,
) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X1,
        X2,
        Z,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        cmap="RdBu_r",
    )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel(title)
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"[plot] {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
