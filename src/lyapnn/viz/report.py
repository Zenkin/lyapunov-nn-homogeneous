from __future__ import annotations

from typing import Optional
import os
import numpy as np
import matplotlib.pyplot as plt


def _extent(X1: np.ndarray, X2: np.ndarray) -> tuple[float, float, float, float]:
    return float(X1.min()), float(X1.max()), float(X2.min()), float(X2.max())


def _bad_mask(V: np.ndarray, Vdot: np.ndarray) -> np.ndarray:
    return (V <= 0.0) | (Vdot >= 0.0)


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
) -> None:
    ext = _extent(X1, X2)
    bad = _bad_mask(V, Vdot)
    bad_mask = np.ma.masked_where(~bad, bad.astype(float))

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle(title)

    im0 = axs[0].imshow(V.T, origin="lower", extent=ext, aspect="auto")
    axs[0].imshow(bad_mask.T, origin="lower", extent=ext, aspect="auto", cmap="Reds", alpha=0.35)
    axs[0].set_title("V")
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel)
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(Vdot.T, origin="lower", extent=ext, aspect="auto", cmap="RdBu_r")
    axs[1].imshow(bad_mask.T, origin="lower", extent=ext, aspect="auto", cmap="Reds", alpha=0.35)
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
    ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel(title)
    ax.set_title(title)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"[plot] {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
