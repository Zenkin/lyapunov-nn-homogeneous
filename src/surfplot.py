from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter


@dataclass(frozen=True)
class Plot3DStyle:
    cmap: str = "RdBu_r"
    font_size: int = 14
    sci: bool = True
    ticks: int = 5


def _apply_axis_formatting(ax: plt.Axes, style: Plot3DStyle) -> None:
    ax.tick_params(labelsize=style.font_size)
    if style.ticks:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=style.ticks))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=style.ticks))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=style.ticks))

    if not style.sci:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.zaxis.set_major_formatter(formatter)


def plot_surface_3d(
    func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    x1_min: float,
    x1_max: float,
    x2_min: float,
    x2_max: float,
    grid: int = 101,
    title_tex: Optional[str] = None,
    cbar_label: Optional[str] = None,
    style: Optional[Plot3DStyle] = None,
    save: Optional[str] = None,
    show: bool = False,
) -> None:
    style = style or Plot3DStyle()
    x1 = np.linspace(x1_min, x1_max, grid)
    x2 = np.linspace(x2_min, x2_max, grid)
    X1, X2 = np.meshgrid(x1, x2, indexing="ij")
    Z = func(X1, X2)

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
        cmap=style.cmap,
    )
    ax.set_xlabel("x1", fontsize=style.font_size)
    ax.set_ylabel("x2", fontsize=style.font_size)

    title = title_tex or ""
    if title:
        ax.set_title(title, fontsize=style.font_size)

    z_label = cbar_label or title_tex or "z"
    ax.set_zlabel(z_label, fontsize=style.font_size)
    _apply_axis_formatting(ax, style)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=style.font_size)
    if style.ticks:
        cbar.locator = MaxNLocator(nbins=style.ticks)
        cbar.update_ticks()
    if not style.sci:
        cbar.formatter = ScalarFormatter(useMathText=True)
        cbar.formatter.set_scientific(False)
        cbar.update_ticks()

    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, dpi=200)
        print(f"[plot] {save}")

    if show:
        plt.show()
    else:
        plt.close(fig)
