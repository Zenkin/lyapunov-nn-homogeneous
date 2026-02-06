# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Dict
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def add_hatch_legend_boxes(
    ax: plt.Axes,
    items: List[Dict],
    loc: str = "upper left",
    box_size: float = 0.04,
    pad: float = 0.012,
    rounding: float = 0.01,
) -> None:
    """Draw a tiny legend made of rounded hatched boxes only (no text)."""
    if not items:
        return

    if loc == "upper right":
        x0, y0, dx, dy = 1.0 - pad - box_size, 1.0 - pad - box_size, 0.0, -(box_size + pad)
    elif loc == "upper left":
        x0, y0, dx, dy = pad, 1.0 - pad - box_size, 0.0, -(box_size + pad)
    elif loc == "lower right":
        x0, y0, dx, dy = 1.0 - pad - box_size, pad, 0.0, +(box_size + pad)
    else:  # lower left
        x0, y0, dx, dy = pad, pad, 0.0, +(box_size + pad)

    for i, it in enumerate(items):
        color = it.get("color", "k")
        hatch = it.get("hatch", "////")
        hatch_lw = float(it.get("hatch_lw", 0.6))

        old = plt.rcParams.get("hatch.linewidth", 1.0)
        plt.rcParams["hatch.linewidth"] = hatch_lw
        try:
            r = FancyBboxPatch(
                (x0 + i * dx, y0 + i * dy),
                box_size,
                box_size,
                boxstyle=f"round,pad=0,rounding_size={rounding}",
                transform=ax.transAxes,
                facecolor="none",
                edgecolor=color,  # hatch color follows edgecolor
                linewidth=1.2,
                hatch=hatch,
                zorder=10,
                clip_on=False,
            )
            ax.add_patch(r)
        finally:
            plt.rcParams["hatch.linewidth"] = old
