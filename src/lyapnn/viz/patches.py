# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path


def add_rect(
    ax: plt.Axes,
    bbox: Tuple[float, float, float, float],
    label: str = "",
    lw: float = 2.0,
    linestyle: str = "-",
    alpha: float = 1.0,
    edgecolor: Optional[str] = None,
    text_color: str = "black",
) -> Rectangle:
    xmin, xmax, ymin, ymax = bbox
    rect = Rectangle(
        (xmin, ymin),
        width=(xmax - xmin),
        height=(ymax - ymin),
        fill=False,
        linewidth=lw,
        linestyle=linestyle,
        alpha=alpha,
    )
    if edgecolor is not None:
        rect.set_edgecolor(edgecolor)
    ax.add_patch(rect)
    if label:
        ax.text(xmin, ymax, f" {label}", va="bottom", ha="left", fontsize=9, color=text_color)
    return rect


def ring_path(
    outer_bbox: Tuple[float, float, float, float],
    inner_bbox: Tuple[float, float, float, float],
) -> Path:
    """Compound path: outer rectangle minus inner rectangle."""
    oxmin, oxmax, oymin, oymax = outer_bbox
    ixmin, ixmax, iymin, iymax = inner_bbox

    verts = [
        (oxmin, oymin), (oxmax, oymin), (oxmax, oymax), (oxmin, oymax), (oxmin, oymin),
        (ixmin, iymin), (ixmin, iymax), (ixmax, iymax), (ixmax, iymin), (ixmin, iymin),
    ]
    codes = [
        Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY,
        Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY,
    ]
    return Path(verts, codes)


def add_hatched_ring_patch(
    ax: plt.Axes,
    outer_bbox: Tuple[float, float, float, float],
    inner_bbox: Tuple[float, float, float, float],
    color: str,
    hatch: str,
    hatch_lw: float = 0.6,
    zorder: int = 3,
) -> PathPatch:
    """Add a hatched rectangular ring with a guaranteed hatch color."""
    path = ring_path(outer_bbox, inner_bbox)

    old = plt.rcParams.get("hatch.linewidth", 1.0)
    plt.rcParams["hatch.linewidth"] = float(hatch_lw)
    try:
        patch = PathPatch(
            path,
            facecolor="none",
            edgecolor=color,
            hatch=hatch,
            linewidth=0.0,
            zorder=int(zorder),
        )
        ax.add_patch(patch)
        return patch
    finally:
        plt.rcParams["hatch.linewidth"] = old
