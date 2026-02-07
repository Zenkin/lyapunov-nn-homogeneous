from __future__ import annotations

from typing import Optional, Sequence
import os
import numpy as np
import matplotlib.pyplot as plt


def _extent(X1: np.ndarray, X2: np.ndarray) -> tuple[float, float, float, float]:
    return float(X1.min()), float(X1.max()), float(X2.min()), float(X2.max())


def _bad_mask(V: np.ndarray, Vdot: np.ndarray) -> np.ndarray:
    return (V <= 0.0) | (Vdot >= 0.0)


def _zoom_bbox(
    X1: np.ndarray,
    X2: np.ndarray,
    Z: np.ndarray,
    zoom_frac: float,
    manual_bounds: Optional[Sequence[float]],
) -> tuple[float, float, float, float]:
    x1_min, x1_max, x2_min, x2_max = _extent(X1, X2)
    if manual_bounds is not None:
        return tuple(float(v) for v in manual_bounds)  # type: ignore[return-value]

    idx = np.unravel_index(np.argmin(Z), Z.shape)
    x1_center = float(X1[idx])
    x2_center = float(X2[idx])
    x1_half = 0.5 * zoom_frac * (x1_max - x1_min)
    x2_half = 0.5 * zoom_frac * (x2_max - x2_min)
    x1_min_zoom = max(x1_min, x1_center - x1_half)
    x1_max_zoom = min(x1_max, x1_center + x1_half)
    x2_min_zoom = max(x2_min, x2_center - x2_half)
    x2_max_zoom = min(x2_max, x2_center + x2_half)
    return x1_min_zoom, x1_max_zoom, x2_min_zoom, x2_max_zoom


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
    plot_inset: bool = False,
    inset_manual: bool = False,
    inset_zoom_frac: float = 0.2,
    inset_x1_min: Optional[float] = None,
    inset_x1_max: Optional[float] = None,
    inset_x2_min: Optional[float] = None,
    inset_x2_max: Optional[float] = None,
    inset_position: Optional[Sequence[float]] = None,
    inset_border_color: str = "red",
    inset_border_lw: float = 3.0,
    inset_connectors: bool = True,
) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d import art3d
    from matplotlib.patches import Rectangle, ConnectionPatch

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel(title)
    ax.set_title(title)

    if plot_inset:
        manual_bounds = None
        if inset_manual:
            if None in (inset_x1_min, inset_x1_max, inset_x2_min, inset_x2_max):
                raise ValueError("Inset manual mode requires all inset_x1/x2 bounds.")
            manual_bounds = (inset_x1_min, inset_x1_max, inset_x2_min, inset_x2_max)

        x1_min_zoom, x1_max_zoom, x2_min_zoom, x2_max_zoom = _zoom_bbox(
            X1, X2, Z, inset_zoom_frac, manual_bounds
        )
        inset_position = inset_position or (0.58, 0.62, 0.33, 0.3)
        inset_ax = fig.add_axes(inset_position, projection="3d")
        inset_ax.plot_surface(
            X1,
            X2,
            Z,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
            cmap=surface.cmap,
            norm=surface.norm,
        )
        inset_ax.set_xlim(x1_min_zoom, x1_max_zoom)
        inset_ax.set_ylim(x2_min_zoom, x2_max_zoom)
        inset_ax.set_zlim(float(Z.min()), float(Z.max()))
        inset_ax.view_init(elev=ax.elev, azim=ax.azim, roll=getattr(ax, "roll", 0.0))
        inset_ax.set_title("")
        inset_ax.set_xlabel("")
        inset_ax.set_ylabel("")
        inset_ax.set_zlabel("")

        for spine in inset_ax.spines.values():
            spine.set_edgecolor(inset_border_color)
            spine.set_linewidth(inset_border_lw)

        z_plane = float(Z.min())
        bbox = Rectangle(
            (x1_min_zoom, x2_min_zoom),
            x1_max_zoom - x1_min_zoom,
            x2_max_zoom - x2_min_zoom,
            fill=False,
            edgecolor=inset_border_color,
            linewidth=inset_border_lw,
        )
        ax.add_patch(bbox)
        art3d.pathpatch_2d_to_3d(bbox, z=z_plane, zdir="z")

        if inset_connectors:
            connectors = [
                ((x1_min_zoom, x2_max_zoom), (0, 1)),
                ((x1_max_zoom, x2_min_zoom), (1, 0)),
            ]
            for data_xy, inset_xy in connectors:
                conn = ConnectionPatch(
                    xyA=inset_xy,
                    coordsA=inset_ax.transAxes,
                    xyB=data_xy,
                    coordsB=ax.transData,
                    color=inset_border_color,
                    linewidth=max(1.0, inset_border_lw * 0.6),
                )
                fig.add_artist(conn)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"[plot] {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
