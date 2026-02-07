from __future__ import annotations

from typing import Optional, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle


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
    plot_inset: bool = False,
    inset_auto: bool = True,
    inset_zoom_frac: float = 0.2,
    inset_manual_bounds: Optional[Tuple[float, float, float, float]] = None,
    inset_position: Tuple[float, float, float, float] = (0.58, 0.62, 0.33, 0.3),
    inset_border_color: str = "red",
    inset_border_lw: float = 3.0,
    inset_connectors: bool = False,
) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d import proj3d

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

    if plot_inset:
        x1_min, x1_max, x2_min, x2_max = _extent(X1, X2)
        if inset_manual_bounds is not None:
            inset_x1_min, inset_x1_max, inset_x2_min, inset_x2_max = inset_manual_bounds
        else:
            if not inset_auto:
                raise ValueError("Inset bounds requested without auto mode or manual bounds.")
            min_idx = np.unravel_index(np.nanargmin(Z), Z.shape)
            x1_center = float(X1[min_idx])
            x2_center = float(X2[min_idx])
            x1_half = 0.5 * inset_zoom_frac * (x1_max - x1_min)
            x2_half = 0.5 * inset_zoom_frac * (x2_max - x2_min)
            inset_x1_min = max(x1_min, x1_center - x1_half)
            inset_x1_max = min(x1_max, x1_center + x1_half)
            inset_x2_min = max(x2_min, x2_center - x2_half)
            inset_x2_max = min(x2_max, x2_center + x2_half)

        ax_inset = fig.add_axes(inset_position, projection="3d")
        ax_inset.plot_surface(
            X1,
            X2,
            Z,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
            cmap=surf.cmap,
            norm=surf.norm,
        )
        ax_inset.set_xlim(inset_x1_min, inset_x1_max)
        ax_inset.set_ylim(inset_x2_min, inset_x2_max)
        ax_inset.set_xlabel("")
        ax_inset.set_ylabel("")
        ax_inset.set_zlabel("")
        ax_inset.set_title("")
        roll = getattr(ax, "roll", None)
        if roll is None:
            ax_inset.view_init(elev=ax.elev, azim=ax.azim)
        else:
            ax_inset.view_init(elev=ax.elev, azim=ax.azim, roll=roll)

        for spine in ax_inset.spines.values():
            spine.set_edgecolor(inset_border_color)
            spine.set_linewidth(inset_border_lw)

        z0 = float(np.nanmin(Z))
        ax.plot(
            [inset_x1_min, inset_x1_max],
            [inset_x2_min, inset_x2_min],
            [z0, z0],
            color=inset_border_color,
            lw=inset_border_lw,
        )
        ax.plot(
            [inset_x1_max, inset_x1_max],
            [inset_x2_min, inset_x2_max],
            [z0, z0],
            color=inset_border_color,
            lw=inset_border_lw,
        )
        ax.plot(
            [inset_x1_max, inset_x1_min],
            [inset_x2_max, inset_x2_max],
            [z0, z0],
            color=inset_border_color,
            lw=inset_border_lw,
        )
        ax.plot(
            [inset_x1_min, inset_x1_min],
            [inset_x2_max, inset_x2_min],
            [z0, z0],
            color=inset_border_color,
            lw=inset_border_lw,
        )

        if inset_connectors:
            def _to_fig_coords(x: float, y: float, z: float) -> Tuple[float, float]:
                xp, yp, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
                x_disp, y_disp = ax.transData.transform((xp, yp))
                return tuple(fig.transFigure.inverted().transform((x_disp, y_disp)))

            upper_left = _to_fig_coords(inset_x1_min, inset_x2_max, z0)
            lower_right = _to_fig_coords(inset_x1_max, inset_x2_min, z0)

            con_ul = ConnectionPatch(
                xyA=(0, 1),
                coordsA=ax_inset.transAxes,
                xyB=upper_left,
                coordsB=fig.transFigure,
                color=inset_border_color,
                lw=max(1.0, inset_border_lw / 2.0),
                clip_on=False,
            )
            con_lr = ConnectionPatch(
                xyA=(1, 0),
                coordsA=ax_inset.transAxes,
                xyB=lower_right,
                coordsB=fig.transFigure,
                color=inset_border_color,
                lw=max(1.0, inset_border_lw / 2.0),
                clip_on=False,
            )
            fig.add_artist(con_ul)
            fig.add_artist(con_lr)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"[plot] {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
