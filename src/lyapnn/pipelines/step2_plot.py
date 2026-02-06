from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import os

import torch

from lyapnn.models.homog_v import HomogV
from lyapnn.systems.duffing_friction import Params, f_inf, f_full, equilibrium_x1
from lyapnn.training.derivatives import Vdot
from lyapnn.viz.grid import make_grid
from lyapnn.viz.heatmaps import plot4_full_with_X_and_glue, plot_vdot_bad_region_hatched
from lyapnn.viz.surfaces import plot_surface3d


@dataclass
class Step2PlotCfg:
    ckpt: str
    device: str = "cpu"
    outdir: str = "runs/step2"

    # What to render
    plot_inf: bool = True
    plot_full: bool = True
    plot_bad_regions: bool = True
    plot_3d: bool = True

    # Grids
    n: int = 301
    inf_x1_lim: Tuple[float, float] = (-6.0, 6.0)
    inf_x2_lim: Tuple[float, float] = (-6.0, 6.0)
    full_x1_lim: Tuple[float, float] = (-20.0, 20.0)
    full_x2_lim: Tuple[float, float] = (-20.0, 20.0)

    # Margin in diagnostics: Vdot + alpha_for_margin * V
    alpha_for_margin: float = 0.2

    # Glue/ring visualization options (used only in diag4_full)
    inner_scale: float = 0.85
    outer_scale: float = 1.15
    bad_bbox_scale: float = 1.25
    glue_labels: Tuple[str, str] = ("X", "B")
    glue_color: str = "tab:purple"


def load_v_ckpt(ckpt_path: str, device: str = "cpu") -> HomogV:
    ckpt = torch.load(ckpt_path, map_location=device)
    mu = float(ckpt.get("meta", {}).get("mu", 2.0))
    V = HomogV(mu=mu, eps=1e-3, hidden=64, depth=3).to(device)
    V.load_state_dict(ckpt["state_dict"])
    V.eval()
    return V


def _ensure_outdir(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir


def plot_step2(cfg: Step2PlotCfg) -> dict:
    """Render Step 2 diagnostics from a saved V checkpoint."""
    outdir = _ensure_outdir(cfg.outdir)

    p = Params()
    xeq = float(equilibrium_x1(p))
    V = load_v_ckpt(cfg.ckpt, device=cfg.device)

    def save(name: str) -> str:
        return os.path.join(outdir, name)

    if cfg.plot_inf:
        X1, X2, Xt = make_grid(cfg.inf_x1_lim, cfg.inf_x2_lim, cfg.n, cfg.device)
        with torch.no_grad():
            Vx = V(Xt).reshape(cfg.n, cfg.n).cpu().numpy()
        Vdx = Vdot(V, Xt, p, f_inf, create_graph=False).reshape(cfg.n, cfg.n).detach().cpu().numpy()
        plot4_full_with_X_and_glue(
            title="f_inf (shifted coords)",
            X1=X1, X2=X2, Vx=Vx, Vdx=Vdx,
            alpha_for_margin=cfg.alpha_for_margin,
            xlabel="x1_tilde", ylabel="x2",
            x1_lim=cfg.inf_x1_lim, x2_lim=cfg.inf_x2_lim,
            inner_scale=0.0, outer_scale=0.0,
            bad_bbox_scale=1.0,
            draw_bad_bbox=False,
            save_path=save("diag4_inf.png"),
        )

    if cfg.plot_full or cfg.plot_bad_regions or cfg.plot_3d:
        X1, X2, X = make_grid(cfg.full_x1_lim, cfg.full_x2_lim, cfg.n, cfg.device)
        Xt = X.clone()
        Xt[:, 0] = Xt[:, 0] - float(xeq)

        def f_full_from_xt(xt: torch.Tensor, pp: Params) -> torch.Tensor:
            xo = xt.clone()
            xo[:, 0] = xo[:, 0] + float(xeq)
            return f_full(xo, pp)

        with torch.no_grad():
            Vx = V(Xt).reshape(cfg.n, cfg.n).cpu().numpy()
        Vdx = Vdot(V, Xt, p, f_full_from_xt, create_graph=False).reshape(cfg.n, cfg.n).detach().cpu().numpy()

        if cfg.plot_full:
            plot4_full_with_X_and_glue(
                title=f"full system (axes original), x_eq={xeq:.6f}",
                X1=X1, X2=X2, Vx=Vx, Vdx=Vdx,
                alpha_for_margin=cfg.alpha_for_margin,
                xlabel="x1", ylabel="x2",
                x1_lim=cfg.full_x1_lim, x2_lim=cfg.full_x2_lim,
                inner_scale=cfg.inner_scale, outer_scale=cfg.outer_scale,
                bad_bbox_scale=cfg.bad_bbox_scale,
                glue_labels=cfg.glue_labels,
                glue_color=cfg.glue_color,
                draw_bad_bbox=False,
                save_path=save("diag4_full.png"),
            )

        if cfg.plot_bad_regions:
            plot_vdot_bad_region_hatched(
                title=f"Vdot bad regions (full system), x_eq={xeq:.6f}",
                X1=X1,
                X2=X2,
                Vdx=Vdx,
                xlabel="x1",
                ylabel="x2",
                x1_lim=cfg.full_x1_lim,
                x2_lim=cfg.full_x2_lim,
                bad_contour_lw=2.6,
                bad_hatch_lw=0.6,
                bad_hatch="////",
                blue_bbox_lw=2.6,
                outer_pad=4.0,
                blue_hatch_lw=0.6,
                blue_hatch="\\",
                save_path=save("vdot_bad_regions_ring.png"),
            )

        if cfg.plot_3d:
            plot_surface3d(
                title=f"3D surface: V(x) (x_eq={xeq:.6f})",
                X1=X1, X2=X2, Z=Vx,
                xlabel="x1", ylabel="x2", zlabel="V",
                save_path=save("V_full_3d.png"),
            )
            plot_surface3d(
                title=f"3D surface: Vdot(x) (x_eq={xeq:.6f})",
                X1=X1, X2=X2, Z=Vdx,
                xlabel="x1", ylabel="x2", zlabel="Vdot",
                sym_clip_q=0.99,
                save_path=save("Vdot_full_3d.png"),
            )

    return {"outdir": outdir, "x_eq": xeq}
