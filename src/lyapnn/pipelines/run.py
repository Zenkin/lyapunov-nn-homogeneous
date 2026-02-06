from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Tuple, Dict, Any

import numpy as np
import torch

from lyapnn.models.homog_v import HomogV
from lyapnn.systems.duffing_friction import Params, f_inf, f_full, equilibrium_x1
from lyapnn.training.derivatives import Vdot
from lyapnn.training.vinf import VinfTrainCfg, train_vinf
from lyapnn.training.w_local import WTrainCfg, WNet, train_w_local
from lyapnn.viz.grid import make_grid
from lyapnn.viz.report import plot_heatmap_pair, plot_surface_3d


@dataclass
class RunCfg:
    outdir: str
    device: str
    dtype: str
    seed: int
    omega: Tuple[float, float, float, float]
    w_box: Tuple[float, float, float, float]
    x_box: Tuple[float, float, float, float]
    grid: int
    vinf_mu: float
    vinf_alpha: float
    vinf_hidden: int
    vinf_depth: int
    vinf_steps: int
    vinf_batch: int
    vinf_lr: float
    vinf_log_every: int
    vinf_normalize_margin: bool
    w_hidden: int
    w_depth: int
    w_steps: int
    w_batch: int
    w_lr: float
    w_log_every: int
    w_r_min: float
    w_margin: float
    w_alpha_pos: float
    w_eps_s: float
    w_lam_s: float
    show: bool = True
    save: bool = True


def _ensure_dir(*parts: str) -> str:
    path = os.path.join(*parts)
    os.makedirs(path, exist_ok=True)
    return path


def _eval_w_and_dot(
    net: WNet,
    xt: torch.Tensor,
    p: Params,
    x_eq: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xt = xt.clone().detach().requires_grad_(True)
    T = net(xt)
    W = torch.sum(T * T, dim=1, keepdim=True)
    gradW = torch.autograd.grad(W.sum(), xt, create_graph=False)[0]
    x = torch.stack([xt[:, 0] + float(x_eq), xt[:, 1]], dim=1)
    fx = f_full(x, p)
    Wdot = torch.sum(gradW * fx, dim=1, keepdim=True)
    return W, Wdot


def _eval_vfull_and_dot(
    V: HomogV,
    xt: torch.Tensor,
    p: Params,
    x_eq: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    def f_full_from_xt(x_tilde: torch.Tensor, pp: Params) -> torch.Tensor:
        x = x_tilde.clone()
        x[:, 0] = x[:, 0] + float(x_eq)
        return f_full(x, pp)

    Vx = V(xt)
    Vdot_full = Vdot(V, xt, p, f_full_from_xt, create_graph=False)
    return Vx, Vdot_full


def _smoothstep01(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _blend_weight(
    x1: np.ndarray,
    x2: np.ndarray,
    w_box: Tuple[float, float, float, float],
    x_box: Tuple[float, float, float, float],
) -> np.ndarray:
    x1_min, x1_max, x2_min, x2_max = w_box
    xi_min, xi_max, yi_min, yi_max = x_box

    inside = (x1 >= xi_min) & (x1 <= xi_max) & (x2 >= yi_min) & (x2 <= yi_max)
    outside = (x1 < x1_min) | (x1 > x1_max) | (x2 < x2_min) | (x2 > x2_max)

    s = np.zeros_like(x1, dtype=float)
    s[inside] = 1.0
    mid = (~inside) & (~outside)
    if not np.any(mid):
        return s

    def axis_u(x, omin, omax, imin, imax):
        u = np.zeros_like(x, dtype=float)
        left = x < imin
        right = x > imax
        den_l = max(1e-12, (imin - omin))
        den_r = max(1e-12, (omax - imax))
        u[left] = (imin - x[left]) / den_l
        u[right] = (x[right] - imax) / den_r
        return np.clip(u, 0.0, 1.0)

    u1 = axis_u(x1[mid], x1_min, x1_max, xi_min, xi_max)
    u2 = axis_u(x2[mid], x2_min, x2_max, yi_min, yi_max)
    u = np.maximum(u1, u2)
    s[mid] = 1.0 - _smoothstep01(u)
    return s


def _blend_v(
    Vfull: np.ndarray,
    dVfull: np.ndarray,
    W: np.ndarray,
    dW: np.ndarray,
    w_box: Tuple[float, float, float, float],
    x_box: Tuple[float, float, float, float],
    x1_tilde: np.ndarray,
    x2_tilde: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    s = _blend_weight(x1_tilde, x2_tilde, w_box, x_box)
    V = Vfull.copy()
    dV = dVfull.copy()

    m_full = dVfull + float(alpha) * Vfull
    m_w = dW + float(alpha) * W

    ok_full = (Vfull > 0.0) & (m_full < 0.0)
    ok_w = (W > 0.0) & (m_w < 0.0)

    only_full = ok_full & (~ok_w)
    only_w = ok_w & (~ok_full)
    both = ok_full & ok_w

    V[only_full] = Vfull[only_full]
    dV[only_full] = dVfull[only_full]
    V[only_w] = W[only_w]
    dV[only_w] = dW[only_w]

    if np.any(both):
        weight = s[both]
        V[both] = (1.0 - weight) * Vfull[both] + weight * W[both]
        dV[both] = (1.0 - weight) * dVfull[both] + weight * dW[both]

    fallback = ~(only_full | only_w | both)
    if np.any(fallback):
        weight = s[fallback]
        V[fallback] = (1.0 - weight) * Vfull[fallback] + weight * W[fallback]
        dV[fallback] = (1.0 - weight) * dVfull[fallback] + weight * dW[fallback]

    return V, dV


def run_pipeline(cfg: RunCfg) -> Dict[str, Any]:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    dev = torch.device(cfg.device)
    dt = torch.float32 if cfg.dtype == "float32" else torch.float64

    omega = cfg.omega
    w_box = cfg.w_box
    x_box = cfg.x_box

    out_root = _ensure_dir(cfg.outdir)
    out_vinf = _ensure_dir(out_root, "vinf")
    out_vfull = _ensure_dir(out_root, "vfull")
    out_w = _ensure_dir(out_root, "w")
    out_final = _ensure_dir(out_root, "final")

    p = Params()
    x_eq = float(equilibrium_x1(p))
    w_box_orig = (
        w_box[0] + x_eq,
        w_box[1] + x_eq,
        w_box[2],
        w_box[3],
    )
    x_box_orig = (
        x_box[0] + x_eq,
        x_box[1] + x_eq,
        x_box[2],
        x_box[3],
    )

    vinf_cfg = VinfTrainCfg(
        seed=cfg.seed,
        device=cfg.device,
        mu=cfg.vinf_mu,
        alpha=cfg.vinf_alpha,
        hidden=cfg.vinf_hidden,
        depth=cfg.vinf_depth,
        steps=cfg.vinf_steps,
        batch=cfg.vinf_batch,
        lr=cfg.vinf_lr,
        log_every=cfg.vinf_log_every,
        normalize_margin=cfg.vinf_normalize_margin,
        box_x1_min=omega[0],
        box_x1_max=omega[1],
        box_x2_min=omega[2],
        box_x2_max=omega[3],
    )
    V = train_vinf(vinf_cfg, save_path=os.path.join(out_vinf, "vinf.pt"))

    X1, X2, Xt = make_grid((omega[0], omega[1]), (omega[2], omega[3]), cfg.grid, cfg.device)
    with torch.no_grad():
        Vinf = V(Xt).reshape(cfg.grid, cfg.grid).detach().cpu().numpy()
    Vdot_inf = Vdot(V, Xt, p, f_inf, create_graph=False).reshape(cfg.grid, cfg.grid).detach().cpu().numpy()

    plot_heatmap_pair(
        X1=X1,
        X2=X2,
        V=Vinf,
        Vdot=Vdot_inf,
        title="V_inf and dV_inf",
        xlabel="x1_tilde",
        ylabel="x2",
        save_path=os.path.join(out_vinf, "vinf_heatmaps.png") if cfg.save else None,
        show=cfg.show,
        w_box=w_box,
        x_box=x_box,
    )
    plot_surface_3d(
        X1=X1,
        X2=X2,
        Z=Vinf,
        title="V_inf (3D)",
        save_path=os.path.join(out_vinf, "vinf_3d.png") if cfg.save else None,
        show=cfg.show,
    )
    plot_surface_3d(
        X1=X1,
        X2=X2,
        Z=Vdot_inf,
        title="dV_inf (3D)",
        save_path=os.path.join(out_vinf, "dvinf_3d.png") if cfg.save else None,
        show=cfg.show,
    )

    X1f, X2f, X = make_grid((omega[0], omega[1]), (omega[2], omega[3]), cfg.grid, cfg.device)
    Xt_full = X.clone()
    Xt_full[:, 0] = Xt_full[:, 0] - float(x_eq)
    with torch.no_grad():
        Vfull = V(Xt_full).reshape(cfg.grid, cfg.grid).detach().cpu().numpy()
    Vdot_full = Vdot(V, Xt_full, p, f_full, create_graph=False).reshape(cfg.grid, cfg.grid).detach().cpu().numpy()

    plot_heatmap_pair(
        X1=X1f,
        X2=X2f,
        V=Vfull,
        Vdot=Vdot_full,
        title="V_full and dV_full",
        xlabel="x1",
        ylabel="x2",
        save_path=os.path.join(out_vfull, "vfull_heatmaps.png") if cfg.save else None,
        show=cfg.show,
        w_box=w_box_orig,
        x_box=x_box_orig,
    )
    plot_surface_3d(
        X1=X1f,
        X2=X2f,
        Z=Vfull,
        title="V_full (3D)",
        save_path=os.path.join(out_vfull, "vfull_3d.png") if cfg.save else None,
        show=cfg.show,
    )
    plot_surface_3d(
        X1=X1f,
        X2=X2f,
        Z=Vdot_full,
        title="dV_full (3D)",
        save_path=os.path.join(out_vfull, "dvfull_3d.png") if cfg.save else None,
        show=cfg.show,
    )

    w_cfg = WTrainCfg(
        seed=cfg.seed,
        device=cfg.device,
        dtype=cfg.dtype,
        hidden=cfg.w_hidden,
        depth=cfg.w_depth,
        steps=cfg.w_steps,
        batch=cfg.w_batch,
        lr=cfg.w_lr,
        log_every=cfg.w_log_every,
        x1_min=w_box[0],
        x1_max=w_box[1],
        x2_min=w_box[2],
        x2_max=w_box[3],
        r_min=cfg.w_r_min,
        margin=cfg.w_margin,
        alpha_pos=cfg.w_alpha_pos,
        eps_s=cfg.w_eps_s,
        lam_s=cfg.w_lam_s,
    )
    Wnet = train_w_local(w_cfg, save_path=os.path.join(out_w, "w_model.pt"))
    Wnet = Wnet.to(device=dev, dtype=dt)

    X1w, X2w, Xt_w = make_grid((omega[0], omega[1]), (omega[2], omega[3]), cfg.grid, cfg.device)
    Xt_w[:, 0] = Xt_w[:, 0] - float(x_eq)
    Wval, Wdot = _eval_w_and_dot(Wnet, Xt_w, p, x_eq)
    Wval = Wval.reshape(cfg.grid, cfg.grid).detach().cpu().numpy()
    Wdot = Wdot.reshape(cfg.grid, cfg.grid).detach().cpu().numpy()

    plot_heatmap_pair(
        X1=X1w,
        X2=X2w,
        V=Wval,
        Vdot=Wdot,
        title="W and dW",
        xlabel="x1",
        ylabel="x2",
        save_path=os.path.join(out_w, "w_heatmaps.png") if cfg.save else None,
        show=cfg.show,
        w_box=w_box_orig,
        x_box=x_box_orig,
    )
    plot_surface_3d(
        X1=X1w,
        X2=X2w,
        Z=Wval,
        title="W (3D)",
        save_path=os.path.join(out_w, "w_3d.png") if cfg.save else None,
        show=cfg.show,
    )
    plot_surface_3d(
        X1=X1w,
        X2=X2w,
        Z=Wdot,
        title="dW (3D)",
        save_path=os.path.join(out_w, "dw_3d.png") if cfg.save else None,
        show=cfg.show,
    )

    Xt_final = X.clone()
    Xt_final[:, 0] = Xt_final[:, 0] - float(x_eq)
    with torch.no_grad():
        Vfull_all = V(Xt_final).reshape(cfg.grid, cfg.grid).detach().cpu().numpy()
    Vdot_full_all = Vdot(V, Xt_final, p, f_full, create_graph=False).reshape(cfg.grid, cfg.grid).detach().cpu().numpy()
    W_all, Wdot_all = _eval_w_and_dot(Wnet, Xt_final, p, x_eq)
    W_all = W_all.reshape(cfg.grid, cfg.grid).detach().cpu().numpy()
    Wdot_all = Wdot_all.reshape(cfg.grid, cfg.grid).detach().cpu().numpy()

    x_tilde = Xt_final.detach().cpu().numpy().reshape(cfg.grid, cfg.grid, 2)
    V_final, dV_final = _blend_v(
        Vfull_all,
        Vdot_full_all,
        W_all,
        Wdot_all,
        w_box,
        x_box,
        x_tilde[:, :, 0],
        x_tilde[:, :, 1],
        cfg.vinf_alpha,
    )

    plot_heatmap_pair(
        X1=X1f,
        X2=X2f,
        V=V_final,
        Vdot=dV_final,
        title="V_final and dV_final",
        xlabel="x1",
        ylabel="x2",
        save_path=os.path.join(out_final, "v_final_heatmaps.png") if cfg.save else None,
        show=cfg.show,
        w_box=w_box_orig,
        x_box=x_box_orig,
        hatch_between=True,
    )
    plot_surface_3d(
        X1=X1f,
        X2=X2f,
        Z=V_final,
        title="V_final (3D)",
        save_path=os.path.join(out_final, "v_final_3d.png") if cfg.save else None,
        show=cfg.show,
    )
    plot_surface_3d(
        X1=X1f,
        X2=X2f,
        Z=dV_final,
        title="dV_final (3D)",
        save_path=os.path.join(out_final, "dv_final_3d.png") if cfg.save else None,
        show=cfg.show,
    )

    return {
        "outdir": out_root,
        "x_eq": x_eq,
        "omega": omega,
        "w_box": w_box,
        "x_box": x_box,
    }
