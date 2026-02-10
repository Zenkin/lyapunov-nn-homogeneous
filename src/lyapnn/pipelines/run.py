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


def _save_plot_data_npz(path: str, **arrays: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"[data] {path}")


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


def _blend_v(
    Vfull: np.ndarray,
    dVfull: np.ndarray,
    W: np.ndarray,
    dW: np.ndarray,
    w_mask: np.ndarray,
    x_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    V = Vfull.copy()
    dV = dVfull.copy()

    inside = x_mask
    between = w_mask & (~x_mask)

    V[inside] = W[inside]
    dV[inside] = dW[inside]

    if np.any(between):
        use_w = W[between] >= Vfull[between]
        V_between = Vfull[between].copy()
        dV_between = dVfull[between].copy()
        V_between[use_w] = W[between][use_w]
        dV_between[use_w] = dW[between][use_w]
        V[between] = V_between
        dV[between] = dV_between

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
    )
    _save_plot_data_npz(
        os.path.join(out_vinf, "vinf_heatmaps.npz"),
        X1=X1,
        X2=X2,
        V=Vinf,
        Vdot=Vdot_inf,
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
    )
    _save_plot_data_npz(
        os.path.join(out_vfull, "vfull_heatmaps.npz"),
        X1=X1f,
        X2=X2f,
        V=Vfull,
        Vdot=Vdot_full,
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

    X1w, X2w, Xt_w = make_grid((w_box[0], w_box[1]), (w_box[2], w_box[3]), cfg.grid, cfg.device)
    Wval, Wdot = _eval_w_and_dot(Wnet, Xt_w, p, x_eq)
    Wval = Wval.reshape(cfg.grid, cfg.grid).detach().cpu().numpy()
    Wdot = Wdot.reshape(cfg.grid, cfg.grid).detach().cpu().numpy()

    plot_heatmap_pair(
        X1=X1w,
        X2=X2w,
        V=Wval,
        Vdot=Wdot,
        title="W and dW",
        xlabel="x1_tilde",
        ylabel="x2",
        save_path=os.path.join(out_w, "w_heatmaps.png") if cfg.save else None,
        show=cfg.show,
    )
    _save_plot_data_npz(
        os.path.join(out_w, "w_heatmaps.npz"),
        X1=X1w,
        X2=X2w,
        V=Wval,
        Vdot=Wdot,
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
    w_mask = (
        (x_tilde[:, :, 0] >= w_box[0]) & (x_tilde[:, :, 0] <= w_box[1]) &
        (x_tilde[:, :, 1] >= w_box[2]) & (x_tilde[:, :, 1] <= w_box[3])
    )
    x_mask = (
        (x_tilde[:, :, 0] >= x_box[0]) & (x_tilde[:, :, 0] <= x_box[1]) &
        (x_tilde[:, :, 1] >= x_box[2]) & (x_tilde[:, :, 1] <= x_box[3])
    )

    V_final, dV_final = _blend_v(Vfull_all, Vdot_full_all, W_all, Wdot_all, w_mask, x_mask)

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
    )
    _save_plot_data_npz(
        os.path.join(out_final, "v_final_heatmaps.npz"),
        X1=X1f,
        X2=X2f,
        V=V_final,
        Vdot=dV_final,
    )

    _save_plot_data_npz(
        os.path.join(out_root, "all_plot_data.npz"),
        vinf_X1=X1,
        vinf_X2=X2,
        vinf_V=Vinf,
        vinf_dV=Vdot_inf,
        vfull_X1=X1f,
        vfull_X2=X2f,
        vfull_V=Vfull,
        vfull_dV=Vdot_full,
        w_X1=X1w,
        w_X2=X2w,
        w_V=Wval,
        w_dV=Wdot,
        final_X1=X1f,
        final_X2=X2f,
        final_V=V_final,
        final_dV=dV_final,
        final_Vfull_all=Vfull_all,
        final_dVfull_all=Vdot_full_all,
        final_W_all=W_all,
        final_dW_all=Wdot_all,
        final_w_mask=w_mask.astype(np.uint8),
        final_x_mask=x_mask.astype(np.uint8),
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
