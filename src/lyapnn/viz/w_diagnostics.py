#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch

from lyapnn.systems.duffing_friction import Params, f_full, equilibrium_x1
from lyapnn.viz.heatmaps import plot4_full_with_X_and_glue, plot_vdot_bad_region_hatched
from lyapnn.viz.surfaces import plot_surface3d


@dataclass
class WPlotCfg:
    # Region is specified in SHIFTED coordinates: x_tilde = [x1 - x_eq, x2]
    x1_min: float
    x1_max: float
    x2_min: float
    x2_max: float
    grid: int = 401
    alpha: float = 1.0

    # 2D saves (optional)
    save_path_4: Optional[str] = None
    save_path_bad: Optional[str] = None

    # 3D toggles + saves
    plot_3d: bool = True
    save_path_3d_w: Optional[str] = None
    save_path_3d_wdot: Optional[str] = None

    xlabel: str = "x1~ (shifted)"
    ylabel: str = "x2"


class TNet(torch.nn.Module):
    """MLP without biases -> T(0)=0 exactly."""

    def __init__(self, hidden: int = 64, depth: int = 2):
        super().__init__()
        assert depth >= 1
        layers = [torch.nn.Linear(2, hidden, bias=False), torch.nn.Tanh()]
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(hidden, hidden, bias=False), torch.nn.Tanh()]
        layers += [torch.nn.Linear(hidden, 2, bias=False)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_tnet_ckpt(ckpt_path: str, device: torch.device, dtype: torch.dtype) -> Tuple[TNet, Dict[str, Any]]:
    data = torch.load(ckpt_path, map_location=device)
    meta: Dict[str, Any] = {}

    if isinstance(data, dict) and ("model_state" in data or "state_dict" in data):
        sd = data.get("model_state", data.get("state_dict"))
        meta = {k: v for k, v in data.items() if k not in ("model_state", "state_dict")}
        args = meta.get("args", {}) if isinstance(meta.get("args"), dict) else {}
        hidden = int(meta.get("hidden", args.get("hidden", 64)))
        depth = int(meta.get("depth", args.get("depth", 2)))
    else:
        sd = data
        hidden, depth = 64, 2

    net = TNet(hidden=hidden, depth=depth).to(device=device, dtype=dtype)
    net.load_state_dict(sd)
    net.eval()
    return net, meta


def _grid_shifted(cfg: WPlotCfg) -> Tuple[np.ndarray, np.ndarray]:
    g = int(cfg.grid)
    x1 = np.linspace(float(cfg.x1_min), float(cfg.x1_max), g)
    x2 = np.linspace(float(cfg.x2_min), float(cfg.x2_max), g)
    X1, X2 = np.meshgrid(x1, x2, indexing="ij")
    return X1, X2


def _eval_W_Wdot_on_grid(
    net: TNet,
    p: Params,
    x_eq: float,
    X1: np.ndarray,
    X2: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[np.ndarray, np.ndarray]:
    # Flatten grid in SHIFTED coords
    x_t = np.stack([X1.reshape(-1), X2.reshape(-1)], axis=1)
    xt = torch.tensor(x_t, device=device, dtype=dtype, requires_grad=True)

    # W = ||T(xt)||^2
    T = net(xt)
    W = (T * T).sum(dim=1, keepdim=True)  # (N,1)

    # grad W wrt shifted coords
    gW = torch.autograd.grad(W.sum(), xt, create_graph=False)[0]  # (N,2)

    # full dynamics computed in ORIGINAL coords, but xdot in shifted coords is same components
    x_orig = torch.stack([xt[:, 0] + float(x_eq), xt[:, 1]], dim=1)
    fx = f_full(x_orig, p)  # (N,2)
    Wdot = (gW * fx).sum(dim=1, keepdim=True)  # (N,1)

    W_np = W.detach().cpu().numpy().reshape(X1.shape)
    Wd_np = Wdot.detach().cpu().numpy().reshape(X1.shape)
    return W_np, Wd_np


def plot_w_diagnostics_from_ckpt(
    ckpt_path: str,
    cfg: WPlotCfg,
    p: Optional[Params] = None,
    device: str = "cpu",
    dtype: str = "float32",
) -> Dict[str, Any]:
    """
    Produce W diagnostics heatmaps and (optionally) 3D surfaces.

    Returns a dict with x_eq and bbox info (when available).
    """
    dev = torch.device(device)
    dt = torch.float32 if str(dtype).lower() in ("float32", "fp32") else torch.float64

    p = Params() if p is None else p
    x_eq = float(equilibrium_x1(p))

    net, meta = _load_tnet_ckpt(ckpt_path, dev, dt)
    X1, X2 = _grid_shifted(cfg)
    W, Wdot = _eval_W_Wdot_on_grid(net, p, x_eq, X1, X2, dev, dt)

    # 2D diagnostics (reuse your existing viz)
    plot4_full_with_X_and_glue(
        title="W diagnostics (shifted grid)",
        X1=X1, X2=X2,
        Vx=W, Vdx=Wdot,
        alpha_for_margin=float(cfg.alpha),
        xlabel=cfg.xlabel, ylabel=cfg.ylabel,
        x1_lim=(float(cfg.x1_min), float(cfg.x1_max)),
        x2_lim=(float(cfg.x2_min), float(cfg.x2_max)),
        save_path=cfg.save_path_4,
    )

    plot_vdot_bad_region_hatched(
        title="BAD region (Wdot >= 0) with rings (shifted grid)",
        X1=X1, X2=X2,
        Vdx=Wdot,
        xlabel=cfg.xlabel, ylabel=cfg.ylabel,
        x1_lim=(float(cfg.x1_min), float(cfg.x1_max)),
        x2_lim=(float(cfg.x2_min), float(cfg.x2_max)),
        save_path=cfg.save_path_bad,
    )

    # 3D surfaces
    if bool(cfg.plot_3d):
        plot_surface3d(
            title="W(x) surface (shifted)",
            X1=X1, X2=X2, Z=W,
            xlabel=cfg.xlabel, ylabel=cfg.ylabel, zlabel="W",
            save_path=cfg.save_path_3d_w,
            cmap="viridis",
            add_colorbar=True,
        )
        plot_surface3d(
            title="Wdot(x) surface (shifted)",
            X1=X1, X2=X2, Z=Wdot,
            xlabel=cfg.xlabel, ylabel=cfg.ylabel, zlabel="Wdot",
            save_path=cfg.save_path_3d_wdot,
            cmap="RdBu_r",
            sym_clip_q=0.99,
            add_colorbar=True,
        )

    return {"x_eq": x_eq, "meta": meta}
