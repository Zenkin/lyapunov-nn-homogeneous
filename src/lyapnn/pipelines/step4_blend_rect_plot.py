#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Tuple

import os
import inspect
import importlib

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from lyapnn.systems.duffing_friction import Params, f_full, equilibrium_x1
from lyapnn.viz.blend_rect import Rect, blend_weight_rect, plot_blend_maps


@dataclass
class Step4Cfg:
    ckpt_V: str
    ckpt_W: str
    outdir: str = "runs/step4"

    # Grid in SHIFTED coords
    x1_min: float = 0.0
    x1_max: float = 10.0
    x2_min: float = -10.0
    x2_max: float = 2.0
    grid: int = 401

    # Rectangles (SHIFTED coords)
    outer: Rect = field(default_factory=lambda: Rect(0.0, 10.0, -10.0, 2.0))
    inner: Rect = field(default_factory=lambda: Rect(0.0, 1.0, -1.0, 1.0))

    # Margin parameter
    alpha: float = 0.2

    # Scale W to stay inside V (NO averages): k = min(V/W) over points in mix band
    eps_scale: float = 1e-12

    # Plots
    plot_3d: bool = False
    show: bool = True
    save: bool = True


class _VWrapper(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if y.ndim == 1:
            y = y[:, None]
        return y


class _TWrapper(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _LocalTNet(nn.Module):
    """
    Local copy of the Step3 TNet architecture to avoid import-path coupling.

    Matches:
      - bias=False in all Linear layers
      - activation: Tanh
      - depth: number of hidden layers (>=1)
      - output dim: 2
    """
    def __init__(self, hidden: int = 64, depth: int = 2):
        super().__init__()
        depth = int(depth)
        hidden = int(hidden)
        if depth < 1:
            raise ValueError("depth must be >= 1")

        layers = [nn.Linear(2, hidden, bias=False), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden, bias=False), nn.Tanh()]
        layers += [nn.Linear(hidden, 2, bias=False)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _try_import_class(candidates: List[Tuple[str, str]]) -> Optional[type]:
    for mod_name, cls_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, cls_name):
                return getattr(mod, cls_name)
        except Exception:
            continue
    return None


def _instantiate_with_args(cls: type, args: Dict[str, Any]) -> nn.Module:
    sig = inspect.signature(cls.__init__)
    kwargs = {}
    for k, v in args.items():
        if k in sig.parameters:
            kwargs[k] = v
    return cls(**kwargs)


def _load_step2_V(path: str, device: torch.device, dtype: torch.dtype) -> nn.Module:
    # 1) Try TorchScript
    try:
        m = torch.jit.load(path, map_location=device)
        if isinstance(m, torch.jit.RecursiveScriptModule):
            return _VWrapper(m).to(device=device, dtype=dtype).eval()
    except Exception:
        pass

    ck = torch.load(path, map_location=device)

    # 2) Full module
    if isinstance(ck, nn.Module):
        return _VWrapper(ck).to(device=device, dtype=dtype).eval()

    # 3) Dict with state_dict
    if isinstance(ck, dict) and "state_dict" in ck:
        state = ck["state_dict"]
        args = {}
        for k in ("args", "model_args", "cfg", "train_cfg"):
            if isinstance(ck.get(k, None), dict):
                args.update(ck[k])

        candidates = [
            ("lyapnn.pipelines.step2_train_v_infty", "HomogV"),
            ("lyapnn.pipelines.step2_train_v_inf", "HomogV"),
            ("lyapnn.pipelines.step2_train", "HomogV"),
            ("lyapnn.models.homog_v", "HomogV"),
            ("lyapnn.models.v_infty", "HomogV"),
            ("lyapnn.models", "HomogV"),
        ]
        Cls = _try_import_class(candidates)
        if Cls is not None:
            try:
                m = _instantiate_with_args(Cls, args)
            except Exception:
                m = Cls()
            m.load_state_dict(state)
            return _VWrapper(m).to(device=device, dtype=dtype).eval()

        keys = list(state.keys())[:20]
        raise ValueError(
            "Unrecognized step2_V checkpoint: could not import model class for state_dict. "
            f"Checkpoint={path}. state_dict sample keys={keys}"
        )

    raise ValueError(f"Unrecognized step2_V checkpoint format: {path}")


def _load_step3_W(path: str, device: torch.device, dtype: torch.dtype) -> nn.Module:
    """
    Robust loader for step3 W checkpoint without importing step3 pipeline module.
    """
    # 1) Try TorchScript
    try:
        m = torch.jit.load(path, map_location=device)
        if isinstance(m, torch.jit.RecursiveScriptModule):
            return _TWrapper(m).to(device=device, dtype=dtype).eval()
    except Exception:
        pass

    ck = torch.load(path, map_location=device)

    # 2) Full module
    if isinstance(ck, nn.Module):
        return _TWrapper(ck).to(device=device, dtype=dtype).eval()

    # 3) Dict with state_dict
    if isinstance(ck, dict) and "state_dict" in ck:
        state = ck["state_dict"]
        args = ck.get("args", {}) if isinstance(ck.get("args", None), dict) else {}
        hidden = int(args.get("hidden", 64))
        depth = int(args.get("depth", 2))
        net = _LocalTNet(hidden=hidden, depth=depth)
        net.load_state_dict(state)
        return _TWrapper(net).to(device=device, dtype=dtype).eval()

    raise ValueError(f"Unrecognized W checkpoint format: {path}")


def _shifted_to_original(x_tilde: torch.Tensor, x_eq: float) -> torch.Tensor:
    return torch.stack([x_tilde[:, 0] + float(x_eq), x_tilde[:, 1]], dim=1)


def plot_step4_rect_blend(cfg: Step4Cfg) -> Dict[str, float]:
    device = torch.device("cpu")
    dtype = torch.float32

    p = Params()
    x_eq = float(equilibrium_x1(p))

    Vnet = _load_step2_V(cfg.ckpt_V, device, dtype)
    Tnet = _load_step3_W(cfg.ckpt_W, device, dtype)

    xs = np.linspace(cfg.x1_min, cfg.x1_max, int(cfg.grid))
    ys = np.linspace(cfg.x2_min, cfg.x2_max, int(cfg.grid))
    X1, X2 = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([X1.reshape(-1), X2.reshape(-1)], axis=1)
    xt = torch.tensor(pts, dtype=dtype, device=device)

    with torch.no_grad():
        V_inf = Vnet(xt).cpu().numpy().reshape(X1.shape)
        T = Tnet(xt)
        W = torch.sum(T * T, dim=1, keepdim=True).cpu().numpy().reshape(X1.shape)

    s = blend_weight_rect(X1, X2, cfg.outer, cfg.inner)

    mid = (s > 0.0) & (s < 1.0)
    if np.any(mid):
        ratio = V_inf[mid] / (W[mid] + float(cfg.eps_scale))
        k = float(np.min(ratio))
    else:
        k = 1.0

    Wk = k * W
    V_blend = (1.0 - s) * V_inf + s * Wk

    # Margin via autograd on blended V
    s_t = torch.tensor(s.reshape(-1, 1), dtype=dtype, device=device)
    xtg = xt.clone().detach().requires_grad_(True)

    Vg = Vnet(xtg)
    Tg = Tnet(xtg)
    Wg = (Tg * Tg).sum(dim=1, keepdim=True) * float(k)
    Vbg = (1.0 - s_t) * Vg + s_t * Wg

    g = torch.autograd.grad(Vbg.sum(), xtg, create_graph=True)[0]
    x_orig = _shifted_to_original(xtg, x_eq)
    fx = f_full(x_orig, p)
    Vdot = (g * fx).sum(dim=1, keepdim=True)
    margin = (Vdot + float(cfg.alpha) * Vbg).detach().cpu().numpy().reshape(X1.shape)

    os.makedirs(cfg.outdir, exist_ok=True)
    diag_path = os.path.join(cfg.outdir, "blend_rect_diag.png") if cfg.save else None

    diag = plot_blend_maps(
        X1=X1, X2=X2,
        V_inf=V_inf,
        W=Wk,
        V_blend=V_blend,
        margin=margin,
        outer=cfg.outer,
        inner=cfg.inner,
        outpath=diag_path,
        show=cfg.show,
    )
    diag["k_scale"] = float(k)

    if cfg.plot_3d:
        _plot_3d_surface(X1, X2, V_blend, os.path.join(cfg.outdir, "Vblend_3d.png") if cfg.save else None, cfg.show, "V_blend")
        _plot_3d_surface(X1, X2, margin, os.path.join(cfg.outdir, "margin_3d.png") if cfg.save else None, cfg.show, "margin")

    return diag


def _plot_3d_surface(X1: np.ndarray, X2: np.ndarray, Z: np.ndarray, outpath: Optional[str], show: bool, title: str) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax.set_xlabel("x1_tilde")
    ax.set_ylabel("x2")
    ax.set_zlabel(title)
    ax.set_title(title)

    if outpath:
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        fig.savefig(outpath, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)
