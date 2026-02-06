from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import os
import numpy as np
import torch

from lyapnn.models.homog_v import HomogV
from lyapnn.systems.duffing_friction import Params, f_full, equilibrium_x1
from lyapnn.viz.grid import make_grid
from lyapnn.viz.surfaces import plot_surface3d
from lyapnn.viz.blend import BlendVizCfg, plot_blend_maps


# --- W model (same as step3) ---
class TNet(torch.nn.Module):
    """MLP without biases -> T(0)=0 exactly. W(x)=||T(x)||^2."""
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


def load_v_ckpt(ckpt_path: str, device: str = "cpu") -> HomogV:
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt.get("meta", {})
    mu = float(meta.get("mu", 2.0))
    hidden = int(meta.get("hidden", 64))
    depth = int(meta.get("depth", 3))
    V = HomogV(mu=mu, eps=1e-3, hidden=hidden, depth=depth).to(device)
    V.load_state_dict(ckpt["state_dict"])
    V.eval()
    return V


def load_w_ckpt(ckpt_path: str, device: str = "cpu", dtype: str = "float32") -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt.get("args", {})
    hidden = int(ckpt.get("hidden", args.get("hidden", 64)))
    depth = int(ckpt.get("depth", args.get("depth", 2)))
    x_eq = float(ckpt.get("x_eq", 0.0))
    p = Params()
    # restore params if present
    if "params" in ckpt and isinstance(ckpt["params"], dict):
        for k, v in ckpt["params"].items():
            if hasattr(p, k):
                setattr(p, k, float(v))
    dt = torch.float32 if dtype == "float32" else torch.float64
    net = TNet(hidden=hidden, depth=depth).to(device=device, dtype=dt)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return {"net": net, "x_eq": x_eq, "p": p, "hidden": hidden, "depth": depth, "dtype": dt}


def shifted_to_original(x_tilde: torch.Tensor, x_eq: float) -> torch.Tensor:
    return torch.stack([x_tilde[:, 0] + float(x_eq), x_tilde[:, 1]], dim=1)


def _smoothstep01(t: torch.Tensor) -> torch.Tensor:
    # clamp to [0,1], then 3t^2-2t^3
    t = torch.clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _pick_scale_k_minratio(
    V: HomogV,
    Wnet: torch.nn.Module,
    Xt: torch.Tensor,
    c1: float,
    band_rel: float,
    rect_mask: torch.Tensor,
) -> float:
    """
    Choose k using REAL sampled points (grid points), without averaging:
      k = min_{i in band} V(Xt_i) / (W(Xt_i)+eps)
    This guarantees k*W <= V on the selected band (conservative, avoids 'bump').
    """
    with torch.no_grad():
        Vv = V(Xt).squeeze(1)
        T = Wnet(Xt)
        Wv = torch.sum(T * T, dim=1)
        eps = 1e-12

        band = torch.abs(Vv - float(c1)) <= (float(band_rel) * max(1e-9, abs(float(c1))))
        sel = rect_mask & band & (Wv > 0)

        if sel.sum().item() < 32:
            # fallback: use nearest points by |V-c1|
            k_near = 4096
            idx = torch.argsort(torch.abs(Vv - float(c1)))[:k_near]
            sel = rect_mask[idx] & (Wv[idx] > 0)
            if sel.sum().item() < 32:
                return 1.0

            Vsel = Vv[idx][sel]
            Wsel = Wv[idx][sel]
        else:
            Vsel = Vv[sel]
            Wsel = Wv[sel]

        ratios = Vsel / (Wsel + eps)
        k = float(torch.min(ratios).item())
        if not np.isfinite(k) or k <= 0:
            k = 1.0
        return k


def _eval_blend_in_batches(
    V: HomogV,
    Wnet: torch.nn.Module,
    Xt: torch.Tensor,
    p: Params,
    x_eq: float,
    alpha: float,
    rect_shifted: Tuple[float, float, float, float],
    c1: float,
    c2: float,
    k: float,
    chunk: int = 8192,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate on a grid (Xt in shifted coords):
      returns (V_inf, W_sc, V_blend, margin) as numpy arrays (N,)
    margin uses full system: Vdot_blend + alpha * V_blend
    """
    n = Xt.shape[0]
    dev = Xt.device
    dt = Xt.dtype

    x1_min, x1_max, x2_min, x2_max = [float(v) for v in rect_shifted]

    V_inf = torch.empty((n,), device=dev, dtype=dt)
    W_sc = torch.empty((n,), device=dev, dtype=dt)
    V_blend = torch.empty((n,), device=dev, dtype=dt)
    margin = torch.empty((n,), device=dev, dtype=dt)

    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        xt = Xt[i0:i1].detach().clone().requires_grad_(True)

        Vv = V(xt).squeeze(1)
        T = Wnet(xt)
        Wv = torch.sum(T * T, dim=1)
        Wv_sc = float(k) * Wv

        # rect mask in shifted coords
        in_rect = (xt[:, 0] >= x1_min) & (xt[:, 0] <= x1_max) & (xt[:, 1] >= x2_min) & (xt[:, 1] <= x2_max)

        # blending weight via V levels (c1 -> 0, c2 -> 1)
        denom = float(c1) - float(c2)
        denom = denom if abs(denom) > 1e-12 else (1e-12 if denom >= 0 else -1e-12)
        t = (float(c1) - Vv) / denom
        s = _smoothstep01(t)

        # outside rect: force V_inf
        s = torch.where(in_rect, s, torch.zeros_like(s))

        Vb = (1.0 - s) * Vv + s * Wv_sc

        # Vdot along full system (original coords): x = (xt1 + xeq, xt2)
        xo = shifted_to_original(xt, float(x_eq))
        fx = f_full(xo, p)
        g = torch.autograd.grad(Vb.sum(), xt, create_graph=False)[0]
        Vdotb = torch.sum(g * fx, dim=1)

        mb = Vdotb + float(alpha) * Vb

        V_inf[i0:i1] = Vv.detach()
        W_sc[i0:i1] = Wv_sc.detach()
        V_blend[i0:i1] = Vb.detach()
        margin[i0:i1] = mb.detach()

    return (
        V_inf.cpu().numpy(),
        W_sc.cpu().numpy(),
        V_blend.cpu().numpy(),
        margin.cpu().numpy(),
    )


@dataclass
class Step4PlotCfg:
    ckpt_V: str
    ckpt_W: str
    outdir: str = "runs/step4"
    device: str = "cpu"
    dtype: str = "float32"

    # plot grid in ORIGINAL axes (like step2 full)
    n: int = 301
    x1_lim: Tuple[float, float] = (-20.0, 20.0)
    x2_lim: Tuple[float, float] = (-20.0, 20.0)

    # blending rectangle in SHIFTED coords
    x1_min: float = 0.0
    x1_max: float = 10.0
    x2_min: float = -10.0
    x2_max: float = 2.0

    # blending levels (in terms of V_inf values)
    c1: float = 2.0
    c2: float = 0.5
    band_rel: float = 0.03  # used to pick scale points around c1

    # margin alpha
    alpha: float = 0.2

    # 3D options
    plot_3d: bool = True
    save: bool = True


def plot_step4_blend(cfg: Step4PlotCfg) -> Dict[str, Any]:
    os.makedirs(cfg.outdir, exist_ok=True)
    dev = torch.device(cfg.device)
    dt = torch.float32 if cfg.dtype == "float32" else torch.float64

    p = Params()
    x_eq_sys = float(equilibrium_x1(p))

    V = load_v_ckpt(cfg.ckpt_V, device=cfg.device).to(dev)
    Winfo = load_w_ckpt(cfg.ckpt_W, device=cfg.device, dtype=cfg.dtype)
    Wnet = Winfo["net"].to(dev)
    x_eq = float(Winfo.get("x_eq", x_eq_sys))
    if abs(x_eq - x_eq_sys) > 1e-6:
        print(f"[warn] x_eq mismatch: system={x_eq_sys:.12g} ckpt_W={x_eq:.12g} (using ckpt_W)")

    # grid in ORIGINAL axes, but evaluate V/W on SHIFTED Xt
    X1, X2, X = make_grid(cfg.x1_lim, cfg.x2_lim, cfg.n, cfg.device)
    Xt = X.clone()
    Xt[:, 0] = Xt[:, 0] - float(x_eq)
    Xt = Xt.to(device=dev, dtype=dt)

    # rect mask for scale picking (in shifted coords)
    rect_mask = (
        (Xt[:, 0] >= float(cfg.x1_min)) & (Xt[:, 0] <= float(cfg.x1_max)) &
        (Xt[:, 1] >= float(cfg.x2_min)) & (Xt[:, 1] <= float(cfg.x2_max))
    )

    k = _pick_scale_k_minratio(V, Wnet, Xt, cfg.c1, cfg.band_rel, rect_mask)

    V_inf, W_sc, V_blend, margin = _eval_blend_in_batches(
        V=V, Wnet=Wnet, Xt=Xt, p=p, x_eq=x_eq, alpha=cfg.alpha,
        rect_shifted=(cfg.x1_min, cfg.x1_max, cfg.x2_min, cfg.x2_max),
        c1=cfg.c1, c2=cfg.c2, k=k,
        chunk=8192,
    )

    V_inf2 = V_inf.reshape(cfg.n, cfg.n)
    W_sc2 = W_sc.reshape(cfg.n, cfg.n)
    Vb2 = V_blend.reshape(cfg.n, cfg.n)
    m2 = margin.reshape(cfg.n, cfg.n)

    # rectangle in ORIGINAL axes (for overlay)
    rect_orig = (float(cfg.x1_min) + x_eq, float(cfg.x1_max) + x_eq, float(cfg.x2_min), float(cfg.x2_max))

    save_path = os.path.join(cfg.outdir, "blend_diag.png") if cfg.save else None
    viz_cfg = BlendVizCfg(
        title=f"Blend: V_inf + W (k={k:.3g}), c1={cfg.c1}, c2={cfg.c2}, x_eq={x_eq:.6g}",
        xlabel="x1", ylabel="x2", save_path=save_path
    )
    diag = plot_blend_maps(
        X1=X1, X2=X2,
        V_inf=V_inf2, W_sc=W_sc2,
        V_blend=Vb2, margin=m2,
        c1=cfg.c1, c2=cfg.c2,
        rect_original=rect_orig,
        cfg=viz_cfg,
    )

    if cfg.plot_3d:
        # 3D V_blend
        plot_surface3d(X1, X2, Vb2, title="V_blend (3D)", save_path=(os.path.join(cfg.outdir, "Vblend_3d.png") if cfg.save else None))
        # 3D margin
        plot_surface3d(X1, X2, m2, title="margin_blend (3D)", save_path=(os.path.join(cfg.outdir, "margin_3d.png") if cfg.save else None))

    info = {
        "outdir": cfg.outdir,
        "x_eq": x_eq,
        "k": k,
        "diag": diag,
    }
    print(f"[step4] k={k:.6g} bad_frac={diag['bad_frac_%']:.3f}% margin_max={diag['margin_max']:.3e}")
    return info
