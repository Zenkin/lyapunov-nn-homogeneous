from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from lyapnn.systems.duffing_friction import Params, f_full, equilibrium_x1


@dataclass
class WTrainCfg:
    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"
    hidden: int = 64
    depth: int = 2
    steps: int = 50000
    batch: int = 2048
    lr: float = 1e-3
    log_every: int = 200
    x1_min: float = -5.0
    x1_max: float = 5.0
    x2_min: float = -5.0
    x2_max: float = 5.0
    r_min: float = 0.0
    margin: float = 0.0
    alpha_pos: float = 1e-3
    eps_s: float = 1e-2
    lam_s: float = 1e-3
    inner_x1_min: Optional[float] = None
    inner_x1_max: Optional[float] = None
    inner_x2_min: Optional[float] = None
    inner_x2_max: Optional[float] = None
    inner_half_ratio: float = 0.5
    eps_transition: float = 1e-2
    lam_transition: float = 1.0
    lam_dom: float = 1.0


class WNet(nn.Module):
    """MLP without biases => T(0)=0 exactly, so W(x)=||T(x)||^2 has W(0)=0."""

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

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _sample_rect(batch: int, x1_min: float, x1_max: float, x2_min: float, x2_max: float,
                 device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    u = torch.rand((batch, 2), device=device, dtype=dtype)
    x1 = x1_min + (x1_max - x1_min) * u[:, 0]
    x2 = x2_min + (x2_max - x2_min) * u[:, 1]
    return torch.stack([x1, x2], dim=1)


def _apply_rmin_mask(x: torch.Tensor, r_min: float) -> torch.Tensor:
    r_min = float(r_min)
    if r_min <= 0.0:
        return x
    r = torch.sqrt(torch.sum(x * x, dim=1) + 1e-12)
    return x[r >= r_min]


def _sigma_min_2x2(J11: torch.Tensor, J12: torch.Tensor, J21: torch.Tensor, J22: torch.Tensor) -> torch.Tensor:
    a = J11 * J11 + J21 * J21
    b = J11 * J12 + J21 * J22
    c = J12 * J12 + J22 * J22
    disc = (a - c) * (a - c) + 4.0 * b * b
    sdisc = torch.sqrt(torch.clamp(disc, min=0.0))
    lam_min = 0.5 * (a + c - sdisc)
    lam_min = torch.clamp(lam_min, min=0.0)
    return torch.sqrt(lam_min + 1e-18)


def _compute_loss(
    model: nn.Module,
    x_tilde: torch.Tensor,
    p: Params,
    x_eq: float,
    margin: float,
    alpha_pos: float,
    eps_s: float,
    lam_s: float,
    v_inner: Optional[torch.Tensor],
    mask_transition: Optional[torch.Tensor],
    mask_outer: Optional[torch.Tensor],
    eps_transition: float,
    lam_transition: float,
    lam_dom: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    x_tilde = x_tilde.requires_grad_(True)

    T = model(x_tilde)
    W = torch.sum(T * T, dim=1)
    gradW = torch.autograd.grad(W.sum(), x_tilde, create_graph=True)[0]

    x = torch.stack([x_tilde[:, 0] + float(x_eq), x_tilde[:, 1]], dim=1)
    f = f_full(x, p)
    Wdot = torch.sum(gradW * f, dim=1)
    S = Wdot + W

    reluS = torch.relu(S + float(margin))
    loss_main = torch.mean(reluS * reluS)

    xnorm2 = torch.sum(x_tilde * x_tilde, dim=1)
    pos_gap = torch.relu(float(alpha_pos) * xnorm2 - W)
    loss_pos = torch.mean(pos_gap * pos_gap)

    T1, T2 = T[:, 0], T[:, 1]
    g1 = torch.autograd.grad(T1.sum(), x_tilde, create_graph=True)[0]
    g2 = torch.autograd.grad(T2.sum(), x_tilde, create_graph=True)[0]
    smin = _sigma_min_2x2(g1[:, 0], g1[:, 1], g2[:, 0], g2[:, 1])
    s_gap = torch.relu(float(eps_s) - smin)
    loss_s = torch.mean(s_gap * s_gap)

    zero = torch.zeros((), device=x_tilde.device, dtype=x_tilde.dtype)

    if mask_transition is not None and torch.any(mask_transition):
        tr_gap = torch.relu(float(eps_transition) - W[mask_transition])
        loss_transition = torch.mean(tr_gap * tr_gap)
    else:
        loss_transition = zero

    if v_inner is not None and mask_outer is not None and torch.any(mask_outer):
        dom_gap = torch.relu(v_inner[mask_outer] - W[mask_outer])
        loss_dom = torch.mean(dom_gap * dom_gap)
    else:
        loss_dom = zero

    total = (
        loss_main
        + loss_pos
        + float(lam_s) * loss_s
        + float(lam_transition) * loss_transition
        + float(lam_dom) * loss_dom
    )

    with torch.no_grad():
        diag = {
            "loss_total": float(total.item()),
            "loss_main": float(loss_main.item()),
            "loss_pos": float(loss_pos.item()),
            "loss_s": float(loss_s.item()),
            "loss_transition": float(loss_transition.item()),
            "loss_dom": float(loss_dom.item()),
            "max_S": float(S.max().item()),
            "frac_S_pos_%": float((S > 0.0).float().mean().item()) * 100.0,
            "min_smin": float(smin.min().item()),
        }
    return total, diag


def _inner_masks(x_tilde: torch.Tensor, cfg: WTrainCfg) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    inner_bounds = (
        cfg.inner_x1_min,
        cfg.inner_x1_max,
        cfg.inner_x2_min,
        cfg.inner_x2_max,
    )
    if any(v is None for v in inner_bounds):
        return None, None

    x1_min, x1_max, x2_min, x2_max = (float(v) for v in inner_bounds)
    in_inner = (
        (x_tilde[:, 0] >= x1_min) & (x_tilde[:, 0] <= x1_max) &
        (x_tilde[:, 1] >= x2_min) & (x_tilde[:, 1] <= x2_max)
    )

    r = float(cfg.inner_half_ratio)
    if not (0.0 < r < 1.0):
        raise ValueError("inner_half_ratio must be in (0,1)")

    hx1_min, hx1_max = r * x1_min, r * x1_max
    hx2_min, hx2_max = r * x2_min, r * x2_max
    in_half = (
        (x_tilde[:, 0] >= hx1_min) & (x_tilde[:, 0] <= hx1_max) &
        (x_tilde[:, 1] >= hx2_min) & (x_tilde[:, 1] <= hx2_max)
    )

    mask_transition = in_inner & (~in_half)
    mask_outer = ~in_inner
    return mask_transition, mask_outer


def train_w_local(cfg: WTrainCfg, save_path: str = "w_model.pt", v_inner_model: Optional[nn.Module] = None) -> WNet:
    device = torch.device(cfg.device)
    dtype = torch.float32 if cfg.dtype == "float32" else torch.float64

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    p = Params()
    x_eq = float(equilibrium_x1(p))

    print(f"Device: {device}, dtype={dtype}")
    print(f"Equilibrium (original coords): x_eq = {x_eq:.12g}, v_eq = 0")
    print("Training region is in SHIFTED coords: x_tilde = [x1 - x_eq, x2]")
    print(f"Region: x1 in [{cfg.x1_min},{cfg.x1_max}], x2 in [{cfg.x2_min},{cfg.x2_max}], r_min={cfg.r_min}")

    model = WNet(hidden=cfg.hidden, depth=cfg.depth).to(device=device, dtype=dtype)
    opt = optim.Adam(model.parameters(), lr=float(cfg.lr))

    if v_inner_model is not None:
        v_inner_model = v_inner_model.to(device=device, dtype=dtype)
        v_inner_model.eval()

    for step in range(int(cfg.steps) + 1):
        x = _sample_rect(int(cfg.batch) * 2, cfg.x1_min, cfg.x1_max, cfg.x2_min, cfg.x2_max, device, dtype)
        x = _apply_rmin_mask(x, cfg.r_min)

        if x.shape[0] < max(128, int(cfg.batch) // 4):
            x = _sample_rect(int(cfg.batch), cfg.x1_min, cfg.x1_max, cfg.x2_min, cfg.x2_max, device, dtype)

        if x.shape[0] > int(cfg.batch):
            idx = torch.randperm(x.shape[0], device=device)[: int(cfg.batch)]
            x = x[idx]

        mask_transition, mask_outer = _inner_masks(x, cfg)
        if v_inner_model is not None:
            with torch.no_grad():
                v_inner = v_inner_model(x).squeeze(-1)
        else:
            v_inner = None

        opt.zero_grad(set_to_none=True)
        loss, diag = _compute_loss(
            model=model,
            x_tilde=x,
            p=p,
            x_eq=x_eq,
            margin=cfg.margin,
            alpha_pos=cfg.alpha_pos,
            eps_s=cfg.eps_s,
            lam_s=cfg.lam_s,
            v_inner=v_inner,
            mask_transition=mask_transition,
            mask_outer=mask_outer,
            eps_transition=cfg.eps_transition,
            lam_transition=cfg.lam_transition,
            lam_dom=cfg.lam_dom,
        )
        loss.backward()
        opt.step()

        if step % int(cfg.log_every) == 0:
            print(
                f"[w {step:6d}/{cfg.steps}] "
                f"loss {diag['loss_total']:.3e} | main {diag['loss_main']:.3e} | "
                f"pos {diag['loss_pos']:.3e} | s {diag['loss_s']:.3e} | "
                f"tr {diag['loss_transition']:.3e} | dom {diag['loss_dom']:.3e} | "
                f"maxS {diag['max_S']:.3e} | frac(S>0) {diag['frac_S_pos_%']:.3f}% | "
                f"min_smin {diag['min_smin']:.3e}"
            )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "params": p.__dict__,
            "x_eq": x_eq,
            "hidden": int(cfg.hidden),
            "depth": int(cfg.depth),
            "dtype": str(cfg.dtype),
        },
        save_path,
    )
    print(f"Saved: {save_path}")
    return model
