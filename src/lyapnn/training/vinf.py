from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os
import numpy as np
import torch
import torch.optim as optim

from lyapnn.geometry.r12 import sample_box
from lyapnn.models.homog_v import HomogV
from lyapnn.systems.duffing_friction import Params, f_inf, equilibrium_x1
from lyapnn.training.derivatives import Vdot


@dataclass
class VinfTrainCfg:
    seed: int = 0
    device: str = "cpu"
    mu: float = 2.0
    alpha: float = 1.0
    hidden: int = 64
    depth: int = 3
    n_train: int = 4000
    n_val: int = 4000
    batch: int = 2048
    steps: int = 1000
    lr: float = 2e-4
    log_every: int = 200
    normalize_margin: bool = True
    box_x1_min: float = -20.0
    box_x1_max: float = 20.0
    box_x2_min: float = -20.0
    box_x2_max: float = 20.0


def train_vinf(cfg: VinfTrainCfg, save_path: Optional[str] = None) -> HomogV:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    dev = torch.device(cfg.device)

    p = Params()
    V = HomogV(mu=cfg.mu, eps=1e-3, hidden=cfg.hidden, depth=cfg.depth).to(dev)
    opt = optim.Adam(V.parameters(), lr=cfg.lr)

    Xtr = torch.from_numpy(
        sample_box(cfg.n_train, cfg.box_x1_min, cfg.box_x1_max, cfg.box_x2_min, cfg.box_x2_max, cfg.seed)
    ).to(dev)
    Xva = torch.from_numpy(
        sample_box(cfg.n_val, cfg.box_x1_min, cfg.box_x1_max, cfg.box_x2_min, cfg.box_x2_max, cfg.seed + 1)
    ).to(dev)

    best_state, best_viol = None, 1e9

    for step in range(1, cfg.steps + 1):
        xb = Xtr[torch.randint(0, Xtr.shape[0], (cfg.batch,), device=dev)]
        Vb = V(xb)
        Vdb = Vdot(V, xb, p, f_inf, create_graph=True)

        margin = Vdb + cfg.alpha * Vb
        if cfg.normalize_margin:
            den = 1.0 + Vb.detach()
            margin = margin / den

        loss_dec = torch.relu(margin).mean()
        loss_scale = torch.relu(Vb - 10.0).mean() + torch.relu(0.05 - Vb).mean()
        loss = loss_dec + 0.5 * loss_scale

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % cfg.log_every == 0:
            with torch.no_grad():
                Vy = V(Xva)
            Vdy = Vdot(V, Xva, p, f_inf, create_graph=False).detach()
            m = (Vdy + cfg.alpha * Vy).squeeze(1)
            viol = (m > 0).float().mean().item()
            print(f"[vinf {step:5d}] loss={loss.item():.3e} viol={viol:.3e} mmax={m.max().item():.3e}")
            if viol < best_viol:
                best_viol = viol
                best_state = {k: v.detach().cpu().clone() for k, v in V.state_dict().items()}

    if best_state is not None:
        V.load_state_dict(best_state)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(
            {
                "state_dict": V.state_dict(),
                "meta": {
                    "mu": cfg.mu,
                    "alpha": cfg.alpha,
                    "hidden": cfg.hidden,
                    "depth": cfg.depth,
                },
            },
            save_path,
        )
        print(f"[save] {save_path}")

    xeq = equilibrium_x1(p)
    print(f"[eq] x_eq={xeq:.12f}, v_eq=0")
    return V
