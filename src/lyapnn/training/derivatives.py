#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Callable
import torch
import torch.nn as nn
from lyapnn.systems.duffing_friction import Params


def Vdot(
    V: nn.Module,
    x: torch.Tensor,
    p: Params,
    f: Callable[[torch.Tensor, Params], torch.Tensor],
    create_graph: bool,
) -> torch.Tensor:
    """Compute Vdot = grad V(x)^T f(x). Returns (N,1)."""
    x_ = x.detach().clone().requires_grad_(True)
    Vx = V(x_)
    g = torch.autograd.grad(Vx.sum(), x_, create_graph=create_graph)[0]
    fx = f(x_, p)
    return (g * fx).sum(dim=1, keepdim=True)
