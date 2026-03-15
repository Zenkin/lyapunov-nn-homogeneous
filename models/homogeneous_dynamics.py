from __future__ import annotations

import torch

from .full_system_dynamics import Params


def _check_xt_shape(xt: torch.Tensor) -> None:
    if xt.ndim != 2 or xt.shape[1] != 2:
        raise ValueError(f"xt must have shape (N,2), got {tuple(xt.shape)}")


def r_norm_r12(xt: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    r-норма для весов r=(1,2):
        ||xt||_r = |x1| + sqrt(|x2|)

    xt: (N,2)
    return: rho формы (N,)
    """
    _check_xt_shape(xt)

    x1 = xt[:, 0]
    x2 = xt[:, 1]

    # eps оставлен в сигнатуре для единообразия с normalize_to_r_sphere_r12.
    # Здесь он не нужен: sqrt(abs(0)) корректно даёт 0.
    _ = eps

    rho = torch.abs(x1) + torch.sqrt(torch.abs(x2))
    return rho


def normalize_to_r_sphere_r12(
    xt: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Нормализация точки xt на гомогенную сферу S_r(1) для r=(1,2).

    Для rho = ||xt||_r > eps:
        y1 = x1 / rho
        y2 = x2 / rho^2

    Для rho <= eps:
        y оставляем нулевым (технический placeholder),
        а mask == False сообщает downstream-коду, что это нулевая точка.

    xt:   (N,2)
    y:    (N,2)
    rho:  (N,)
    mask: (N,) bool, где rho > eps
    """
    _check_xt_shape(xt)

    rho = r_norm_r12(xt, eps=eps)
    mask = rho > eps

    y = torch.zeros_like(xt)

    if mask.any():
        rho_m = rho[mask]
        y[mask, 0] = xt[mask, 0] / rho_m
        y[mask, 1] = xt[mask, 1] / (rho_m * rho_m)

    return y, rho, mask


class HomogeneousInfinitySystem:
    """
    Гомогенное приближение на бесконечности в смещённых координатах xt:

        x1_dot = x2
        x2_dot = -a3 * x1^3 - a2 * |x2|^(1/2) * x2

    Вход/выход батчевые: xt имеет форму (N,2), результат тоже (N,2).
    """

    def __init__(self, p: Params):
        self.p = p

    def f_inf(self, xt: torch.Tensor) -> torch.Tensor:
        _check_xt_shape(xt)

        x1 = xt[:, 0]
        x2 = xt[:, 1]

        dx1 = x2
        dx2 = -self.p.a3 * (x1 ** 3) - self.p.a2 * torch.sqrt(torch.abs(x2)) * x2

        return torch.stack([dx1, dx2], dim=1)