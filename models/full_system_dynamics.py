from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Params:
    a1: float = 1.0
    a2: float = 2.0
    a3: float = 1.0
    c1: float = 1.0
    c2: float = 2.0
    fc: float = 0.8
    vs: float = 0.5


def smooth_abs(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(x * x + eps)


def phi(v: torch.Tensor, p: Params) -> torch.Tensor:
    return p.fc * torch.tanh(v / (p.vs + 1e-12))


def drag_term(x2: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(smooth_abs(x2) + 1e-12) * x2


class FullSystem:
    """
    Полная система x_dot = f(x).
    Умеет:
      - считать f(x) батчево
      - находить равновесие (x2*=0, x1* решаем Ньютонам)
      - работать в смещённых координатах x_tilde = x - x_eq
    """

    def __init__(self, p: Params):
        self.p = p

    def f(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError(f"x must have shape (N,2), got {tuple(x.shape)}")

        x1 = x[:, 0]
        x2 = x[:, 1]

        dx1 = x2
        dx2 = (
            -phi(x2, self.p)
            + self.p.a1 * (x1 - self.p.c1)
            - self.p.a2 * drag_term(x2)
            - self.p.a3 * (x1 - self.p.c2) ** 3
        )
        return torch.stack([dx1, dx2], dim=1)

    def equilibrium(self, x1_init: float | None = None,
                    tol: float = 1e-12, max_iter: int = 50) -> torch.Tensor:
        """
        Возвращает x_eq как torch.Tensor формы (2,)
        Для этой системы x2*=0.
        x1* решаем методом Ньютона для уравнения:
            a1*(x1-c1) - a3*(x1-c2)^3 = 0
        """
        if x1_init is None:
            x1_init = float(self.p.c1)

        # dtype/device берём "нейтральные" (CPU/float64) — это просто число.
        x = torch.tensor(x1_init, dtype=torch.float64)

        for _ in range(max_iter):
            f = self.p.a1 * (x - self.p.c1) - self.p.a3 * (x - self.p.c2) ** 3
            df = self.p.a1 - 3.0 * self.p.a3 * (x - self.p.c2) ** 2

            if abs(df.item()) < 1e-14:
                break

            x_new = x - f / df
            if abs((x_new - x).item()) < tol:
                x = x_new
                break
            x = x_new

        return torch.tensor([x.item(), 0.0], dtype=torch.float64)

    @staticmethod
    def to_tilde(x: torch.Tensor, x_eq: torch.Tensor) -> torch.Tensor:
        """
        x -> x_tilde = x - x_eq
        x: (N,2), x_eq: (2,)
        """
        return x - x_eq.view(1, 2)

    @staticmethod
    def from_tilde(x_tilde: torch.Tensor, x_eq: torch.Tensor) -> torch.Tensor:
        """
        x_tilde -> x = x_tilde + x_eq
        """
        return x_tilde + x_eq.view(1, 2)

    def f_tilde(self, x_tilde: torch.Tensor, x_eq: torch.Tensor) -> torch.Tensor:
        """
        Динамика в смещённых координатах:
            x = x_tilde + x_eq
            x_tilde_dot = f(x)
        (поскольку x_eq константа, производная не меняется)
        """
        x = self.from_tilde(x_tilde, x_eq)
        return self.f(x)

    @staticmethod
    def match_like(x_eq: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Bring x_eq to same dtype/device as 'like'."""
        return x_eq.to(dtype=like.dtype, device=like.device)