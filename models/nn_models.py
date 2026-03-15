import torch
import torch.nn as nn

from .homogeneous_dynamics import normalize_to_r_sphere_r12


class WArticle(nn.Module):
    """
    T(x) = W2 * tanh(W1 x + b1) + b2,  enforce T(0)=0
    W(x) = ||T(x)||^2
    """

    def __init__(self, n=2, hidden=32):
        super().__init__()
        self.n = n
        self.fc1 = nn.Linear(n, hidden)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden, n)

        self.project_T0()

    def project_T0(self):
        # Enforce T(0)=0: b2 = -W2 * tanh(b1)
        with torch.no_grad():
            self.fc2.bias.copy_(- self.fc2.weight @ self.act(self.fc1.bias))

    def T(self, x):
        h = self.act(self.fc1(x))
        return self.fc2(h)

    def forward(self, x):
        T = self.T(x)
        W = (T * T).sum(dim=1, keepdim=True)
        return W


class VHomogeneousSphere(nn.Module):
    """
    Скалярная нейросеть для homogeneous-stage.

    sphere_forward(y):
        Значение V(y) на гомогенной сфере S_r(1), y имеет форму (N, 2).

    forward(xt):
        Гомогенное продолжение на всю плоскость в смещённых координатах xt:
            V(xt) = rho^mu * V(y),
        где
            rho = ||xt||_r = |x1| + sqrt(|x2|),
            y   = Lambda_r^{-1}(rho) xt.

        В точке xt = 0 возвращается 0.
    """

    def __init__(self, n=2, hidden=32, mu=2.0, eps=1e-12):
        super().__init__()
        if n != 2:
            raise ValueError(f"VHomogeneousSphere supports only n=2, got n={n}")

        self.n = n
        self.mu = float(mu)
        self.eps = float(eps)

        self.fc1 = nn.Linear(n, hidden)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden, hidden)
        self.act2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden, 1)

    def sphere_forward(self, y):
        if y.ndim != 2 or y.shape[1] != self.n:
            raise ValueError(f"y must have shape (N,{self.n}), got {tuple(y.shape)}")

        h = self.act1(self.fc1(y))
        h = self.act2(self.fc2(h))
        v = self.fc3(h)
        return v

    def forward(self, xt):
        if xt.ndim != 2 or xt.shape[1] != self.n:
            raise ValueError(f"xt must have shape (N,{self.n}), got {tuple(xt.shape)}")

        y, rho, mask = normalize_to_r_sphere_r12(xt, eps=self.eps)

        v = torch.zeros((xt.shape[0], 1), dtype=xt.dtype, device=xt.device)

        if mask.any():
            y_m = y[mask]
            rho_m = rho[mask].unsqueeze(1)
            v_sphere = self.sphere_forward(y_m)
            v[mask] = (rho_m ** self.mu) * v_sphere

        return v


# --- Фабрика моделей ---
def build_W(model_name: str, *, hidden: int, n: int = 2, mu: float = 2.0, eps: float = 1e-12):
    if model_name == "article":
        return WArticle(n=n, hidden=hidden)
    if model_name == "homogeneous":
        return VHomogeneousSphere(n=n, hidden=hidden, mu=mu, eps=eps)
    raise ValueError(f"Unknown W model: {model_name}")
