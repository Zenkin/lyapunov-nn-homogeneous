import torch
import torch.nn as nn


class W(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


class WArticle(nn.Module):
    """
    Реализация из статьи:
    T(x) = w2 * tanh(w1 x + b1) + b2
    W(x) = T(x)^T T(x)
    """

    def __init__(self, n=2, hidden=32):
        super().__init__()

        self.n = n

        # T_theta
        self.fc1 = nn.Linear(n, hidden)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden, n)

        # Обеспечиваем T(0)=0 как в статье
        with torch.no_grad():
            self.fc2.bias.copy_(
                - self.fc2.weight @ self.act(self.fc1.bias)
            )

    def forward(self, x):
        h = self.act(self.fc1(x))
        T = self.fc2(h)
        W = (T * T).sum(dim=1, keepdim=True)
        return W


# --- Фабрика моделей ---
def build_W(model_name: str, hidden: int, n: int = 2):
    model_name = model_name.lower()

    if model_name == "deep":
        return W(hidden=hidden)

    if model_name == "article":
        return WArticle(n=n, hidden=hidden)

    raise ValueError(f"Unknown model: {model_name}")
