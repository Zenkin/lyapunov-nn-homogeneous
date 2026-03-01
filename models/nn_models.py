import torch
import torch.nn as nn


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


# --- Фабрика моделей ---
def build_W(model_name: str, *, hidden: int, n: int = 2):
    if model_name == "article":
        return WArticle(n=n, hidden=hidden)
    raise ValueError(f"Unknown W model: {model_name}")