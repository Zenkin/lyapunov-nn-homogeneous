import torch

from models.homogeneous_dynamics import normalize_to_r_sphere_r12
from models.nn_models import WArticle, VHomogeneousSphere, build_W


def main():
    dtype = torch.float64

    # -------------------------------------------------
    # 1) Проверка фабрики
    # -------------------------------------------------
    article = build_W("article", hidden=16, n=2)
    homogeneous = build_W("homogeneous", hidden=16, n=2, mu=2.0, eps=1e-12)

    print("article class =", article.__class__.__name__)
    print("homogeneous class =", homogeneous.__class__.__name__)
    print()

    assert isinstance(article, WArticle)
    assert isinstance(homogeneous, VHomogeneousSphere)

    # -------------------------------------------------
    # 2) Проверка sphere_forward(y)
    # -------------------------------------------------
    y = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.25],
        [-0.5, -0.25],
    ], dtype=dtype)

    homogeneous = homogeneous.to(dtype=dtype)

    v_sphere = homogeneous.sphere_forward(y)

    print("v_sphere.shape =")
    print(v_sphere.shape)
    print()

    print("v_sphere =")
    print(v_sphere)
    print()

    assert v_sphere.shape == (4, 1)
    assert torch.isfinite(v_sphere).all().item()

    # -------------------------------------------------
    # 3) Проверка forward(xt) на плоскости
    # -------------------------------------------------
    xt = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [2.0, 4.0],
        [0.0, 0.0],   # нулевая точка
        [-3.0, 9.0],
        [0.0, -1.0],
    ], dtype=dtype)

    v_xt = homogeneous(xt)

    print("v_xt.shape =")
    print(v_xt.shape)
    print()

    print("v_xt =")
    print(v_xt)
    print()

    assert v_xt.shape == (xt.shape[0], 1)
    assert torch.isfinite(v_xt).all().item()

    # В нуле должно быть ровно 0
    assert torch.allclose(v_xt[3], torch.tensor([0.0], dtype=dtype), atol=1e-12), \
        f"V(0) must be 0, got {v_xt[3]}"

    # -------------------------------------------------
    # 4) Проверка внутренней согласованности continuation:
    #    V(xt) = rho^mu * V(y)
    # -------------------------------------------------
    y_norm, rho, mask = normalize_to_r_sphere_r12(xt, eps=1e-12)

    v_expected = torch.zeros_like(v_xt)
    if mask.any():
        rho_m = rho[mask].unsqueeze(1)
        v_expected[mask] = (rho_m ** homogeneous.mu) * homogeneous.sphere_forward(y_norm[mask])

    print("v_expected =")
    print(v_expected)
    print()

    assert torch.allclose(v_xt, v_expected, atol=1e-12), \
        f"continuation mismatch:\n got      {v_xt}\n expected {v_expected}"

    # -------------------------------------------------
    # 5) Проверка article-модели не сломана
    # -------------------------------------------------
    x_article = torch.tensor([
        [0.0, 0.0],
        [1.0, 2.0],
        [-1.0, 3.0],
    ], dtype=dtype)

    article = article.to(dtype=dtype)
    w_article = article(x_article)

    print("w_article.shape =")
    print(w_article.shape)
    print()

    print("w_article =")
    print(w_article)
    print()

    assert w_article.shape == (3, 1)
    assert torch.isfinite(w_article).all().item()
    assert torch.all(w_article >= 0.0).item(), "WArticle output must be nonnegative"

    print("OK: nn_models sanity checks passed.")


if __name__ == "__main__":
    main()
