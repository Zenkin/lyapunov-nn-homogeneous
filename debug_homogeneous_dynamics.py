import torch

from models.full_system_dynamics import Params
from models.homogeneous_dynamics import (
    r_norm_r12,
    normalize_to_r_sphere_r12,
    HomogeneousInfinitySystem,
)


def main():
    dtype = torch.float64

    xt = torch.tensor([
        [1.0, 0.0],    # rho = 1
        [0.0, 1.0],    # rho = 1
        [2.0, 4.0],    # rho = 4
        [0.0, 0.0],    # rho = 0
        [-3.0, 9.0],   # rho = 6
    ], dtype=dtype)

    print("xt =")
    print(xt)
    print()

    # -------------------------------------------------
    # 1) Проверка r-нормы
    # -------------------------------------------------
    rho = r_norm_r12(xt)
    print("rho =")
    print(rho)
    print()

    rho_expected = torch.tensor([1.0, 1.0, 4.0, 0.0, 6.0], dtype=dtype)
    assert torch.allclose(rho, rho_expected, atol=1e-12), \
        f"rho mismatch: got {rho}, expected {rho_expected}"

    # -------------------------------------------------
    # 2) Проверка нормализации на сферу
    # -------------------------------------------------
    y, rho2, mask = normalize_to_r_sphere_r12(xt)

    print("y =")
    print(y)
    print()
    print("rho2 =")
    print(rho2)
    print()
    print("mask =")
    print(mask)
    print()

    assert torch.allclose(rho, rho2, atol=1e-12)

    y_nonzero = y[mask]
    sphere_val = torch.abs(y_nonzero[:, 0]) + torch.sqrt(torch.abs(y_nonzero[:, 1]))

    print("sphere check =")
    print(sphere_val)
    print()

    ones = torch.ones_like(sphere_val)
    assert torch.allclose(sphere_val, ones, atol=1e-12), \
        f"points are not on S_r(1): got {sphere_val}"

    idx = 2
    y_expected = torch.tensor([0.5, 0.25], dtype=dtype)
    assert torch.allclose(y[idx], y_expected, atol=1e-12), \
        f"y[{idx}] mismatch: got {y[idx]}, expected {y_expected}"

    assert mask[3].item() is False
    assert torch.allclose(y[3], torch.tensor([0.0, 0.0], dtype=dtype), atol=1e-12)

    # -------------------------------------------------
    # 3) Доп. проверка: отрицательный x2
    # -------------------------------------------------
    xt_neg = torch.tensor([[0.0, -1.0]], dtype=dtype)
    rho_neg = r_norm_r12(xt_neg)
    y_neg, _, mask_neg = normalize_to_r_sphere_r12(xt_neg)

    print("xt_neg =")
    print(xt_neg)
    print("rho_neg =")
    print(rho_neg)
    print("y_neg =")
    print(y_neg)
    print("mask_neg =")
    print(mask_neg)
    print()

    assert torch.allclose(rho_neg, torch.tensor([1.0], dtype=dtype), atol=1e-12)
    assert torch.allclose(y_neg[0], torch.tensor([0.0, -1.0], dtype=dtype), atol=1e-12)
    assert mask_neg[0].item() is True

    sphere_val_neg = torch.abs(y_neg[:, 0]) + torch.sqrt(torch.abs(y_neg[:, 1]))
    assert torch.allclose(sphere_val_neg, torch.tensor([1.0], dtype=dtype), atol=1e-12)

    # -------------------------------------------------
    # 4) Доп. проверка: обратная сборка xt из rho и y
    # -------------------------------------------------
    xt_rec = torch.zeros_like(xt)
    xt_rec[mask, 0] = rho[mask] * y[mask, 0]
    xt_rec[mask, 1] = (rho[mask] ** 2) * y[mask, 1]

    print("xt reconstructed =")
    print(xt_rec)
    print()

    assert torch.allclose(xt_rec[mask], xt[mask], atol=1e-12), \
        f"reconstruction mismatch:\n got      {xt_rec[mask]}\n expected {xt[mask]}"

    # -------------------------------------------------
    # 5) Проверка f_inf
    # -------------------------------------------------
    p = Params(a2=2.0, a3=1.0)
    hom_sys = HomogeneousInfinitySystem(p)

    x_test = torch.tensor([
        [1.0, 4.0],    # ожидаем [4, -17]
        [0.0, 1.0],    # ожидаем [1, -2]
        [2.0, 0.0],    # ожидаем [0, -8]
        [0.0, -1.0],   # ожидаем [-1, 2]
    ], dtype=dtype)

    f = hom_sys.f_inf(x_test)

    print("f_inf(x_test) =")
    print(f)
    print()

    f_expected = torch.tensor([
        [4.0, -17.0],
        [1.0, -2.0],
        [0.0, -8.0],
        [-1.0, 2.0],
    ], dtype=dtype)

    assert torch.allclose(f, f_expected, atol=1e-12), \
        f"f_inf mismatch:\n got      {f}\n expected {f_expected}"

    print("OK: all sanity checks passed.")


if __name__ == "__main__":
    main()
