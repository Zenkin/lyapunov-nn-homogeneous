import torch

from grid import sample_r_sphere_r12


def main():
    dtype = torch.float64
    n_samples = 10000

    y = sample_r_sphere_r12(
        n_samples,
        device="cpu",
        dtype=dtype,
    )

    print("y.shape =")
    print(y.shape)
    print()

    assert y.shape == (n_samples, 2), \
        f"shape mismatch: got {y.shape}, expected {(n_samples, 2)}"

    # -------------------------------------------------
    # 1) Проверка условия сферы:
    #    |y1| + sqrt(|y2|) = 1
    # -------------------------------------------------
    sphere_val = torch.abs(y[:, 0]) + torch.sqrt(torch.abs(y[:, 1]))

    print("sphere_val min/max =")
    print(sphere_val.min().item(), sphere_val.max().item())
    print()

    ones = torch.ones_like(sphere_val)
    assert torch.allclose(sphere_val, ones, atol=1e-12), \
        "sphere condition failed"

    # -------------------------------------------------
    # 2) Проверка диапазонов
    # -------------------------------------------------
    abs_y1 = torch.abs(y[:, 0])
    abs_y2 = torch.abs(y[:, 1])

    print("|y1| min/max =")
    print(abs_y1.min().item(), abs_y1.max().item())
    print()

    print("|y2| min/max =")
    print(abs_y2.min().item(), abs_y2.max().item())
    print()

    assert torch.all(abs_y1 >= 0.0).item()
    assert torch.all(abs_y1 <= 1.0).item()
    assert torch.all(abs_y2 >= 0.0).item()
    assert torch.all(abs_y2 <= 1.0).item()

    # -------------------------------------------------
    # 3) Проверка знаков:
    #    у обеих координат должны встречаться и +, и -
    # -------------------------------------------------
    y1_pos = int((y[:, 0] > 0).sum().item())
    y1_neg = int((y[:, 0] < 0).sum().item())
    y2_pos = int((y[:, 1] > 0).sum().item())
    y2_neg = int((y[:, 1] < 0).sum().item())

    print("sign counts:")
    print(f"y1_pos={y1_pos}, y1_neg={y1_neg}")
    print(f"y2_pos={y2_pos}, y2_neg={y2_neg}")
    print()

    assert y1_pos > 0 and y1_neg > 0, "y1 does not contain both signs"
    assert y2_pos > 0 and y2_neg > 0, "y2 does not contain both signs"

    # -------------------------------------------------
    # 4) Покажем первые несколько точек
    # -------------------------------------------------
    print("first 10 samples =")
    print(y[:10])
    print()

    print("OK: sample_r_sphere_r12 sanity checks passed.")


if __name__ == "__main__":
    main()
