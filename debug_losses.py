import torch

from grid import sample_r_sphere_r12
from models.full_system_dynamics import Params
from models.homogeneous_dynamics import HomogeneousInfinitySystem
from models.losses import scalar_lie_derivative, loss_homogeneous_sphere
from models.nn_models import VHomogeneousSphere


class QuadraticV(torch.nn.Module):
    """
    Тестовая модель для проверки scalar_lie_derivative.
    V(x1, x2) = x1^2 + 3 x2^2
    """

    def sphere_forward(self, y):
        x1 = y[:, 0]
        x2 = y[:, 1]
        return (x1 * x1 + 3.0 * x2 * x2).unsqueeze(1)



def main():
    dtype = torch.float64

    # -------------------------------------------------
    # 1) Локальная аналитическая проверка scalar_lie_derivative
    # -------------------------------------------------
    y_test = torch.tensor([
        [1.0, 2.0],
        [-1.0, 0.5],
        [0.0, -3.0],
    ], dtype=dtype).requires_grad_(True)

    f_test = torch.tensor([
        [0.5, -1.0],
        [2.0, 4.0],
        [-3.0, 1.5],
    ], dtype=dtype)

    quad = QuadraticV()
    V_test = quad.sphere_forward(y_test)
    dV = scalar_lie_derivative(
        values=V_test,
        inputs=y_test,
        vector_field=f_test,
        create_graph=False,
    )

    # grad V = [2*x1, 6*x2]
    dV_expected = torch.tensor([
        2.0 * 1.0 * 0.5 + 6.0 * 2.0 * (-1.0),
        2.0 * (-1.0) * 2.0 + 6.0 * 0.5 * 4.0,
        2.0 * 0.0 * (-3.0) + 6.0 * (-3.0) * 1.5,
    ], dtype=dtype)

    print("dV =")
    print(dV)
    print()

    print("dV_expected =")
    print(dV_expected)
    print()

    assert torch.allclose(dV, dV_expected, atol=1e-12), \
        f"scalar_lie_derivative mismatch:\n got      {dV}\n expected {dV_expected}"

    # -------------------------------------------------
    # 2) Проверка loss_homogeneous_sphere на реальных y из S_r(1)
    # -------------------------------------------------
    y = sample_r_sphere_r12(
        2048,
        device="cpu",
        dtype=dtype,
    )

    p = Params(a2=2.0, a3=1.0)
    hom_sys = HomogeneousInfinitySystem(p)
    f_inf_y = hom_sys.f_inf(y)

    model = VHomogeneousSphere(n=2, hidden=16, mu=2.0, eps=1e-12).to(dtype=dtype)

    metrics = loss_homogeneous_sphere(model, y, f_inf_y)

    print("homogeneous loss metrics:")
    for k, v in metrics.items():
        print(f"{k}: {float(v.item()):.12f}")
    print()

    expected_keys = {
        "loss",
        "loss_decay",
        "loss_pos",
        "v_min",
        "v_mean",
        "dV_inf_max",
        "viol_decay_rate",
        "viol_pos_rate",
    }
    assert set(metrics.keys()) == expected_keys, \
        f"metrics keys mismatch: got {set(metrics.keys())}, expected {expected_keys}"

    for k, v in metrics.items():
        assert isinstance(v, torch.Tensor), f"metric {k} must be a tensor"
        assert v.ndim == 0, f"metric {k} must be scalar, got ndim={v.ndim}"
        assert torch.isfinite(v).item(), f"metric {k} is not finite: {v}"

    assert metrics["loss"].requires_grad, "loss must require grad for backward()"
    assert metrics["loss_decay"].item() >= 0.0
    assert metrics["loss_pos"].item() >= 0.0
    assert 0.0 <= metrics["viol_decay_rate"].item() <= 1.0
    assert 0.0 <= metrics["viol_pos_rate"].item() <= 1.0

    metrics["loss"].backward()

    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += float(param.grad.norm().item())

    print(f"parameter grad_norm sum = {grad_norm:.12f}")
    print()

    assert grad_norm > 0.0, "backward produced zero gradients for all parameters"

    print("OK: losses sanity checks passed.")


if __name__ == "__main__":
    main()
