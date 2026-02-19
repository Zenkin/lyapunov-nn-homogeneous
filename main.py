from __future__ import annotations

import torch
from models import build_W, Params, FullSystem, cfg
from viz import plot_heatmap, VIZ_DEBUG, VIZ_SLIDES
from grid import make_grid

# system params
A1, A2, A3 = 1.0, 2.0, 1.0
C1, C2 = 1.0, 2.0
FC, VS = 0.8, 0.5


def main():
    # Параметры системы (коэффициенты модели)
    p = Params(a1=A1, a2=A2, a3=A3, c1=C1, c2=C2, vs=VS, fc=FC)
    sys = FullSystem(p)

    # Вычисляем положение равновесия в исходных координатах x
    # x_eq имеет форму (2,) и возвращается как torch.Tensor (float64, CPU)
    x_eq_torch = sys.equilibrium(x1_init=C1)
    print("x_eq:", x_eq_torch.tolist())

    # Преобразуем в numpy для построения сетки и осей
    x_eq_np = x_eq_torch.detach().cpu().numpy()

    # ------------------------------------------------------------
    # Область задаём в смещённых координатах:
    # xt = x - x_eq
    # То есть равновесие будет в центре области (xt = 0)
    # ------------------------------------------------------------
    x1_t_min, x1_t_max = -4.0, 4.0
    x2_t_min, x2_t_max = -4.0, 4.0

    # Переводим окно из tilde-координат в исходные координаты x:
    # x = xt + x_eq
    x1_min = x_eq_np[0] + x1_t_min
    x1_max = x_eq_np[0] + x1_t_max
    x2_min = x_eq_np[1] + x2_t_min
    x2_max = x_eq_np[1] + x2_t_max

    # Строим сетку в исходных координатах x
    # (динамика системы определена именно в x)
    grid = 101
    x1_x, x2_x, X1_x, X2_x, pts_x_np = make_grid(
        x1_min=x1_min, x1_max=x1_max,
        x2_min=x2_min, x2_max=x2_max,
        grid=grid
    )

    # Формируем оси в tilde-координатах:
    # xt = x - x_eq
    # Теперь (0,0) соответствует равновесию
    X1_t = X1_x - x_eq_np[0]
    X2_t = X2_x - x_eq_np[1]

    # Переводим numpy-сетку в torch (рабочий dtype/device)
    pts_x_torch = torch.from_numpy(pts_x_np).to(torch.float64)

    # Приводим x_eq к тому же dtype/device, что и рабочие тензоры
    # (иначе возможна ошибка device/dtype mismatch)
    x_eq_td = x_eq_torch.to(
        dtype=pts_x_torch.dtype,
        device=pts_x_torch.device
    )

    # Переход к смещённым координатам: xt = x - x_eq (работаем в координатах относительно равновесия)
    pts_tilda_torch = sys.to_tilde(pts_x_torch, x_eq_td)

    # Создаём модель W(xt); приводим ее к тому же dtype/device,
    # что и рабочие тензоры (чтобы избежать mismatch)
    w = build_W(cfg.W_MODEL, hidden=cfg.W_HIDDEN, n=2).to(
        device=pts_x_torch.device,
        dtype=pts_x_torch.dtype
    )

    # Включаем вычисление градиента по xt,
    # так как далее нам нужна ∇W(xt) для вычисления dW
    pts_tilda_torch_with_grad = (
        pts_tilda_torch.detach().requires_grad_(True)
    )

    # xt_dot = f(x), где x = xt + x_eq, xt = x_tilda
    f_tilde = sys.f_tilde(pts_tilda_torch_with_grad, x_eq_td)

    # Значения W(xt) в каждой точке сетки
    # (выход модели имеет форму (N,1) → убираем последнюю размерность)
    W_flat = w(pts_tilda_torch_with_grad).squeeze(-1)

    # Градиент ∇W(xt) = dW/d(xt)
    # Используем сумму, чтобы получить корректный батчевый градиент
    gradW = torch.autograd.grad(
        outputs=W_flat.sum(),
        inputs=pts_tilda_torch_with_grad,
        create_graph=True,  # нужно, для дальнейшей оптимизации по dW
    )[0]  # форма (N,2)

    # Производная Ляпунова:
    # dW(xt) = ∇W(xt) · f_tilde(xt)
    dW_flat = (gradW * f_tilde).sum(dim=1)

    # Возвращаем значения к форме (grid, grid)
    # для визуализации на двумерной сетке
    W_grid = W_flat.reshape(grid, grid).detach().cpu().numpy()
    dW_grid = dW_flat.reshape(grid, grid).detach().cpu().numpy()

    plot_heatmap(
        X1_t,
        X2_t,
        W_grid,
        title="W(xt)",
        cbar_label="W(xt)",
        xlabel="x_{1t}",
        ylabel="x_{2t}",
        save_path="runs/figs/W_tilde.png",
        eq_point=(0.0, 0.0),
        cfg=VIZ_SLIDES,
    )

    plot_heatmap(
        X1_t,
        X2_t,
        dW_grid,
        title="dW(xt)",
        cbar_label="dW(xt)",
        xlabel="x_{1t}",
        ylabel="x_{2t}",
        save_path="runs/figs/dW_tilde.png",
        eq_point=(0.0, 0.0),
        cfg=VIZ_SLIDES,
    )

    plot_heatmap(
        X1_x,
        X2_x,
        W_grid,
        title="W(x)",
        cbar_label="W(x)",
        xlabel="x_{1}",
        ylabel="x_{2}",
        save_path="runs/figs/W.png",
        eq_point=x_eq_np,
        cfg=VIZ_SLIDES,
    )

    plot_heatmap(
        X1_x,
        X2_x,
        dW_grid,
        title="dW(x)",
        cbar_label="dW(x)",
        xlabel="x_{1}",
        ylabel="x_{2}",
        save_path="runs/figs/dW.png",
        eq_point=x_eq_np,
        cfg=VIZ_SLIDES,
    )


if __name__ == "__main__":
    main()
