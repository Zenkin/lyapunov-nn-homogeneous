from __future__ import annotations

import torch
from models import build_W, Params, FullSystem, cfg, loss_L2_article
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

    L2 = loss_L2_article(w, pts_tilda_torch_with_grad, f_tilde)
    print("L2:", float(L2.detach().cpu()))

    import os

    os.makedirs("runs", exist_ok=True)

    lr = 1e-3
    steps = 100000
    print_every = 100

    opt = torch.optim.Adam(w.parameters(), lr=lr)

    best_loss = float("inf")
    best_iter = -1
    best_path = "runs/W_article_best.pt"

    for it in range(steps):
        opt.zero_grad(set_to_none=True)

        # ----- семплируем точки внутри области -----
        N = 4096
        xt = (2.0 * torch.rand(N, 2, device=pts_x_torch.device, dtype=pts_x_torch.dtype) - 1.0) * 4.0
        xt.requires_grad_(True)

        f_t = sys.f_tilde(xt, x_eq_td)

        # ====== L2 ======
        w_vec = w.T(xt)

        jv_parts = []
        for k in range(w_vec.shape[1]):
            Tk = w_vec[:, k]
            grad_Tk = torch.autograd.grad(
                Tk.sum(),
                xt,
                create_graph=True,
                retain_graph=True
            )[0]
            jv_parts.append((grad_Tk * f_t).sum(dim=1))

        jv = torch.stack(jv_parts, dim=1)

        w2 = (w_vec * w_vec).sum(dim=1)
        wf = (w_vec * jv).sum(dim=1)

        term = 2.0 * wf + w2
        L2 = torch.relu(term).mean()

        # backward
        L2.backward()
        opt.step()

        # поддерживаем T(0)=0
        w.project_T0()

        # ----- сохраняем лучший вариант -----
        current_loss = float(L2.detach().cpu())

        if current_loss < best_loss:
            best_loss = current_loss
            best_iter = it

            torch.save({
                "model_state_dict": w.state_dict(),
                "iter": it,
                "loss": best_loss,
            }, best_path)

            print(f"[best] saved at iter={it}, L2={best_loss:.6e}")

        # ----- лог -----
        if (it % print_every) == 0 or it == steps - 1:
            with torch.no_grad():
                violation = (term > 0).float().mean()

                print(
                    f"[{it:5d}] "
                    f"L2={current_loss:.6e}  "
                    f"W_mean={float(w2.mean()):.3e}  "
                    f"W_max={float(w2.max()):.3e}  "
                    f"viol_rate={float(violation):.3f}"
                )

    print(f"\nTraining finished.")
    print(f"Best model: iter={best_iter}, L2={best_loss:.6e}")
    print(f"Saved to: {best_path}")

    checkpoint = torch.load("runs/W_article_best.pt", map_location=pts_x_torch.device)
    w.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded best model from iter:", checkpoint["iter"])

    # ===== eval on grid after training =====
    xtg = pts_tilda_torch.detach().clone().requires_grad_(True)  # (N,2) grid points, requires grad
    ftg = sys.f_tilde(xtg, x_eq_td)  # (N,2)

    w_vec = w.T(xtg)  # (N,2)
    w2 = (w_vec * w_vec).sum(dim=1)  # W (N,)

    # jv = DT f
    jv_parts = []
    for k in range(w_vec.shape[1]):
        Tk = w_vec[:, k]
        grad_Tk = torch.autograd.grad(
            Tk.sum(), xtg,
            create_graph=False,
            retain_graph=True
        )[0]
        jv_parts.append((grad_Tk * ftg).sum(dim=1))
    jv = torch.stack(jv_parts, dim=1)  # (N,2)

    wf = (w_vec * jv).sum(dim=1)  # (N,)
    dW = 2.0 * wf  # dW = 2 T^T DT f
    term = dW + w2  # what L2 enforces <= 0
    viol = (term > 0).to(w2.dtype)

    W_grid = w2.reshape(grid, grid).detach().cpu().numpy()
    dW_grid = dW.reshape(grid, grid).detach().cpu().numpy()
    term_grid = term.reshape(grid, grid).detach().cpu().numpy()
    viol_grid = viol.reshape(grid, grid).detach().cpu().numpy()

    plot_heatmap(X1_t, X2_t, term_grid, title="dW(xt)+W(xt)", cbar_label="dW+W",
                 xlabel="x_{1t}", ylabel="x_{2t}", save_path="runs/figs/term_tilde.png",
                 eq_point=(0.0, 0.0), cfg=VIZ_SLIDES)

    plot_heatmap(X1_t, X2_t, dW_grid, title="dW_grid", cbar_label="dW_grid",
                 xlabel="x_{1t}", ylabel="x_{2t}", save_path="runs/figs/dW_grid.png",
                 eq_point=(0.0, 0.0), cfg=VIZ_SLIDES)


if __name__ == "__main__":
    main()
