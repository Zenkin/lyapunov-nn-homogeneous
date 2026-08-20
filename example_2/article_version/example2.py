"""Literal mathematical core of the article's nonlinear-pendulum example.

This module contains only formulas and numerical values explicitly given in
the article. Parameters absent from the article are required as arguments and
are not assigned inferred defaults here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
import torch
from torch import nn


matplotlib.use("Agg")
torch.set_default_dtype(torch.float64)


@dataclass(frozen=True)
class ArticleSpecification:
    """Numerical values stated in Section V-B of the article."""

    omega: float = 1.0
    angle_min: float = -math.pi
    angle_max: float = math.pi
    velocity_min: float = -4.0
    velocity_max: float = 4.0
    grid_points_per_axis: int = 100
    lyapunov_hidden_units: int = 32
    controller_hidden_units: int = 20


@dataclass(frozen=True)
class ImplementationConfig:
    """New numerical choices required to make the incomplete example runnable."""

    seed: int = 20260820
    kappa: float = 0.05
    epsilon: float = 0.1
    training_steps: int = 5000
    learning_rate: float = 1e-3
    validation_points_per_axis: int = 201
    plot_points_per_axis: int = 301
    log_every: int = 500


def nonlinear_field(
    x: torch.Tensor,
    u: torch.Tensor,
    omega: float,
) -> torch.Tensor:
    """Return the nonlinear field printed in the article.

    The state is ``x=(theta, theta_dot)`` and

    ``F(x,u)=(x2, omega**2*sin(x1)+cos(x1)*u)``.
    """

    if x.shape[-1] != 2:
        raise ValueError("The state must have two components")

    if u.ndim == x.ndim and u.shape[-1] == 1:
        u = u.squeeze(-1)

    x1, x2 = x[..., 0], x[..., 1]
    second = omega**2 * torch.sin(x1) + torch.cos(x1) * u
    return torch.stack((x2, second), dim=-1)


def article_linear_matrices(
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the matrices printed in Section V-B without correcting them."""

    selected_dtype = dtype or torch.get_default_dtype()
    a = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=selected_dtype, device=device)
    b = torch.tensor([[0.0], [1.0]], dtype=selected_dtype, device=device)
    return a, b


def true_jacobian_matrices(
    omega: float,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the Jacobian of ``nonlinear_field`` at ``(x,u)=(0,0)``.

    This function is an audit reference. It is not substituted for the matrix
    printed in the article version.
    """

    selected_dtype = dtype or torch.get_default_dtype()
    a = torch.tensor(
        [[0.0, 1.0], [omega**2, 0.0]], dtype=selected_dtype, device=device
    )
    b = torch.tensor([[0.0], [1.0]], dtype=selected_dtype, device=device)
    return a, b


def implementation_local_design(
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the documented current choices of ``K`` and ``P``.

    ``K=(-2,-3)`` places the poles of ``A_article+BK`` at ``-1`` and ``-2``.
    ``P`` is the exact solution of the Lyapunov equation with right-hand side
    ``-I``. These values are derived for this implementation and are not
    reported numerical parameters of the article.
    """

    selected_dtype = dtype or torch.get_default_dtype()
    k = torch.tensor([-2.0, -3.0], dtype=selected_dtype, device=device)
    p = torch.tensor(
        [[5.0 / 4.0, 1.0 / 4.0], [1.0 / 4.0, 1.0 / 4.0]],
        dtype=selected_dtype,
        device=device,
    )
    return k, p


def article_linear_field(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Return ``A_article x + B u`` using the displayed matrices."""

    a, b = article_linear_matrices(dtype=x.dtype, device=x.device)
    if u.ndim == x.ndim and u.shape[-1] == 1:
        u = u.squeeze(-1)
    return x @ a.T + u.unsqueeze(-1) * b[:, 0]


class ZeroAtOriginNetwork(nn.Module):
    """One-hidden-layer network from equation (8), with ``T(0)=0``.

    Writing the output layer without a free bias and subtracting the hidden
    activation at zero implements the article's constraint
    ``b2 = -w2*tanh(b1)`` exactly.
    """

    def __init__(self, input_dimension: int, hidden_units: int, output_dimension: int):
        super().__init__()
        self.hidden = nn.Linear(input_dimension, hidden_units)
        self.output = nn.Linear(hidden_units, output_dimension, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_at_x = torch.tanh(self.hidden(x))
        hidden_at_zero = torch.tanh(self.hidden.bias)
        return self.output(hidden_at_x - hidden_at_zero)


class LyapunovNetwork(nn.Module):
    """Candidate ``W(x;theta)=T_theta(x)^T T_theta(x)`` from equation (9)."""

    def __init__(self, hidden_units: int = 32):
        super().__init__()
        self.transform = ZeroAtOriginNetwork(2, hidden_units, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformed = self.transform(x)
        return torch.sum(transformed * transformed, dim=-1)


class ControllerNetwork(ZeroAtOriginNetwork):
    """Scalar control network ``N_theta2'(x)=T_theta2'(x)`` from equation (8)."""

    def __init__(self, hidden_units: int = 20):
        super().__init__(2, hidden_units, 1)


def lyapunov_value_and_derivative(
    lyapunov: LyapunovNetwork,
    controller: ControllerNetwork,
    x: torch.Tensor,
    omega: float,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate ``W`` and ``DW F(x,N(x))`` by automatic differentiation."""

    differentiable_x = x.detach().clone().requires_grad_(True)
    value = lyapunov(differentiable_x)
    gradient = torch.autograd.grad(
        value.sum(), differentiable_x, create_graph=create_graph
    )[0]
    control = controller(differentiable_x)
    field = nonlinear_field(differentiable_x, control, omega)
    derivative = torch.sum(gradient * field, dim=-1)
    return value, derivative


def article_pointwise_loss(
    lyapunov: LyapunovNetwork,
    controller: ControllerNetwork,
    x: torch.Tensor,
    omega: float,
    epsilon: float,
) -> torch.Tensor:
    """Return the local-stabilization loss displayed in Section IV-B.

    With ``[s]_+=max(0,s)`` and ``[s]_- = min(0,s)``, the formula is

    ``[DW F + W]_+ - [W-epsilon]_-``.

    The function returns one loss value per sample and does not select an
    unreported reduction rule.
    """

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    value, derivative = lyapunov_value_and_derivative(
        lyapunov, controller, x, omega, create_graph=True
    )
    return torch.relu(derivative + value) + torch.relu(epsilon - value)


def local_quadratic_value(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Evaluate the article's local candidate ``V_l(x)=x^T P x``."""

    if p.shape != (2, 2):
        raise ValueError("P must be a 2 by 2 matrix")
    return torch.einsum("...i,ij,...j->...", x, p, x)


def article_local_lyapunov_matrix(k: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Return ``(A_article+BK)^T P + P(A_article+BK)``.

    The function evaluates the matrix inequality stated in Section V-B but
    does not infer the unreported matrices ``K`` or ``P``.
    """

    if k.shape not in ((2,), (1, 2)):
        raise ValueError("K must have shape (2,) or (1,2)")
    if p.shape != (2, 2):
        raise ValueError("P must be a 2 by 2 matrix")
    a, b = article_linear_matrices(dtype=p.dtype, device=p.device)
    closed_loop = a + b @ k.reshape(1, 2).to(dtype=p.dtype, device=p.device)
    return closed_loop.T @ p + p @ closed_loop


def uniform_rectangle_grid(
    specification: ArticleSpecification,
    *,
    include_boundary: bool,
) -> torch.Tensor:
    """Construct one explicit interpretation of a uniform ``100 by 100`` grid.

    The article does not state whether boundary points are included. Requiring
    ``include_boundary`` prevents that implementation choice from being hidden.
    """

    count = specification.grid_points_per_axis
    if include_boundary:
        angle = torch.linspace(specification.angle_min, specification.angle_max, count)
        velocity = torch.linspace(
            specification.velocity_min, specification.velocity_max, count
        )
    else:
        angle_step = (specification.angle_max - specification.angle_min) / count
        velocity_step = (
            specification.velocity_max - specification.velocity_min
        ) / count
        angle = torch.linspace(
            specification.angle_min + 0.5 * angle_step,
            specification.angle_max - 0.5 * angle_step,
            count,
        )
        velocity = torch.linspace(
            specification.velocity_min + 0.5 * velocity_step,
            specification.velocity_max - 0.5 * velocity_step,
            count,
        )
    x1, x2 = torch.meshgrid(angle, velocity, indexing="ij")
    return torch.stack((x1.reshape(-1), x2.reshape(-1)), dim=1)


def article_training_domain(
    grid: torch.Tensor,
    p: torch.Tensor,
    kappa: float,
) -> torch.Tensor:
    """Select ``X \\ B_(kappa/2)`` with ``B_c={x:x^T P x<=c}``."""

    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    return grid[local_quadratic_value(grid, p) > 0.5 * kappa]


def local_linear_control(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Evaluate the article's local controller ``u_l(x)=Kx``."""

    if k.shape not in ((2,), (1, 2)):
        raise ValueError("K must have shape (2,) or (1,2)")
    return x @ k.reshape(2, 1)


def switched_control(
    x: torch.Tensor,
    lyapunov: LyapunovNetwork,
    learned_controller: Callable[[torch.Tensor], torch.Tensor],
    k: torch.Tensor,
    kappa: float,
) -> torch.Tensor:
    """Apply the local switching rule printed for ``lambda_l=0``.

    The local linear control is active where ``W(x)<kappa``. At equality the
    article assigns the learned controller.
    """

    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    local = local_linear_control(x, k)
    learned = learned_controller(x)
    if learned.ndim == x.ndim - 1:
        learned = learned.unsqueeze(-1)
    return torch.where((lyapunov(x) < kappa).unsqueeze(-1), local, learned)


def train_networks(
    specification: ArticleSpecification,
    config: ImplementationConfig,
) -> tuple[LyapunovNetwork, ControllerNetwork, dict[str, float | int | str]]:
    """Train both networks on the article's ``100 by 100`` domain grid."""

    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    _, p = implementation_local_design()
    full_grid = uniform_rectangle_grid(specification, include_boundary=False)
    training_points = article_training_domain(full_grid, p, config.kappa)

    lyapunov = LyapunovNetwork(specification.lyapunov_hidden_units)
    controller = ControllerNetwork(specification.controller_hidden_units)
    optimizer = torch.optim.Adam(
        [*lyapunov.parameters(), *controller.parameters()],
        lr=config.learning_rate,
    )

    final_pointwise_loss = None
    for step in range(1, config.training_steps + 1):
        pointwise_loss = article_pointwise_loss(
            lyapunov,
            controller,
            training_points,
            specification.omega,
            config.epsilon,
        )
        loss = pointwise_loss.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_pointwise_loss = pointwise_loss.detach()

        if step == 1 or step % config.log_every == 0 or step == config.training_steps:
            print(
                f"[train {step:5d}] mean_loss={loss.detach().item():.6e} "
                f"max_pointwise_loss={pointwise_loss.detach().max().item():.6e}"
            )

    if final_pointwise_loss is None:
        raise RuntimeError("training_steps must be positive")
    return lyapunov, controller, {
        "training_grid_convention": "cell midpoints",
        "training_grid_points_before_filter": len(full_grid),
        "training_grid_points_after_filter": len(training_points),
        "loss_reduction": "arithmetic mean of the displayed pointwise loss",
        "completed_training_steps": config.training_steps,
        "final_mean_pointwise_loss": final_pointwise_loss.mean().item(),
        "final_maximum_pointwise_loss": final_pointwise_loss.max().item(),
    }


def _derivative_under_control(
    lyapunov: LyapunovNetwork,
    points: torch.Tensor,
    control: Callable[[torch.Tensor], torch.Tensor],
    omega: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    differentiable_points = points.detach().clone().requires_grad_(True)
    value = lyapunov(differentiable_points)
    gradient = torch.autograd.grad(value.sum(), differentiable_points)[0]
    field = nonlinear_field(
        differentiable_points, control(differentiable_points), omega
    )
    return value.detach(), torch.sum(gradient * field, dim=-1).detach()


def _local_value_and_derivative(
    points: torch.Tensor,
    k: torch.Tensor,
    p: torch.Tensor,
    omega: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    differentiable_points = points.detach().clone().requires_grad_(True)
    value = local_quadratic_value(differentiable_points, p)
    gradient = torch.autograd.grad(value.sum(), differentiable_points)[0]
    field = nonlinear_field(
        differentiable_points,
        local_linear_control(differentiable_points, k),
        omega,
    )
    return value.detach(), torch.sum(gradient * field, dim=-1).detach()


def _save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _remove_generated_svg_trailing_whitespace(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def _draw_domain_boundaries(
    axis,
    angle: np.ndarray,
    velocity: np.ndarray,
    local_value: np.ndarray,
    learned_value: np.ndarray,
    kappa: float,
    extra_handles: tuple = (),
) -> None:
    from matplotlib.lines import Line2D

    axis.contour(
        angle,
        velocity,
        local_value,
        levels=[0.5 * kappa],
        colors=["#00A6A6"],
        linewidths=1.8,
        linestyles="--",
    )
    axis.contour(
        angle,
        velocity,
        local_value,
        levels=[kappa],
        colors=["#007C83"],
        linewidths=2.0,
    )
    axis.contour(
        angle,
        velocity,
        learned_value,
        levels=[kappa],
        colors=["#F2B134"],
        linewidths=1.8,
    )
    axis.scatter([0.0], [0.0], s=22, c="#111111", zorder=5)
    handles = [
        Line2D([0], [0], color="#00A6A6", linestyle="--", lw=1.8, label=r"$V_\ell=\kappa/2$"),
        Line2D([0], [0], color="#007C83", lw=2.0, label=r"$V_\ell=\kappa$"),
        Line2D([0], [0], color="#F2B134", lw=1.8, label=r"$W=\kappa$ (switch)"),
    ]
    handles.extend(extra_handles)
    axis.legend(handles=handles, loc="upper right", framealpha=0.94, fontsize=8)
    axis.set_xlabel(r"angle $\theta$")
    axis.set_ylabel(r"angular velocity $\dot{\theta}$")
    axis.set_aspect("auto")


def validate_and_plot(
    lyapunov: LyapunovNetwork,
    controller: ControllerNetwork,
    specification: ArticleSpecification,
    config: ImplementationConfig,
    outdir: Path,
) -> dict[str, float | int | str]:
    """Run boundary-including finite-grid checks and save article-style figures."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    k, p = implementation_local_design()
    validation_specification = ArticleSpecification(
        grid_points_per_axis=config.validation_points_per_axis
    )
    validation_grid = uniform_rectangle_grid(
        validation_specification, include_boundary=True
    )
    local_value, local_derivative = _local_value_and_derivative(
        validation_grid, k, p, specification.omega
    )
    outside_half_level = local_value > 0.5 * config.kappa
    outer_points = validation_grid[outside_half_level]
    learned_value, learned_derivative = _derivative_under_control(
        lyapunov, outer_points, controller, specification.omega
    )
    all_learned_value, switched_derivative = _derivative_under_control(
        lyapunov,
        validation_grid,
        lambda x: switched_control(
            x, lyapunov, controller, k, config.kappa
        ),
        specification.omega,
    )
    nonzero = torch.sum(validation_grid * validation_grid, dim=1) > 1e-14
    local_domain = (local_value <= config.kappa) & nonzero
    switch_to_local = all_learned_value < config.kappa

    metrics: dict[str, float | int | str] = {
        "validation_grid_convention": (
            "boundary-including grid; contains the midpoint training grid"
        ),
        "validation_grid_points": len(validation_grid),
        "validation_points_in_X_minus_B_kappa_over_2": len(outer_points),
        "minimum_W_on_X_minus_B_kappa_over_2": learned_value.min().item(),
        "maximum_DW_F_learned_on_X_minus_B_kappa_over_2": learned_derivative.max().item(),
        "maximum_DW_F_plus_W_on_X_minus_B_kappa_over_2": (
            learned_derivative + learned_value
        ).max().item(),
        "W_below_epsilon_fraction_on_X_minus_B_kappa_over_2": (
            learned_value < config.epsilon
        ).double().mean().item(),
        "nonnegative_learned_derivative_fraction_on_X_minus_B_kappa_over_2": (
            learned_derivative >= 0.0
        ).double().mean().item(),
        "maximum_DV_local_F_local_on_B_kappa": local_derivative[
            local_domain
        ].max().item(),
        "nonnegative_local_derivative_fraction_on_B_kappa": (
            local_derivative[local_domain] >= 0.0
        ).double().mean().item(),
        "minimum_W_away_from_origin_on_X": all_learned_value[nonzero].min().item(),
        "maximum_DW_F_under_article_switch_on_X": switched_derivative[
            nonzero
        ].max().item(),
        "nonnegative_switched_derivative_fraction_on_X": (
            switched_derivative[nonzero] >= 0.0
        ).double().mean().item(),
        "points_using_local_control": int(switch_to_local.sum().item()),
        "local_control_points_outside_B_kappa": int(
            (switch_to_local & (local_value > config.kappa)).sum().item()
        ),
        "interpretation": "finite-grid numerical evidence, not a continuous-domain certificate",
    }

    plot_specification = ArticleSpecification(
        grid_points_per_axis=config.plot_points_per_axis
    )
    plot_grid = uniform_rectangle_grid(plot_specification, include_boundary=True)
    plot_local_value, _ = _local_value_and_derivative(
        plot_grid, k, p, specification.omega
    )
    plot_learned_value, plot_switched_derivative = _derivative_under_control(
        lyapunov,
        plot_grid,
        lambda x: switched_control(
            x, lyapunov, controller, k, config.kappa
        ),
        specification.omega,
    )
    count = config.plot_points_per_axis
    angle = plot_grid[:, 0].reshape(count, count).numpy()
    velocity = plot_grid[:, 1].reshape(count, count).numpy()
    value_image = plot_learned_value.reshape(count, count).numpy()
    derivative_image = plot_switched_derivative.reshape(count, count).numpy()
    local_image = plot_local_value.reshape(count, count).numpy()

    def draw_value(axis):
        image = axis.contourf(angle, velocity, value_image, levels=32, cmap="viridis")
        contours = axis.contour(
            angle, velocity, value_image, levels=12, colors="#1B263B", linewidths=0.45, alpha=0.55
        )
        axis.clabel(contours, inline=True, fontsize=6, fmt="%.2g")
        axis.set_title(r"Learned candidate $W(x;\theta)$")
        _draw_domain_boundaries(
            axis, angle, velocity, local_image, value_image, config.kappa
        )
        return image

    def draw_derivative(axis):
        from matplotlib.lines import Line2D

        minimum = float(np.min(derivative_image))
        maximum = float(np.max(derivative_image))
        if minimum < 0.0 < maximum:
            norm = TwoSlopeNorm(vmin=minimum, vcenter=0.0, vmax=maximum)
        else:
            norm = None
        image = axis.contourf(
            angle,
            velocity,
            derivative_image,
            levels=32,
            cmap="coolwarm",
            norm=norm,
        )
        if minimum <= 0.0 <= maximum:
            axis.contour(
                angle,
                velocity,
                derivative_image,
                levels=[0.0],
                colors="#8B1E3F",
                linewidths=1.1,
            )
        axis.set_title(r"$DW\,F$ under the article switching rule")
        _draw_domain_boundaries(
            axis,
            angle,
            velocity,
            local_image,
            value_image,
            config.kappa,
            extra_handles=(
                Line2D(
                    [0],
                    [0],
                    color="#8B1E3F",
                    lw=1.1,
                    label=r"$DW\,F=0$",
                ),
            ),
        )
        return image

    outdir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    value_plot = draw_value(axes[0])
    derivative_plot = draw_derivative(axes[1])
    figure.colorbar(value_plot, ax=axes[0], label=r"$W(x;\theta)$")
    figure.colorbar(derivative_plot, ax=axes[1], label=r"$DW(x;\theta)F(x,U(x))$")
    figure.savefig(outdir / "combined_validation.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
    value_plot = draw_value(axis)
    figure.colorbar(value_plot, ax=axis, label=r"$W(x;\theta)$")
    figure.savefig(outdir / "figure_3_learned_W.png", dpi=220)
    figure_3_svg = outdir / "figure_3_learned_W.svg"
    figure.savefig(figure_3_svg)
    _remove_generated_svg_trailing_whitespace(figure_3_svg)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
    derivative_plot = draw_derivative(axis)
    figure.colorbar(
        derivative_plot, ax=axis, label=r"$DW(x;\theta)F(x,U(x))$"
    )
    figure.savefig(outdir / "figure_4_switched_derivative.png", dpi=220)
    figure_4_svg = outdir / "figure_4_switched_derivative.svg"
    figure.savefig(figure_4_svg)
    _remove_generated_svg_trailing_whitespace(figure_4_svg)
    plt.close(figure)
    np.savez_compressed(
        outdir / "validation_arrays.npz",
        x=validation_grid.numpy(),
        W=all_learned_value.numpy(),
        dW_switched=switched_derivative.numpy(),
        V_local=local_value.numpy(),
        dV_local=local_derivative.numpy(),
    )
    return metrics


def run(
    specification: ArticleSpecification,
    config: ImplementationConfig,
    outdir: Path,
) -> dict[str, object]:
    lyapunov, controller, training_metrics = train_networks(specification, config)
    validation_metrics = validate_and_plot(
        lyapunov, controller, specification, config, outdir
    )
    result: dict[str, object] = {
        "article_specification": specification.__dict__,
        "implementation_config": config.__dict__,
        "local_design": {
            "K": [-2.0, -3.0],
            "P": [[1.25, 0.25], [0.25, 0.25]],
            "origin": "current reproducible design, not values reported in the article",
        },
        "training": training_metrics,
        "validation": validation_metrics,
    }
    _save_json(outdir / "run_record.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("example_2/article_version/figures/reference"),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run a short installation check; its output is not a reference result",
    )
    arguments = parser.parse_args()
    specification = ArticleSpecification()
    config = ImplementationConfig()
    if arguments.quick:
        specification = ArticleSpecification(grid_points_per_axis=30)
        config = ImplementationConfig(
            training_steps=20,
            validation_points_per_axis=41,
            plot_points_per_axis=81,
            log_every=10,
        )
    result = run(specification, config, arguments.outdir)
    print(json.dumps(result["validation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
