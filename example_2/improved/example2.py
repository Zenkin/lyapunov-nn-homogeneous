"""Domain-aligned neural stabilization experiment for Example 2.

The construction keeps the two-controller principle of Section IV-B:

* ``u=Kx`` and ``V_l=x^T P x`` are used near the origin;
* a neural controller ``N`` and neural candidate ``W`` are used outside;
* the control switches between the two domains.

The switching set is changed from the learned condition ``W<kappa`` to the
verified local condition ``V_l<=kappa``.  The outer controller is trained to
point into that set on its boundary and to match the local controller there.
Periodic angle features make the learned functions single-valued on the
pendulum state cylinder.

All conclusions produced by this module are finite-sample numerical checks,
not continuous-domain stability certificates.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import nn

import example_2.article_version.example2 as article
from example_2.corrected_matrix.example2 import (
    consistent_corrected_local_design,
)
from example_2.corrected_matrix.switching_audit import local_level_boundary


matplotlib.use("Agg")
torch.set_default_dtype(torch.float64)


@dataclass(frozen=True)
class ImprovedConfig:
    """Declared numerical choices absent from the article."""

    seed: int = 20260820
    kappa: float = 0.05
    epsilon: float = 0.05
    decay_rate: float = 0.10
    boundary_decay_rate: float = 0.10
    training_points_per_axis: int = 100
    boundary_training_points: int = 512
    training_steps: int = 6000
    learning_rate: float = 2e-3
    worst_fraction: float = 0.05
    worst_weight: float = 2.0
    boundary_match_weight: float = 5.0
    boundary_inward_weight: float = 5.0
    control_weight: float = 1e-5
    validation_points_per_axis: int = 211
    boundary_validation_points: int = 2048
    trajectory_angle_points: int = 25
    trajectory_velocity_points: int = 21
    trajectory_final_time: float = 20.0
    trajectory_step: float = 0.01
    trajectory_target_level: float = 1e-4
    log_every: int = 500


def local_analytic_certificate(kappa: float) -> dict[str, float | str]:
    """Return a continuous decay bound for the selected local controller.

    With ``R=||x||`` and the corrected Lyapunov equation, direct expansion
    gives

    ``DV_l F(x,Kx) <= -R^2 + c R^4``

    where ``c=sqrt(13)/18+13/6``.  Since
    ``V_l>=lambda_min(P) R^2``, the level ``V_l<=kappa`` has the declared
    strict quadratic decay whenever the returned margin is positive.
    """

    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    minimum_eigenvalue = (13.0 - 3.0 * math.sqrt(13.0)) / 12.0
    nonlinear_coefficient = math.sqrt(13.0) / 18.0 + 13.0 / 6.0
    maximum_squared_norm = kappa / minimum_eigenvalue
    decay_margin = 1.0 - nonlinear_coefficient * maximum_squared_norm
    return {
        "bound": "DV_l F(x,Kx) <= -decay_margin*||x||^2 on V_l<=kappa",
        "minimum_eigenvalue_P": minimum_eigenvalue,
        "nonlinear_remainder_coefficient": nonlinear_coefficient,
        "maximum_squared_norm_in_B_kappa": maximum_squared_norm,
        "decay_margin": decay_margin,
    }


def periodic_features(x: torch.Tensor) -> torch.Tensor:
    """Map ``(theta, theta_dot)`` to periodic, scaled network features.

    The feature vector ``(sin(theta), 1-cos(theta), theta_dot/4)`` is exactly
    equal at ``theta=-pi`` and ``theta=pi`` and is zero at the target state.
    """

    theta = x[..., 0]
    velocity = x[..., 1]
    return torch.stack(
        (torch.sin(theta), 1.0 - torch.cos(theta), velocity / 4.0), dim=-1
    )


class PeriodicZeroAtOriginNetwork(nn.Module):
    """One-hidden-layer tanh network, periodic and exactly zero at the target."""

    def __init__(self, hidden_units: int, output_dimension: int):
        super().__init__()
        self.hidden = nn.Linear(3, hidden_units)
        self.output = nn.Linear(hidden_units, output_dimension, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = periodic_features(x)
        hidden = torch.tanh(self.hidden(features))
        hidden_at_target = torch.tanh(self.hidden.bias)
        return self.output(hidden - hidden_at_target)


class PeriodicLyapunovNetwork(nn.Module):
    """Nonnegative outer candidate ``W=T^T T`` with periodic angle input."""

    def __init__(self, hidden_units: int = 64):
        super().__init__()
        self.transform = PeriodicZeroAtOriginNetwork(hidden_units, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformed = self.transform(x)
        return torch.sum(transformed * transformed, dim=-1)


class PeriodicControllerNetwork(PeriodicZeroAtOriginNetwork):
    """Periodic scalar outer control law."""

    def __init__(self, hidden_units: int = 32):
        super().__init__(hidden_units, 1)


def wrapped_angle(theta: torch.Tensor) -> torch.Tensor:
    """Return the representative of an angle in ``[-pi,pi)``."""

    return torch.remainder(theta + math.pi, 2.0 * math.pi) - math.pi


def principal_state(x: torch.Tensor) -> torch.Tensor:
    """Use a principal angle while leaving angular velocity unchanged."""

    return torch.stack((wrapped_angle(x[..., 0]), x[..., 1]), dim=-1)


def improved_switched_control(
    x: torch.Tensor,
    controller: PeriodicControllerNetwork,
    k: torch.Tensor,
    p: torch.Tensor,
    kappa: float,
) -> torch.Tensor:
    """Use ``Kx`` exactly on ``B_kappa`` and the neural control outside."""

    state = principal_state(x)
    local = article.local_linear_control(state, k)
    learned = controller(state)
    use_local = article.local_quadratic_value(state, p) <= kappa
    return torch.where(use_local.unsqueeze(-1), local, learned)


def value_gradient_and_field(
    lyapunov: PeriodicLyapunovNetwork,
    controller: PeriodicControllerNetwork,
    points: torch.Tensor,
    omega: float,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``W``, ``DW F(x,N(x))``, and ``N(x)``."""

    differentiable = points.detach().clone().requires_grad_(True)
    value = lyapunov(differentiable)
    gradient = torch.autograd.grad(
        value.sum(), differentiable, create_graph=create_graph
    )[0]
    control = controller(differentiable)
    field = article.nonlinear_field(differentiable, control, omega)
    derivative = torch.sum(gradient * field, dim=1)
    return value, derivative, control


def local_value_derivative(
    points: torch.Tensor,
    control: torch.Tensor,
    p: torch.Tensor,
    omega: float,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``V_l`` and its derivative for a supplied control tensor."""

    differentiable = points.detach().clone().requires_grad_(True)
    value = article.local_quadratic_value(differentiable, p)
    gradient = torch.autograd.grad(
        value.sum(), differentiable, create_graph=create_graph
    )[0]
    field = article.nonlinear_field(differentiable, control, omega)
    return value, torch.sum(gradient * field, dim=1)


def midpoint_grid(points_per_axis: int) -> torch.Tensor:
    """Return a cell-midpoint grid on the article rectangle."""

    specification = article.ArticleSpecification(
        grid_points_per_axis=points_per_axis
    )
    return article.uniform_rectangle_grid(specification, include_boundary=False)


def top_fraction_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    """Mean of the largest declared fraction of pointwise values."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must belong to (0,1]")
    count = max(1, math.ceil(fraction * len(values)))
    return torch.topk(values, count, sorted=False).values.mean()


def training_terms(
    lyapunov: PeriodicLyapunovNetwork,
    controller: PeriodicControllerNetwork,
    outer_points: torch.Tensor,
    boundary_points: torch.Tensor,
    config: ImprovedConfig,
) -> dict[str, torch.Tensor]:
    """Evaluate every explicitly declared term of the improved objective."""

    k, p = consistent_corrected_local_design()
    value, derivative, control = value_gradient_and_field(
        lyapunov, controller, outer_points, 1.0, create_graph=True
    )
    decay_violation = torch.relu(derivative + config.decay_rate * value)
    positivity_violation = torch.relu(config.epsilon - value)
    pointwise = decay_violation + positivity_violation

    boundary_control = controller(boundary_points)
    local_boundary_control = article.local_linear_control(boundary_points, k)
    _, boundary_local_derivative = local_value_derivative(
        boundary_points,
        boundary_control,
        p,
        1.0,
        create_graph=True,
    )
    inward_violation = torch.relu(
        boundary_local_derivative + config.boundary_decay_rate * config.kappa
    )

    return {
        "outer_mean": pointwise.mean(),
        "outer_worst_fraction": top_fraction_mean(
            pointwise, config.worst_fraction
        ),
        "boundary_control_match": torch.mean(
            (boundary_control - local_boundary_control) ** 2
        ),
        "boundary_inward": torch.mean(inward_violation**2),
        "control_regularization": torch.mean(control**2),
    }


def combined_objective(
    terms: dict[str, torch.Tensor], config: ImprovedConfig
) -> torch.Tensor:
    """Combine the named terms without hiding their weights."""

    return (
        terms["outer_mean"]
        + config.worst_weight * terms["outer_worst_fraction"]
        + config.boundary_match_weight * terms["boundary_control_match"]
        + config.boundary_inward_weight * terms["boundary_inward"]
        + config.control_weight * terms["control_regularization"]
    )


def train(
    config: ImprovedConfig,
) -> tuple[
    PeriodicLyapunovNetwork,
    PeriodicControllerNetwork,
    dict[str, float | int | str],
]:
    """Train the outer candidate and controller on the declared samples."""

    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    _, p = consistent_corrected_local_design()
    full_grid = midpoint_grid(config.training_points_per_axis)
    local_value = article.local_quadratic_value(full_grid, p)
    outer_points = full_grid[local_value > config.kappa]
    boundary_points = local_level_boundary(
        p, config.kappa, config.boundary_training_points
    )

    lyapunov = PeriodicLyapunovNetwork()
    controller = PeriodicControllerNetwork()
    optimizer = torch.optim.Adam(
        [*lyapunov.parameters(), *controller.parameters()],
        lr=config.learning_rate,
    )

    for step in range(1, config.training_steps + 1):
        terms = training_terms(
            lyapunov, controller, outer_points, boundary_points, config
        )
        objective = combined_objective(terms, config)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        if step == 1 or step % config.log_every == 0 or step == config.training_steps:
            print(
                f"[train {step:5d}] objective={objective.detach().item():.6e} "
                f"outer={terms['outer_mean'].detach().item():.6e} "
                f"tail={terms['outer_worst_fraction'].detach().item():.6e} "
                f"match={terms['boundary_control_match'].detach().item():.6e} "
                f"inward={terms['boundary_inward'].detach().item():.6e}"
            )

    final_terms = training_terms(
        lyapunov, controller, outer_points, boundary_points, config
    )
    return lyapunov, controller, {
        "completed_steps": config.training_steps,
        "training_grid_convention": (
            f"{config.training_points_per_axis}x"
            f"{config.training_points_per_axis} cell midpoints"
        ),
        "training_points_outside_B_kappa": len(outer_points),
        "boundary_training_points": len(boundary_points),
        "final_metric_timing": "after the last optimizer update",
        **{name: value.detach().item() for name, value in final_terms.items()},
        "combined_objective": combined_objective(
            final_terms, config
        ).detach().item(),
    }


def validate(
    lyapunov: PeriodicLyapunovNetwork,
    controller: PeriodicControllerNetwork,
    config: ImprovedConfig,
) -> tuple[dict[str, float | int | str], dict[str, np.ndarray]]:
    """Check disjoint midpoint and shifted-boundary validation samples."""

    k, p = consistent_corrected_local_design()
    points = midpoint_grid(config.validation_points_per_axis)
    training_grid = midpoint_grid(config.training_points_per_axis)
    training_angles = torch.unique(training_grid[:, 0])
    training_velocities = torch.unique(training_grid[:, 1])
    validation_angles = torch.unique(points[:, 0])
    validation_velocities = torch.unique(points[:, 1])
    angle_coordinates_overlap = torch.isclose(
        training_angles.unsqueeze(1),
        validation_angles.unsqueeze(0),
        atol=1e-14,
        rtol=0.0,
    ).any()
    velocity_coordinates_overlap = torch.isclose(
        training_velocities.unsqueeze(1),
        validation_velocities.unsqueeze(0),
        atol=1e-14,
        rtol=0.0,
    ).any()
    rectangular_grids_are_disjoint = not (
        bool(angle_coordinates_overlap) and bool(velocity_coordinates_overlap)
    )
    local_value = article.local_quadratic_value(points, p)
    local_domain = local_value <= config.kappa
    outer_points = points[~local_domain]

    learned_value, learned_derivative, learned_control = value_gradient_and_field(
        lyapunov, controller, outer_points, 1.0, create_graph=False
    )
    local_points = points[local_domain]
    local_control = article.local_linear_control(local_points, k)
    _, local_derivative = local_value_derivative(
        local_points, local_control, p, 1.0, create_graph=False
    )
    local_nonzero = torch.sum(local_points * local_points, dim=1) > 1e-14

    boundary = local_level_boundary(
        p,
        config.kappa,
        config.boundary_validation_points,
        phase=0.5,
    )
    training_boundary = local_level_boundary(
        p, config.kappa, config.boundary_training_points
    )
    minimum_boundary_sample_distance = torch.cdist(
        training_boundary, boundary
    ).min().item()
    training_outer = training_grid[
        article.local_quadratic_value(training_grid, p) > config.kappa
    ]
    training_sample_set = {
        tuple(row) for row in torch.cat((training_outer, training_boundary)).tolist()
    }
    all_validation_samples_are_new = all(
        tuple(row) not in training_sample_set
        for row in torch.cat((points, boundary)).tolist()
    )
    boundary_learned_control = controller(boundary).detach()
    boundary_local_control = article.local_linear_control(boundary, k)
    _, boundary_derivative = local_value_derivative(
        boundary,
        boundary_learned_control,
        p,
        1.0,
        create_graph=False,
    )

    seam_velocity = torch.linspace(-4.0, 4.0, 2001)
    left = torch.stack((torch.full_like(seam_velocity, -math.pi), seam_velocity), dim=1)
    right = torch.stack((torch.full_like(seam_velocity, math.pi), seam_velocity), dim=1)
    with torch.no_grad():
        seam_w_error = (lyapunov(left) - lyapunov(right)).abs()
        seam_u_error = (controller(left) - controller(right)).abs()

    decay_residual = learned_derivative + config.decay_rate * learned_value
    boundary_residual = (
        boundary_derivative + config.boundary_decay_rate * config.kappa
    )
    metrics: dict[str, float | int | str] = {
        "interpretation": "independent finite-sample check, not a theorem",
        "validation_grid_convention": (
            f"{config.validation_points_per_axis}x"
            f"{config.validation_points_per_axis} cell midpoints"
        ),
        "validation_grid_disjoint_from_training_grid": (
            rectangular_grids_are_disjoint
        ),
        "all_validation_samples_disjoint_from_all_training_samples": (
            all_validation_samples_are_new
        ),
        "validation_points": len(points),
        "validation_outer_points": len(outer_points),
        "minimum_W_outside_B_kappa": learned_value.min().item(),
        "maximum_positivity_margin_violation_outside_B_kappa": torch.relu(
            config.epsilon - learned_value
        ).max().item(),
        "W_below_epsilon_fraction_outside_B_kappa": (
            learned_value < config.epsilon
        ).double().mean().item(),
        "maximum_DW_F_outside_B_kappa": learned_derivative.max().item(),
        "maximum_DW_F_plus_decay_W_outside_B_kappa": decay_residual.max().item(),
        "nonnegative_DW_F_fraction_outside_B_kappa": (
            learned_derivative >= 0.0
        ).double().mean().item(),
        "positive_decay_residual_fraction_outside_B_kappa": (
            decay_residual > 0.0
        ).double().mean().item(),
        "maximum_DV_local_F_local_inside_B_kappa_except_origin": (
            local_derivative[local_nonzero].max().item()
        ),
        "nonnegative_DV_local_fraction_inside_B_kappa": (
            local_derivative[local_nonzero] >= 0.0
        ).double().mean().item(),
        "boundary_validation_points": len(boundary),
        "boundary_validation_disjoint_from_boundary_training_samples": (
            minimum_boundary_sample_distance > 1e-12
        ),
        "minimum_boundary_validation_distance_from_training_samples": (
            minimum_boundary_sample_distance
        ),
        "maximum_abs_control_jump_on_boundary": (
            boundary_learned_control - boundary_local_control
        ).abs().max().item(),
        "maximum_DV_local_F_learned_on_boundary": boundary_derivative.max().item(),
        "maximum_boundary_inward_residual": boundary_residual.max().item(),
        "positive_boundary_inward_residual_fraction": (
            boundary_residual > 0.0
        ).double().mean().item(),
        "maximum_periodic_edge_W_mismatch": seam_w_error.max().item(),
        "maximum_periodic_edge_control_mismatch": seam_u_error.max().item(),
    }
    arrays = {
        "x": points.numpy(),
        "V_local": local_value.numpy(),
        "outer_x": outer_points.numpy(),
        "outer_W": learned_value.detach().numpy(),
        "outer_dW": learned_derivative.detach().numpy(),
        "outer_decay_residual": decay_residual.detach().numpy(),
        "outer_u": learned_control.detach().squeeze(-1).numpy(),
        "boundary_x": boundary.numpy(),
        "boundary_dV_learned": boundary_derivative.detach().numpy(),
        "boundary_u_jump": (
            boundary_learned_control - boundary_local_control
        ).detach().squeeze(-1).numpy(),
    }
    return metrics, arrays


def closed_loop_field(
    x: torch.Tensor,
    controller: PeriodicControllerNetwork,
    k: torch.Tensor,
    p: torch.Tensor,
    kappa: float,
) -> torch.Tensor:
    """Evaluate the physical field under the domain-aligned switch."""

    state = principal_state(x)
    control = improved_switched_control(state, controller, k, p, kappa)
    return article.nonlinear_field(state, control, 1.0)


def simulate_trajectories(
    controller: PeriodicControllerNetwork,
    config: ImprovedConfig,
) -> tuple[dict[str, float | int | str], dict[str, np.ndarray]]:
    """Integrate a declared initial-condition grid by fixed-step RK4."""

    k, p = consistent_corrected_local_design()
    # The endpoints -pi and pi represent the same physical angle.  Use one
    # endpoint only so the empirical fraction does not count that state twice.
    angle = -math.pi + (2.0 * math.pi / config.trajectory_angle_points) * torch.arange(
        config.trajectory_angle_points
    )
    velocity = torch.linspace(-4.0, 4.0, config.trajectory_velocity_points)
    angle_grid, velocity_grid = torch.meshgrid(angle, velocity, indexing="ij")
    initial = torch.stack(
        (angle_grid.reshape(-1), velocity_grid.reshape(-1)), dim=1
    )
    state = initial.clone()
    entered_local = torch.full((len(state),), -1.0)
    entered_target = torch.full((len(state),), -1.0)
    finite = torch.ones(len(state), dtype=torch.bool)
    maximum_abs_velocity = state[:, 1].abs().clone()
    exceeded_velocity_domain = torch.zeros(len(state), dtype=torch.bool)
    sample_stride = max(1, round(0.1 / config.trajectory_step))
    sampled_states = [state.clone()]
    sampled_times = [0.0]
    steps = round(config.trajectory_final_time / config.trajectory_step)

    with torch.no_grad():
        for step in range(1, steps + 1):
            h = config.trajectory_step
            f1 = closed_loop_field(state, controller, k, p, config.kappa)
            f2 = closed_loop_field(state + 0.5 * h * f1, controller, k, p, config.kappa)
            f3 = closed_loop_field(state + 0.5 * h * f2, controller, k, p, config.kappa)
            f4 = closed_loop_field(state + h * f3, controller, k, p, config.kappa)
            state = state + (h / 6.0) * (f1 + 2.0 * f2 + 2.0 * f3 + f4)
            state[:, 0] = wrapped_angle(state[:, 0])
            finite &= torch.isfinite(state).all(dim=1)
            maximum_abs_velocity = torch.maximum(
                maximum_abs_velocity, state[:, 1].abs()
            )
            exceeded_velocity_domain |= state[:, 1].abs() > 4.0
            local_value = article.local_quadratic_value(state, p)
            time = step * h
            newly_local = (entered_local < 0.0) & (local_value <= config.kappa)
            newly_target = (
                (entered_target < 0.0)
                & (local_value <= config.trajectory_target_level)
            )
            entered_local[newly_local] = time
            entered_target[newly_target] = time
            if step % sample_stride == 0:
                sampled_states.append(state.clone())
                sampled_times.append(time)

    final_local_value = article.local_quadratic_value(state, p)
    success = finite & (final_local_value <= config.trajectory_target_level)
    metrics: dict[str, float | int | str] = {
        "interpretation": "empirical fixed-step trajectory check, not an ROA proof",
        "integration_method": "classical RK4 with wrapped angle",
        "integration_step": config.trajectory_step,
        "final_time": config.trajectory_final_time,
        "initial_condition_grid": (
            f"{config.trajectory_angle_points}x{config.trajectory_velocity_points}; "
            "periodic angle endpoint counted once"
        ),
        "initial_conditions": len(initial),
        "target_level_V_local": config.trajectory_target_level,
        "trajectories_entering_B_kappa": int((entered_local >= 0.0).sum().item()),
        "trajectories_reaching_target_level": int(
            (entered_target >= 0.0).sum().item()
        ),
        "successful_at_final_time": int(success.sum().item()),
        "successful_without_leaving_training_domain": int(
            (success & ~exceeded_velocity_domain).sum().item()
        ),
        "success_fraction": success.double().mean().item(),
        "nonfinite_trajectories": int((~finite).sum().item()),
        "trajectories_exceeding_training_velocity_domain": int(
            exceeded_velocity_domain.sum().item()
        ),
        "maximum_absolute_velocity_over_trajectories": (
            maximum_abs_velocity.max().item()
        ),
        "maximum_final_V_local": final_local_value.max().item(),
    }
    arrays = {
        "initial": initial.numpy(),
        "final": state.numpy(),
        "final_V_local": final_local_value.numpy(),
        "success": success.numpy(),
        "entry_time_B_kappa": entered_local.numpy(),
        "entry_time_target": entered_target.numpy(),
        "maximum_abs_velocity": maximum_abs_velocity.numpy(),
        "exceeded_velocity_domain": exceeded_velocity_domain.numpy(),
        "sampled_time": np.asarray(sampled_times),
        "sampled_state": torch.stack(sampled_states).numpy(),
    }
    return metrics, arrays


def equilibrium_audit(
    controller: PeriodicControllerNetwork,
    config: ImprovedConfig,
    scan_points: int = 20001,
) -> dict[str, object]:
    """Locate sampled closed-loop equilibria on the zero-velocity line.

    Every equilibrium must have ``theta_dot=0``.  Sign changes of the second
    field component are refined by bisection and retained only when the final
    field residual is small.  This detects roots on a dense one-dimensional
    scan; it is not a symbolic root-count proof.
    """

    if scan_points < 3 or scan_points % 2 == 0:
        raise ValueError("scan_points must be an odd integer of at least three")
    k, p = consistent_corrected_local_design()

    def acceleration(theta: torch.Tensor) -> torch.Tensor:
        points = torch.stack((theta, torch.zeros_like(theta)), dim=1)
        return closed_loop_field(
            points, controller, k, p, config.kappa
        )[:, 1]

    angle = torch.linspace(-math.pi, math.pi, scan_points)
    with torch.no_grad():
        values = acceleration(angle)
    brackets: list[tuple[float, float]] = []
    exact = torch.nonzero(values.abs() <= 1e-13).flatten().tolist()
    for index in exact:
        value = angle[index].item()
        brackets.append((value, value))
    changes = torch.nonzero(values[:-1] * values[1:] < 0.0).flatten().tolist()
    for index in changes:
        brackets.append((angle[index].item(), angle[index + 1].item()))

    roots: list[float] = []
    for left, right in brackets:
        if left != right:
            left_value = acceleration(torch.tensor([left]))[0].item()
            for _ in range(60):
                middle = 0.5 * (left + right)
                middle_value = acceleration(torch.tensor([middle]))[0].item()
                if left_value * middle_value <= 0.0:
                    right = middle
                else:
                    left = middle
                    left_value = middle_value
            candidate = 0.5 * (left + right)
        else:
            candidate = left
        residual = abs(acceleration(torch.tensor([candidate]))[0].item())
        distinct = all(
            abs((candidate - root + math.pi) % (2.0 * math.pi) - math.pi)
            > 1e-7
            for root in roots
        )
        if residual <= 1e-9 and distinct:
            roots.append(candidate)
    roots.sort()

    records: list[dict[str, object]] = []
    for root in roots:
        point = torch.tensor([root, 0.0], requires_grad=True)
        jacobian = torch.autograd.functional.jacobian(
            lambda state: closed_loop_field(
                state.unsqueeze(0), controller, k, p, config.kappa
            )[0],
            point,
        )
        eigenvalues = torch.linalg.eigvals(jacobian).detach()
        records.append(
            {
                "theta": root,
                "field_residual": abs(
                    acceleration(torch.tensor([root]))[0].item()
                ),
                "jacobian": jacobian.detach().tolist(),
                "jacobian_eigenvalues": [
                    {"real": value.real.item(), "imag": value.imag.item()}
                    for value in eigenvalues
                ],
            }
        )
    return {
        "interpretation": "dense zero-velocity scan with bisection; not symbolic root isolation",
        "scan_points": scan_points,
        "detected_equilibria": len(records),
        "equilibria": records,
    }


def save_figure(
    validation: dict[str, np.ndarray],
    trajectories: dict[str, np.ndarray],
    equilibria: dict[str, object],
    config: ImprovedConfig,
    outdir: Path,
) -> None:
    """Save a compact four-panel scientific audit figure."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    count = config.validation_points_per_axis
    x = validation["x"]
    local_value = validation["V_local"].reshape(count, count)
    outer_w = validation["outer_W"]
    outer_derivative = validation["outer_dW"]
    residual = validation["outer_decay_residual"]
    angle_grid = x[:, 0].reshape(count, count)
    velocity_grid = x[:, 1].reshape(count, count)
    outer_mask = local_value > config.kappa
    w_grid = np.full((count, count), np.nan)
    residual_grid = np.full((count, count), np.nan)
    derivative_grid = np.full((count, count), np.nan)
    w_grid[outer_mask] = outer_w
    residual_grid[outer_mask] = residual
    derivative_grid[outer_mask] = outer_derivative

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)
    figure.suptitle("Domain-aligned neural stabilization: finite-sample audit", fontsize=14)

    first = axes[0, 0].contourf(
        angle_grid,
        velocity_grid,
        np.ma.masked_invalid(w_grid),
        levels=28,
        cmap="viridis",
    )
    axes[0, 0].contour(
        angle_grid,
        velocity_grid,
        local_value,
        levels=[config.kappa],
        colors=["#F2B134"],
        linewidths=2.0,
    )
    axes[0, 0].set_title(r"Outer candidate $W$ and switch $V_\ell=\kappa$")
    figure.colorbar(first, ax=axes[0, 0], label=r"$W$")

    condition_grid = np.full((count, count), np.nan)
    condition_grid[outer_mask & (residual_grid <= 0.0)] = 0.0
    condition_grid[
        outer_mask & (residual_grid > 0.0) & (derivative_grid < 0.0)
    ] = 1.0
    condition_grid[outer_mask & (derivative_grid >= 0.0)] = 2.0
    axes[0, 1].contourf(
        angle_grid,
        velocity_grid,
        np.ma.masked_invalid(condition_grid),
        levels=[-0.5, 0.5, 1.5, 2.5],
        colors=["#74ADD1", "#F4A261", "#D1495B"],
    )
    axes[0, 1].contour(
        angle_grid,
        velocity_grid,
        np.ma.masked_invalid(residual_grid),
        levels=[0.0],
        colors=["#A65F00"],
        linewidths=1.1,
    )
    axes[0, 1].contour(
        angle_grid,
        velocity_grid,
        np.ma.masked_invalid(derivative_grid),
        levels=[0.0],
        colors=["#7A0019"],
        linewidths=1.1,
    )
    axes[0, 1].contour(
        angle_grid,
        velocity_grid,
        local_value,
        levels=[config.kappa],
        colors=["#F2B134"],
        linewidths=1.8,
    )
    axes[0, 1].set_title("Outer Lyapunov-condition map")

    positive_count = int(np.count_nonzero(residual > 0.0))
    nondecreasing_count = int(np.count_nonzero(outer_derivative >= 0.0))
    slow_decrease_count = positive_count - nondecreasing_count
    residual_count = int(residual.size)
    satisfied_count = residual_count - positive_count
    axes[0, 1].legend(
        handles=[
            Patch(
                facecolor="#74ADD1",
                label=(
                    rf"$\dot W+0.1W\leq0$: satisfied — "
                    rf"{satisfied_count:,} ({100.0 * satisfied_count / residual_count:.2f}\%)"
                ),
            ),
            Patch(
                facecolor="#F4A261",
                label=(
                    rf"$-0.1W<\dot W<0$: too slow — "
                    rf"{slow_decrease_count:,} ({100.0 * slow_decrease_count / residual_count:.2f}\%)"
                ),
            ),
            Patch(
                facecolor="#D1495B",
                label=(
                    rf"$\dot W\geq0$: not decreasing — "
                    rf"{nondecreasing_count:,} ({100.0 * nondecreasing_count / residual_count:.2f}\%)"
                ),
            ),
            Patch(
                facecolor="white",
                edgecolor="#F2B134",
                linewidth=1.8,
                label=r"$V_\ell\leq\kappa$: local domain",
            ),
        ],
        loc="upper left",
        fontsize=7.2,
        framealpha=0.95,
    )
    axes[0, 1].text(
        0.98,
        0.97,
        (
            rf"$\max(\dot W+0.1W)={float(residual.max()):+.4f}$"
            "\n"
            rf"$\max\dot W={float(outer_derivative.max()):+.4f}$"
        ),
        transform=axes[0, 1].transAxes,
        ha="right",
        va="top",
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.94, "edgecolor": "#AAB2BD"},
    )

    boundary = validation["boundary_x"]
    boundary_derivative = validation["boundary_dV_learned"]
    negative = boundary_derivative < 0.0
    axes[1, 0].scatter(
        boundary[negative, 0], boundary[negative, 1], s=7, color="#2B7A9B", label="negative"
    )
    axes[1, 0].scatter(
        boundary[~negative, 0], boundary[~negative, 1], s=9, color="#C44536", label="nonnegative"
    )
    axes[1, 0].set_title(r"$DV_\ell F(x,N(x))$ on $V_\ell=\kappa$")
    axes[1, 0].legend(loc="upper right")

    initial = trajectories["initial"]
    success = trajectories["success"]
    extrapolated = trajectories["exceeded_velocity_domain"]
    verified_success = success & ~extrapolated
    extrapolated_success = success & extrapolated
    sampled = trajectories["sampled_state"]
    selected = np.linspace(0, len(initial) - 1, min(45, len(initial)), dtype=int)
    for index in selected:
        if not success[index]:
            color = "#C44536"
        elif extrapolated[index]:
            color = "#D89000"
        else:
            color = "#2B7A9B"
        axes[1, 1].plot(
            sampled[:, index, 0], sampled[:, index, 1], color=color, alpha=0.55, lw=0.7
        )
    axes[1, 1].scatter(
        initial[verified_success, 0],
        initial[verified_success, 1],
        s=8,
        color="#2B7A9B",
        label="success in domain",
    )
    axes[1, 1].scatter(
        initial[extrapolated_success, 0],
        initial[extrapolated_success, 1],
        s=10,
        color="#D89000",
        label="success after extrapolation",
    )
    axes[1, 1].scatter(initial[~success, 0], initial[~success, 1], s=11, color="#C44536", label="not reached")
    equilibrium_angles = np.asarray(
        [item["theta"] for item in equilibria["equilibria"]]
    )
    axes[1, 1].scatter(
        equilibrium_angles,
        np.zeros_like(equilibrium_angles),
        marker="x",
        s=45,
        linewidths=1.4,
        color="#111111",
        label="equilibria",
        zorder=6,
    )
    axes[1, 1].set_title("Closed-loop trajectories and initial states")
    axes[1, 1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncols=2,
        fontsize=7.5,
    )

    for axis in axes.flat:
        axis.set_xlim(-math.pi, math.pi)
        axis.set_ylim(-4.0, 4.0)
        axis.set_xlabel(r"angle $\theta$")
        axis.set_ylabel(r"angular velocity $\dot{\theta}$")
        axis.grid(color="#D8DEE4", linewidth=0.5, alpha=0.55)

    outdir.mkdir(parents=True, exist_ok=True)
    figure.savefig(outdir / "improved_audit.png", dpi=220)
    figure.savefig(outdir / "improved_audit.svg")
    plt.close(figure)


def run(config: ImprovedConfig, outdir: Path) -> dict[str, object]:
    """Train, validate, simulate, and save all reproducibility artifacts."""

    lyapunov, controller, training = train(config)
    validation_metrics, validation_arrays = validate(
        lyapunov, controller, config
    )
    trajectory_metrics, trajectory_arrays = simulate_trajectories(
        controller, config
    )
    equilibria = equilibrium_audit(controller, config)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "lyapunov_state_dict": lyapunov.state_dict(),
            "controller_state_dict": controller.state_dict(),
            "config": asdict(config),
        },
        outdir / "model_state.pt",
    )
    np.savez_compressed(outdir / "validation_arrays.npz", **validation_arrays)
    np.savez_compressed(outdir / "trajectory_arrays.npz", **trajectory_arrays)
    save_figure(
        validation_arrays, trajectory_arrays, equilibria, config, outdir
    )
    result: dict[str, object] = {
        "mathematical_changes": [
            "corrected Jacobian and consistent local P",
            "switch on V_l<=kappa instead of W<kappa",
            "periodic angle features for W and N",
            "outer tail loss in addition to mean loss",
            "boundary inward and controller-matching losses",
        ],
        "config": asdict(config),
        "local_design": {
            "K": consistent_corrected_local_design()[0].tolist(),
            "P": consistent_corrected_local_design()[1].tolist(),
        },
        "local_analytic_certificate": local_analytic_certificate(
            config.kappa
        ),
        "training": training,
        "validation": validation_metrics,
        "trajectories": trajectory_metrics,
        "equilibrium_audit": equilibria,
    }
    (outdir / "run_record.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("example_2/improved/results/reference"),
    )
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = ImprovedConfig()
    if arguments.quick:
        config = ImprovedConfig(
            training_points_per_axis=30,
            boundary_training_points=64,
            training_steps=20,
            validation_points_per_axis=41,
            boundary_validation_points=128,
            trajectory_angle_points=7,
            trajectory_velocity_points=7,
            trajectory_final_time=0.2,
            trajectory_step=0.02,
            log_every=10,
        )
    result = run(config, arguments.outdir)
    print(json.dumps({
        "validation": result["validation"],
        "trajectories": result["trajectories"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
