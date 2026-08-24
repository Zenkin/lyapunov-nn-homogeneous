"""Domain-aligned neural stabilization experiment for Example 2.

The construction keeps the two-controller principle of Section IV-B:

* ``u=Kx`` and ``V_l=x^T P x`` are used near the origin;
* a neural controller ``N`` and neural candidate ``W=T^T T`` are used outside;
* a smooth level-set transition joins the two domains.

The transition is tied to the verified local level ``V_l=kappa``, not to a
learned level set.  A single continuously differentiable candidate ``V_NN``
is differentiated in the loss under the same smoothly blended feedback used
for simulation.  Periodic angle features make the outer functions
single-valued on the pendulum state cylinder.

All conclusions produced by this module are finite-sample numerical checks,
not continuous-domain stability certificates.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
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
    transition_upper_multiplier: float = 2.0
    epsilon: float = 0.05
    decay_rate: float = 0.10
    training_points_per_axis: int = 100
    training_steps: int = 6000
    learning_rate: float = 2e-3
    worst_fraction: float = 0.05
    worst_weight: float = 2.0
    control_weight: float = 1e-5
    validation_points_per_axis: int = 401
    audit_validation_points_per_axis: int = 801
    boundary_validation_points: int = 2048
    roa_refinement_seed_count: int = 24
    roa_level_decimal_places: int = 3
    trajectory_angle_points: int = 25
    trajectory_velocity_points: int = 21
    trajectory_final_time: float = 40.0
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


def positive_part(value: torch.Tensor) -> torch.Tensor:
    """Return ``[value]_+=max(0,value)`` from the article's notation.

    This operator belongs to the loss.  It is not an activation function of
    either neural network; both hidden layers use ``tanh``.
    """

    return torch.clamp_min(value, 0.0)


def wrapped_angle(theta: torch.Tensor) -> torch.Tensor:
    """Return the representative of an angle in ``[-pi,pi)``."""

    return torch.remainder(theta + math.pi, 2.0 * math.pi) - math.pi


def principal_state(x: torch.Tensor) -> torch.Tensor:
    """Use a principal angle while leaving angular velocity unchanged."""

    return torch.stack((wrapped_angle(x[..., 0]), x[..., 1]), dim=-1)


def transition_weight(
    local_value: torch.Tensor,
    kappa: float,
    upper_multiplier: float,
) -> torch.Tensor:
    """Return a C1 smoothstep from zero to one across local Lyapunov levels."""

    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    if upper_multiplier <= 1.0:
        raise ValueError("upper_multiplier must be greater than one")
    upper = upper_multiplier * kappa
    coordinate = torch.clamp((local_value - kappa) / (upper - kappa), 0.0, 1.0)
    return coordinate * coordinate * (3.0 - 2.0 * coordinate)


def periodic_base_value(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Return a periodic positive extension of the local quadratic form.

    ``z=(sin(theta),theta_dot)`` has the same first-order expansion as the
    state at the target.  The additional ``(1-cos(theta))^2`` term removes the
    otherwise spurious zero at ``theta=pi, theta_dot=0`` without changing the
    quadratic term at the target.
    """

    theta = x[..., 0]
    velocity = x[..., 1]
    periodic_state = torch.stack((torch.sin(theta), velocity), dim=-1)
    return article.local_quadratic_value(periodic_state, p) + (
        1.0 - torch.cos(theta)
    ) ** 2


def composite_value_and_control(
    x: torch.Tensor,
    lyapunov: PeriodicLyapunovNetwork,
    controller: PeriodicControllerNetwork,
    k: torch.Tensor,
    p: torch.Tensor,
    config: ImprovedConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``V_NN``, actual control, raw ``W``, and transition weight.

    For ``V_l<=kappa`` this is exactly ``(V_l,Kx)``.  Above
    ``V_l>=transition_upper_multiplier*kappa`` it is
    ``(V_periodic+W,N)``.  The cubic smoothstep has zero derivative at both
    endpoints, so the joined value and control are continuous and the value
    is continuously differentiable.
    """

    state = principal_state(x)
    local_value = article.local_quadratic_value(state, p)
    weight = transition_weight(
        local_value, config.kappa, config.transition_upper_multiplier
    )
    learned_value = lyapunov(state)
    outer_value = periodic_base_value(state, p) + learned_value
    combined_value = (1.0 - weight) * local_value + weight * outer_value
    local = article.local_linear_control(state, k)
    learned = controller(state)
    combined_control = (1.0 - weight).unsqueeze(-1) * local + weight.unsqueeze(
        -1
    ) * learned
    return combined_value, combined_control, learned_value, weight


def improved_control(
    x: torch.Tensor,
    controller: PeriodicControllerNetwork,
    k: torch.Tensor,
    p: torch.Tensor,
    config: ImprovedConfig,
) -> torch.Tensor:
    """Evaluate the same smooth feedback used during training."""

    state = principal_state(x)
    local_value = article.local_quadratic_value(state, p)
    weight = transition_weight(
        local_value, config.kappa, config.transition_upper_multiplier
    )
    local = article.local_linear_control(state, k)
    learned = controller(state)
    return (1.0 - weight).unsqueeze(-1) * local + weight.unsqueeze(-1) * learned


def composite_value_derivative(
    lyapunov: PeriodicLyapunovNetwork,
    controller: PeriodicControllerNetwork,
    points: torch.Tensor,
    config: ImprovedConfig,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``V_NN``, its actual derivative, control, raw ``W``, and weight."""

    k, p = consistent_corrected_local_design()
    differentiable = points.detach().clone().requires_grad_(True)
    value, control, learned_value, weight = composite_value_and_control(
        differentiable, lyapunov, controller, k, p, config
    )
    gradient = torch.autograd.grad(
        value.sum(), differentiable, create_graph=create_graph
    )[0]
    field = article.nonlinear_field(differentiable, control, 1.0)
    derivative = torch.sum(gradient * field, dim=1)
    return value, derivative, control, learned_value, weight


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
    config: ImprovedConfig,
) -> dict[str, torch.Tensor]:
    """Evaluate every explicitly declared term of the improved objective."""

    value, derivative, control, learned_value, _ = composite_value_derivative(
        lyapunov, controller, outer_points, config, create_graph=True
    )
    decay_violation = positive_part(derivative + config.decay_rate * value)
    positivity_violation = positive_part(config.epsilon - learned_value)
    pointwise = decay_violation + positivity_violation

    return {
        "outer_mean": pointwise.mean(),
        "outer_worst_fraction": top_fraction_mean(
            pointwise, config.worst_fraction
        ),
        "control_regularization": torch.mean(control**2),
    }


def combined_objective(
    terms: dict[str, torch.Tensor], config: ImprovedConfig
) -> torch.Tensor:
    """Combine the named terms without hiding their weights."""

    return (
        terms["outer_mean"]
        + config.worst_weight * terms["outer_worst_fraction"]
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
    lyapunov = PeriodicLyapunovNetwork()
    controller = PeriodicControllerNetwork()
    optimizer = torch.optim.Adam(
        [*lyapunov.parameters(), *controller.parameters()],
        lr=config.learning_rate,
    )

    for step in range(1, config.training_steps + 1):
        terms = training_terms(lyapunov, controller, outer_points, config)
        objective = combined_objective(terms, config)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        if step == 1 or step % config.log_every == 0 or step == config.training_steps:
            print(
                f"[train {step:5d}] objective={objective.detach().item():.6e} "
                f"outer={terms['outer_mean'].detach().item():.6e} "
                f"tail={terms['outer_worst_fraction'].detach().item():.6e}"
            )

    final_terms = training_terms(lyapunov, controller, outer_points, config)
    return lyapunov, controller, {
        "completed_steps": config.training_steps,
        "training_grid_convention": (
            f"{config.training_points_per_axis}x"
            f"{config.training_points_per_axis} cell midpoints"
        ),
        "training_points_outside_B_kappa": len(outer_points),
        "final_metric_timing": "after the last optimizer update",
        **{name: value.detach().item() for name, value in final_terms.items()},
        "combined_objective": combined_objective(
            final_terms, config
        ).detach().item(),
    }


def _origin_component(mask: np.ndarray, origin: tuple[int, int]) -> np.ndarray:
    """Return the four-neighbour component on a periodic angle grid."""

    rows, columns = mask.shape
    component = np.zeros_like(mask, dtype=bool)
    if not mask[origin]:
        return component
    stack = [origin]
    component[origin] = True
    while stack:
        row, column = stack.pop()
        neighbours = [
            ((row - 1) % rows, column),
            ((row + 1) % rows, column),
        ]
        if column > 0:
            neighbours.append((row, column - 1))
        if column + 1 < columns:
            neighbours.append((row, column + 1))
        for neighbour in neighbours:
            if mask[neighbour] and not component[neighbour]:
                component[neighbour] = True
                stack.append(neighbour)
    return component


def _sampled_sublevel_component(
    value_grid: np.ndarray,
    angle_grid: np.ndarray,
    velocity_grid: np.ndarray,
    level: float,
) -> np.ndarray:
    """Return the component of ``V<=level`` containing the sampled target."""

    origin_flat = int(np.argmin(angle_grid**2 + velocity_grid**2))
    origin = np.unravel_index(origin_flat, value_grid.shape)
    return _origin_component(value_grid <= level, origin)


def _sampled_component_metrics(
    value_grid: np.ndarray,
    derivative_grid: np.ndarray,
    angle_grid: np.ndarray,
    velocity_grid: np.ndarray,
    level: float,
) -> tuple[dict[str, float | int | bool], np.ndarray]:
    """Audit one sampled origin-connected sublevel."""

    component = _sampled_sublevel_component(
        value_grid, angle_grid, velocity_grid, level
    )
    sublevel = value_grid <= level
    nonzero = angle_grid**2 + velocity_grid**2 > 1e-14
    checked = component & nonzero
    if not np.any(checked):
        raise RuntimeError("The sampled sublevel contains no non-target point")
    return {
        "component_validation_points": int(component.sum()),
        "sublevel_validation_points": int(sublevel.sum()),
        "disconnected_sublevel_points": int(
            np.count_nonzero(sublevel & ~component)
        ),
        "maximum_dV_in_component_except_target": float(
            derivative_grid[checked].max()
        ),
        "minimum_V_in_component_except_target": float(value_grid[checked].min()),
        "nonnegative_dV_points_in_component_except_target": int(
            np.count_nonzero(derivative_grid[checked] >= 0.0)
        ),
        "touches_validation_velocity_boundary": bool(
            component[:, 0].any() or component[:, -1].any()
        ),
    }, component


def _value_derivative_with_spatial_gradients(
    point: np.ndarray,
    lyapunov: PeriodicLyapunovNetwork,
    controller: PeriodicControllerNetwork,
    config: ImprovedConfig,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate ``V``, ``dV``, and both spatial gradients at one point."""

    k, p = consistent_corrected_local_design()
    state = torch.tensor(point, requires_grad=True).reshape(1, 2)
    value, control, _, _ = composite_value_and_control(
        state, lyapunov, controller, k, p, config
    )
    value_gradient = torch.autograd.grad(
        value.sum(), state, create_graph=True
    )[0]
    field = article.nonlinear_field(state, control, 1.0)
    derivative = torch.sum(value_gradient * field, dim=1)
    derivative_gradient = torch.autograd.grad(derivative.sum(), state)[0]
    return (
        value.item(),
        derivative.item(),
        value_gradient.detach().numpy().reshape(2),
        derivative_gradient.detach().numpy().reshape(2),
    )


def refine_critical_roa_level(
    lyapunov: PeriodicLyapunovNetwork,
    controller: PeriodicControllerNetwork,
    config: ImprovedConfig,
    points: np.ndarray,
    values: np.ndarray,
    derivatives: np.ndarray,
) -> dict[str, float | int | str | list[float]]:
    """Numerically refine the first low-``V`` contact with ``dV=0``.

    The lowest sampled nonnegative-derivative points seed local constrained
    minimizations of ``V`` subject to ``dV=0``.  Each optimization is confined
    to a small box around its seed so it cannot collapse to the target, where
    both quantities are zero.  This is numerical refinement, not root
    isolation or a continuous-domain proof.
    """

    from scipy.optimize import minimize

    nonzero = np.sum(points * points, axis=1) > 1e-14
    bad_indices = np.flatnonzero(nonzero & (derivatives >= 0.0))
    if len(bad_indices) == 0:
        raise RuntimeError("No nonnegative-derivative seed exists on the grid")
    ordered = bad_indices[np.argsort(values[bad_indices])]
    seed_indices = ordered[: min(config.roa_refinement_seed_count, len(ordered))]
    angle_step = 2.0 * math.pi / config.validation_points_per_axis
    velocity_step = 8.0 / config.validation_points_per_axis
    angle_radius = 12.0 * angle_step
    velocity_radius = 12.0 * velocity_step
    candidates: list[tuple[float, np.ndarray, float]] = []

    for seed_index in seed_indices:
        seed = points[seed_index]
        lower_angle = max(-math.pi, float(seed[0] - angle_radius))
        upper_angle = min(math.pi, float(seed[0] + angle_radius))
        lower_velocity = max(-4.0, float(seed[1] - velocity_radius))
        upper_velocity = min(4.0, float(seed[1] + velocity_radius))
        cached_point: np.ndarray | None = None
        cached_result: tuple[float, float, np.ndarray, np.ndarray] | None = None

        def evaluate(candidate: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
            nonlocal cached_point, cached_result
            candidate = np.asarray(candidate, dtype=float)
            if cached_point is None or not np.array_equal(candidate, cached_point):
                cached_point = candidate.copy()
                cached_result = _value_derivative_with_spatial_gradients(
                    candidate, lyapunov, controller, config
                )
            assert cached_result is not None
            return cached_result

        result = minimize(
            fun=lambda candidate: evaluate(candidate)[0],
            x0=seed,
            jac=lambda candidate: evaluate(candidate)[2],
            method="SLSQP",
            bounds=[
                (lower_angle, upper_angle),
                (lower_velocity, upper_velocity),
            ],
            constraints={
                "type": "eq",
                "fun": lambda candidate: evaluate(candidate)[1],
                "jac": lambda candidate: evaluate(candidate)[3],
            },
            options={"ftol": 1e-13, "maxiter": 200},
        )
        value, derivative, _, _ = evaluate(result.x)
        if result.success and abs(derivative) <= 1e-8 and value > config.kappa:
            candidates.append((value, np.asarray(result.x), derivative))

    if not candidates:
        raise RuntimeError("Constrained dV=0 refinement did not converge")
    critical_value, critical_point, critical_derivative = min(
        candidates, key=lambda item: item[0]
    )
    scale = 10**config.roa_level_decimal_places
    conservative_level = math.floor((critical_value - 1e-12) * scale) / scale
    return {
        "interpretation": (
            "local constrained minimization of V subject to dV=0, seeded by "
            "the independent grid; numerical estimate, not root isolation"
        ),
        "successful_local_refinements": len(candidates),
        "critical_level_estimate": critical_value,
        "critical_point_estimate": critical_point.tolist(),
        "dV_at_critical_point": critical_derivative,
        "reported_conservative_level": conservative_level,
        "rounding_rule": (
            f"critical level rounded downward to "
            f"{config.roa_level_decimal_places} decimal places"
        ),
    }


def estimate_grid_roa(
    value_grid: np.ndarray,
    derivative_grid: np.ndarray,
    angle_grid: np.ndarray,
    velocity_grid: np.ndarray,
) -> tuple[dict[str, float | int | str], np.ndarray]:
    """Find the largest sampled origin component with strict decrease.

    The result is deliberately named a grid estimate.  It is not a
    continuous-domain certificate between validation samples.
    """

    origin_flat = int(np.argmin(angle_grid**2 + velocity_grid**2))
    origin = np.unravel_index(origin_flat, value_grid.shape)
    nonzero = angle_grid**2 + velocity_grid**2 > 1e-14
    candidates = np.unique(value_grid)

    def component_for(level: float) -> tuple[bool, np.ndarray]:
        component = _origin_component(value_grid <= level, origin)
        checked = component & nonzero
        strictly_decreasing = not np.any(derivative_grid[checked] >= 0.0)
        positive = not np.any(value_grid[checked] <= 0.0)
        touches_velocity_edge = bool(component[:, 0].any() or component[:, -1].any())
        return strictly_decreasing and positive and not touches_velocity_edge, component

    low = 0
    high = len(candidates) - 1
    best = -1
    best_component = np.zeros_like(value_grid, dtype=bool)
    while low <= high:
        middle = (low + high) // 2
        passed, component = component_for(float(candidates[middle]))
        if passed:
            best = middle
            best_component = component
            low = middle + 1
        else:
            high = middle - 1

    if best < 0:
        raise RuntimeError("No positive sampled Lyapunov sublevel passed the audit")
    level = float(candidates[best])
    checked = best_component & nonzero
    return {
        "interpretation": (
            "largest origin-connected sampled sublevel with V>0 and dV<0; "
            "finite-grid estimate, not a continuous-domain proof"
        ),
        "c_grid": level,
        "component_validation_points": int(best_component.sum()),
        "maximum_dV_in_component_except_target": float(
            derivative_grid[checked].max()
        ),
        "minimum_V_in_component_except_target": float(value_grid[checked].min()),
        "touches_validation_velocity_boundary": bool(
            best_component[:, 0].any() or best_component[:, -1].any()
        ),
    }, best_component


def validate(
    lyapunov: PeriodicLyapunovNetwork,
    controller: PeriodicControllerNetwork,
    config: ImprovedConfig,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Check the actual composite candidate on an independent midpoint grid."""

    k, p = consistent_corrected_local_design()
    count = config.validation_points_per_axis
    points = midpoint_grid(count)
    training_grid = midpoint_grid(config.training_points_per_axis)
    rectangular_grids_are_disjoint = not (
        torch.isin(training_grid[:, 0].unique(), points[:, 0].unique()).any()
        and torch.isin(training_grid[:, 1].unique(), points[:, 1].unique()).any()
    ).item()
    local_value = article.local_quadratic_value(points, p)
    value, derivative, control, learned_value, weight = composite_value_derivative(
        lyapunov, controller, points, config, create_graph=False
    )
    residual = derivative + config.decay_rate * value
    nonzero = torch.sum(points * points, dim=1) > 1e-14
    local = local_value <= config.kappa
    transition = (local_value > config.kappa) & (
        local_value < config.transition_upper_multiplier * config.kappa
    )
    outer = local_value >= config.transition_upper_multiplier * config.kappa
    objective_points = ~local
    validation_pointwise_loss = positive_part(residual[objective_points]) + positive_part(
        config.epsilon - learned_value[objective_points]
    )
    validation_objective_terms = {
        "outer_mean": validation_pointwise_loss.mean().item(),
        "outer_worst_fraction": top_fraction_mean(
            validation_pointwise_loss, config.worst_fraction
        ).item(),
        "control_regularization": torch.mean(
            control[objective_points] ** 2
        ).item(),
    }
    validation_objective_terms["combined_objective"] = (
        validation_objective_terms["outer_mean"]
        + config.worst_weight
        * validation_objective_terms["outer_worst_fraction"]
        + config.control_weight
        * validation_objective_terms["control_regularization"]
    )

    boundary = local_level_boundary(
        p, config.kappa, config.boundary_validation_points, phase=0.5
    )
    with torch.no_grad():
        boundary_value, boundary_control, _, boundary_weight = composite_value_and_control(
            boundary, lyapunov, controller, k, p, config
        )
        local_boundary_control = article.local_linear_control(boundary, k)

    seam_velocity = torch.linspace(-4.0, 4.0, 2001)
    left = torch.stack((torch.full_like(seam_velocity, -math.pi), seam_velocity), dim=1)
    right = torch.stack((torch.full_like(seam_velocity, math.pi), seam_velocity), dim=1)
    with torch.no_grad():
        left_value, left_control, _, _ = composite_value_and_control(
            left, lyapunov, controller, k, p, config
        )
        right_value, right_control, _, _ = composite_value_and_control(
            right, lyapunov, controller, k, p, config
        )

    angle_grid = points[:, 0].reshape(count, count).numpy()
    velocity_grid = points[:, 1].reshape(count, count).numpy()
    value_grid = value.detach().reshape(count, count).numpy()
    derivative_grid = derivative.detach().reshape(count, count).numpy()
    grid_roa_metrics, _ = estimate_grid_roa(
        value_grid, derivative_grid, angle_grid, velocity_grid
    )
    critical_refinement = refine_critical_roa_level(
        lyapunov,
        controller,
        config,
        points.numpy(),
        value.detach().numpy(),
        derivative.detach().numpy(),
    )
    roa_level = float(critical_refinement["reported_conservative_level"])
    conservative_metrics, roa_component = _sampled_component_metrics(
        value_grid,
        derivative_grid,
        angle_grid,
        velocity_grid,
        roa_level,
    )
    roa_metrics: dict[str, object] = {
        "interpretation": (
            "origin-connected sampled sublevel at a locally refined critical "
            "level rounded downward; finite numerical estimate, not a "
            "continuous-domain proof"
        ),
        "c_grid_pass": grid_roa_metrics["c_grid"],
        "c_conservative": roa_level,
        "critical_refinement": critical_refinement,
        **conservative_metrics,
    }

    def region_metrics(mask: torch.Tensor) -> dict[str, float | int]:
        checked = mask & nonzero
        return {
            "points": int(checked.sum().item()),
            "minimum_V": value[checked].min().item(),
            "maximum_dV": derivative[checked].max().item(),
            "nonnegative_dV_fraction": (
                derivative[checked] >= 0.0
            ).double().mean().item(),
            "positive_decay_residual_fraction": (
                residual[checked] > 0.0
            ).double().mean().item(),
        }

    metrics: dict[str, object] = {
        "interpretation": "independent finite-sample check, not a theorem",
        "validation_grid_convention": f"{count}x{count} cell midpoints",
        "validation_grid_disjoint_from_training_grid": rectangular_grids_are_disjoint,
        "validation_points": len(points),
        "objective_on_validation_points_outside_B_kappa": (
            validation_objective_terms
        ),
        "full_domain": region_metrics(torch.ones_like(nonzero)),
        "local_domain": region_metrics(local),
        "transition_annulus": region_metrics(transition),
        "outer_domain": region_metrics(outer),
        "minimum_raw_W_outside_B_kappa": learned_value[~local].min().item(),
        "raw_W_below_epsilon_fraction_outside_B_kappa": (
            learned_value[~local] < config.epsilon
        ).double().mean().item(),
        "maximum_raw_W_margin_violation_outside_B_kappa": positive_part(
            config.epsilon - learned_value[~local]
        ).max().item(),
        "maximum_abs_VNN_minus_kappa_on_inner_boundary": (
            boundary_value - config.kappa
        ).abs().max().item(),
        "maximum_abs_control_difference_on_inner_boundary": (
            boundary_control - local_boundary_control
        ).abs().max().item(),
        "maximum_transition_weight_on_inner_boundary": boundary_weight.max().item(),
        "maximum_periodic_edge_VNN_mismatch": (
            left_value - right_value
        ).abs().max().item(),
        "maximum_periodic_edge_control_mismatch": (
            left_control - right_control
        ).abs().max().item(),
        "roa_estimate": roa_metrics,
    }
    arrays = {
        "x": points.numpy(),
        "V_local": local_value.numpy(),
        "V_NN": value.detach().numpy(),
        "dV_NN": derivative.detach().numpy(),
        "decay_residual": residual.detach().numpy(),
        "raw_W": learned_value.detach().numpy(),
        "transition_weight": weight.detach().numpy(),
        "control": control.detach().squeeze(-1).numpy(),
        "roa_component": roa_component,
        "roa_critical_point": np.asarray(
            critical_refinement["critical_point_estimate"], dtype=float
        ),
    }
    return metrics, arrays


def closed_loop_field(
    x: torch.Tensor,
    controller: PeriodicControllerNetwork,
    k: torch.Tensor,
    p: torch.Tensor,
    config: ImprovedConfig,
) -> torch.Tensor:
    """Evaluate the physical field under the smooth domain-aligned control."""

    state = principal_state(x)
    control = improved_control(state, controller, k, p, config)
    return article.nonlinear_field(state, control, 1.0)


def simulate_trajectories(
    lyapunov: PeriodicLyapunovNetwork,
    controller: PeriodicControllerNetwork,
    config: ImprovedConfig,
    roa_level: float,
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
    crossed_loss_of_authority = torch.zeros(len(state), dtype=torch.bool)
    minimum_crossing_speed = torch.full((len(state),), math.inf)
    sample_stride = max(1, round(0.1 / config.trajectory_step))
    sampled_states = [state.clone()]
    sampled_times = [0.0]
    steps = round(config.trajectory_final_time / config.trajectory_step)

    initial_local_value = article.local_quadratic_value(state, p)
    entered_local[initial_local_value <= config.kappa] = 0.0
    entered_target[initial_local_value <= config.trajectory_target_level] = 0.0
    with torch.no_grad():
        initial_vnn = composite_value_and_control(
            state, lyapunov, controller, k, p, config
        )[0]
    initial_in_roa = initial_vnn <= roa_level
    initial_near_inner_boundary = initial_in_roa & (
        initial_vnn >= 0.95 * roa_level
    )
    initial_just_outside = (initial_vnn > roa_level) & (
        initial_vnn <= 1.05 * roa_level
    )

    with torch.no_grad():
        for step in range(1, steps + 1):
            h = config.trajectory_step
            previous = state
            f1 = closed_loop_field(state, controller, k, p, config)
            f2 = closed_loop_field(state + 0.5 * h * f1, controller, k, p, config)
            f3 = closed_loop_field(state + 0.5 * h * f2, controller, k, p, config)
            f4 = closed_loop_field(state + h * f3, controller, k, p, config)
            state = state + (h / 6.0) * (f1 + 2.0 * f2 + 2.0 * f3 + f4)
            state[:, 0] = wrapped_angle(state[:, 0])
            previous_cosine = torch.cos(previous[:, 0])
            current_cosine = torch.cos(state[:, 0])
            crossed = previous_cosine * current_cosine <= 0.0
            denominator = previous_cosine.abs() + current_cosine.abs()
            fraction = torch.where(
                denominator > 0.0,
                previous_cosine.abs() / denominator,
                torch.zeros_like(denominator),
            )
            crossing_velocity = previous[:, 1] + fraction * (
                state[:, 1] - previous[:, 1]
            )
            crossed_loss_of_authority |= crossed
            minimum_crossing_speed[crossed] = torch.minimum(
                minimum_crossing_speed[crossed], crossing_velocity[crossed].abs()
            )
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
    sampled_state_tensor = torch.stack(sampled_states)
    flat_sampled = sampled_state_tensor.reshape(-1, 2)
    sampled_values: list[torch.Tensor] = []
    sampled_derivatives: list[torch.Tensor] = []
    sampled_controls: list[torch.Tensor] = []
    for start in range(0, len(flat_sampled), 20000):
        value, derivative, control, _, _ = composite_value_derivative(
            lyapunov,
            controller,
            flat_sampled[start : start + 20000],
            config,
            create_graph=False,
        )
        sampled_values.append(value.detach())
        sampled_derivatives.append(derivative.detach())
        sampled_controls.append(control.detach().squeeze(-1))
    sampled_vnn = torch.cat(sampled_values).reshape(sampled_state_tensor.shape[:-1])
    sampled_dvnn = torch.cat(sampled_derivatives).reshape(
        sampled_state_tensor.shape[:-1]
    )
    sampled_control = torch.cat(sampled_controls).reshape(
        sampled_state_tensor.shape[:-1]
    )
    successful_crossings = success & crossed_loss_of_authority
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
        "initial_conditions_in_roa_grid_level": int(initial_in_roa.sum().item()),
        "successful_initial_conditions_in_roa_grid_level": int(
            (success & initial_in_roa).sum().item()
        ),
        "initial_conditions_in_inner_boundary_band_0.95c_to_c": int(
            initial_near_inner_boundary.sum().item()
        ),
        "successful_initial_conditions_in_inner_boundary_band": int(
            (success & initial_near_inner_boundary).sum().item()
        ),
        "initial_conditions_in_outer_boundary_band_c_to_1.05c": int(
            initial_just_outside.sum().item()
        ),
        "successful_initial_conditions_in_outer_boundary_band": int(
            (success & initial_just_outside).sum().item()
        ),
        "roa_grid_level": roa_level,
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
        "trajectories_crossing_theta_plus_or_minus_pi_over_2": int(
            crossed_loss_of_authority.sum().item()
        ),
        "successful_trajectories_crossing_theta_plus_or_minus_pi_over_2": int(
            successful_crossings.sum().item()
        ),
        "minimum_crossing_speed_among_successful_crossings": (
            minimum_crossing_speed[successful_crossings].min().item()
            if successful_crossings.any()
            else math.nan
        ),
        "maximum_final_V_local": final_local_value.max().item(),
    }
    arrays = {
        "initial": initial.numpy(),
        "final": state.numpy(),
        "final_V_local": final_local_value.numpy(),
        "success": success.numpy(),
        "initial_V_NN": initial_vnn.numpy(),
        "initial_in_roa_grid_level": initial_in_roa.numpy(),
        "initial_in_inner_boundary_band": initial_near_inner_boundary.numpy(),
        "initial_in_outer_boundary_band": initial_just_outside.numpy(),
        "entry_time_B_kappa": entered_local.numpy(),
        "entry_time_target": entered_target.numpy(),
        "maximum_abs_velocity": maximum_abs_velocity.numpy(),
        "exceeded_velocity_domain": exceeded_velocity_domain.numpy(),
        "crossed_loss_of_authority": crossed_loss_of_authority.numpy(),
        "minimum_crossing_speed": minimum_crossing_speed.numpy(),
        "sampled_time": np.asarray(sampled_times),
        "sampled_state": sampled_state_tensor.numpy(),
        "sampled_V_NN": sampled_vnn.numpy(),
        "sampled_dV_NN": sampled_dvnn.numpy(),
        "sampled_control": sampled_control.numpy(),
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
            points, controller, k, p, config
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
                state.unsqueeze(0), controller, k, p, config
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
    roa_level: float,
    outdir: Path,
) -> None:
    """Save the Lyapunov-domain comparison and a trajectory diagnostic."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    count = config.validation_points_per_axis
    x = validation["x"]
    angle_grid = x[:, 0].reshape(count, count)
    velocity_grid = x[:, 1].reshape(count, count)
    local_grid = validation["V_local"].reshape(count, count)
    value_grid = validation["V_NN"].reshape(count, count)
    derivative_grid = validation["dV_NN"].reshape(count, count)
    residual_grid = validation["decay_residual"].reshape(count, count)
    roa_component = validation["roa_component"]
    roa_critical_point = validation["roa_critical_point"]
    initial = trajectories["initial"]
    success = trajectories["success"]
    extrapolated = trajectories["exceeded_velocity_domain"]
    inner_boundary_band = trajectories["initial_in_inner_boundary_band"]
    outer_boundary_band = trajectories["initial_in_outer_boundary_band"]
    sampled = trajectories["sampled_state"]
    equilibrium_angles = np.asarray(
        [item["theta"] for item in equilibria["equilibria"]]
    )

    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.4), constrained_layout=True)
    figure.suptitle("Example 2: Lyapunov-domain and trajectory audit", fontsize=14)

    value_plot = axes[0, 0].contourf(
        angle_grid, velocity_grid, value_grid, levels=30, cmap="viridis"
    )
    axes[0, 0].contour(
        angle_grid, velocity_grid, roa_component.astype(float), levels=[0.5],
        colors=["#F2B134"], linewidths=2.1
    )
    axes[0, 0].contour(
        angle_grid,
        velocity_grid,
        local_grid,
        levels=[config.kappa, config.transition_upper_multiplier * config.kappa],
        colors=["#FFFFFF", "#FFFFFF"],
        linestyles=["solid", "dashed"],
        linewidths=[1.3, 1.0],
    )
    axes[0, 0].set_title(r"$V_{NN}$; gold: sampled-component boundary")
    figure.colorbar(value_plot, ax=axes[0, 0], label=r"$V_{NN}$")

    condition_grid = np.zeros_like(value_grid)
    condition_grid[(residual_grid > 0.0) & (derivative_grid < 0.0)] = 1.0
    condition_grid[derivative_grid >= 0.0] = 2.0
    axes[0, 1].contourf(
        angle_grid,
        velocity_grid,
        condition_grid,
        levels=[-0.5, 0.5, 1.5, 2.5],
        colors=["#74ADD1", "#F4A261", "#D1495B"],
    )
    axes[0, 1].contour(
        angle_grid, velocity_grid, roa_component.astype(float), levels=[0.5],
        colors=["#F2B134"], linewidths=2.1
    )
    axes[0, 1].scatter(
        roa_critical_point[0], roa_critical_point[1], marker="*", s=70,
        facecolors="#FFFFFF", edgecolors="#111111", linewidths=0.9, zorder=7,
    )
    axes[0, 1].set_title(r"Actual condition map for $\dot V_{NN}$")
    residual = validation["decay_residual"]
    derivative = validation["dV_NN"]
    nondecreasing_count = int(np.count_nonzero(derivative >= 0.0))
    positive_count = int(np.count_nonzero(residual > 0.0))
    slow_count = positive_count - nondecreasing_count
    total_count = int(len(residual))
    axes[0, 1].legend(
        handles=[
            Patch(
                facecolor="#74ADD1",
                label=rf"$\dot V_{{NN}}+0.1V_{{NN}}\leq0$: "
                rf"{total_count-positive_count:,} ({100*(total_count-positive_count)/total_count:.2f}\%)",
            ),
            Patch(
                facecolor="#F4A261",
                label=rf"$-0.1V_{{NN}}<\dot V_{{NN}}<0$: "
                rf"{slow_count:,} ({100*slow_count/total_count:.2f}\%)",
            ),
            Patch(
                facecolor="#D1495B",
                label=rf"$\dot V_{{NN}}\geq0$: "
                rf"{nondecreasing_count:,} ({100*nondecreasing_count/total_count:.2f}\%)",
            ),
            Patch(
                facecolor="none", edgecolor="#F2B134", linewidth=2.0,
                label=rf"sampled component: $c={roa_level:.3f}$",
            ),
            Line2D(
                [0], [0], marker="*", linestyle="none", markersize=7,
                markerfacecolor="#FFFFFF", markeredgecolor="#111111",
                label=r"first refined contact with $\dot V_{NN}=0$",
            ),
        ],
        loc="upper left",
        fontsize=7.2,
        framealpha=0.95,
    )

    axes[1, 0].contourf(
        angle_grid,
        velocity_grid,
        roa_component.astype(float),
        levels=[0.5, 1.5],
        colors=["#B8D8C0"],
        alpha=0.68,
    )
    axes[1, 0].contour(
        angle_grid, velocity_grid, roa_component.astype(float), levels=[0.5],
        colors=["#2F6F4E"], linewidths=1.8
    )
    axes[1, 0].scatter(
        initial[success, 0], initial[success, 1], s=10,
        color="#2B7A9B", label=f"converged — {int(success.sum())}/{len(success)}"
    )
    axes[1, 0].scatter(
        initial[~success, 0], initial[~success, 1], s=16,
        color="#C44536", label=f"not reached — {int((~success).sum())}"
    )
    axes[1, 0].scatter(
        initial[inner_boundary_band, 0], initial[inner_boundary_band, 1],
        s=42, facecolors="none", edgecolors="#F2B134", linewidths=1.2,
        label=rf"inside boundary band — {int(inner_boundary_band.sum())}",
    )
    axes[1, 0].scatter(
        initial[outer_boundary_band, 0], initial[outer_boundary_band, 1],
        s=52, facecolors="none", edgecolors="#6C5AA7", linewidths=1.2,
        label=rf"just outside — {int(outer_boundary_band.sum())}",
    )
    axes[1, 0].scatter(
        equilibrium_angles, np.zeros_like(equilibrium_angles), marker="x",
        s=42, linewidths=1.3, color="#111111", label="equilibria", zorder=6
    )
    axes[1, 0].set_title("Green: sampled origin component; dots: simulations")
    axes[1, 0].legend(loc="upper left", fontsize=7.0)

    selected = np.linspace(0, len(initial) - 1, min(45, len(initial)), dtype=int)
    for index in selected:
        color = "#C44536" if not success[index] else (
            "#D89000" if extrapolated[index] else "#2B7A9B"
        )
        path = sampled[:, index].copy()
        path[1:][np.abs(np.diff(path[:, 0])) > math.pi] = np.nan
        axes[1, 1].plot(path[:, 0], path[:, 1], color=color, alpha=0.55, lw=0.7)
    contour_levels = np.linspace(
        0.05, max(0.06, np.quantile(value_grid, 0.75)), 7
    )
    axes[1, 1].contour(
        angle_grid, velocity_grid, value_grid, levels=contour_levels,
        colors=["#6B7280"], linewidths=0.55, alpha=0.55
    )
    axes[1, 1].contour(
        angle_grid, velocity_grid, roa_component.astype(float), levels=[0.5],
        colors=["#F2B134"], linewidths=1.6
    )
    axes[1, 1].scatter(
        initial[success & ~extrapolated, 0], initial[success & ~extrapolated, 1],
        s=8, color="#2B7A9B", label="success in domain"
    )
    axes[1, 1].scatter(
        initial[success & extrapolated, 0], initial[success & extrapolated, 1],
        s=10, color="#D89000", label="success after extrapolation"
    )
    axes[1, 1].scatter(
        initial[~success, 0], initial[~success, 1], s=11,
        color="#C44536", label="not reached"
    )
    axes[1, 1].set_title("Closed-loop trajectories and Lyapunov level sets")
    axes[1, 1].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2, fontsize=7.5
    )

    for axis in axes.flat:
        axis.axvline(-0.5 * math.pi, color="#6B7280", linestyle="--", linewidth=0.8)
        axis.axvline(0.5 * math.pi, color="#6B7280", linestyle="--", linewidth=0.8)
        axis.set_xlim(-math.pi, math.pi)
        axis.set_ylim(-4.0, 4.0)
        axis.set_xlabel(r"angle $\theta$")
        axis.set_ylabel(r"angular velocity $\dot\theta$")
        axis.grid(color="#D8DEE4", linewidth=0.5, alpha=0.55)

    outdir.mkdir(parents=True, exist_ok=True)
    figure.savefig(outdir / "improved_audit.png", dpi=220)
    figure.savefig(outdir / "improved_audit.svg")
    plt.close(figure)

    maximum_sampled_derivative = trajectories["sampled_dV_NN"].max(axis=0)
    lower_left = (
        (initial[:, 0] < -1.0)
        & (initial[:, 1] < -1.0)
        & success
        & ~extrapolated
        & (maximum_sampled_derivative <= 1e-10)
    )
    candidates = np.flatnonzero(lower_left)
    if len(candidates) == 0:
        candidates = np.flatnonzero(success)
    if len(candidates) == 0:
        candidates = np.arange(len(initial))
    target = np.asarray([-1.4, -1.8])
    diagnostic_index = int(
        candidates[np.argmin(np.sum((initial[candidates] - target) ** 2, axis=1))]
    )
    diagnostic_state = sampled[:, diagnostic_index].copy()
    diagnostic_state[1:][np.abs(np.diff(diagnostic_state[:, 0])) > math.pi] = np.nan
    time = trajectories["sampled_time"]
    sampled_value = trajectories["sampled_V_NN"][:, diagnostic_index]
    sampled_derivative = trajectories["sampled_dV_NN"][:, diagnostic_index]
    active_value_indices = np.flatnonzero(
        sampled_value > config.trajectory_target_level
    )
    diagnostic_time_end = float(time[-1])
    if len(active_value_indices) > 0:
        diagnostic_time_end = min(
            diagnostic_time_end,
            max(2.0, float(time[active_value_indices[-1]]) + 0.5),
        )

    diagnostic, diagnostic_axes = plt.subplot_mosaic(
        [["phase", "value"], ["phase", "derivative"]],
        figsize=(11.5, 7.0),
        constrained_layout=True,
    )
    diagnostic.suptitle(
        "Decreasing trajectory: phase motion versus Lyapunov value", fontsize=14
    )
    phase_axis = diagnostic_axes["phase"]
    phase_axis.contour(
        angle_grid, velocity_grid, value_grid, levels=contour_levels,
        colors=["#6B7280"], linewidths=0.65, alpha=0.65
    )
    phase_axis.contour(
        angle_grid, velocity_grid, roa_component.astype(float), levels=[0.5],
        colors=["#F2B134"], linewidths=2.0
    )
    phase_axis.plot(
        diagnostic_state[:, 0], diagnostic_state[:, 1],
        color="#2B7A9B", linewidth=1.5
    )
    phase_axis.scatter(
        initial[diagnostic_index, 0], initial[diagnostic_index, 1], marker="o",
        s=45, color="#D89000", label="initial state", zorder=5
    )
    phase_axis.scatter(
        0.0, 0.0, marker="x", s=55, color="#111111", label="target", zorder=5
    )
    phase_axis.axvline(-0.5 * math.pi, color="#6B7280", linestyle="--", linewidth=0.9)
    phase_axis.axvline(0.5 * math.pi, color="#6B7280", linestyle="--", linewidth=0.9)
    phase_axis.set_xlim(-math.pi, math.pi)
    phase_axis.set_ylim(-4.0, 4.0)
    phase_axis.set_xlabel(r"angle $\theta$")
    phase_axis.set_ylabel(r"angular velocity $\dot\theta$")
    phase_axis.set_title("Motion across coordinates while descending level sets")
    phase_axis.legend(loc="upper left", fontsize=8)
    phase_axis.grid(color="#D8DEE4", linewidth=0.5, alpha=0.55)

    diagnostic_axes["value"].plot(time, sampled_value, color="#2B7A9B", linewidth=1.5)
    diagnostic_axes["value"].axhline(
        roa_level, color="#F2B134", linewidth=1.5,
        label=rf"$c={roa_level:.3f}$"
    )
    diagnostic_axes["value"].set_ylabel(r"$V_{NN}(t)$")
    diagnostic_axes["value"].set_title("Lyapunov value along the selected trajectory")
    diagnostic_axes["value"].set_xlim(float(time[0]), diagnostic_time_end)
    diagnostic_axes["value"].legend(loc="upper right", fontsize=8)
    diagnostic_axes["value"].grid(color="#D8DEE4", linewidth=0.5, alpha=0.55)

    diagnostic_axes["derivative"].plot(
        time, sampled_derivative, color="#6C5AA7", linewidth=1.2
    )
    diagnostic_axes["derivative"].axhline(0.0, color="#C44536", linewidth=1.0)
    diagnostic_axes["derivative"].fill_between(
        time, sampled_derivative, 0.0, where=sampled_derivative <= 0.0,
        color="#74ADD1", alpha=0.25
    )
    diagnostic_axes["derivative"].set_xlabel("time")
    diagnostic_axes["derivative"].set_ylabel(r"$\dot V_{NN}(t)$")
    diagnostic_axes["derivative"].set_title("Negative derivative confirms descent")
    diagnostic_axes["derivative"].set_xlim(float(time[0]), diagnostic_time_end)
    diagnostic_axes["derivative"].grid(
        color="#D8DEE4", linewidth=0.5, alpha=0.55
    )
    diagnostic.savefig(
        outdir / "trajectory_diagnostic.png", dpi=220, bbox_inches="tight"
    )
    diagnostic.savefig(outdir / "trajectory_diagnostic.svg", bbox_inches="tight")
    plt.close(diagnostic)

    crossing_candidates = np.flatnonzero(
        success
        & trajectories["crossed_loss_of_authority"]
        & ~extrapolated
        & (maximum_sampled_derivative <= 1e-10)
    )
    if len(crossing_candidates) == 0:
        crossing_candidates = np.flatnonzero(
            success & trajectories["crossed_loss_of_authority"]
        )
    crossing_observed = len(crossing_candidates) > 0
    if not crossing_observed:
        crossing_candidates = np.flatnonzero(success)
    if len(crossing_candidates) == 0:
        crossing_candidates = np.arange(len(initial))
    crossing_target = np.asarray([-1.9, 1.6])
    crossing_index = int(
        crossing_candidates[
            np.argmin(
                np.sum((initial[crossing_candidates] - crossing_target) ** 2, axis=1)
            )
        ]
    )
    crossing_state = sampled[:, crossing_index].copy()
    crossing_path = crossing_state.copy()
    crossing_path[1:][np.abs(np.diff(crossing_path[:, 0])) > math.pi] = np.nan
    crossing_control = trajectories["sampled_control"][:, crossing_index]
    theta = crossing_state[:, 0]
    velocity = crossing_state[:, 1]
    cosine = np.cos(theta)
    gravity_term = np.sin(theta)
    control_term = cosine * crossing_control
    total_acceleration = gravity_term + control_term
    crossing_intervals = np.flatnonzero(cosine[:-1] * cosine[1:] <= 0.0)
    crossing_times: list[float] = []
    crossing_speeds: list[float] = []
    crossing_gravity: list[float] = []
    for index in crossing_intervals:
        denominator = abs(cosine[index]) + abs(cosine[index + 1])
        fraction = abs(cosine[index]) / denominator if denominator > 0.0 else 0.0
        crossing_times.append(float(time[index] + fraction * (time[index + 1] - time[index])))
        crossing_speeds.append(
            float(abs(velocity[index] + fraction * (velocity[index + 1] - velocity[index])))
        )
        crossing_gravity.append(
            1.0 if np.sin(theta[index] + fraction * (theta[index + 1] - theta[index])) >= 0.0 else -1.0
        )

    crossing_figure, crossing_axes = plt.subplots(
        1, 2, figsize=(11.5, 4.8), constrained_layout=True
    )
    crossing_figure.suptitle(
        (
            r"Passing the loss-of-authority lines $\theta=\pm\pi/2$"
            if crossing_observed
            else r"No $\theta=\pm\pi/2$ crossing in the sampled horizon"
        ),
        fontsize=14,
    )
    crossing_axes[0].contour(
        angle_grid, velocity_grid, value_grid, levels=contour_levels,
        colors=["#6B7280"], linewidths=0.6, alpha=0.6
    )
    crossing_axes[0].plot(
        crossing_path[:, 0], crossing_path[:, 1], color="#2B7A9B", linewidth=1.5
    )
    crossing_axes[0].scatter(
        initial[crossing_index, 0], initial[crossing_index, 1],
        s=44, color="#D89000", label="initial state", zorder=5
    )
    crossing_axes[0].scatter(0.0, 0.0, marker="x", s=52, color="#111111", label="target")
    crossing_axes[0].axvline(-0.5 * math.pi, color="#6B7280", linestyle="--", linewidth=1.0)
    crossing_axes[0].axvline(0.5 * math.pi, color="#6B7280", linestyle="--", linewidth=1.0)
    crossing_axes[0].set_xlim(-math.pi, math.pi)
    crossing_axes[0].set_ylim(-4.0, 4.0)
    crossing_axes[0].set_xlabel(r"angle $\theta$")
    crossing_axes[0].set_ylabel(r"angular velocity $\dot\theta$")
    crossing_axes[0].set_title(
        "The trajectory crosses with nonzero angular velocity"
        if crossing_observed
        else "A successful trajectory from the short audit"
    )
    crossing_axes[0].legend(loc="upper left", fontsize=8)
    crossing_axes[0].grid(color="#D8DEE4", linewidth=0.5, alpha=0.55)

    crossing_axes[1].plot(time, gravity_term, color="#2B7A9B", label=r"gravity $\sin\theta$")
    crossing_axes[1].plot(time, control_term, color="#D89000", label=r"control $\cos\theta\,u$")
    crossing_axes[1].plot(
        time, total_acceleration, color="#6C5AA7", linewidth=1.2,
        label=r"total $\ddot\theta$"
    )
    for crossing_time, speed, gravity in zip(
        crossing_times, crossing_speeds, crossing_gravity
    ):
        crossing_axes[1].axvline(
            crossing_time, color="#6B7280", linestyle="--", linewidth=0.8
        )
        crossing_axes[1].scatter(crossing_time, 0.0, color="#D89000", s=24, zorder=5)
        crossing_axes[1].scatter(crossing_time, gravity, color="#2B7A9B", s=24, zorder=5)
        crossing_axes[1].annotate(
            rf"$|\dot\theta|\approx{speed:.2f}$",
            (crossing_time, gravity), xytext=(4, 7), textcoords="offset points",
            fontsize=8,
        )
    crossing_axes[1].axhline(0.0, color="#AAB2BD", linewidth=0.8)
    crossing_time_end = min(
        float(time[-1]),
        max(2.0, max(crossing_times, default=0.0) + 1.0),
    )
    crossing_axes[1].set_xlim(float(time[0]), crossing_time_end)
    crossing_axes[1].set_xlabel("time")
    crossing_axes[1].set_ylabel("acceleration contribution")
    crossing_axes[1].set_title(
        r"At each crossing: $\cos\theta\,u=0$, while $\sin\theta=\pm1$"
        if crossing_observed
        else "Acceleration terms along the selected trajectory"
    )
    crossing_axes[1].legend(loc="upper right", fontsize=8)
    crossing_axes[1].grid(color="#D8DEE4", linewidth=0.5, alpha=0.55)
    crossing_figure.savefig(outdir / "loss_of_authority_crossing.png", dpi=220)
    crossing_figure.savefig(outdir / "loss_of_authority_crossing.svg")
    plt.close(crossing_figure)


def run(config: ImprovedConfig, outdir: Path) -> dict[str, object]:
    """Train, validate, simulate, and save all reproducibility artifacts."""

    lyapunov, controller, training = train(config)
    validation_metrics, validation_arrays = validate(
        lyapunov, controller, config
    )
    audit_config = replace(
        config,
        validation_points_per_axis=config.audit_validation_points_per_axis,
    )
    high_resolution_validation, _ = validate(
        lyapunov, controller, audit_config
    )
    primary_level = float(validation_metrics["roa_estimate"]["c_conservative"])
    audit_level = float(
        high_resolution_validation["roa_estimate"]["c_conservative"]
    )
    roa_level = min(primary_level, audit_level)
    validation_metrics["roa_estimate"]["selected_level"] = roa_level
    validation_metrics["roa_estimate"]["selection_rule"] = (
        "minimum of the downward-rounded "
        f"{config.validation_points_per_axis}x{config.validation_points_per_axis} "
        "and "
        f"{config.audit_validation_points_per_axis}x"
        f"{config.audit_validation_points_per_axis} estimates"
    )
    if roa_level != primary_level:
        count = config.validation_points_per_axis
        points = validation_arrays["x"]
        angle_grid = points[:, 0].reshape(count, count)
        velocity_grid = points[:, 1].reshape(count, count)
        conservative_metrics, component = _sampled_component_metrics(
            validation_arrays["V_NN"].reshape(count, count),
            validation_arrays["dV_NN"].reshape(count, count),
            angle_grid,
            velocity_grid,
            roa_level,
        )
        validation_metrics["roa_estimate"].update(conservative_metrics)
        validation_arrays["roa_component"] = component
    trajectory_metrics, trajectory_arrays = simulate_trajectories(
        lyapunov, controller, config, roa_level
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
        validation_arrays, trajectory_arrays, equilibria, config, roa_level, outdir
    )
    result: dict[str, object] = {
        "mathematical_changes": [
            "corrected Jacobian and consistent local P",
            "C1 level-set blend tied to the verified local V_l levels",
            "single composite V_NN differentiated under the simulated control",
            "periodic positive base plus the nonnegative neural W=T^T T",
            "periodic angle features for W and N",
            "outer tail loss in addition to mean loss",
            "grid-seeded refinement of the origin-connected decreasing sublevel",
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
        "high_resolution_validation": high_resolution_validation,
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
            training_steps=20,
            validation_points_per_axis=41,
            audit_validation_points_per_axis=61,
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
