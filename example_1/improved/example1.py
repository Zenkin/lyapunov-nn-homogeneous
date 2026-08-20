"""Smooth-gluing variant of the first numerical example.

The model, training, validation, and plotting are kept in one file so the
mathematical construction and its numerical checks remain explicit.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


torch.set_default_dtype(torch.float64)


@dataclass(frozen=True)
class SystemParameters:
    # a1, a2, a3, c1, c2 are printed in the article.
    a1: float = 1.0
    a2: float = 2.0
    a3: float = 1.0
    c1: float = 1.0
    c2: float = 2.0
    # Fc and vs are reconstructed from branch dev_w, not from the article.
    friction_amplitude: float = 0.8
    friction_velocity: float = 0.5


@dataclass(frozen=True)
class RunConfig:
    seed: int = 20260820
    homogeneous_hidden: int = 32
    homogeneous_train_points: int = 2048
    homogeneous_validation_points: int = 8192
    homogeneous_steps: int = 5000
    homogeneous_learning_rate: float = 1e-3
    outer_hidden: int = 32
    outer_grid_points_per_axis: int = 100
    outer_validation_points_per_axis: int = 200
    outer_steps: int = 12000
    outer_learning_rate: float = 1e-3
    outer_fixed_quadratic: float = 1e-3
    outer_decay_margin: float = 5e-2
    radial_angles: int = 512
    radial_points: int = 256
    radial_min: float = 0.25
    radial_max: float = 8.0
    kappa_padding: float = 1.01
    log_every: int = 500


def dry_friction(v: torch.Tensor, p: SystemParameters) -> torch.Tensor:
    """Bounded friction phi(v)=Fc*tanh(v/vs), copied from dev_w."""
    return p.friction_amplitude * torch.tanh(v / p.friction_velocity)


def nonlinear_drag(v: torch.Tensor) -> torch.Tensor:
    """Exact article term sqrt(|v|)*v; no smoothing is inserted."""
    return torch.sqrt(torch.abs(v)) * v


def original_field(x: torch.Tensor, p: SystemParameters) -> torch.Tensor:
    """Full vector field in the article's original coordinates."""
    x1, x2 = x[..., 0], x[..., 1]
    dx1 = x2
    dx2 = (
        -dry_friction(x2, p)
        + p.a1 * (x1 - p.c1)
        - p.a2 * nonlinear_drag(x2)
        - p.a3 * (x1 - p.c2) ** 3
    )
    return torch.stack((dx1, dx2), dim=-1)


def equilibrium_position(p: SystemParameters) -> float:
    """Find the unique real equilibrium for the printed numerical parameters.

    Bisection is used instead of the erroneous shifted closed form printed in
    the article.  The bracket is expanded deterministically if necessary.
    """

    def residual(value: float) -> float:
        return p.a1 * (value - p.c1) - p.a3 * (value - p.c2) ** 3

    left = p.c2
    right = p.c2 + 1.0
    while residual(left) * residual(right) > 0.0:
        right = p.c2 + 2.0 * (right - p.c2)
        if right - p.c2 > 1e6:
            raise RuntimeError("Could not bracket the equilibrium")

    for _ in range(100):
        middle = 0.5 * (left + right)
        if residual(left) * residual(middle) <= 0.0:
            right = middle
        else:
            left = middle
    return 0.5 * (left + right)


def shifted_field(z: torch.Tensor, p: SystemParameters, x_equilibrium: float) -> torch.Tensor:
    """Full vector field in z=(x1-x_equilibrium, x2)."""
    x = torch.stack((z[..., 0] + x_equilibrium, z[..., 1]), dim=-1)
    return original_field(x, p)


def homogeneous_field(z: torch.Tensor, p: SystemParameters) -> torch.Tensor:
    """(r=(1,2), nu=1)-homogeneous approximation at infinity."""
    z1, z2 = z[..., 0], z[..., 1]
    return torch.stack(
        (z2, -p.a3 * z1**3 - p.a2 * nonlinear_drag(z2)), dim=-1
    )


def homogeneous_norm(z: torch.Tensor) -> torch.Tensor:
    """Definition 4 gauge for r=(1,2) with varpi=2."""
    return torch.sqrt(z[..., 0] ** 2 + torch.abs(z[..., 1]))


def homogeneous_normalize(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return rho and Lambda_r(rho)^(-1) z for nonzero z.

    At z=0 the normalized direction is defined as zero only to keep program
    values finite; V(0) is assigned separately and does not use that direction.
    """
    rho = homogeneous_norm(z)
    nonzero = rho > 0.0
    safe_rho = torch.where(nonzero, rho, torch.ones_like(rho))
    y = torch.stack((z[..., 0] / safe_rho, z[..., 1] / safe_rho**2), dim=-1)
    return rho, torch.where(nonzero[..., None], y, torch.zeros_like(y))


def homogeneous_sphere(number: int, offset: float = 0.5) -> torch.Tensor:
    """Deterministic points satisfying y1^2+|y2|=1 exactly."""
    angle = 2.0 * math.pi * (torch.arange(number) + offset) / number
    sine = torch.sin(angle)
    return torch.stack((torch.cos(angle), torch.sign(sine) * sine**2), dim=1)


class SphereNetwork(nn.Module):
    """One-hidden-layer scalar network W_theta from article equation (6)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.input = nn.Linear(2, hidden)
        self.output = nn.Linear(hidden, 1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.output(torch.tanh(self.input(y))).squeeze(-1)


class OuterNetwork(nn.Module):
    """Positive-definite inner candidate used in the modified experiment.

    With s=z/input_scale and R_theta(0)=0, the represented function is

        U_theta(z) = epsilon*||s||^2 + ||R_theta(s)||^2.

    Therefore U_theta(0)=0 and U_theta(z)>0 for z!=0 independently of the
    trainable parameters.  The fixed quadratic term also prevents the
    candidate from becoming flat merely because the neural residual is zero.
    """

    def __init__(
        self,
        hidden: int,
        scale_z1: float = 1.0,
        scale_z2: float = 1.0,
        fixed_quadratic: float = 1e-3,
    ):
        super().__init__()
        if fixed_quadratic <= 0.0:
            raise ValueError("fixed_quadratic must be positive")
        self.register_buffer("input_scale", torch.tensor([scale_z1, scale_z2]))
        self.fixed_quadratic = fixed_quadratic
        self.input = nn.Linear(2, hidden)
        self.output = nn.Linear(hidden, 2, bias=False)

    def transform(self, z: torch.Tensor) -> torch.Tensor:
        scaled = z / self.input_scale
        activation = torch.tanh(self.input(scaled))
        activation_at_zero = torch.tanh(self.input.bias)
        return self.output(activation - activation_at_zero)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        scaled = z / self.input_scale
        transform = self.transform(z)
        return self.fixed_quadratic * torch.sum(scaled * scaled, dim=-1) + torch.sum(
            transform * transform, dim=-1
        )


def lie_derivative(
    values_function,
    points: torch.Tensor,
    vector_field,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return function values and grad(V)*f at points away from nonsmooth axes."""
    differentiable_points = points.detach().clone().requires_grad_(True)
    values = values_function(differentiable_points)
    gradient = torch.autograd.grad(values.sum(), differentiable_points, create_graph=create_graph)[0]
    derivative = torch.sum(gradient * vector_field(differentiable_points), dim=-1)
    return values, derivative


def homogeneous_candidate(network: SphereNetwork, z: torch.Tensor) -> torch.Tensor:
    rho, y = homogeneous_normalize(z)
    value = rho**2 * network(y)
    return torch.where(rho > 0.0, value, torch.zeros_like(value))


def quintic_smoothstep(t: torch.Tensor) -> torch.Tensor:
    """C1 transition equal to zero for t<=0 and one for t>=1.

    On [0,1] this is 6*t**5-15*t**4+10*t**3.  Its first derivative is zero
    at both endpoints, so composing it with clamp remains continuously
    differentiable there.
    """
    clipped = torch.clamp(t, 0.0, 1.0)
    return clipped**3 * (10.0 + clipped * (-15.0 + 6.0 * clipped))


def quintic_smoothstep_derivative(t: torch.Tensor) -> torch.Tensor:
    """Derivative of quintic_smoothstep with respect to its argument."""
    inside = (t > 0.0) & (t < 1.0)
    clipped = torch.clamp(t, 0.0, 1.0)
    derivative = 30.0 * clipped**2 * (1.0 - clipped) ** 2
    return torch.where(inside, derivative, torch.zeros_like(derivative))


def smooth_combined_candidate(
    homogeneous_network: SphereNetwork,
    inner_network: OuterNetwork,
    z: torch.Tensor,
    kappa: float,
) -> torch.Tensor:
    """Smoothly unite U_theta inside with V_infinity outside.

    The result equals U_theta on V_infinity<=kappa and V_infinity on
    V_infinity>=2*kappa.  Both components are positive away from the origin,
    hence their convex combination is positive there as well.
    """
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    homogeneous_value = homogeneous_candidate(homogeneous_network, z)
    gate = quintic_smoothstep((homogeneous_value - kappa) / kappa)
    inner_value = inner_network(z)
    return (1.0 - gate) * inner_value + gate * homogeneous_value


def train_homogeneous(
    config: RunConfig, p: SystemParameters
) -> tuple[SphereNetwork, dict[str, float]]:
    network = SphereNetwork(config.homogeneous_hidden)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.homogeneous_learning_rate)
    train_points = homogeneous_sphere(config.homogeneous_train_points, offset=0.5)

    for step in range(1, config.homogeneous_steps + 1):
        values, derivative = lie_derivative(
            lambda y: homogeneous_candidate(network, y),
            train_points,
            lambda y: homogeneous_field(y, p),
            create_graph=True,
        )
        loss_decay = torch.relu(derivative + 1.0).mean()
        loss_positive = torch.relu(1.0 - values).mean()
        loss = loss_decay + loss_positive
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % config.log_every == 0 or step == config.homogeneous_steps:
            print(
                f"[homogeneous {step:5d}] loss={loss.item():.6e} "
                f"min_Vinf={values.min().item():.6e} "
                f"max_DVinf_finf={derivative.max().item():.6e}"
            )

    validation_points = homogeneous_sphere(
        config.homogeneous_validation_points, offset=0.25
    )
    values, derivative = lie_derivative(
        lambda y: homogeneous_candidate(network, y),
        validation_points,
        lambda y: homogeneous_field(y, p),
        create_graph=False,
    )
    _, raw_network_derivative = lie_derivative(
        network,
        validation_points,
        lambda y: homogeneous_field(y, p),
        create_graph=False,
    )
    metrics = {
        "minimum_W_on_independent_sphere_grid": values.min().item(),
        "maximum_actual_DV_finf_on_independent_sphere_grid": derivative.max().item(),
        "maximum_actual_decay_constraint_DV_finf_plus_1": (derivative + 1.0).max().item(),
        "maximum_raw_network_DW_finf_for_comparison": raw_network_derivative.max().item(),
        "positive_violation_fraction": (values < 1.0).double().mean().item(),
        "decay_violation_fraction": (derivative > -1.0).double().mean().item(),
    }
    return network, metrics


def mesh_rectangle(
    x1_min: float,
    x1_max: float,
    x2_min: float,
    x2_max: float,
    number: int,
    midpoint: bool,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    if midpoint:
        step1 = (x1_max - x1_min) / number
        step2 = (x2_max - x2_min) / number
        axis1 = np.linspace(x1_min, x1_max, number, endpoint=False) + 0.5 * step1
        axis2 = np.linspace(x2_min, x2_max, number, endpoint=False) + 0.5 * step2
    else:
        axis1 = np.linspace(x1_min, x1_max, number)
        axis2 = np.linspace(x2_min, x2_max, number)
    grid1, grid2 = np.meshgrid(axis1, axis2, indexing="xy")
    points = torch.from_numpy(np.column_stack((grid1.ravel(), grid2.ravel())))
    return grid1, grid2, points


def choose_empirical_kappa(
    network: SphereNetwork,
    config: RunConfig,
    p: SystemParameters,
    x_equilibrium: float,
) -> tuple[float, dict[str, float]]:
    # X is the article's 1-by-1 rectangle, now centered at the true equilibrium.
    _, _, x_points = mesh_rectangle(-0.5, 0.5, -0.5, 0.5, 201, midpoint=False)
    with torch.no_grad():
        maximum_on_x = homogeneous_candidate(network, x_points).max().item()

    directions = homogeneous_sphere(config.radial_angles, offset=0.5)
    radii = torch.logspace(
        math.log10(config.radial_min),
        math.log10(config.radial_max),
        config.radial_points,
    )
    y = directions[None, :, :].expand(config.radial_points, -1, -1)
    radius = radii[:, None]
    radial_points = torch.stack(
        (radius * y[..., 0], radius**2 * y[..., 1]), dim=-1
    ).reshape(-1, 2)
    values, derivative = lie_derivative(
        lambda z: homogeneous_candidate(network, z),
        radial_points,
        lambda z: shifted_field(z, p, x_equilibrium),
        create_graph=False,
    )
    violations = derivative >= 0.0
    largest_violating_level = (
        values[violations].max().item() if torch.any(violations) else 0.0
    )
    kappa = config.kappa_padding * max(maximum_on_x, largest_violating_level)
    outside = values >= kappa
    if not torch.any(outside):
        raise RuntimeError("Radial validation range does not reach the selected kappa")

    metrics = {
        "maximum_V_on_X_grid": maximum_on_x,
        "largest_V_level_with_nonnegative_derivative": largest_violating_level,
        "selected_kappa": kappa,
        "radial_validation_minimum_rho": config.radial_min,
        "radial_validation_maximum_rho": config.radial_max,
        "maximum_DV_f_on_sampled_V_ge_kappa": derivative[outside].max().item(),
        "sample_count_V_ge_kappa": int(outside.sum().item()),
    }
    return kappa, metrics


def level_boundary(
    network: SphereNetwork, level: float, number: int, offset: float
) -> torch.Tensor:
    directions = homogeneous_sphere(number, offset=offset)
    with torch.no_grad():
        sphere_values = network(directions)
    if torch.any(sphere_values <= 0.0):
        raise RuntimeError("W_theta must be positive before constructing a level set")
    radius = torch.sqrt(level / sphere_values)
    return torch.stack(
        (radius * directions[:, 0], radius**2 * directions[:, 1]), dim=1
    )


def level_angle_grid(
    homogeneous_network: SphereNetwork,
    maximum_level: float,
    level_count: int,
    angle_count: int,
    level_offset: float,
    angle_offset: float,
) -> torch.Tensor:
    """Grid uniform in Lyapunov level and homogeneous-sphere angle."""
    directions = homogeneous_sphere(angle_count, offset=angle_offset)
    with torch.no_grad():
        sphere_values = homogeneous_network(directions)
    if torch.any(sphere_values <= 0.0):
        raise RuntimeError("W_theta must be positive before constructing level coordinates")
    levels = maximum_level * (torch.arange(level_count) + level_offset) / level_count
    radius = torch.sqrt(levels[:, None] / sphere_values[None, :])
    directions_2d = directions[None, :, :].expand(level_count, -1, -1)
    points = torch.stack(
        (radius * directions_2d[..., 0], radius**2 * directions_2d[..., 1]), dim=-1
    )
    return points.reshape(-1, 2)


def outer_training_grid(
    homogeneous_network: SphereNetwork,
    kappa: float,
    number: int,
) -> tuple[torch.Tensor, float, float]:
    directions = homogeneous_sphere(8192, offset=0.25)
    with torch.no_grad():
        minimum_sphere_value = homogeneous_network(directions).min().item()
    maximum_rho = math.sqrt(2.0 * kappa / minimum_sphere_value)
    level_grid = level_angle_grid(
        homogeneous_network,
        maximum_level=2.0 * kappa,
        level_count=number,
        angle_count=number,
        level_offset=0.5,
        angle_offset=0.5,
    )
    outside_x = (torch.abs(level_grid[:, 0]) > 0.5) | (torch.abs(level_grid[:, 1]) > 0.5)
    selected = level_grid[outside_x]
    return selected, maximum_rho, minimum_sphere_value


def train_inner_for_smooth_candidate(
    homogeneous_network: SphereNetwork,
    config: RunConfig,
    p: SystemParameters,
    x_equilibrium: float,
    kappa: float,
) -> tuple[OuterNetwork, dict[str, float], tuple[float, float]]:
    """Train the inner network through the exact united candidate.

    No pointwise dominance constraints are imposed.  The loss differentiates
    smooth_combined_candidate itself, so the transition-gate chain-rule term
    is included by construction.
    """
    grid, maximum_rho, minimum_sphere_value = outer_training_grid(
        homogeneous_network, kappa, config.outer_grid_points_per_axis
    )
    for parameter in homogeneous_network.parameters():
        parameter.requires_grad_(False)
    network = OuterNetwork(
        config.outer_hidden,
        scale_z1=maximum_rho,
        scale_z2=maximum_rho**2,
        fixed_quadratic=config.outer_fixed_quadratic,
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=config.outer_learning_rate)
    completed_steps = 0
    final_training_loss = math.inf

    for step in range(1, config.outer_steps + 1):
        values, derivative = lie_derivative(
            lambda z: smooth_combined_candidate(
                homogeneous_network, network, z, kappa
            ),
            grid,
            lambda z: shifted_field(z, p, x_equilibrium),
            create_graph=True,
        )
        decay_loss = torch.relu(derivative + config.outer_decay_margin).mean()
        optimizer.zero_grad(set_to_none=True)
        decay_loss.backward()
        optimizer.step()
        completed_steps = step
        final_training_loss = decay_loss.item()

        if step == 1 or step % config.log_every == 0 or step == config.outer_steps:
            print(
                f"[smooth {step:5d}] loss={decay_loss.item():.6e} "
                f"min_V={values.min().item():.3e} "
                f"max_dV={torch.max(derivative).item():.3e}"
            )

    metrics = {
        "training_grid_points_after_filter": len(grid),
        "minimum_W_theta_on_sphere_used_for_box": minimum_sphere_value,
        "bounding_box_abs_z1": maximum_rho,
        "bounding_box_abs_z2": maximum_rho**2,
        "fixed_quadratic_coefficient": config.outer_fixed_quadratic,
        "trained_expression": "relu(dV_smooth/dt + decay_margin)",
        "completed_training_steps": completed_steps,
        "final_training_loss": final_training_loss,
    }
    return network, metrics, (maximum_rho, maximum_rho**2)


def explicit_smooth_lie_derivative(
    homogeneous_network: SphereNetwork,
    inner_network: OuterNetwork,
    points: torch.Tensor,
    vector_field,
    kappa: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the united value and its explicitly expanded derivative.

    This function is an audit implementation independent of differentiating
    smooth_combined_candidate as a single expression.
    """
    homogeneous_value, homogeneous_derivative = lie_derivative(
        lambda z: homogeneous_candidate(homogeneous_network, z),
        points,
        vector_field,
        create_graph=False,
    )
    inner_value, inner_derivative = lie_derivative(
        inner_network,
        points,
        vector_field,
        create_graph=False,
    )
    transition_coordinate = (homogeneous_value - kappa) / kappa
    gate = quintic_smoothstep(transition_coordinate)
    gate_derivative_by_value = (
        quintic_smoothstep_derivative(transition_coordinate) / kappa
    )
    value = (1.0 - gate) * inner_value + gate * homogeneous_value
    derivative = (
        (1.0 - gate) * inner_derivative
        + gate * homogeneous_derivative
        + gate_derivative_by_value
        * homogeneous_derivative
        * (homogeneous_value - inner_value)
    )
    return value, derivative


def near_velocity_axis_grid(
    homogeneous_network: SphereNetwork,
    maximum_level: float,
    level_count: int = 64,
    offset_count: int = 24,
) -> torch.Tensor:
    """Independent points approaching z2=0 from both sides.

    Exact z2=0 is excluded because rho contains abs(z2).  The offsets span
    1e-12 through 1e-2 on the homogeneous sphere.
    """
    offsets = torch.logspace(-12.0, -2.0, offset_count)
    first = torch.sqrt(1.0 - offsets)
    directions = torch.cat(
        (
            torch.stack((first, offsets), dim=1),
            torch.stack((first, -offsets), dim=1),
            torch.stack((-first, offsets), dim=1),
            torch.stack((-first, -offsets), dim=1),
        ),
        dim=0,
    )
    with torch.no_grad():
        sphere_values = homogeneous_network(directions)
    levels = maximum_level * (torch.arange(level_count) + 0.375) / level_count
    radius = torch.sqrt(levels[:, None] / sphere_values[None, :])
    directions_2d = directions[None, :, :].expand(level_count, -1, -1)
    return torch.stack(
        (radius * directions_2d[..., 0], radius**2 * directions_2d[..., 1]),
        dim=-1,
    ).reshape(-1, 2)


def exact_axis_forward_difference_audit(
    homogeneous_network: SphereNetwork,
    inner_network: OuterNetwork,
    p: SystemParameters,
    x_equilibrium: float,
    kappa: float,
    maximum_abs_z1: float,
    number: int = 2048,
) -> dict[str, object]:
    """Sample the upper directional derivative on the nonsmooth axis.

    At z2=0 PyTorch's selected derivative of abs(z2) is not used.  Instead we
    evaluate [V(z+h*f(z))-V(z)]/h for positive h approaching zero, which is the
    expression in the upper Dini directional derivative.
    """
    axis_z1 = torch.linspace(-maximum_abs_z1, maximum_abs_z1, number)
    axis_points = torch.stack((axis_z1, torch.zeros_like(axis_z1)), dim=1)
    domain = torch.abs(axis_z1) > 0.5
    axis_points = axis_points[domain]
    field = shifted_field(axis_points, p, x_equilibrium)
    steps = (1e-4, 3e-5, 1e-5, 3e-6)
    with torch.no_grad():
        base_values = smooth_combined_candidate(
            homogeneous_network, inner_network, axis_points, kappa
        )
        quotients = []
        for step in steps:
            advanced_values = smooth_combined_candidate(
                homogeneous_network,
                inner_network,
                axis_points + step * field,
                kappa,
            )
            quotients.append((advanced_values - base_values) / step)
    maximum_by_step = [value.max().item() for value in quotients]
    return {
        "axis_points": len(axis_points),
        "positive_steps": list(steps),
        "maximum_forward_quotient_by_step": maximum_by_step,
        "maximum_forward_quotient_at_smallest_step": maximum_by_step[-1],
        "maximum_pointwise_change_between_two_smallest_steps": torch.max(
            torch.abs(quotients[-1] - quotients[-2])
        ).item(),
    }


def validate_and_plot(
    homogeneous_network: SphereNetwork,
    inner_network: OuterNetwork,
    config: RunConfig,
    p: SystemParameters,
    x_equilibrium: float,
    kappa: float,
    box: tuple[float, float],
    outdir: Path,
) -> dict[str, object]:
    level_points = level_angle_grid(
        homogeneous_network,
        maximum_level=2.0 * kappa,
        level_count=config.outer_validation_points_per_axis,
        angle_count=config.outer_validation_points_per_axis,
        level_offset=0.25,
        angle_offset=0.25,
    )
    level_outside_x = (torch.abs(level_points[:, 0]) > 0.5) | (
        torch.abs(level_points[:, 1]) > 0.5
    )
    level_points = level_points[level_outside_x]

    # A second coordinate system is used only for independent validation and
    # plotting. It is deliberately different from the training coordinates.
    grid1, grid2, rectangle = mesh_rectangle(
        -box[0], box[0], -box[1], box[1], config.outer_validation_points_per_axis, midpoint=True
    )
    with torch.no_grad():
        homogeneous_values_all = homogeneous_candidate(homogeneous_network, rectangle)
    outside_x = (torch.abs(rectangle[:, 0]) > 0.5) | (torch.abs(rectangle[:, 1]) > 0.5)
    domain = outside_x & (homogeneous_values_all <= 2.0 * kappa)
    cartesian_points = rectangle[domain]
    near_axis_points = near_velocity_axis_grid(
        homogeneous_network, maximum_level=2.0 * kappa
    )
    near_axis_outside_x = (torch.abs(near_axis_points[:, 0]) > 0.5) | (
        torch.abs(near_axis_points[:, 1]) > 0.5
    )
    near_axis_points = near_axis_points[near_axis_outside_x]
    points = torch.cat((level_points, cartesian_points, near_axis_points), dim=0)
    homogeneous_values, homogeneous_derivative = lie_derivative(
        lambda z: homogeneous_candidate(homogeneous_network, z),
        points,
        lambda z: shifted_field(z, p, x_equilibrium),
        create_graph=False,
    )
    inner_values, inner_derivative = lie_derivative(
        inner_network,
        points,
        lambda z: shifted_field(z, p, x_equilibrium),
        create_graph=False,
    )
    combined_values, combined_derivative = lie_derivative(
        lambda z: smooth_combined_candidate(
            homogeneous_network, inner_network, z, kappa
        ),
        points,
        lambda z: shifted_field(z, p, x_equilibrium),
        create_graph=False,
    )
    explicit_values, explicit_derivative = explicit_smooth_lie_derivative(
        homogeneous_network,
        inner_network,
        points,
        lambda z: shifted_field(z, p, x_equilibrium),
        kappa,
    )

    inner_boundary = level_boundary(
        homogeneous_network,
        kappa,
        config.homogeneous_validation_points,
        offset=0.25,
    )
    outer_boundary = level_boundary(
        homogeneous_network,
        2.0 * kappa,
        config.homogeneous_validation_points,
        offset=0.25,
    )
    with torch.no_grad():
        inner_boundary_u = inner_network(inner_boundary)
        inner_boundary_combined = smooth_combined_candidate(
            homogeneous_network, inner_network, inner_boundary, kappa
        )
        outer_boundary_v = homogeneous_candidate(homogeneous_network, outer_boundary)
        outer_boundary_combined = smooth_combined_candidate(
            homogeneous_network, inner_network, outer_boundary, kappa
        )

    transition = (homogeneous_values > kappa) & (
        homogeneous_values < 2.0 * kappa
    )
    axis_audit = exact_axis_forward_difference_audit(
        homogeneous_network,
        inner_network,
        p,
        x_equilibrium,
        kappa,
        maximum_abs_z1=config.radial_max,
    )
    level_stop = len(level_points)
    cartesian_stop = level_stop + len(cartesian_points)
    level_derivative = combined_derivative[:level_stop]
    cartesian_derivative = combined_derivative[level_stop:cartesian_stop]
    near_axis_derivative = combined_derivative[cartesian_stop:]

    metrics = {
        "validation_points_in_B2kappa_minus_X": len(points),
        "validation_level_angle_points": len(level_points),
        "validation_cartesian_points": len(cartesian_points),
        "validation_near_velocity_axis_points": len(near_axis_points),
        "validation_transition_points": int(transition.sum().item()),
        "minimum_inner_U": inner_values.min().item(),
        "minimum_smooth_V": combined_values.min().item(),
        "maximum_smooth_DV_f": combined_derivative.max().item(),
        "maximum_smooth_DV_f_on_level_angle_grid": level_derivative.max().item(),
        "maximum_smooth_DV_f_on_cartesian_grid": cartesian_derivative.max().item(),
        "maximum_smooth_DV_f_on_near_axis_grid": near_axis_derivative.max().item(),
        "maximum_smooth_decay_DV_f_plus_V": (
            combined_derivative + combined_values
        ).max().item(),
        "maximum_trained_decay_expression_with_margin": (
            combined_derivative + config.outer_decay_margin
        ).max().item(),
        "independent_hinge_loss_with_training_margin": torch.relu(
            combined_derivative + config.outer_decay_margin
        ).mean().item(),
        "independent_training_margin_violation_fraction": (
            combined_derivative > -config.outer_decay_margin
        ).double().mean().item(),
        "smooth_nonpositive_fraction": (
            combined_values <= 0.0
        ).double().mean().item(),
        "smooth_nonnegative_derivative_fraction": (
            combined_derivative >= 0.0
        ).double().mean().item(),
        "maximum_value_formula_disagreement": torch.max(
            torch.abs(combined_values - explicit_values)
        ).item(),
        "maximum_derivative_formula_disagreement": torch.max(
            torch.abs(combined_derivative - explicit_derivative)
        ).item(),
        "maximum_inner_boundary_identity_error": torch.max(
            torch.abs(inner_boundary_combined - inner_boundary_u)
        ).item(),
        "maximum_outer_boundary_identity_error": torch.max(
            torch.abs(outer_boundary_combined - outer_boundary_v)
        ).item(),
        "exact_velocity_axis_forward_difference": axis_audit,
    }

    shape = grid1.shape
    combined_image = np.full(rectangle.shape[0], np.nan)
    derivative_image = np.full(rectangle.shape[0], np.nan)
    cartesian_count = len(cartesian_points)
    cartesian_start = len(level_points)
    cartesian_stop = cartesian_start + cartesian_count
    combined_image[domain.numpy()] = (
        combined_values[cartesian_start:cartesian_stop].detach().numpy()
    )
    derivative_image[domain.numpy()] = (
        combined_derivative[cartesian_start:cartesian_stop].detach().numpy()
    )
    combined_image = combined_image.reshape(shape)
    derivative_image = derivative_image.reshape(shape)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    first = axes[0].pcolormesh(grid1, grid2, combined_image, shading="nearest")
    axes[0].set_title("smoothly united V on B(2 kappa) \\ X")
    axes[0].set_xlabel("z1 = x1 - x_eq")
    axes[0].set_ylabel("z2 = x2")
    figure.colorbar(first, ax=axes[0])
    second = axes[1].pcolormesh(grid1, grid2, derivative_image, shading="nearest")
    axes[1].set_title("directional derivative of united V")
    axes[1].set_xlabel("z1 = x1 - x_eq")
    axes[1].set_ylabel("z2 = x2")
    figure.colorbar(second, ax=axes[1])
    figure.savefig(outdir / "combined_validation.png", dpi=180)
    plt.close(figure)

    np.savez_compressed(
        outdir / "validation_arrays.npz",
        z=points.detach().numpy(),
        V_homogeneous=homogeneous_values.detach().numpy(),
        dV_homogeneous=homogeneous_derivative.detach().numpy(),
        U_inner=inner_values.detach().numpy(),
        dU_inner=inner_derivative.detach().numpy(),
        V_combined=combined_values.detach().numpy(),
        dV_combined=combined_derivative.detach().numpy(),
    )
    return metrics


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(config: RunConfig, outdir: Path, strict_audit: bool = True) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.use_deterministic_algorithms(True)
    outdir.mkdir(parents=True, exist_ok=True)
    p = SystemParameters()
    x_equilibrium = equilibrium_position(p)
    residual = p.a1 * (x_equilibrium - p.c1) - p.a3 * (x_equilibrium - p.c2) ** 3
    print(f"equilibrium x_eq={x_equilibrium:.15f}, residual={residual:.3e}")

    homogeneous_network, homogeneous_metrics = train_homogeneous(config, p)
    kappa, kappa_metrics = choose_empirical_kappa(
        homogeneous_network, config, p, x_equilibrium
    )
    print(f"empirical kappa={kappa:.12g}")
    inner_network, inner_training_metrics, box = train_inner_for_smooth_candidate(
        homogeneous_network, config, p, x_equilibrium, kappa
    )
    validation_metrics = validate_and_plot(
        homogeneous_network,
        inner_network,
        config,
        p,
        x_equilibrium,
        kappa,
        box,
        outdir,
    )

    configuration = {
        "system": asdict(p),
        "run": asdict(config),
        "parameter_sources": {
            "article": [
                "a1, a2, a3, c1, c2",
                "r=(1,2), nu=1, mu=2",
                "outer hidden width 32",
                "outer grid 100x100",
                "tanh activation",
            ],
            "repository_dev_w": [
                "phi(v)=Fc*tanh(v/vs)",
                "Fc=0.8, vs=0.5",
                "drag=sqrt(abs(v))*v",
            ],
            "implementation_choices": [
                "seed",
                "homogeneous hidden width and all optimizer settings",
                "training steps and validation densities",
                "kappa sampling range and padding",
                "inner optimizer settings, decay margin, and fixed quadratic coefficient",
                "quintic smooth transition between kappa and 2*kappa",
            ],
        },
    }
    audit = {
        "equilibrium_residual_below_1e-12": abs(residual) < 1e-12,
        "homogeneous_positive_on_validation_grid": (
            homogeneous_metrics["minimum_W_on_independent_sphere_grid"] >= 1.0
        ),
        "homogeneous_decay_on_validation_grid": (
            homogeneous_metrics["maximum_actual_decay_constraint_DV_finf_plus_1"] <= 0.0
        ),
        "full_candidate_decay_for_sampled_V_ge_kappa": (
            kappa_metrics["maximum_DV_f_on_sampled_V_ge_kappa"] < 0.0
        ),
        "inner_positive_on_validation_grid": (
            validation_metrics["minimum_inner_U"] > 0.0
        ),
        "smooth_candidate_positive_on_validation_grid": (
            validation_metrics["minimum_smooth_V"] > 0.0
        ),
        "smooth_candidate_decay_on_validation_grid": (
            validation_metrics["maximum_smooth_DV_f"] < 0.0
        ),
        "explicit_derivative_matches_autograd": (
            validation_metrics["maximum_derivative_formula_disagreement"] < 1e-10
        ),
        "inner_boundary_identity": (
            validation_metrics["maximum_inner_boundary_identity_error"] < 1e-12
        ),
        "outer_boundary_identity": (
            validation_metrics["maximum_outer_boundary_identity_error"] < 1e-12
        ),
        "negative_forward_quotient_on_exact_velocity_axis": (
            validation_metrics["exact_velocity_axis_forward_difference"]
            ["maximum_forward_quotient_at_smallest_step"]
            < 0.0
        ),
    }
    audit["all_finite_grid_checks_passed"] = all(audit.values())
    metrics = {
        "equilibrium": {
            "x1_original_coordinate": x_equilibrium,
            "x2": 0.0,
            "equilibrium_equation_residual": residual,
        },
        "homogeneous_sphere": homogeneous_metrics,
        "empirical_kappa": kappa_metrics,
        "inner_training_set": inner_training_metrics,
        "independent_validation": validation_metrics,
        "automated_audit": audit,
        "interpretation": (
            "All extrema and violation fractions are finite-grid observations, "
            "not continuous-domain certificates."
        ),
    }
    save_json(outdir / "config.json", configuration)
    save_json(outdir / "metrics.json", metrics)
    torch.save(
        {
            "homogeneous_network": homogeneous_network.state_dict(),
            "inner_network": inner_network.state_dict(),
            "config": configuration,
            "metrics": metrics,
        },
        outdir / "models.pt",
    )
    if strict_audit and not audit["all_finite_grid_checks_passed"]:
        failed = [name for name, passed in audit.items() if not passed]
        raise RuntimeError(f"Finite-grid audit failed: {failed}")
    return metrics


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("example_1/improved/results/reference"),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="short smoke run; its metrics are not reference results",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = RunConfig()
    if arguments.quick:
        config = RunConfig(
            homogeneous_train_points=256,
            homogeneous_validation_points=512,
            homogeneous_steps=50,
            outer_grid_points_per_axis=30,
            outer_validation_points_per_axis=40,
            outer_steps=50,
            radial_angles=64,
            radial_points=64,
            log_every=25,
        )
    run(config, arguments.outdir, strict_audit=not arguments.quick)


if __name__ == "__main__":
    main()
