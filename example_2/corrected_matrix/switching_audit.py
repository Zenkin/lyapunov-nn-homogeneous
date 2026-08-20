"""Audit the domains used by the two controllers in Example 2.

This module does not change the neural architecture or the loss.  It compares
the switching rule printed in the article with the direct local-level rule

    use Kx if V_l(x) <= kappa, otherwise use N(x).

The latter aligns the controller domains, but it is not by itself a proof that
``V_l`` and ``W`` form a continuous composite Lyapunov function.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import torch

import example_2.article_version.example2 as article
from example_2.corrected_matrix.example2 import (
    consistent_corrected_local_design,
    corrected_pipeline,
)


torch.set_default_dtype(torch.float64)


def local_level_switched_control(
    x: torch.Tensor,
    learned_controller: Callable[[torch.Tensor], torch.Tensor],
    k: torch.Tensor,
    p: torch.Tensor,
    kappa: float,
) -> torch.Tensor:
    """Use the local controller exactly on ``B_kappa={V_l<=kappa}``."""

    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    local = article.local_linear_control(x, k)
    learned = learned_controller(x)
    if learned.ndim == x.ndim - 1:
        learned = learned.unsqueeze(-1)
    use_local = article.local_quadratic_value(x, p) <= kappa
    return torch.where(use_local.unsqueeze(-1), local, learned)


def _value_and_derivative(
    value_function: Callable[[torch.Tensor], torch.Tensor],
    control: Callable[[torch.Tensor], torch.Tensor],
    points: torch.Tensor,
    omega: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    differentiable_points = points.detach().clone().requires_grad_(True)
    value = value_function(differentiable_points)
    gradient = torch.autograd.grad(value.sum(), differentiable_points)[0]
    field = article.nonlinear_field(
        differentiable_points, control(differentiable_points), omega
    )
    derivative = torch.sum(gradient * field, dim=-1)
    return value.detach(), derivative.detach()


def local_level_boundary(
    p: torch.Tensor,
    kappa: float,
    count: int = 2048,
    phase: float = 0.0,
) -> torch.Tensor:
    """Parameterize the exact ellipse ``x^T P x=kappa``.

    ``phase`` is measured in fractions of one angular grid step.  It permits a
    boundary audit that is shifted away from the boundary-training points.
    """

    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    eigenvalues, eigenvectors = torch.linalg.eigh(p)
    if torch.any(eigenvalues <= 0.0):
        raise ValueError("P must be positive definite")
    inverse_square_root = (
        eigenvectors
        @ torch.diag(torch.rsqrt(eigenvalues))
        @ eigenvectors.T
    )
    angles = (
        torch.arange(count, dtype=p.dtype, device=p.device) + phase
    ) * (2.0 * math.pi / count)
    unit_circle = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    return math.sqrt(kappa) * unit_circle @ inverse_square_root


def audit_trained_switches(
    lyapunov: article.LyapunovNetwork,
    controller: article.ControllerNetwork,
    specification: article.ArticleSpecification,
    config: article.ImplementationConfig,
) -> dict[str, float | int | str]:
    """Compare domain alignment and within-branch derivatives on one grid."""

    k, p = consistent_corrected_local_design()
    validation_specification = article.ArticleSpecification(
        grid_points_per_axis=config.validation_points_per_axis
    )
    points = article.uniform_rectangle_grid(
        validation_specification, include_boundary=True
    )
    local_value, local_derivative = _value_and_derivative(
        lambda x: article.local_quadratic_value(x, p),
        lambda x: article.local_linear_control(x, k),
        points,
        specification.omega,
    )
    learned_value, learned_derivative = _value_and_derivative(
        lyapunov, controller, points, specification.omega
    )

    inner = local_value <= 0.5 * config.kappa
    local_domain = local_value <= config.kappa
    article_local = learned_value < config.kappa
    aligned_local = local_domain
    nonzero = torch.sum(points * points, dim=1) > 1e-14

    # This is a branchwise diagnostic only.  A discontinuity at the switching
    # boundary would require a separate composite-Lyapunov construction.
    active_branch_derivative = torch.where(
        aligned_local, local_derivative, learned_derivative
    )

    outside_local_domain = ~local_domain
    sampled_safe_w_threshold = learned_value[outside_local_domain].min()
    boundary = local_level_boundary(p, config.kappa, phase=0.5)
    boundary_w = lyapunov(boundary).detach()

    return {
        "validation_grid_points": len(points),
        "article_switch_local_points": int(article_local.sum().item()),
        "article_switch_neural_points_inside_B_kappa_over_2": int(
            (inner & ~article_local).sum().item()
        ),
        "article_switch_local_points_outside_B_kappa": int(
            (article_local & outside_local_domain).sum().item()
        ),
        "aligned_switch_local_points": int(aligned_local.sum().item()),
        "aligned_switch_neural_points_inside_B_kappa_over_2": int(
            (inner & ~aligned_local).sum().item()
        ),
        "aligned_switch_local_points_outside_B_kappa": int(
            (aligned_local & outside_local_domain).sum().item()
        ),
        "maximum_local_derivative_on_B_kappa": local_derivative[
            local_domain & nonzero
        ].max().item(),
        "maximum_learned_derivative_outside_B_kappa": learned_derivative[
            outside_local_domain
        ].max().item(),
        "maximum_active_branch_derivative": active_branch_derivative[
            nonzero
        ].max().item(),
        "nonnegative_active_branch_derivative_fraction": (
            active_branch_derivative[nonzero] >= 0.0
        ).double().mean().item(),
        "sampled_W_threshold_with_strict_local_set_contained_in_B_kappa": (
            sampled_safe_w_threshold.item()
        ),
        "minimum_W_on_V_local_equals_kappa": boundary_w.min().item(),
        "maximum_W_on_V_local_equals_kappa": boundary_w.max().item(),
        "maximum_abs_W_minus_kappa_on_V_local_equals_kappa": (
            boundary_w - config.kappa
        ).abs().max().item(),
        "boundary_audit_points": len(boundary),
        "boundary_audit_sampling": (
            "uniform parameter angle, shifted by half an audit-grid step"
        ),
        "interpretation": (
            "finite-grid branch-domain audit; not a composite Lyapunov certificate"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "example_2/corrected_matrix/results/reference/switching_audit.json"
        ),
    )
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    specification = article.ArticleSpecification()
    config = article.ImplementationConfig()
    if arguments.quick:
        specification = article.ArticleSpecification(grid_points_per_axis=30)
        config = article.ImplementationConfig(
            training_steps=20,
            validation_points_per_axis=41,
            plot_points_per_axis=81,
            log_every=10,
        )

    with corrected_pipeline(consistent_corrected_local_design):
        lyapunov, controller, _ = article.train_networks(specification, config)
    result = audit_trained_switches(lyapunov, controller, specification, config)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
