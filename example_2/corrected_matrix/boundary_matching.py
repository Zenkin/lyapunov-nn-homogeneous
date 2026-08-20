"""Add only a sampled boundary-matching term to the corrected experiment.

The added term is

    mean(abs(W(x)-kappa)) for sampled x with V_l(x)=kappa.

Its weight is an explicit experimental parameter; it is not reported in the
article.  Network architecture and all original pointwise-loss terms remain
unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import example_2.article_version.example2 as article
from example_2.corrected_matrix.example2 import (
    consistent_corrected_local_design,
    corrected_pipeline,
)
from example_2.corrected_matrix.switching_audit import (
    audit_trained_switches,
    local_level_boundary,
)


torch.set_default_dtype(torch.float64)


def boundary_matching_loss(
    lyapunov: article.LyapunovNetwork,
    boundary_points: torch.Tensor,
    kappa: float,
) -> torch.Tensor:
    """Return the sampled mean absolute mismatch on ``V_l=kappa``."""

    return torch.mean(torch.abs(lyapunov(boundary_points) - kappa))


def train_with_boundary_matching(
    specification: article.ArticleSpecification,
    config: article.ImplementationConfig,
    boundary_weight: float,
    boundary_points_count: int,
) -> tuple[
    article.LyapunovNetwork,
    article.ControllerNetwork,
    dict[str, float | int | str],
]:
    """Train the unchanged networks with one additional boundary term."""

    if boundary_weight < 0.0:
        raise ValueError("boundary_weight must be nonnegative")
    if boundary_points_count < 4:
        raise ValueError("boundary_points_count must be at least four")

    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    _, p = consistent_corrected_local_design()
    full_grid = article.uniform_rectangle_grid(specification, include_boundary=False)
    training_points = article.article_training_domain(
        full_grid, p, config.kappa
    )
    boundary_points = local_level_boundary(
        p, config.kappa, count=boundary_points_count
    )

    lyapunov = article.LyapunovNetwork(specification.lyapunov_hidden_units)
    controller = article.ControllerNetwork(specification.controller_hidden_units)
    optimizer = torch.optim.Adam(
        [*lyapunov.parameters(), *controller.parameters()],
        lr=config.learning_rate,
    )

    for step in range(1, config.training_steps + 1):
        pointwise_loss = article.article_pointwise_loss(
            lyapunov,
            controller,
            training_points,
            specification.omega,
            config.epsilon,
        )
        original_mean_loss = pointwise_loss.mean()
        matching_loss = boundary_matching_loss(
            lyapunov, boundary_points, config.kappa
        )
        objective = original_mean_loss + boundary_weight * matching_loss
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()

        if step == 1 or step % config.log_every == 0 or step == config.training_steps:
            print(
                f"[weight={boundary_weight:g} step={step:5d}] "
                f"original={original_mean_loss.detach().item():.6e} "
                f"boundary={matching_loss.detach().item():.6e} "
                f"objective={objective.detach().item():.6e}"
            )

    # Recompute after the last Adam update.  The article-version record is
    # evaluated before that update; this experiment labels the difference.
    final_pointwise = article.article_pointwise_loss(
        lyapunov,
        controller,
        training_points,
        specification.omega,
        config.epsilon,
    ).detach()
    boundary_residual = (
        lyapunov(boundary_points).detach() - config.kappa
    ).abs()
    metrics: dict[str, float | int | str] = {
        "boundary_weight": boundary_weight,
        "boundary_points": boundary_points_count,
        "boundary_sampling": "uniform parameter angle on the exact V_l=kappa ellipse",
        "training_grid_points": len(training_points),
        "completed_training_steps": config.training_steps,
        "final_metric_timing": "after the last Adam update",
        "final_original_mean_pointwise_loss": final_pointwise.mean().item(),
        "final_original_maximum_pointwise_loss": final_pointwise.max().item(),
        "final_boundary_mean_absolute_error": boundary_residual.mean().item(),
        "final_boundary_maximum_absolute_error": boundary_residual.max().item(),
    }
    return lyapunov, controller, metrics


def run(
    boundary_weight: float,
    boundary_points_count: int,
    specification: article.ArticleSpecification,
    config: article.ImplementationConfig,
) -> dict[str, object]:
    with corrected_pipeline(consistent_corrected_local_design):
        lyapunov, controller, training = train_with_boundary_matching(
            specification,
            config,
            boundary_weight,
            boundary_points_count,
        )
    audit = audit_trained_switches(lyapunov, controller, specification, config)
    return {
        "change_from_corrected_matrix_experiment": (
            "sampled boundary-matching term only"
        ),
        "training": training,
        "switching_audit": audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--boundary-points", type=int, default=512)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", type=Path)
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

    result = run(
        arguments.weight,
        arguments.boundary_points,
        specification,
        config,
    )
    output = arguments.out or Path(
        "example_2/corrected_matrix/results/reference/"
        f"boundary_matching_weight_{arguments.weight:g}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
