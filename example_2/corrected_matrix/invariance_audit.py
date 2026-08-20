"""Finite-grid audit of the switching surface ``W(x)=kappa``.

The loss, networks, controller, and switching rule are not modified.  The
script extracts every level-set component detected on a rectangular grid,
refines its points using the network gradient, and evaluates ``DW F`` under
both the local and learned controls.

This is a numerical audit, not a continuous-domain invariance certificate.
Components smaller than the extraction-grid cells can remain undetected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import contourpy
import numpy as np
from scipy import ndimage
import torch

import example_2.article_version.example2 as article
from example_2.corrected_matrix.example2 import (
    consistent_corrected_local_design,
    corrected_pipeline,
)


torch.set_default_dtype(torch.float64)


def _network_value_and_gradient(
    lyapunov: article.LyapunovNetwork,
    points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    differentiable = points.detach().clone().requires_grad_(True)
    value = lyapunov(differentiable)
    gradient = torch.autograd.grad(value.sum(), differentiable)[0]
    return value.detach(), gradient.detach()


def _network_values_in_batches(
    lyapunov: article.LyapunovNetwork,
    points: torch.Tensor,
    batch_size: int = 65536,
) -> torch.Tensor:
    values = []
    with torch.no_grad():
        for start in range(0, len(points), batch_size):
            values.append(lyapunov(points[start : start + batch_size]))
    return torch.cat(values)


def refine_level_points(
    lyapunov: article.LyapunovNetwork,
    points: torch.Tensor,
    level: float,
    specification: article.ArticleSpecification,
    iterations: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project approximate contour points toward ``W=level`` by Newton steps."""

    refined = points.detach().clone()
    for _ in range(iterations):
        value, gradient = _network_value_and_gradient(lyapunov, refined)
        squared_norm = torch.sum(gradient * gradient, dim=1)
        safe = squared_norm > 1e-20
        correction = torch.zeros_like(refined)
        correction[safe] = (
            ((value[safe] - level) / squared_norm[safe]).unsqueeze(1)
            * gradient[safe]
        )
        refined = refined - correction
        refined[:, 0].clamp_(specification.angle_min, specification.angle_max)
        refined[:, 1].clamp_(
            specification.velocity_min, specification.velocity_max
        )
    value, gradient = _network_value_and_gradient(lyapunov, refined)
    return refined, torch.linalg.vector_norm(gradient, dim=1)


def extract_level_components(
    lyapunov: article.LyapunovNetwork,
    specification: article.ArticleSpecification,
    level: float,
    points_per_axis: int,
) -> tuple[list[torch.Tensor], np.ndarray, np.ndarray, np.ndarray]:
    """Extract all contour components detected by a marching-squares grid."""

    angle = np.linspace(
        specification.angle_min, specification.angle_max, points_per_axis
    )
    velocity = np.linspace(
        specification.velocity_min, specification.velocity_max, points_per_axis
    )
    angle_grid, velocity_grid = np.meshgrid(angle, velocity, indexing="ij")
    points = torch.from_numpy(
        np.stack((angle_grid.reshape(-1), velocity_grid.reshape(-1)), axis=1)
    )
    value = _network_values_in_batches(lyapunov, points).reshape(
        points_per_axis, points_per_axis
    ).numpy()

    # contourpy uses z[row_y, column_x], whereas the array above is indexed as
    # value[angle, velocity].
    generator = contourpy.contour_generator(x=angle, y=velocity, z=value.T)
    raw_components = generator.lines(level)
    components: list[torch.Tensor] = []
    for component in raw_components:
        approximate = torch.from_numpy(np.asarray(component, dtype=np.float64))
        refined, _ = refine_level_points(
            lyapunov, approximate, level, specification
        )
        components.append(refined)
    return components, angle, velocity, value


def _component_lengths(components: list[torch.Tensor]) -> list[float]:
    lengths: list[float] = []
    for component in components:
        if len(component) < 2:
            lengths.append(0.0)
            continue
        differences = component[1:] - component[:-1]
        lengths.append(torch.linalg.vector_norm(differences, dim=1).sum().item())
    return lengths


def _closed_component_flags(
    components: list[torch.Tensor],
    specification: article.ArticleSpecification,
    points_per_axis: int,
) -> list[bool]:
    angle_step = (
        specification.angle_max - specification.angle_min
    ) / (points_per_axis - 1)
    velocity_step = (
        specification.velocity_max - specification.velocity_min
    ) / (points_per_axis - 1)
    tolerance = 2.0 * max(angle_step, velocity_step)
    return [
        len(component) > 2
        and torch.linalg.vector_norm(component[0] - component[-1]).item()
        <= tolerance
        for component in components
    ]


def audit_switching_surface(
    lyapunov: article.LyapunovNetwork,
    controller: article.ControllerNetwork,
    specification: article.ArticleSpecification,
    config: article.ImplementationConfig,
    contour_points_per_axis: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Evaluate topology and directional derivatives on the detected contour."""

    components, angle, velocity, value_grid = extract_level_components(
        lyapunov,
        specification,
        config.kappa,
        contour_points_per_axis,
    )
    if not components:
        raise RuntimeError("No W=kappa component was detected on the audit grid")

    component_ids = torch.cat(
        [torch.full((len(component),), index, dtype=torch.int64)
         for index, component in enumerate(components)]
    )
    contour_points = torch.cat(components, dim=0)
    contour_value, gradient = _network_value_and_gradient(lyapunov, contour_points)
    gradient_norm = torch.linalg.vector_norm(gradient, dim=1)
    k, p = consistent_corrected_local_design()
    local_value = article.local_quadratic_value(contour_points, p)
    local_control = article.local_linear_control(contour_points, k)
    learned_control = controller(contour_points).detach()
    local_field = article.nonlinear_field(
        contour_points, local_control, specification.omega
    )
    learned_field = article.nonlinear_field(
        contour_points, learned_control, specification.omega
    )
    derivative_local = torch.sum(gradient * local_field, dim=1)
    derivative_learned = torch.sum(gradient * learned_field, dim=1)

    inside_mask = value_grid < config.kappa
    labels, component_count_inside = ndimage.label(
        inside_mask, structure=np.ones((3, 3), dtype=np.int8)
    )
    inside_sizes = [
        int(np.sum(labels == label)) for label in range(1, component_count_inside + 1)
    ]
    inside_touches_boundary = [
        bool(
            np.any(labels[0, :] == label)
            or np.any(labels[-1, :] == label)
            or np.any(labels[:, 0] == label)
            or np.any(labels[:, -1] == label)
        )
        for label in range(1, component_count_inside + 1)
    ]
    origin_angle_index = int(np.argmin(np.abs(angle)))
    origin_velocity_index = int(np.argmin(np.abs(velocity)))
    origin_inside_component = int(
        labels[origin_angle_index, origin_velocity_index]
    )

    edge_velocity = torch.from_numpy(velocity)
    left_edge = torch.stack(
        (torch.full_like(edge_velocity, specification.angle_min), edge_velocity), dim=1
    )
    right_edge = torch.stack(
        (torch.full_like(edge_velocity, specification.angle_max), edge_velocity), dim=1
    )
    with torch.no_grad():
        left_w = lyapunov(left_edge)
        right_w = lyapunov(right_edge)
        left_u = controller(left_edge).squeeze(-1)
        right_u = controller(right_edge).squeeze(-1)

    closed_flags = _closed_component_flags(
        components, specification, contour_points_per_axis
    )
    component_metrics: list[dict[str, object]] = []
    for index, component in enumerate(components):
        selected = component_ids == index
        selected_points = contour_points[selected]
        selected_local_value = local_value[selected]
        selected_local_derivative = derivative_local[selected]
        selected_learned_derivative = derivative_learned[selected]
        component_metrics.append(
            {
                "component_id": index,
                "point_count": len(component),
                "closed_on_extraction_grid": closed_flags[index],
                "length": _component_lengths([component])[0],
                "angle_min": selected_points[:, 0].min().item(),
                "angle_max": selected_points[:, 0].max().item(),
                "velocity_min": selected_points[:, 1].min().item(),
                "velocity_max": selected_points[:, 1].max().item(),
                "minimum_V_local": selected_local_value.min().item(),
                "maximum_V_local": selected_local_value.max().item(),
                "maximum_DW_F_local": selected_local_derivative.max().item(),
                "nonnegative_DW_F_local_fraction": (
                    selected_local_derivative >= 0.0
                ).double().mean().item(),
                "maximum_DW_F_learned": selected_learned_derivative.max().item(),
                "nonnegative_DW_F_learned_fraction": (
                    selected_learned_derivative >= 0.0
                ).double().mean().item(),
            }
        )
    angle_cell = (specification.angle_max - specification.angle_min) / (
        contour_points_per_axis - 1
    )
    velocity_cell = (specification.velocity_max - specification.velocity_min) / (
        contour_points_per_axis - 1
    )
    metrics: dict[str, object] = {
        "interpretation": (
            "finite-grid contour audit; not a continuous-domain invariance certificate"
        ),
        "contour_grid_points_per_axis": contour_points_per_axis,
        "detected_level_components": len(components),
        "detected_closed_level_components": int(sum(closed_flags)),
        "detected_open_or_boundary_touching_level_components": int(
            len(closed_flags) - sum(closed_flags)
        ),
        "level_component_point_counts": [len(component) for component in components],
        "level_component_lengths": _component_lengths(components),
        "level_components": component_metrics,
        "maximum_abs_W_minus_kappa_after_refinement": (
            contour_value - config.kappa
        ).abs().max().item(),
        "minimum_gradient_norm_on_detected_level": gradient_norm.min().item(),
        "minimum_V_local_on_detected_level": local_value.min().item(),
        "maximum_V_local_on_detected_level": local_value.max().item(),
        "detected_level_points_outside_B_kappa": int(
            (local_value > config.kappa).sum().item()
        ),
        "minimum_DW_F_local_on_detected_level": derivative_local.min().item(),
        "maximum_DW_F_local_on_detected_level": derivative_local.max().item(),
        "positive_DW_F_local_points_on_detected_level": int(
            (derivative_local > 0.0).sum().item()
        ),
        "nonnegative_DW_F_local_fraction_on_detected_level": (
            derivative_local >= 0.0
        ).double().mean().item(),
        "minimum_DW_F_learned_on_detected_level": derivative_learned.min().item(),
        "maximum_DW_F_learned_on_detected_level": derivative_learned.max().item(),
        "nonnegative_DW_F_learned_points_on_detected_level": int(
            (derivative_learned >= 0.0).sum().item()
        ),
        "nonnegative_DW_F_learned_fraction_on_detected_level": (
            derivative_learned >= 0.0
        ).double().mean().item(),
        "detected_components_of_W_below_kappa": int(component_count_inside),
        "W_below_kappa_component_grid_sizes": inside_sizes,
        "W_below_kappa_component_approximate_areas": [
            size * angle_cell * velocity_cell for size in inside_sizes
        ],
        "W_below_kappa_components_touching_X_boundary": inside_touches_boundary,
        "origin_W_below_kappa_component_label": origin_inside_component,
        "maximum_periodic_edge_W_mismatch": (left_w - right_w).abs().max().item(),
        "maximum_periodic_edge_learned_control_mismatch": (
            left_u - right_u
        ).abs().max().item(),
    }
    arrays = {
        "x": contour_points.numpy(),
        "component_id": component_ids.numpy(),
        "W": contour_value.numpy(),
        "gradient_W": gradient.numpy(),
        "V_local": local_value.numpy(),
        "u_local": local_control.squeeze(-1).numpy(),
        "u_learned": learned_control.squeeze(-1).numpy(),
        "dW_local": derivative_local.numpy(),
        "dW_learned": derivative_learned.numpy(),
    }
    return metrics, arrays


def save_surface_figure(
    lyapunov: article.LyapunovNetwork,
    specification: article.ArticleSpecification,
    config: article.ImplementationConfig,
    arrays: dict[str, np.ndarray],
    outdir: Path,
) -> None:
    """Save a four-panel geometry and directional-derivative audit figure."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plot_count = 401
    angle = np.linspace(specification.angle_min, specification.angle_max, plot_count)
    velocity = np.linspace(
        specification.velocity_min, specification.velocity_max, plot_count
    )
    angle_grid, velocity_grid = np.meshgrid(angle, velocity, indexing="ij")
    plot_points = torch.from_numpy(
        np.stack((angle_grid.reshape(-1), velocity_grid.reshape(-1)), axis=1)
    )
    value = _network_values_in_batches(lyapunov, plot_points).reshape(
        plot_count, plot_count
    ).numpy()
    _, p = consistent_corrected_local_design()
    local_boundary = article.local_quadratic_value(
        plot_points, p
    ).reshape(plot_count, plot_count).numpy()

    contour_points = arrays["x"]
    derivative_local = arrays["dW_local"]
    derivative_learned = arrays["dW_learned"]

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    figure, axes = plt.subplots(
        2, 2, figsize=(12.0, 8.0), constrained_layout=True
    )

    def draw_geometry(axis, *, zoom: bool) -> None:
        axis.contourf(
            angle_grid,
            velocity_grid,
            value,
            levels=[value.min() - 1.0, config.kappa],
            colors=["#DCEEF8"],
            alpha=0.95,
        )
        axis.contour(
            angle_grid,
            velocity_grid,
            value,
            levels=[config.kappa],
            colors=["#263238"],
            linewidths=1.8,
        )
        axis.contour(
            angle_grid,
            velocity_grid,
            local_boundary,
            levels=[config.kappa],
            colors=["#008C95"],
            linewidths=1.7,
            linestyles="--",
        )
        axis.scatter([0.0], [0.0], s=22, color="#111111", zorder=5)
        if zoom:
            axis.set_xlim(-0.55, 0.55)
            axis.set_ylim(-0.4, 0.4)
        else:
            axis.set_xlim(specification.angle_min, specification.angle_max)
            axis.set_ylim(specification.velocity_min, specification.velocity_max)
        axis.set_xlabel(r"angle $\theta$")
        axis.set_ylabel(r"angular velocity $\dot{\theta}$")
        axis.grid(color="#D8DEE4", linewidth=0.5, alpha=0.55)

    draw_geometry(axes[0, 0], zoom=False)
    axes[0, 0].set_title(r"Detected geometry of $\{W<\kappa\}$ on $X$")
    draw_geometry(axes[0, 1], zoom=True)
    axes[0, 1].set_title("Central component and local certified set")

    def draw_derivative(axis, derivative: np.ndarray, title: str) -> None:
        negative = derivative < 0.0
        nonnegative = ~negative
        axis.scatter(
            contour_points[negative, 0],
            contour_points[negative, 1],
            s=5,
            color="#2B7A9B",
            label="negative",
            rasterized=True,
        )
        axis.scatter(
            contour_points[nonnegative, 0],
            contour_points[nonnegative, 1],
            s=8,
            color="#C44536",
            label="nonnegative",
            rasterized=True,
        )
        axis.set_xlim(specification.angle_min, specification.angle_max)
        axis.set_ylim(specification.velocity_min, specification.velocity_max)
        axis.set_xlabel(r"angle $\theta$")
        axis.set_ylabel(r"angular velocity $\dot{\theta}$")
        axis.set_title(title)
        axis.grid(color="#D8DEE4", linewidth=0.5, alpha=0.55)

    draw_derivative(
        axes[1, 0],
        derivative_local,
        r"Sign of $DW\,F(x,Kx)$ on detected $W=\kappa$",
    )
    draw_derivative(
        axes[1, 1],
        derivative_learned,
        r"Sign of $DW\,F(x,N(x))$ on detected $W=\kappa$",
    )

    geometry_handles = [
        Line2D([0], [0], color="#263238", lw=1.8, label=r"$W=\kappa$"),
        Line2D(
            [0], [0], color="#008C95", lw=1.7, ls="--", label=r"$V_l=\kappa$"
        ),
        Line2D([0], [0], color="#DCEEF8", lw=7, label=r"$W<\kappa$"),
    ]
    axes[0, 0].legend(handles=geometry_handles, loc="upper center", ncols=3)
    axes[0, 1].legend(handles=geometry_handles, loc="upper right")
    sign_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2B7A9B", label="negative"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#C44536", label="nonnegative"),
    ]
    axes[1, 0].legend(handles=sign_handles, loc="upper center", ncols=2)
    axes[1, 1].legend(handles=sign_handles, loc="upper center", ncols=2)

    figure.suptitle(
        "Finite-grid audit of the switching surface (corrected local design)",
        fontsize=13,
    )
    figure.savefig(outdir / "switching_surface_audit.png", dpi=220)
    figure.savefig(outdir / "switching_surface_audit.svg")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(
            "example_2/corrected_matrix/results/reference/invariance_audit"
        ),
    )
    parser.add_argument("--contour-grid", type=int, default=601)
    parser.add_argument("--model-state", type=Path)
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    specification = article.ArticleSpecification()
    config = article.ImplementationConfig()
    if arguments.quick and arguments.model_state is not None:
        parser.error("--quick and --model-state cannot be used together")
    if arguments.quick:
        specification = article.ArticleSpecification(grid_points_per_axis=30)
        config = article.ImplementationConfig(
            training_steps=20,
            validation_points_per_axis=41,
            plot_points_per_axis=81,
            log_every=10,
        )

    if arguments.model_state is None:
        with corrected_pipeline(consistent_corrected_local_design):
            lyapunov, controller, training = article.train_networks(
                specification, config
            )
    else:
        checkpoint = torch.load(arguments.model_state, weights_only=True)
        specification = article.ArticleSpecification(
            **checkpoint["article_specification"]
        )
        config = article.ImplementationConfig(
            **checkpoint["implementation_config"]
        )
        lyapunov = article.LyapunovNetwork(specification.lyapunov_hidden_units)
        controller = article.ControllerNetwork(specification.controller_hidden_units)
        lyapunov.load_state_dict(checkpoint["lyapunov_state_dict"])
        controller.load_state_dict(checkpoint["controller_state_dict"])
        training = {
            "model_source": str(arguments.model_state),
            "training_not_repeated": True,
        }

    metrics, arrays = audit_switching_surface(
        lyapunov,
        controller,
        specification,
        config,
        arguments.contour_grid,
    )
    arguments.outdir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "lyapunov_state_dict": lyapunov.state_dict(),
            "controller_state_dict": controller.state_dict(),
            "article_specification": specification.__dict__,
            "implementation_config": config.__dict__,
            "local_design": {
                "K": consistent_corrected_local_design()[0].tolist(),
                "P": consistent_corrected_local_design()[1].tolist(),
            },
        },
        arguments.outdir / "model_state.pt",
    )
    np.savez_compressed(arguments.outdir / "contour_arrays.npz", **arrays)
    save_surface_figure(
        lyapunov, specification, config, arrays, arguments.outdir
    )
    result = {"training": training, "surface_audit": metrics}
    (arguments.outdir / "run_record.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
