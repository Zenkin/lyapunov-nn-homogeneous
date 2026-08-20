"""Repeat Example 2 while changing only the local linear model.

Two experiments are intentionally separated:

``matrix_only``
    Correct the displayed matrix A but freeze every reported numerical choice,
    including the previously selected numerical K and P.  Since A is not used
    by the neural training pipeline, this run must equal the article-version
    run.

``consistent_local_design``
    Correct A, keep K=(-2,-3), and recompute P from the same Lyapunov equation
    with right-hand side -I.  All neural, optimizer, grid, seed, kappa, and
    epsilon choices remain unchanged.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Iterator

import torch

import example_2.article_version.example2 as article


torch.set_default_dtype(torch.float64)


def corrected_linear_matrices(
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the Jacobian of the printed field for the printed omega=1."""

    selected_dtype = dtype or torch.get_default_dtype()
    a = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=selected_dtype, device=device)
    b = torch.tensor([[0.0], [1.0]], dtype=selected_dtype, device=device)
    return a, b


def frozen_article_local_design(
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact K and P already used by ``article_version``."""

    selected_dtype = dtype or torch.get_default_dtype()
    k = torch.tensor([-2.0, -3.0], dtype=selected_dtype, device=device)
    p = torch.tensor(
        [[5.0 / 4.0, 1.0 / 4.0], [1.0 / 4.0, 1.0 / 4.0]],
        dtype=selected_dtype,
        device=device,
    )
    return k, p


def consistent_corrected_local_design(
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep K and solve the Lyapunov equation for corrected A.

    For K=(-2,-3), the exact result is

    P = [[11/6, 1/2], [1/2, 1/3]],

    and ``(A+BK)^T P + P(A+BK) = -I``.
    """

    selected_dtype = dtype or torch.get_default_dtype()
    k = torch.tensor([-2.0, -3.0], dtype=selected_dtype, device=device)
    p = torch.tensor(
        [[11.0 / 6.0, 1.0 / 2.0], [1.0 / 2.0, 1.0 / 3.0]],
        dtype=selected_dtype,
        device=device,
    )
    return k, p


def corrected_lyapunov_matrix(k: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Return the local Lyapunov matrix using corrected A."""

    a, b = corrected_linear_matrices(dtype=p.dtype, device=p.device)
    closed_loop = a + b @ k.reshape(1, 2).to(dtype=p.dtype, device=p.device)
    return closed_loop.T @ p + p @ closed_loop


@contextmanager
def corrected_pipeline(
    local_design,
) -> Iterator[None]:
    """Temporarily inject only the explicitly selected local formulas.

    The article-version training and plotting code is reused verbatim.  The
    original module globals are restored even if the experiment fails.
    """

    original_matrices = article.article_linear_matrices
    original_design = article.implementation_local_design
    article.article_linear_matrices = corrected_linear_matrices
    article.implementation_local_design = local_design
    try:
        yield
    finally:
        article.article_linear_matrices = original_matrices
        article.implementation_local_design = original_design


def run_one(
    name: str,
    local_design,
    specification: article.ArticleSpecification,
    config: article.ImplementationConfig,
    outdir: Path,
) -> dict[str, object]:
    """Run the unchanged article pipeline with one selected local design."""

    selected_outdir = outdir / name
    with corrected_pipeline(local_design):
        result = article.run(specification, config, selected_outdir)

    k, p = local_design()
    result["local_design"] = {
        "K": k.tolist(),
        "P": p.tolist(),
        "origin": (
            "frozen article-version numerical choice"
            if name == "matrix_only"
            else "recomputed for the corrected Jacobian"
        ),
    }
    result["controlled_experiment"] = {
        "name": name,
        "corrected_A": [[0.0, 1.0], [1.0, 0.0]],
        "K": k.tolist(),
        "P": p.tolist(),
        "lyapunov_matrix_for_corrected_A": corrected_lyapunov_matrix(k, p).tolist(),
        "unchanged_items": [
            "nonlinear field",
            "network architectures and widths",
            "pointwise loss and arithmetic-mean reduction",
            "kappa and epsilon",
            "training grid convention",
            "Adam learning rate and number of steps",
            "random seed",
            "W-based switching rule",
        ],
    }
    (selected_outdir / "run_record.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("example_2/corrected_matrix/results/reference"),
    )
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()

    # This workload is faster and remains deterministic with one CPU thread.
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

    results = {
        "matrix_only": run_one(
            "matrix_only",
            frozen_article_local_design,
            specification,
            config,
            arguments.outdir,
        ),
        "consistent_local_design": run_one(
            "consistent_local_design",
            consistent_corrected_local_design,
            specification,
            config,
            arguments.outdir,
        ),
    }
    comparison = {
        name: value["validation"] for name, value in results.items()
    }
    (arguments.outdir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
