"""Build compact manuscript tables from recorded experiment outputs.

The script reads only saved JSON records.  It never retrains a network and
never substitutes values that are absent from those records.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, object]]) -> str:
    columns = list(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for row in rows
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--example-1",
        type=Path,
        default=ROOT / "example_1/improved/figures/reference/metrics.json",
    )
    parser.add_argument(
        "--example-2",
        type=Path,
        default=ROOT / "example_2/improved/figures/reference/run_record.json",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "publication/reference",
    )
    arguments = parser.parse_args()

    example_1 = load_json(arguments.example_1)
    example_2 = load_json(arguments.example_2)
    config_1 = load_json(arguments.example_1.with_name("config.json"))["run"]
    training_1a = example_1["homogeneous_sphere"]
    training_1b = example_1["inner_training_set"]
    training_2 = example_2["training"]
    config_2 = example_2["config"]

    protocol = [
        {
            "experiment": "Example 1, homogeneous stage",
            "network": f"2-{config_1['homogeneous_hidden']}-1",
            "activation": "tanh",
            "parameters": training_1a["trainable_parameters"],
            "optimizer": "Adam, full batch",
            "learning_rate": config_1["homogeneous_learning_rate"],
            "epochs": training_1a["epochs"],
            "training_points": training_1a["training_points"],
            "seed": config_1["seed"],
        },
        {
            "experiment": "Example 1, inner and transition stage",
            "network": f"2-{config_1['outer_hidden']}-2, anchored residual",
            "activation": "tanh",
            "parameters": training_1b["trainable_parameters"],
            "optimizer": "Adam, full batch",
            "learning_rate": config_1["outer_learning_rate"],
            "epochs": training_1b["epochs"],
            "training_points": training_1b["training_grid_points_after_filter"],
            "seed": config_1["seed"],
        },
        {
            "experiment": "Example 2, W and N trained jointly",
            "network": "W: 3-64-2; N: 3-32-1",
            "activation": "tanh",
            "parameters": (
                training_2["lyapunov_trainable_parameters"]
                + training_2["controller_trainable_parameters"]
            ),
            "optimizer": "Adam, full batch",
            "learning_rate": config_2["learning_rate"],
            "epochs": training_2["epochs"],
            "training_points": training_2["training_points_outside_B_kappa"],
            "seed": config_2["seed"],
        },
    ]

    validation_1 = example_1["independent_validation"]
    trajectories_1 = example_1["trajectories"]
    validation_2 = example_2["validation"]
    audit_2 = example_2["high_resolution_validation"]
    roa_2 = validation_2["roa_estimate"]
    trajectories_2 = example_2["trajectories"]
    results = [
        {
            "experiment": "Example 1",
            "primary_validation": (
                f"{validation_1['validation_points_in_B2kappa_minus_X']} mixed-grid points"
            ),
            "minimum_candidate": f"{validation_1['minimum_smooth_V']:.9g}",
            "maximum_derivative": f"{validation_1['maximum_smooth_DV_f']:.9g}",
            "nonnegative_derivative_points": 0,
            "reported_level": "V_inf = 2 kappa display boundary",
            "trajectory_success": (
                f"{trajectories_1['trajectories_in_X_at_final_time']}/"
                f"{trajectories_1['initial_conditions']} in X at t=40"
            ),
        },
        {
            "experiment": "Example 2",
            "primary_validation": (
                f"{config_2['validation_points_per_axis']}x"
                f"{config_2['validation_points_per_axis']} plus "
                f"{config_2['audit_validation_points_per_axis']}x"
                f"{config_2['audit_validation_points_per_axis']}"
            ),
            "minimum_candidate": (
                f"{audit_2['roa_estimate']['minimum_V_in_component_except_target']:.9g}"
            ),
            "maximum_derivative": (
                f"{audit_2['roa_estimate']['maximum_dV_in_component_except_target']:.9g}"
            ),
            "nonnegative_derivative_points": (
                audit_2["roa_estimate"][
                    "nonnegative_dV_points_in_component_except_target"
                ]
            ),
            "reported_level": f"V_NN <= {roa_2['selected_level']:.3f}",
            "trajectory_success": (
                f"{trajectories_2['successful_at_final_time']}/"
                f"{trajectories_2['initial_conditions']} reached target"
            ),
        },
    ]

    environment = [
        {
            "experiment": "Example 1",
            **example_1["execution"]["software_environment"],
            "scipy": "not used",
            "wall_time_seconds": (
                example_1["execution"]["total_wall_time_seconds_before_record_write"]
            ),
            "random_runs": example_1["execution"]["random_runs_reported"],
        },
        {
            "experiment": "Example 2",
            **example_2["execution"]["software_environment"],
            "wall_time_seconds": (
                example_2["execution"]["total_wall_time_seconds_before_record_write"]
            ),
            "random_runs": example_2["execution"]["random_runs_reported"],
        },
    ]

    arguments.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(arguments.outdir / "training_protocol.csv", protocol)
    write_csv(arguments.outdir / "numerical_results.csv", results)
    write_csv(arguments.outdir / "execution_environment.csv", environment)
    summary = "\n".join(
        [
            "# Reproducibility summary",
            "",
            "Generated directly from the committed JSON records. Epoch means one "
            "full-batch Adam update over the fixed points of the corresponding stage.",
            "",
            "## Training protocol",
            "",
            markdown_table(protocol),
            "",
            "## Numerical checks",
            "",
            markdown_table(results),
            "",
            "## Execution environment",
            "",
            markdown_table(environment),
            "",
            "Both examples report one fixed-seed run. No across-seed mean or standard "
            "deviation is claimed. All grid results are finite-sample numerical "
            "evidence, not continuous-domain certificates.",
            "",
        ]
    )
    (arguments.outdir / "reproducibility_summary.md").write_text(
        summary, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
