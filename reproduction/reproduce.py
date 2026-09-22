"""Re-evaluate preserved checkpoints; never train or overwrite reference files."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from example_1.improved import example1 as e1
from example_2.improved import example2 as e2


def save(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_models():
    one_path = ROOT / "example_1/improved/figures/reference/models.pt"
    two_path = ROOT / "reproduction/checkpoints/example2-d22b775/model_state.pt"
    manifest = json.loads((ROOT / "reproduction/checkpoints/SHA256.json").read_text())
    for path in (one_path, two_path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest[path.relative_to(ROOT).as_posix()]:
            raise ValueError(f"Checkpoint checksum mismatch: {path}")
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        one = torch.load(one_path, map_location="cpu", weights_only=True)
    two = torch.load(two_path, map_location="cpu", weights_only=True)
    config1 = e1.RunConfig(**one["config"]["run"])
    h = e1.SphereNetwork(config1.homogeneous_hidden)
    n = e1.OuterNetwork(
        config1.outer_hidden, fixed_quadratic=config1.outer_fixed_quadratic
    )
    h.load_state_dict(one["homogeneous_network"])
    n.load_state_dict(one["inner_network"])
    l = e2.PeriodicLyapunovNetwork()
    u = e2.PeriodicControllerNetwork()
    l.load_state_dict(two["lyapunov_state_dict"])
    u.load_state_dict(two["controller_state_dict"])
    for model in (h, n, l, u):
        model.eval()
        model.requires_grad_(False)
    return one, config1, h, n, two, e2.ImprovedConfig(**two["config"]), l, u


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    out = args.outdir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    one, c1, h, n, two, c2, l, u = load_models()
    p = e1.SystemParameters(**one["config"]["system"])
    eq = e1.equilibrium_position(p)
    kappa = one["metrics"]["empirical_kappa"]["selected_kappa"]
    box = tuple(float(v) for v in n.input_scale)
    folder1, folder2 = out / "example1", out / "example2"
    folder1.mkdir()
    folder2.mkdir()
    print("Example 1: independent validation of the saved networks...", flush=True)
    validation1 = e1.validate_and_plot(h, n, c1, p, eq, kappa, box, folder1)
    # Use the display rectangle and horizon recorded in reproduction/README.md.
    paper_c1 = replace(c1, trajectory_final_time=20.0)
    trajectories1, arrays1 = e1.simulate_to_compact_set(
        h, n, paper_c1, p, eq, kappa, (4.0, 12.0)
    )
    np.savez_compressed(folder1 / "trajectory_arrays.npz", **arrays1)
    print(
        "Example 2: 401 x 401 and 801 x 801 validation of the historical checkpoint...",
        flush=True,
    )
    validation2, arrays2 = e2.validate(l, u, c2)
    high2, _ = e2.validate(
        l,
        u,
        replace(c2, validation_points_per_axis=c2.audit_validation_points_per_axis),
    )
    level = min(
        validation2["roa_estimate"]["c_conservative"],
        high2["roa_estimate"]["c_conservative"],
    )
    count = c2.validation_points_per_axis
    xgrid = arrays2["x"].reshape(count, count, 2)
    component_metrics, component = e2._sampled_component_metrics(
        arrays2["V_NN"].reshape(count, count),
        arrays2["dV_NN"].reshape(count, count),
        xgrid[:, :, 0],
        xgrid[:, :, 1],
        level,
    )
    validation2["roa_estimate"].update(component_metrics)
    validation2["roa_estimate"]["selected_level"] = level
    arrays2["roa_component"] = component
    print(f"Example 2: c={level:.3f}; integrating 525 trajectories...", flush=True)
    trajectories2, paths2 = e2.simulate_trajectories(l, u, c2, level)
    equilibria = e2.equilibrium_audit(u, c2)
    np.savez_compressed(folder2 / "validation_arrays.npz", **arrays2)
    np.savez_compressed(folder2 / "trajectory_arrays.npz", **paths2)
    from reproduction.paper_figures import make_figures

    paper = make_figures(
        out, h, n, p, eq, kappa, l, u, c2, level, arrays1, arrays2, paths2, equilibria
    )
    expected1 = one["metrics"]["independent_validation"]
    old2 = json.loads(
        (ROOT / "reproduction/checkpoints/example2-d22b775/run_record.json").read_text()
    )
    checks = {
        "example1_validation_matches_saved_maximum": bool(
            np.isclose(
                validation1["maximum_smooth_DV_f"],
                expected1["maximum_smooth_DV_f"],
                rtol=1e-9,
                atol=1e-11,
            )
        ),
        "example1_validation_decay": validation1["maximum_smooth_DV_f"] < 0,
        "example1_validation_positive": validation1["minimum_smooth_V"] > 0,
        "example1_derivative_matches_autograd": validation1[
            "maximum_derivative_formula_disagreement"
        ]
        < 1e-10,
        "example1_paper_audit_decay": paper["example1"][
            "maximum_derivative_in_audit_set"
        ]
        < 0,
        "example1_all_625_reach_X_at_20": trajectories1[
            "trajectories_in_X_at_final_time"
        ]
        == 625,
        "example1_paper_250_inside": paper["example1"]["initial_inside"] == 250,
        "example2_level_matches_historical_record": level
        == old2["validation"]["roa_estimate"]["selected_level"],
        "example2_primary_component_decay": component_metrics[
            "nonnegative_dV_points_in_component_except_target"
        ]
        == 0,
        "example2_component_matches_historical_maximum": bool(
            np.isclose(
                component_metrics["maximum_dV_in_component_except_target"],
                old2["validation"]["roa_estimate"][
                    "maximum_dV_in_component_except_target"
                ],
                rtol=1e-9,
                atol=1e-11,
            )
        ),
        "example2_high_resolution_component_decay": high2["roa_estimate"][
            "nonnegative_dV_points_in_component_except_target"
        ]
        == 0,
        "example2_all_525_reach_target": trajectories2["successful_at_final_time"]
        == old2["trajectories"]["successful_at_final_time"]
        == 525,
        "example2_343_stay_in_domain": trajectories2[
            "successful_without_leaving_training_domain"
        ]
        == old2["trajectories"]["successful_without_leaving_training_domain"]
        == 343,
    }
    result = {
        "mode": "checkpoint replay; no training",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "software": e2.software_environment(),
        "checkpoint_sha256": json.loads(
            (ROOT / "reproduction/checkpoints/SHA256.json").read_text()
        ),
        "example1": {
            "config": asdict(paper_c1),
            "system": asdict(p),
            "trajectory_rectangle": [4, 12],
            "validation": validation1,
            "trajectories": trajectories1,
            "paper": paper["example1"],
        },
        "example2": {
            "config": asdict(c2),
            "validation": validation2,
            "high_resolution_validation": high2,
            "trajectories": trajectories2,
            "equilibria": equilibria,
            "paper": paper["example2"],
        },
        "interpretation": "Finite-grid and finite-horizon observations, not a continuous-domain certificate.",
    }
    save(out / "verification.json", result)
    print(json.dumps({"checks": checks, "paper": paper}, indent=2), flush=True)
    from reproduction.paper_figures import gallery

    gallery(out)
    if not all(checks.values()):
        raise RuntimeError(
            "Reproduction checks failed. See verification.json; results are preserved."
        )
    print(f'Verified figures: {out / "RESULTS.html"}', flush=True)


if __name__ == "__main__":
    main()
