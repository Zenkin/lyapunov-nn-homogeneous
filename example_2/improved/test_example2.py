import math
import unittest

import torch

import example_2.article_version.example2 as article
from example_2.corrected_matrix.example2 import consistent_corrected_local_design
from example_2.improved.example2 import (
    ImprovedConfig,
    PeriodicControllerNetwork,
    PeriodicLyapunovNetwork,
    combined_objective,
    equilibrium_audit,
    improved_switched_control,
    local_analytic_certificate,
    midpoint_grid,
    periodic_features,
    top_fraction_mean,
)


class ImprovedExampleTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_periodic_features_identify_angle_seam(self):
        velocity = torch.linspace(-4.0, 4.0, 17)
        left = torch.stack((torch.full_like(velocity, -math.pi), velocity), dim=1)
        right = torch.stack((torch.full_like(velocity, math.pi), velocity), dim=1)
        self.assertTrue(torch.allclose(periodic_features(left), periodic_features(right), atol=1e-15))

    def test_both_networks_are_exactly_zero_at_target(self):
        target = torch.zeros((1, 2))
        self.assertEqual(PeriodicLyapunovNetwork()(target).item(), 0.0)
        self.assertEqual(PeriodicControllerNetwork()(target).item(), 0.0)

    def test_switch_uses_only_local_quadratic_level(self):
        class FixedController(torch.nn.Module):
            def forward(self, x):
                return torch.full((len(x), 1), 7.0, dtype=x.dtype)

        k, p = consistent_corrected_local_design()
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        control = improved_switched_control(
            points, FixedController(), k, p, kappa=0.05
        )
        self.assertEqual(control[0].item(), 0.0)
        self.assertEqual(control[1].item(), 7.0)

    def test_top_fraction_mean_uses_largest_values(self):
        values = torch.tensor([1.0, 2.0, 8.0, 9.0])
        self.assertEqual(top_fraction_mean(values, 0.5).item(), 8.5)

    def test_combined_objective_has_declared_weights(self):
        config = ImprovedConfig(
            worst_weight=2.0,
            boundary_match_weight=3.0,
            boundary_inward_weight=4.0,
            control_weight=5.0,
        )
        terms = {
            "outer_mean": torch.tensor(1.0),
            "outer_worst_fraction": torch.tensor(2.0),
            "boundary_control_match": torch.tensor(3.0),
            "boundary_inward": torch.tensor(4.0),
            "control_regularization": torch.tensor(5.0),
        }
        self.assertEqual(combined_objective(terms, config).item(), 55.0)

    def test_training_and_validation_midpoint_grids_are_disjoint(self):
        training = midpoint_grid(100)[:, 0].unique()
        validation = midpoint_grid(211)[:, 0].unique()
        self.assertFalse(torch.isin(training, validation).any().item())

    def test_local_design_is_consistent_with_corrected_jacobian(self):
        k, p = consistent_corrected_local_design()
        a = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        b = torch.tensor([[0.0], [1.0]])
        closed = a + b @ k.reshape(1, 2)
        self.assertTrue(torch.allclose(closed.T @ p + p @ closed, -torch.eye(2), atol=1e-14))

    def test_local_continuous_decay_bound_has_positive_margin(self):
        certificate = local_analytic_certificate(0.05)
        self.assertGreater(certificate["decay_margin"], 0.349)
        self.assertLess(certificate["decay_margin"], 0.350)

        k, p = consistent_corrected_local_design()
        coordinate = torch.linspace(-0.55, 0.55, 101)
        angle, velocity = torch.meshgrid(coordinate, coordinate, indexing="ij")
        points = torch.stack((angle.reshape(-1), velocity.reshape(-1)), dim=1)
        points = points[article.local_quadratic_value(points, p) <= 0.05]
        gradient = 2.0 * points @ p
        field = article.nonlinear_field(
            points, article.local_linear_control(points, k), 1.0
        )
        derivative = torch.sum(gradient * field, dim=1)
        squared_norm = torch.sum(points * points, dim=1)
        upper_bound = -certificate["decay_margin"] * squared_norm
        self.assertTrue(torch.all(derivative <= upper_bound + 1e-14))

    def test_equilibrium_scan_detects_the_target(self):
        controller = PeriodicControllerNetwork()
        audit = equilibrium_audit(
            controller, ImprovedConfig(), scan_points=401
        )
        angles = [item["theta"] for item in audit["equilibria"]]
        self.assertTrue(any(abs(angle) < 1e-12 for angle in angles))


if __name__ == "__main__":
    unittest.main()
