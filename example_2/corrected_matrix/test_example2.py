import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import example_2.article_version.example2 as article
from example_2.corrected_matrix.example2 import (
    consistent_corrected_local_design,
    corrected_linear_matrices,
    corrected_lyapunov_matrix,
    frozen_article_local_design,
    run_one,
)
from example_2.corrected_matrix.switching_audit import (
    local_level_boundary,
    local_level_switched_control,
)
from example_2.corrected_matrix.boundary_matching import boundary_matching_loss
from example_2.corrected_matrix.invariance_audit import (
    extract_level_components,
    refine_level_points,
)


class CorrectedMatrixTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_corrected_A_matches_finite_difference_jacobian(self):
        a, b = corrected_linear_matrices()
        true_a, true_b = article.true_jacobian_matrices(omega=1.0)
        self.assertTrue(torch.equal(a, true_a))
        self.assertTrue(torch.equal(b, true_b))

    def test_consistent_P_solves_corrected_equation_exactly(self):
        k, p = consistent_corrected_local_design()
        self.assertTrue(
            torch.allclose(
                corrected_lyapunov_matrix(k, p),
                -torch.eye(2),
                atol=1e-14,
                rtol=1e-14,
            )
        )
        self.assertTrue(torch.all(torch.linalg.eigvalsh(p) > 0.0))

    def test_frozen_P_remains_stable_but_no_longer_solves_minus_I(self):
        k, p = frozen_article_local_design()
        actual = corrected_lyapunov_matrix(k, p)
        expected = torch.tensor([[-0.5, 0.25], [0.25, -1.0]])
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.all(torch.linalg.eigvalsh(actual) < 0.0))
        self.assertFalse(torch.equal(actual, -torch.eye(2)))

    def test_neural_training_does_not_read_A(self):
        specification = article.ArticleSpecification(grid_points_per_axis=10)
        config = article.ImplementationConfig(
            training_steps=1,
            validation_points_per_axis=11,
            plot_points_per_axis=21,
            log_every=1,
        )
        with patch.object(
            article,
            "article_linear_matrices",
            side_effect=AssertionError("A must not be read during neural training"),
        ):
            article.train_networks(specification, config)

    def test_run_record_uses_the_selected_local_design(self):
        stale = {
            "local_design": {
                "K": [-2.0, -3.0],
                "P": [[1.25, 0.25], [0.25, 0.25]],
            }
        }

        def fake_run(specification, config, outdir):
            outdir.mkdir(parents=True, exist_ok=True)
            return stale.copy()

        with TemporaryDirectory() as directory, patch.object(
            article, "run", side_effect=fake_run
        ):
            result = run_one(
                "consistent_local_design",
                consistent_corrected_local_design,
                article.ArticleSpecification(),
                article.ImplementationConfig(training_steps=1),
                Path(directory),
            )
        _, expected_p = consistent_corrected_local_design()
        self.assertEqual(result["local_design"]["P"], expected_p.tolist())

    def test_local_level_boundary_is_exact(self):
        _, p = consistent_corrected_local_design()
        boundary = local_level_boundary(p, kappa=0.05, count=257, phase=0.5)
        values = article.local_quadratic_value(boundary, p)
        self.assertTrue(
            torch.allclose(values, torch.full_like(values, 0.05), atol=1e-14)
        )

    def test_shifted_boundary_audit_does_not_reuse_training_points(self):
        _, p = consistent_corrected_local_design()
        training = local_level_boundary(p, kappa=0.05, count=512)
        audit = local_level_boundary(p, kappa=0.05, count=2048, phase=0.5)
        distances = torch.cdist(training, audit)
        self.assertGreater(distances.min().item(), 1e-8)

    def test_local_level_switch_uses_only_B_kappa(self):
        k, p = consistent_corrected_local_design()
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]])

        def learned(x):
            return torch.full((len(x), 1), 7.0, dtype=x.dtype)

        control = local_level_switched_control(
            points, learned, k, p, kappa=0.05
        )
        self.assertEqual(control[0].item(), 0.0)
        self.assertEqual(control[1].item(), 7.0)

    def test_boundary_matching_loss_is_exact_mean_absolute_error(self):
        class FixedLyapunov(torch.nn.Module):
            def forward(self, x):
                return x[:, 0]

        points = torch.tensor([[0.04, 0.0], [0.08, 0.0]])
        loss = boundary_matching_loss(FixedLyapunov(), points, kappa=0.05)
        self.assertAlmostEqual(loss.item(), 0.02)

    def test_detects_and_refines_a_circular_level_set(self):
        class RadiusSquared(torch.nn.Module):
            def forward(self, x):
                return torch.sum(x * x, dim=1)

        specification = article.ArticleSpecification(
            angle_min=-2.0,
            angle_max=2.0,
            velocity_min=-2.0,
            velocity_max=2.0,
        )
        model = RadiusSquared()
        components, _, _, _ = extract_level_components(
            model, specification, level=1.0, points_per_axis=81
        )
        self.assertEqual(len(components), 1)
        refined, gradient_norm = refine_level_points(
            model, components[0], 1.0, specification
        )
        residual = torch.sum(refined * refined, dim=1) - 1.0
        self.assertLess(residual.abs().max().item(), 1e-12)
        self.assertGreater(gradient_norm.min().item(), 1.9)


if __name__ == "__main__":
    unittest.main()
