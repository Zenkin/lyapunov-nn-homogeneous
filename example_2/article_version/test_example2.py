import math
import unittest

import torch

from example_2.article_version.example2 import (
    ArticleSpecification,
    ControllerNetwork,
    LyapunovNetwork,
    article_linear_field,
    article_linear_matrices,
    article_local_lyapunov_matrix,
    article_pointwise_loss,
    article_training_domain,
    implementation_local_design,
    local_quadratic_value,
    nonlinear_field,
    switched_control,
    true_jacobian_matrices,
    uniform_rectangle_grid,
)


class ArticleFormulaTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self.specification = ArticleSpecification()

    def test_nonlinear_field_matches_printed_formula(self):
        x = torch.tensor([[0.3, -0.7], [-0.4, 1.2]])
        u = torch.tensor([[0.5], [-0.2]])
        value = nonlinear_field(x, u, omega=self.specification.omega)
        expected_second = torch.sin(x[:, 0]) + torch.cos(x[:, 0]) * u[:, 0]
        expected = torch.stack((x[:, 1], expected_second), dim=1)
        self.assertTrue(torch.allclose(value, expected, atol=1e-14, rtol=1e-14))

    def test_article_matrices_are_reproduced_literally(self):
        a, b = article_linear_matrices()
        self.assertTrue(torch.equal(a, torch.tensor([[0.0, 1.0], [0.0, 0.0]])))
        self.assertTrue(torch.equal(b, torch.tensor([[0.0], [1.0]])))

    def test_article_matrix_is_not_the_jacobian_of_the_printed_system(self):
        omega = self.specification.omega
        a_article, b_article = article_linear_matrices()
        a_jacobian, b_jacobian = true_jacobian_matrices(omega)

        origin = torch.zeros(2)
        zero_control = torch.tensor(0.0)
        step = 1e-6
        finite_difference_columns = []
        for index in range(2):
            direction = torch.zeros(2)
            direction[index] = step
            plus = nonlinear_field(origin + direction, zero_control, omega)
            minus = nonlinear_field(origin - direction, zero_control, omega)
            finite_difference_columns.append((plus - minus) / (2.0 * step))
        a_finite_difference = torch.stack(finite_difference_columns, dim=1)

        plus_u = nonlinear_field(origin, torch.tensor(step), omega)
        minus_u = nonlinear_field(origin, torch.tensor(-step), omega)
        b_finite_difference = ((plus_u - minus_u) / (2.0 * step)).reshape(2, 1)

        self.assertTrue(
            torch.allclose(a_finite_difference, a_jacobian, atol=1e-10, rtol=1e-10)
        )
        self.assertTrue(
            torch.allclose(b_finite_difference, b_jacobian, atol=1e-10, rtol=1e-10)
        )
        self.assertTrue(torch.equal(b_article, b_jacobian))
        self.assertAlmostEqual((a_finite_difference - a_article).abs().max().item(), 1.0)

    def test_article_linear_field_uses_displayed_a(self):
        x = torch.tensor([[0.4, -0.6]])
        u = torch.tensor([[0.3]])
        self.assertTrue(
            torch.equal(article_linear_field(x, u), torch.tensor([[-0.6, 0.3]]))
        )
        self.assertTrue(
            torch.equal(
                article_linear_field(x[0], u[0, 0]), torch.tensor([-0.6, 0.3])
            )
        )

    def test_networks_satisfy_the_imposed_zero_at_origin(self):
        torch.manual_seed(7)
        lyapunov = LyapunovNetwork(self.specification.lyapunov_hidden_units)
        controller = ControllerNetwork(self.specification.controller_hidden_units)
        origin = torch.zeros((1, 2))
        self.assertTrue(torch.equal(lyapunov.transform(origin), torch.zeros((1, 2))))
        self.assertTrue(torch.equal(lyapunov(origin), torch.zeros(1)))
        self.assertTrue(torch.equal(controller(origin), torch.zeros((1, 1))))

    def test_local_lyapunov_matrix_matches_displayed_inequality(self):
        k = torch.zeros(2)
        p = torch.eye(2)
        expected = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        self.assertTrue(torch.equal(article_local_lyapunov_matrix(k, p), expected))

    def test_documented_local_design_has_stated_poles_and_lyapunov_equation(self):
        k, p = implementation_local_design()
        a, b = article_linear_matrices()
        closed_loop = a + b @ k.reshape(1, 2)
        eigenvalues = torch.linalg.eigvals(closed_loop).real.sort().values
        self.assertTrue(torch.allclose(eigenvalues, torch.tensor([-2.0, -1.0])))
        self.assertTrue(
            torch.allclose(
                closed_loop.T @ p + p @ closed_loop,
                -torch.eye(2),
                atol=1e-14,
                rtol=1e-14,
            )
        )
        self.assertTrue(torch.all(torch.linalg.eigvalsh(p) > 0.0))

    def test_square_candidate_is_nonnegative(self):
        torch.manual_seed(11)
        lyapunov = LyapunovNetwork(self.specification.lyapunov_hidden_units)
        points = torch.randn((100, 2))
        self.assertTrue(torch.all(lyapunov(points) >= 0.0))

    def test_pointwise_loss_matches_bracket_definition(self):
        torch.manual_seed(13)
        lyapunov = LyapunovNetwork(self.specification.lyapunov_hidden_units)
        controller = ControllerNetwork(self.specification.controller_hidden_units)
        points = torch.tensor([[0.2, -0.3], [-0.5, 0.7]])
        epsilon = 0.1

        actual = article_pointwise_loss(
            lyapunov, controller, points, self.specification.omega, epsilon
        )

        differentiable_points = points.detach().clone().requires_grad_(True)
        value = lyapunov(differentiable_points)
        gradient = torch.autograd.grad(value.sum(), differentiable_points)[0]
        control = controller(differentiable_points)
        derivative = torch.sum(
            gradient
            * nonlinear_field(differentiable_points, control, self.specification.omega),
            dim=1,
        )
        expected = torch.clamp_min(derivative + value, 0.0) - torch.clamp_max(
            value - epsilon, 0.0
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-13, rtol=1e-13))

        actual.mean().backward()
        for parameter in list(lyapunov.parameters()) + list(controller.parameters()):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.all(torch.isfinite(parameter.grad)))

    def test_grid_choice_and_training_domain_are_explicit(self):
        grid_with_boundary = uniform_rectangle_grid(
            self.specification, include_boundary=True
        )
        grid_at_midpoints = uniform_rectangle_grid(
            self.specification, include_boundary=False
        )
        self.assertEqual(grid_with_boundary.shape, (10000, 2))
        self.assertEqual(grid_at_midpoints.shape, (10000, 2))
        self.assertAlmostEqual(grid_with_boundary[:, 0].min().item(), -math.pi)
        self.assertGreater(grid_at_midpoints[:, 0].min().item(), -math.pi)

        p = torch.eye(2)
        selected = article_training_domain(grid_at_midpoints, p, kappa=1.0)
        self.assertTrue(torch.all(local_quadratic_value(selected, p) > 0.5))

    def test_switch_uses_learned_control_at_equality(self):
        class ConstantLyapunov(torch.nn.Module):
            def forward(self, x):
                return torch.tensor([0.5, 1.0, 1.5], dtype=x.dtype)

        points = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        k = torch.tensor([2.0, 0.0])

        def learned_controller(x):
            return torch.full((x.shape[0], 1), -3.0, dtype=x.dtype)

        control = switched_control(
            points,
            ConstantLyapunov(),
            learned_controller,
            k,
            kappa=1.0,
        )
        self.assertTrue(torch.equal(control[:, 0], torch.tensor([2.0, -3.0, -3.0])))


if __name__ == "__main__":
    unittest.main()
