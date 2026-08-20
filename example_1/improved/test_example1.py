import math
import unittest

import torch

from example_1.improved.example1 import (
    OuterNetwork,
    SystemParameters,
    dry_friction,
    equilibrium_position,
    homogeneous_field,
    homogeneous_candidate,
    homogeneous_norm,
    homogeneous_normalize,
    homogeneous_sphere,
    explicit_smooth_lie_derivative,
    lie_derivative,
    level_boundary,
    quintic_smoothstep,
    quintic_smoothstep_derivative,
    smooth_combined_candidate,
    SphereNetwork,
    original_field,
    shifted_field,
)


class MathematicalIdentityTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self.parameters = SystemParameters()

    def test_equilibrium_is_in_original_coordinates(self):
        equilibrium = equilibrium_position(self.parameters)
        self.assertAlmostEqual(equilibrium, 3.324717957244746, places=13)
        state = torch.tensor([[equilibrium, 0.0]])
        self.assertLess(torch.linalg.vector_norm(original_field(state, self.parameters)).item(), 1e-13)

    def test_shifted_origin_is_equilibrium(self):
        equilibrium = equilibrium_position(self.parameters)
        value = shifted_field(torch.zeros((1, 2)), self.parameters, equilibrium)
        self.assertLess(torch.linalg.vector_norm(value).item(), 1e-13)

    def test_reconstructed_friction_is_bounded_odd_and_zero_at_zero(self):
        velocity = torch.tensor([-100.0, -0.3, 0.0, 0.3, 100.0])
        value = dry_friction(velocity, self.parameters)
        self.assertEqual(value[2].item(), 0.0)
        self.assertTrue(torch.allclose(value, -torch.flip(value, dims=(0,)), atol=1e-14))
        self.assertLessEqual(torch.max(torch.abs(value)).item(), self.parameters.friction_amplitude)

    def test_homogeneous_norm_scaling(self):
        z = torch.tensor([[0.7, -1.3], [-1.1, 0.2]])
        for scale in (0.4, 2.0, 3.0):
            dilated = torch.stack((scale * z[:, 0], scale**2 * z[:, 1]), dim=1)
            self.assertTrue(
                torch.allclose(homogeneous_norm(dilated), scale * homogeneous_norm(z), atol=1e-13)
            )

    def test_homogeneous_field_degree_one(self):
        z = torch.tensor([[0.7, -1.3], [-1.1, 0.2]])
        field = homogeneous_field(z, self.parameters)
        for scale in (0.4, 2.0, 3.0):
            dilated_z = torch.stack((scale * z[:, 0], scale**2 * z[:, 1]), dim=1)
            left = homogeneous_field(dilated_z, self.parameters)
            right = torch.stack((scale**2 * field[:, 0], scale**3 * field[:, 1]), dim=1)
            self.assertTrue(torch.allclose(left, right, atol=1e-12, rtol=1e-12))

    def test_homogeneous_candidate_has_degree_two(self):
        torch.manual_seed(2)
        network = SphereNetwork(hidden=7)
        z = torch.tensor([[0.7, -1.3], [-1.1, 0.2]])
        value = homogeneous_candidate(network, z)
        for scale in (0.4, 2.0, 3.0):
            dilated = torch.stack((scale * z[:, 0], scale**2 * z[:, 1]), dim=1)
            scaled_value = homogeneous_candidate(network, dilated)
            self.assertTrue(torch.allclose(scaled_value, scale**2 * value, atol=1e-12, rtol=1e-12))

    def test_normalization_lies_on_homogeneous_sphere(self):
        z = torch.tensor([[0.7, -1.3], [-1.1, 0.2]])
        _, y = homogeneous_normalize(z)
        sphere_equation = y[:, 0] ** 2 + torch.abs(y[:, 1])
        self.assertTrue(torch.allclose(sphere_equation, torch.ones_like(sphere_equation), atol=1e-13))

    def test_sphere_parameterization_is_exact(self):
        y = homogeneous_sphere(1000)
        sphere_equation = y[:, 0] ** 2 + torch.abs(y[:, 1])
        self.assertTrue(torch.allclose(sphere_equation, torch.ones_like(sphere_equation), atol=1e-13))
        self.assertGreater(torch.min(torch.abs(y[:, 1])).item(), 0.0)

    def test_outer_transform_is_zero_at_origin(self):
        network = OuterNetwork(hidden=4)
        value = network.transform(torch.zeros((1, 2)))
        self.assertTrue(torch.equal(value, torch.zeros_like(value)))

    def test_modified_inner_candidate_is_positive_definite_by_construction(self):
        network = OuterNetwork(
            hidden=4, scale_z1=2.0, scale_z2=3.0, fixed_quadratic=1e-3
        )
        points = torch.tensor([[0.0, 0.0], [0.2, 0.0], [0.0, -0.3], [0.4, 0.7]])
        values = network(points)
        self.assertEqual(values[0].item(), 0.0)
        self.assertTrue(torch.all(values[1:] > 0.0))

    def test_quintic_gate_values_and_endpoint_derivatives(self):
        coordinate = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0])
        values = quintic_smoothstep(coordinate)
        derivatives = quintic_smoothstep_derivative(coordinate)
        expected = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0])
        self.assertTrue(torch.allclose(values, expected, atol=1e-15))
        self.assertEqual(derivatives[0].item(), 0.0)
        self.assertEqual(derivatives[1].item(), 0.0)
        self.assertEqual(derivatives[3].item(), 0.0)
        self.assertEqual(derivatives[4].item(), 0.0)

    @staticmethod
    def _positive_constant_sphere_network() -> SphereNetwork:
        network = SphereNetwork(hidden=3)
        with torch.no_grad():
            network.input.weight.zero_()
            network.input.bias.zero_()
            network.output.weight.zero_()
            network.output.bias.fill_(1.5)
        return network

    def test_smooth_candidate_matches_components_at_both_boundaries(self):
        homogeneous_network = self._positive_constant_sphere_network()
        inner_network = OuterNetwork(hidden=4, fixed_quadratic=1e-3)
        kappa = 1.0
        inner_boundary = level_boundary(homogeneous_network, kappa, 128, 0.25)
        outer_boundary = level_boundary(homogeneous_network, 2.0 * kappa, 128, 0.25)
        with torch.no_grad():
            at_inner = smooth_combined_candidate(
                homogeneous_network, inner_network, inner_boundary, kappa
            )
            at_outer = smooth_combined_candidate(
                homogeneous_network, inner_network, outer_boundary, kappa
            )
            self.assertTrue(torch.allclose(at_inner, inner_network(inner_boundary), atol=1e-14))
            self.assertTrue(
                torch.allclose(
                    at_outer,
                    homogeneous_candidate(homogeneous_network, outer_boundary),
                    atol=1e-14,
                )
            )

    def test_expanded_smooth_derivative_matches_direct_autograd(self):
        torch.manual_seed(7)
        homogeneous_network = self._positive_constant_sphere_network()
        inner_network = OuterNetwork(hidden=5, fixed_quadratic=1e-3)
        equilibrium = equilibrium_position(self.parameters)
        points = torch.tensor([[0.8, 0.4], [-1.1, 0.3], [0.7, -0.6]])
        kappa = 1.0
        vector_field = lambda z: shifted_field(z, self.parameters, equilibrium)
        direct_value, direct_derivative = lie_derivative(
            lambda z: smooth_combined_candidate(
                homogeneous_network, inner_network, z, kappa
            ),
            points,
            vector_field,
            create_graph=False,
        )
        expanded_value, expanded_derivative = explicit_smooth_lie_derivative(
            homogeneous_network,
            inner_network,
            points,
            vector_field,
            kappa,
        )
        self.assertTrue(torch.allclose(direct_value, expanded_value, atol=1e-13))
        self.assertTrue(
            torch.allclose(direct_derivative, expanded_derivative, atol=1e-12, rtol=1e-12)
        )


if __name__ == "__main__":
    unittest.main()
