import math
import unittest

import torch

from example_1.original.example1 import (
    OuterNetwork,
    SystemParameters,
    dry_friction,
    equilibrium_position,
    homogeneous_field,
    homogeneous_candidate,
    homogeneous_norm,
    homogeneous_normalize,
    homogeneous_sphere,
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


if __name__ == "__main__":
    unittest.main()
