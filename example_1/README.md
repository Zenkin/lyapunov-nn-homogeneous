# Example 1: stability analysis

A mechanical system with bounded dry friction, nonlinear drag, and a cubic
restoring term. Its approximation at infinity is homogeneous with `r=(1,2)`.

- [`original`](original): minimum-based gluing from equation (16).
- [`improved`](improved): a subsequent extension with a positive-definite
  inner candidate and smooth gluing.

Both use the same system parameters, equilibrium shift, homogeneous outer
candidate, seed, and level-angle grid.
