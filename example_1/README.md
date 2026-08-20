# Example 1

The first example studies a mechanical system with bounded dry friction,
nonlinear drag, and a cubic restoring term. It admits an `r=(1,2)` homogeneous
approximation at infinity.

Two implementations are provided:

- [`original`](original): the minimum-gluing construction from equation (16);
- [`improved`](improved): a positive-definite inner candidate with smooth
  level-set gluing.

The two folders use the same system parameters, equilibrium correction,
homogeneous outer candidate, deterministic seed, and level-angle grid. This
makes the effect of the changed inner architecture and gluing rule explicit.
