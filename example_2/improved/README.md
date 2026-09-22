# Smooth neural extension of Example 2

This extension joins a local Lyapunov design to a neural candidate and
controller through the local level sets.

The hidden layers use `tanh`; the loss uses `[s]_+=max(0,s)`.

## Composite construction

Let

```text
V_l(x) = x^T P x,
W(x)   = T(x)^T T(x),
z(x)   = (sin(theta), theta_dot),
V_p(x) = z(x)^T P z(x) + (1-cos(theta))^2.
```

The last term removes the zero of `z` at the inverted configuration without
changing the quadratic expansion at the target. A cubic smoothstep `s` equals
zero on `V_l<=kappa` and one on `V_l>=2 kappa`. The candidate and feedback used
both in training and simulation are

```text
V_NN = (1-s) V_l + s (V_p + W),
u    = (1-s) Kx  + s N(x).
```

The smoothstep has endpoint values 0 and 1 and zero derivative at both
endpoints. The candidate is continuously differentiable, the control is
continuous, and both recover the local formulas on `V_l<=kappa`.
The outer networks use the periodic features

```text
(sin(theta), 1-cos(theta), theta_dot/4).
```

The trained pointwise expression is

```text
[dV_NN/dt + 0.1 V_NN]_+ + [epsilon-W]_+,
epsilon = 0.05.
```

The fixed terms `V_l` and `V_p` make the composite positive definite;
`W=T^T T` is nonnegative. The epsilon penalty and its gradient at zero are
discussed in the [audit](AUDIT.md#loss-and-exact-zero).

## Reproduce

After [setup](../../README.md#setup), run from the repository root:

```bash
python -m unittest example_2.improved.test_example2
python -m example_2.improved.example2 --outdir example_2/improved/results/reference
```

The deterministic reference run uses a `100x100` training midpoint grid,
6,000 Adam steps, an independent `401x401` validation midpoint grid, and a
second `801x801` resolution audit.

The final combined objective is `0.1641872` on the training grid, `0.1676893`
on the `401x401` grid, and `0.1677449` on the `801x801` grid. These values use
the same mean, worst-5% tail, and control-regularization weights.

## Recorded result

A grid-seeded local minimization of `V_NN` on `dV_NN/dt=0` gives

```text
V_NN = 3.4369631633566,
x    = (1.4678147012, -0.2972368229).
```

The reported level is rounded downward to

```text
c = 3.436.
```

Both validation resolutions contain zero samples with `dV_NN/dt>=0` in the
origin-connected component of `V_NN<=c`. The component does not touch the
velocity boundary. This is a grid-seeded numerical estimate, not a
continuous-domain certificate between samples.

All `525/525` sampled trajectories reach `V_l<=1e-4` by `t=40`.
Of these, 181 leave the training velocity interval before converging; their
success is an extrapolation result. The [audit](AUDIT.md#trajectories) gives
the initial conditions, step-size comparison, and boundary-band checks.

The committed figures are:

- [`improved_audit.png`](figures/reference/improved_audit.png): Lyapunov
  sublevel, condition map, and comparison with simulated outcomes;
- [`trajectory_diagnostic.png`](figures/reference/trajectory_diagnostic.png):
  a trajectory whose coordinates move non-monotonically while `V_NN`
  decreases;
- [`loss_of_authority_crossing.png`](figures/reference/loss_of_authority_crossing.png):
  a successful passage through `theta=+-pi/2`, where the control contribution
  is zero but gravity is nonzero.
- [`training_history.png`](figures/reference/training_history.png): the recorded
  full-batch objective and sampled violation fractions over 6,000 epochs.

Detailed limitations and numerical checks are recorded in [`AUDIT.md`](AUDIT.md).
