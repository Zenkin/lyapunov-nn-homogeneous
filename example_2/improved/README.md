# Smooth neural extension of Example 2

This implementation retains the principle of Section IV-B: a verified local
Lyapunov design is extended by a neural Lyapunov term and a neural controller.
The extension is joined through the known local level sets rather than through
a learned switching condition.

The hidden layers use the hyperbolic tangent. The `positive_part` helper in the
loss implements the article's bracket `[s]_+=max(0,s)`; it is not a ReLU
activation in either network.

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

The smoothstep and its first derivative vanish at the endpoints. Therefore the
joined candidate is continuously differentiable, the control is continuous,
and the local formulas are recovered exactly on `V_l<=kappa`. The outer neural
maps use the periodic features

```text
(sin(theta), 1-cos(theta), theta_dot/4).
```

The trained pointwise expression is

```text
[dV_NN/dt + 0.1 V_NN]_+ + [epsilon-W]_+,
epsilon = 0.05.
```

Thus the simulation uses the positive margin
`-[W-epsilon]_-=[epsilon-W]_+`, not the identically zero term `-[W]_-` from
the basic loss (10). Positive definiteness of the composite candidate does not
depend on moving an exact zero of `W`: the fixed terms `V_l` and `V_p` provide
the positive base, while `W=T^T T` remains a nonnegative neural extension.

## Reproduce

From the repository root:

```bash
python -m unittest example_2.improved.test_example2
python -m example_2.improved.example2 \
  --outdir example_2/improved/figures/reference
```

The deterministic reference run uses a `100x100` training midpoint grid,
6,000 Adam steps, an independent `401x401` validation midpoint grid, and a
second `801x801` resolution audit.

The final combined objective is `0.1641872` on the training grid, `0.1676893`
on the `401x401` grid, and `0.1677449` on the `801x801` grid. These values use
the same mean, worst-5% tail, and control-regularization weights.

## Recorded result

The first numerically refined contact with `dV_NN/dt=0` occurs at

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

All `525/525` trajectories on the declared periodic initial-condition grid
reach `V_l<=1e-4` by `t=40`. Of these trajectories, 313 cross at least one of
the lines `theta=+-pi/2` with nonzero angular velocity. The count at `t=20` is
`524/525` for integration steps `0.02`, `0.01`, and `0.005`; the remaining
trajectory reaches the target when the horizon is extended to `t=40`.
All 3 sampled initial conditions in `0.95c<=V_NN<=c` and all 10 sampled
conditions in `c<V_NN<=1.05c` also reach the target. The latter are empirical
results outside the Lyapunov-audited component, not certificate points.

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
