# Audit of the domain-aligned improvement

## What is retained

The experiment retains the article's essential stabilization principle:

1. a quadratic Lyapunov function and linear feedback are used near the
   origin;
2. a neural candidate `W=T^T T` and neural feedback are used on the outer
   bounded domain;
3. the two feedback laws are joined by state-dependent switching.

The corrected Jacobian, `K=(-2,-3)`, and

```text
P = [[11/6, 1/2],
     [ 1/2, 1/3]]
```

give `(A+BK)^T P+P(A+BK)=-I` exactly.

The local set also has a continuous, rather than merely gridded, decay
estimate. Writing `R=||x||`, the nonlinear remainder relative to the corrected
closed-loop linearization is

```text
r(x) = sin(theta)-theta + (1-cos(theta))(2 theta+3 theta_dot).
```

Using

```text
|sin(theta)-theta| <= |theta|^3/6,
|1-cos(theta)|     <= theta^2/2,
|(P x)_2|          <= (sqrt(13)/6) R,
```

gives

```text
DV_l F(x,Kx) <= -R^2 + c R^4,
c = sqrt(13)/18 + 13/6.
```

Moreover,

```text
lambda_min(P) = (13-3 sqrt(13))/12,
R^2 <= kappa/lambda_min(P) on B_kappa.
```

For `kappa=0.05`, therefore,

```text
DV_l F(x,Kx) <= -0.3495373937 ||x||^2 < 0
```

for every nonzero `x` in `B_kappa`. This part is a continuous analytic bound;
the neural outer-domain conclusions below remain numerical.

## Explicit changes

The changes are implementation choices introduced after auditing the literal
version. They are not attributed to the article.

1. The switch uses `V_l<=kappa`, not `W<kappa`. Thus the local controller is
   never activated outside its checked local set.
2. `W` and `N` use periodic features
   `(sin(theta),1-cos(theta),theta_dot/4)`. This removes the artificial seam
   mismatch between `theta=-pi` and `theta=pi`.
3. Training includes the mean outer loss and the mean of its worst 5% of
   pointwise values. This reduces the ability of an arithmetic mean to hide
   sparse violations.
4. On the exact ellipse `V_l=kappa`, separate losses penalize the control jump
   and violation of
   `DV_l F(x,N(x))+0.1 kappa<=0`.
5. Validation uses a `211x211` midpoint grid whose points do not coincide with
   the `100x100` training midpoint grid. Boundary validation uses 2,048 points
   shifted away from the 512 boundary-training points.

## Structural obstruction to a global continuous result

For any continuous periodic feedback, define its acceleration on the
zero-velocity line by

```text
h(theta) = sin(theta) + cos(theta) U(theta,0).
```

At the loss-of-authority angles, independently of the value of `U`,

```text
h(-pi/2) = -1,     h(pi/2) = 1.
```

The selected local controller gives

```text
h_l(theta) = sin(theta)-2 theta cos(theta),
```

which is positive immediately to the left of zero and negative immediately
to the right. If the outer feedback is continuous and agrees with the local
feedback at the switching boundary, the intermediate value theorem forces at
least one additional zero between the positive switching boundary and
`pi/2`, and at least one between `-pi/2` and the negative switching boundary.
These zeros are additional closed-loop equilibria.

Consequently, a continuous periodic controller agreeing with this local
controller cannot make the origin the only equilibrium on the full pendulum
cylinder. A continuously differentiable function also cannot have a strictly
negative derivative at any additional equilibrium, because the vector field
is zero there. Therefore the improved experiment is evaluated as a local or
empirical almost-global construction, not as a global strict Lyapunov
certificate on the whole rectangle.

## Scope of numerical conclusions

The saved record reports three distinct checks:

- local and outer inequalities on a finite independent grid;
- direction and controller mismatch on the switching ellipse;
- fixed-step RK4 trajectories from a declared periodic initial-condition
  grid.

Trajectory integration is repeated with smaller steps and longer final times
in the self-audit. Agreement of those runs checks numerical robustness but
does not prove a region of attraction between sampled initial conditions.

## Recorded result

The corrected domain alignment works as intended on the sampled checks:

```text
max DV_l F(x,Kx) in B_kappa, excluding the origin = -0.0008860817918446897
max DV_l F(x,N(x)) on V_l=kappa                  = -0.022883994983153563
max boundary inward residual                     = -0.017883994983153562
max sampled control jump on V_l=kappa             =  0.015526928475840918
periodic seam mismatch for W                      =  1.33e-15
periodic seam mismatch for N                      =  8.88e-16
```

Thus the local controller is never selected outside `B_kappa`, the sampled
local derivative is negative away from the origin, and the learned vector
field points into `B_kappa` on all 2,048 shifted boundary points. Controller
matching is approximate, not exact.

The outer strict Lyapunov inequalities do not pass on the full validation
rectangle:

```text
min W outside B_kappa                              = 0.0017880185058213414
fraction W<epsilon, epsilon=0.05                  = 0.011222001445086706
max DW F outside B_kappa                          = 0.008026269737122498
fraction DW F>=0 outside B_kappa                  = 0.01905708092485549
max (DW F+0.1 W) outside B_kappa                  = 0.053640605507874355
fraction DW F+0.1 W>0 outside B_kappa             = 0.06240968208092486
```

The sampled `W` remains positive outside the local set, but it does not reach
the deliberately stronger margin `epsilon=0.05` everywhere. More
importantly, its derivative is not strictly negative everywhere. This is
consistent with the structural obstruction above and is reported as a failed
global-certificate check.

The dense zero-velocity scan detects four equilibria:

```text
theta = -1.3302206164   saddle
theta =  0              asymptotically stable target
theta =  1.0657639785   saddle
theta =  1.8588048108   unstable focus by linearization
```

The classifications above use the eigenvalues of the numerically evaluated
closed-loop Jacobian saved in `run_record.json`. They are local linear
classifications of the detected equilibria, not a global phase-portrait
proof.

On the periodic `25x21` initial-condition grid, with `-pi` and `pi` counted
only once, 500 of 525 trajectories enter `B_kappa` and reach
`V_l<=1e-4` by `t=20`. The remaining 25 trajectories do not reach the local
set:

```text
empirical success fraction = 500/525 = 0.9523809523809523.
```

Ninety-one trajectories temporarily exceed the training interval
`|theta_dot|<=4`; the maximum observed absolute velocity is approximately
`5.6933`. All 91 later reach the target, so 409 successful trajectories stay
inside the training rectangle for the complete simulation. Their convergence
is the part supported by both the trajectory calculation and the rectangular
validation domain; the other 91 successes involve neural extrapolation and
are labelled accordingly in the numerical record.

The unsuccessful initial conditions form a visible cluster in the saved
phase portrait. This is an empirical sampled basin estimate, not an assertion
that exactly 95.24% of the continuous state space belongs to the region of
attraction.

## Numerical self-audit

The trajectory count is unchanged under step refinement and a longer
integration interval:

| RK4 step | Final time | Successful trajectories |
| ---: | ---: | ---: |
| 0.02 | 20 | 500/525 |
| 0.01 | 20 | 500/525 |
| 0.005 | 20 | 500/525 |
| 0.01 | 40 | 500/525 |
| 0.01 | 80 | 500/525 |

At 41 deterministic validation points, the automatic directional derivative
`DW F` was compared with the centered difference

```text
[W(x+hF(x))-W(x-hF(x))]/(2h),  h=1e-6.
```

The maximum absolute discrepancy was `9.00e-10`. This checks the derivative
implementation but does not certify unsampled points.

Finally, on the two zero-velocity intersections of `V_l=kappa`,
`theta=+-0.1651445648`, the evaluated local closed-loop accelerations are
`+0.1614004832` on the negative side and `-0.1614004832` on the positive
side. Together with the control-independent values at `+-pi/2`, these signs
numerically confirm the hypotheses used in the intermediate-value argument.
