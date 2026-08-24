# Audit of the smooth neural extension

## Scope

The implementation keeps the article's local-plus-neural stabilization
principle, but the smooth composite architecture, periodic base, corrected
Jacobian, and numerical hyperparameters are later implementation choices.
They are not attributed to the printed article.

The conclusions have two different strengths:

1. the local decay estimate and positivity of the composite architecture are
   analytic;
2. the outer derivative inequalities, value of `c`, equilibria, and
   trajectories are finite numerical checks.

## Local analytic part

For the corrected Jacobian,

```text
K = (-2,-3),
P = [[11/6, 1/2],
     [ 1/2, 1/3]],
```

and `(A+BK)^T P+P(A+BK)=-I` exactly. Direct remainder bounds give

```text
DV_l F(x,Kx) <= -0.3495373937 ||x||^2 < 0
```

for every nonzero point in `V_l<=kappa`, with `kappa=0.05`.

## Why the composite is positive

The neural term alone is `W=T^T T>=0`, but this does not exclude zeros of `T`
away from the target. The implemented candidate is instead

```text
V_NN = (1-s) V_l + s (V_p+W),
V_p  = (sin(theta),theta_dot)^T P (sin(theta),theta_dot)
       + (1-cos(theta))^2.
```

In the transition, `0<s<1` and `V_l>0` away from the target. In the outer
region, `V_p>0` on the pendulum cylinder away from the target and `W>=0`.
Hence the composite is positive definite independently of whether the neural
map has an exact zero.

The cubic transition has zero endpoint derivatives. On 2,048 samples of
`V_l=kappa`, the numerical identities are

```text
max |V_NN-kappa|       = 1.53e-16,
max |u-Kx|             = 4.44e-16,
max transition weight = 2.31e-29.
```

The periodic seam mismatches of both `V_NN` and the actual control are zero to
the recorded floating-point precision.

## Loss and exact zero

The hidden activations are `tanh`. The positive-part operation in the loss is
implemented by `torch.clamp_min`, not by a ReLU network layer.

The outer margin is

```text
-[W-epsilon]_- = [epsilon-W]_+,  epsilon=0.05.
```

At an exact scalar example `W=a^2`, the penalty has value `epsilon` at `a=0`
but derivative zero. At a small nonzero `a`, its derivative is `-2a` and
gradient descent pushes `|a|` upward. A unit test records this distinction.
The composite architecture avoids relying on this gradient to establish
positive definiteness.

On the `801x801` audit grid, raw `W` is below `epsilon` at approximately
0.174% of the samples outside `V_l<=kappa`; the maximum margin violation is
about 0.04990. This residual does not make `V_NN` nonpositive because of the fixed
positive base, but it is retained as a failed margin check.

## Independent validation

The primary validation grid contains `401x401=160,801` cell midpoints and is
disjoint from the `100x100` training midpoint grid. A second `801x801`
midpoint grid is used as a resolution audit.

The objective values evaluated after the final optimizer update are

```text
training 100x100 combined objective     = 0.16608341281928593
validation 401x401 combined objective   = 0.16936016274451050
audit 801x801 combined objective        = 0.16945453931764880
```

The two validation values use exactly the training objective weights; neither
grid participates in gradient updates.

```text
local points, excluding target       = 836
max dV_NN in local region            = -0.0002454607918731659

transition points                    = 840
max dV_NN in transition              = -0.09210354617765754

full nonzero validation points       = 160,800
max dV_NN on full rectangle          = +0.6337056599161834
fraction dV_NN >= 0                  = 0.00333955223880597
fraction dV_NN+0.1 V_NN > 0         = 0.009894278606965174
```

Thus the full rectangle does not have a strict neural Lyapunov certificate.
The red and orange regions remain visible in the committed condition map.

## Sampled Lyapunov domain

For each candidate level, the audit forms the periodic-angle, four-neighbour
component of `{V_NN<=c}` that contains the target. It requires

```text
V_NN>0 and dV_NN<0
```

at every non-target sample in that component and rejects components touching
the upper or lower velocity boundary. The first low-value samples with
`dV_NN>=0` seed local constrained minimizations of `V_NN` subject to
`dV_NN=0`. The two resolutions give the same refined contact to the shown
precision:

```text
critical level estimate                     = 3.4664871442366
critical point estimate                     = (1.4681556877,-0.2980136606)
reported level, rounded downward             = 3.466
401x401 points in connected component        = 18,893
801x801 points in connected component        = 75,361
nonnegative-dV points in either component    = 0
disconnected points in either sublevel       = 0
touches velocity boundary                  = false
```

This is a grid-seeded local numerical refinement. It does not isolate all
roots or certify values between samples. A formal continuous-domain result
would additionally require interval or Lipschitz bounds.

## Trajectories

Fixed-step RK4 with wrapped angle was applied to a periodic `25x21` grid of
initial conditions.

```text
final time                                  = 40
integration step                            = 0.01
successful trajectories                     = 525/525
initial conditions inside V_NN<=c           = 59
successes among those 59                     = 59
initial conditions in 0.95c<=V_NN<=c        = 6/6 successful
initial conditions in c<V_NN<=1.05c         = 7/7 successful
trajectories crossing theta=+-pi/2          = 313
minimum recorded crossing speed             = 0.1111082705
trajectories leaving |theta_dot|<=4          = 182
maximum observed |theta_dot|                 = 8.1556534353
```

Only 343 trajectories stay inside the velocity interval used for training and
rectangular validation. The other 182 successful trajectories use the learned
controller outside that interval and are therefore labelled as extrapolation,
not as validation-domain evidence.

The time-horizon and step-size audit is

| RK4 step | Final time | Reached target |
| ---: | ---: | ---: |
| 0.02 | 20 | 524/525 |
| 0.01 | 20 | 524/525 |
| 0.005 | 20 | 524/525 |
| 0.01 | 40 | 525/525 |

The unchanged `t=20` count under step refinement indicates that the remaining
state is slow rather than a time-discretization artifact. Extending the horizon
to `t=40` brings it into the target level.

## Loss-of-authority lines

The plant is

```text
theta_dot    = velocity,
velocity_dot = sin(theta)+cos(theta) u.
```

At `theta=+-pi/2`, the control contribution is zero, while gravity is `+-1`.
These points are not equilibria. The crossing figure shows one successful
trajectory passing the line with nonzero velocity and separately plots the
gravity and control contributions.

## Additional equilibria and global claim

The dense zero-velocity scan detects four equilibria:

```text
theta = -1.4932806571   saddle
theta =  0              asymptotically stable target
theta =  1.4313954898   saddle
theta =  1.7075779461   unstable focus by linearization
```

The classifications use eigenvalues of numerically evaluated closed-loop
Jacobians. They are local numerical classifications, not symbolic isolation of
all roots.

The extra equilibria prevent a global strict Lyapunov claim for this
continuous periodic static feedback. They do not contradict the sampled
regional level `c` or the empirical convergence of the declared finite
trajectory grid. Accordingly, the result is reported as a regional
finite-sample Lyapunov estimate together with a broader empirical trajectory
result, not as a global proof.

## Software audit

- all 56 repository unit tests pass in the recorded environment;
- two complete 6,000-step runs produced identical numerical tensor hashes:
  `d675ad589fb684a6f661d3e85fe212aaf2e484f0a8171bd6e8e2171aa5ba709c`
  for the Lyapunov network and
  `afa25a66afdd6832bd770ab629d08f603b7d97b2133c1f76fce4ef65c4b82244`
  for the controller;
- the positive-part/epsilon zero-gradient distinction has a dedicated test;
- the local matrix equation, analytic local bound, periodic seam, smoothstep
  endpoint slopes, and target zeros have dedicated tests;
- all saved metrics are evaluated after the final optimizer update;
- training and validation midpoint grids are disjoint.
