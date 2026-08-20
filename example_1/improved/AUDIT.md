# Validation record: smooth-gluing variant

This file records the numerical checks for the smooth-gluing implementation.
The values below are evidence on stated finite grids; they are not a proof on
a continuous or unbounded domain.

## Construction being checked

The inner candidate is parameterized as

```text
U_theta(z) = epsilon_q ||z/R||^2 + ||R_theta(z/R)||^2,
epsilon_q = 0.001.
```

Let `q(z)=V_inf(z)`. In the transition layer the two candidates are joined by

```text
t = clip((q-kappa)/kappa, 0, 1),
sigma(t) = 6 t^5 - 15 t^4 + 10 t^3,
V_smooth = (1-sigma) U_theta + sigma V_inf.
```

The complete expression is differentiated during training. An independent
expanded calculation checks

```text
D V_smooth f = (1-sigma) D U_theta f + sigma D V_inf f
             + (d sigma/dq) D V_inf f (V_inf-U_theta).
```

## Deterministic tests

All 13 unit tests pass. They cover the vector field, the equilibrium,
homogeneity, positivity of the inner parameterization, endpoint values and
derivatives of the quintic gate, boundary matching, and agreement between the
expanded derivative and direct automatic differentiation.

```text
x_eq                         = 3.3247179572447463
equilibrium residual         = -8.881784197001252e-16
```

## Homogeneous stage

The homogeneous network is trained for 5,000 full-grid Adam steps. On an
independent 8,192-point homogeneous-sphere grid:

```text
min V_inf                    = +1.0010020055247113
max D V_inf f_inf            = -1.0005125421955627
positivity violation fraction = 0
decay violation fraction       = 0
```

The raw derivative of `W_theta` printed in loss (12) is not the derivative of
the complete homogeneous extension. Its sampled maximum is

```text
max D W_theta f_inf          = +3.577872410662481
```

The implementation therefore trains and validates the derivative of the full
candidate `V_inf`, as defined above.

The empirical level search gives

```text
kappa                        = 9.201080709618669
max D V_inf f_full for sampled V_inf >= kappa
                             = -0.15854100291437767
```

## Inner and transition stages

The inner stage uses 12,000 deterministic full-grid Adam steps and 9,452
training points. The final training hinge loss is zero for
`relu(D V_smooth f_full + 0.05)`.

The independent validation set contains 54,140 points:

```text
37,800  level-angle points
10,388  Cartesian points
 5,952  points approaching z2=0 logarithmically
29,877  points in the transition region
```

Observed extrema are

```text
min U_theta                  = +0.012761223688969375
min V_smooth                 = +0.012761223688969375
max D V_smooth f, all        = -0.04414346562709255
max D V_smooth f, level-angle = -0.04414346562709255
max D V_smooth f, Cartesian  = -0.05134973483000782
max D V_smooth f, near z2=0  = -0.09373390271227824
nonpositive-value fraction   = 0
nonnegative-derivative fraction = 0
```

The training reserve `0.05` is not fully preserved on the independent grid:

```text
independent hinge loss       = 4.68501486891905e-7
reserve-violation fraction   = 0.00016623568526043592
worst reserve shortfall      = 0.0058565343729074515
```

This does not change the observed strict sign `D V_smooth f<0`, but the reserve
shortfall is retained here as part of the numerical record.

Independent consistency checks give

```text
max value-formula disagreement = 0
max derivative disagreement  = 3.751665644813329e-12
inner boundary identity error = 0
outer boundary identity error = 2.1316282072803006e-14
```

## Nonsmooth axis

Since `rho(z)` contains `abs(z2)`, automatic differentiation at `z2=0` selects
a subgradient. The exact axis is therefore checked separately at 1,920 sampled
points using one-sided directional quotients:

```text
h=1e-4   : -0.08367280149093603
h=3e-5   : -0.08367987483770545
h=1e-5   : -0.08368189579610608
h=3e-6   : -0.0836826031293721
```

These values support negativity of the sampled upper directional derivative;
a finite sequence of quotients is not a proof of its limiting value.

## Scope and limitations

1. The dry-friction law and its constants are reconstructed from repository
   branch `dev_w`; the article does not report them.
2. `kappa` is selected by a finite search over `0.25 <= rho <= 8`.
3. Validation concerns `R^2 \ X`, as in the practical GAS example, rather than
   asymptotic stability of the equilibrium point itself.
4. The nonsmooth axis is checked numerically, not by a verified nonsmooth bound.
5. The recorded result uses one fixed random seed.
6. The smooth gate, quadratic coefficient, optimizer settings, and grids are
   implementation choices introduced for this variant.
7. Passing a finite grid does not certify inequalities between grid points or
   beyond the sampled radial range.

Within this scope, the smooth-gluing variant has positive sampled values and
strictly negative sampled directional derivatives outside `X`. It preserves
the article's central construction—learning on the homogeneous sphere and
uniting an outer homogeneous candidate with an inner neural candidate—while
replacing the pointwise minimum by a differentiable transition.
