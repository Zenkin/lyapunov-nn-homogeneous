# Validation record: minimum-gluing baseline

This file records the independent finite-grid checks for the reference
implementation. It separates properties verified by identities or tests from
properties observed only on sampled points.

## Deterministic checks

Nine unit tests cover:

1. the equilibrium in original and shifted coordinates;
2. boundedness, oddness, and the zero of the reconstructed dry friction;
3. scaling of the homogeneous gauge;
4. degree-one homogeneity of `f_inf`;
5. degree-two homogeneity of the complete candidate;
6. normalization and sphere parameterization;
7. the identity `T_theta(0)=0`.

The equilibrium calculation gives

```text
x_eq = 3.3247179572447463
residual = -8.881784197001252e-16
```

## Homogeneous stage

The loss is evaluated on the complete extension

```text
V_inf(z) = rho(z)^2 W_theta(Lambda_r(rho)^(-1) z).
```

On an independent sphere grid:

```text
min V_inf                         = +1.0010020055247113
max D V_inf f_inf                 = -1.0005125421955627
positivity violation fraction     = 0
decay violation fraction          = 0
```

The raw network derivative printed in loss (12) is a different quantity:

```text
max D W_theta f_inf               = +3.577872410662481
```

A literal run based only on the raw derivative reached
`max D W_theta f_inf=-1.00033`, while the derivative of the complete extension
reached `max D V_inf f_inf=+0.27823` and was positive at 3.125% of the
independent sphere points. The retained implementation differentiates the
complete function.

## Empirical level selection

The finite radial-angle search produced

```text
kappa                              = 9.201080709618669
max D V_inf f_full for sampled
points with V_inf >= kappa         = -0.15854100291437767
```

This establishes the sign on the sampled range only.

## United candidate

With 12,000 deterministic full-grid Adam steps, the training loss decreased
from approximately `18.42` to `4.52e-4`. The independent validation reported:

```text
max D U_theta f_full               = +0.023851386365289263
max (D U_theta f_full + U_theta)   = +0.02848055938555221
max (U_theta - V_inf) in B_kappa   = +0.10292857152044554
fraction with selected dV >= 0     = 0.0014111397028305802
min combined candidate             = +1.7769647360975886e-05
```

Thus the sampled candidate remains positive, while the decay and inner
dominance conditions are not satisfied everywhere on the independent grids.
The normal run reports these checks and exits nonzero.

## Additional experiments

The following alternatives were evaluated during reconstruction:

- a Cartesian `100 x 100` grid, which undersampled a thin region near `z2=0`;
- a free output bias in the square map;
- a positive softplus scalar output;
- counterexample-guided continuation;
- constant rescaling of a separately checked analytic candidate.

Counterexample-guided continuation reduced the fresh-grid maximum selected
derivative to approximately `+0.00554`, but did not make it negative.

For the analytic candidate, the outer boundary required a scale of at least
`5.900728431496378`, while the inner region required a scale of at most
`0.24638186248318356`. This shows incompatibility for that candidate and those
sampled constraints; it is not a general impossibility result for all neural
networks.

## Interpretation

The baseline reproduces the structure of the published minimum-gluing method
with explicitly documented numerical choices. It provides a useful reference
for comparing architectures. The recorded run is not presented as a
violation-free Lyapunov certificate for the complete united candidate.
