# Improved smooth-gluing construction

This folder contains a subsequent modification developed by the authors after
revisiting the first numerical example. It preserves the homogeneous outer
candidate and the central sphere-learning principle of the article, while
changing two parts of the bounded-region construction:

1. the inner function is positive definite by parameterization;
2. the pointwise minimum is replaced by a smooth transition between the level
   sets `V_inf=kappa` and `V_inf=2 kappa`.

The complete candidate is differentiated during training, including the
chain-rule term introduced by the transition weight. This version is an
author-developed extension; it is not the minimum-gluing algorithm printed in
the article.

## Recorded result

The independent validation set contains 54,140 points outside `X`, including
level-angle points, Cartesian points, and points approaching the nonsmooth
velocity axis. For the recorded seed:

```text
min U_theta                       = +0.012761223688969375
min V_smooth                      = +0.012761223688969375
max D V_smooth f                  = -0.04414346562709255
nonpositive-value fraction        = 0
nonnegative-derivative fraction   = 0
```

Detailed grids, formula checks, and limitations are listed in
[`AUDIT.md`](AUDIT.md).

## Run

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r example_1/improved/requirements.txt

python example_1/improved/example1.py \
  --outdir example_1/improved/results/reference
python -m unittest example_1.improved.test_example1
```

For a short installation check:

```bash
python example_1/improved/example1.py --quick \
  --outdir example_1/improved/results/quick
```

The run saves `config.json`, `metrics.json`, model weights, validation arrays,
and the validation figure. A normal run exits nonzero if any finite-grid audit
condition is not satisfied.

## System and homogeneous candidate

The system parameters are

```text
a1=1, a2=2, a3=1, c1=1, c2=2,
phi(v)=0.8 tanh(v/0.5).
```

The equilibrium is solved in the original coordinate and equals
`x_eq=3.324717957244746...`. The networks use `z=(x1-x_eq,x2)`.

For `r=(1,2)` and `mu=2`,

```text
rho(z) = sqrt(z1^2 + abs(z2)),
y = (z1/rho, z2/rho^2),
V_inf(z) = rho(z)^2 W_theta(y).
```

This is the `varpi=2` gauge from Definition 4. The application paragraph of
the article writes another equivalent `r`-homogeneous gauge,
`|z1|+sqrt(|z2|)`. The gauge used by the experiment is stored as an explicit
numerical choice.

The homogeneous loss differentiates the complete extension `V_inf`; it does
not substitute the ambient derivative of the sphere network for the derivative
of the extended function.

## Positive-definite inner model

Let `s(z)=(z1/R1,z2/R2)`. The residual network is anchored at zero:

```text
R_theta(z) = W2 (tanh(W1 s(z)+b1) - tanh(b1)).
```

The inner candidate is

```text
U_theta(z) = epsilon_q ||s(z)||^2 + ||R_theta(z)||^2,
epsilon_q = 0.001.
```

Therefore `U_theta(0)=0` and `U_theta(z)>0` for every `z!=0`, independently of
the trainable weights.

## Smooth level-set gluing

Set `q(z)=V_inf(z)` and

```text
t(z) = clip((q(z)-kappa)/kappa, 0, 1),
sigma(t) = 6 t^5 - 15 t^4 + 10 t^3.
```

The united candidate is

```text
V_smooth(z) = (1-sigma(t(z))) U_theta(z) + sigma(t(z)) V_inf(z).
```

It equals `U_theta` for `V_inf<=kappa`, equals `V_inf` for
`V_inf>=2 kappa`, and is a convex combination in the transition region.
The zero endpoint derivatives of `sigma` remove a derivative jump at both
joining level sets wherever the component functions are differentiable.

The expanded directional derivative is

```text
D V_smooth f = (1-sigma) D U_theta f + sigma D V_inf f
             + (d sigma/d q) D V_inf f (V_inf-U_theta).
```

The code compares this formula against direct automatic differentiation.
Training uses the complete expression

```text
relu(D V_smooth f_full + 0.05).
```

No dominance penalty or separate scale `chi` is needed.

## Scope

The smooth gate, fixed quadratic coefficient, seed, optimizer settings, grid
coordinates, and empirical `kappa` are choices made for this public
implementation. The validation is finite-grid numerical evidence for one
seed, not a continuous-domain or unbounded-domain proof.
