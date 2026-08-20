# Original minimum-gluing construction

This folder is a self-contained reference implementation of the construction
used in equations (9), (11), (12), and (16) of the paper. It combines an
`r=(1,2)` homogeneous candidate at infinity with a one-hidden-layer square
candidate on `B_(2 kappa) \ X` through a pointwise minimum.

The folder represents the published method, not recovered historical source
code. Numerical choices that are absent from the paper are listed below and
written to `config.json` on every run.

## Run

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r example_1/original/requirements.txt

python example_1/original/example1.py \
  --outdir example_1/original/results/reference
python -m unittest example_1.original.test_example1
```

For a short installation check:

```bash
python example_1/original/example1.py --quick \
  --outdir example_1/original/results/quick
```

The quick mode is a smoke test only. A normal run applies the full validation
checks and exits nonzero if any of them is not satisfied.

## System

With `x=(x1,x2)` and `x2` denoting velocity,

```text
x1_dot = x2
x2_dot = -phi(x2) + a1 (x1-c1)
         - a2 sqrt(abs(x2)) x2 - a3 (x1-c2)^3
```

The paper specifies

```text
a1=1, a2=2, a3=1, c1=1, c2=2.
```

The bounded dry-friction law used here is taken from the repository's earlier
`dev_w` implementation:

```text
phi(v) = 0.8 tanh(v/0.5).
```

The equilibrium is computed in the original coordinate by solving

```text
a1 (x_eq-c1) - a3 (x_eq-c2)^3 = 0,
```

which gives `x_eq=3.324717957244746...`. The neural networks use the shifted
coordinate `z=(x1-x_eq,x2)`, and the compact set is `X=[-0.5,0.5]^2` in `z`.

## Homogeneous candidate

The approximation at infinity is

```text
f_inf(z) = (z2, -a3 z1^3 - a2 sqrt(abs(z2)) z2).
```

The implementation uses the homogeneous gauge obtained from Definition 4 with
`varpi=2`:

```text
rho(z) = sqrt(z1^2 + abs(z2)),
y = (z1/rho, z2/rho^2),
V_inf(z) = rho(z)^2 W_theta(y).
```

The application paragraph of the paper writes the equivalent homogeneous
gauge `|z1|+sqrt(|z2|)`. Both scale linearly under
`diag(lambda,lambda^2)`, but they define different unit spheres. The selected
gauge is therefore an explicit numerical choice rather than a claim about the
unavailable source code.

The loss is applied to the directional derivative of the complete homogeneous
extension `V_inf`. This includes the chain rule through `rho` and `y`; the raw
quantity `D W_theta(y) f_inf(y)` is recorded separately for comparison.

## Inner candidate and minimum gluing

The inner model follows the square architecture

```text
T_theta(z) = W2 (tanh(W1 z+b1) - tanh(b1)),
U_theta(z) = ||T_theta(z)||^2.
```

The scalar factor `chi` from equation (16) is absorbed into `W2`. The final
candidate is

```text
V(z) = V_inf(z)                   outside B_(2 kappa),
V(z) = min(V_inf(z), U_theta(z))  on B_(2 kappa) \ X.
```

Training includes decay, positivity, inner-dominance, and outer-boundary
dominance terms. The grid is uniform in homogeneous level and sphere angle;
the paper reports `100 x 100` points but does not specify the grid coordinates.

## Validation note

The homogeneous outer candidate satisfies the sampled positivity and decay
conditions. On the independent validation grids, the united baseline retains
a small set of decay violations and does not satisfy the inner-dominance
condition everywhere. The exact values and the tested alternatives are kept in
[`AUDIT.md`](AUDIT.md).

The baseline remains useful because it isolates the effect of the published
minimum-gluing rule and provides a direct comparison for the improved version.
