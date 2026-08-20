# Neural Lyapunov functions with homogeneous approximation

This is the official source-code repository accompanying the article
*A neural network-based stability analysis and stabilization through
homogeneous approximations*. The repository is maintained by the authors of
the article and contains the public implementations of its numerical
examples.

The code is organized around two versions of the same idea: learn a Lyapunov
candidate for the homogeneous approximation at infinity, learn a second
candidate on the bounded transition region, and unite the two functions.

| Version | Construction | Purpose |
| --- | --- | --- |
| [`example_1/original`](example_1/original) | Minimum-based gluing from equation (16) | Public implementation of the construction presented in the article |
| [`example_1/improved`](example_1/improved) | Positive-definite inner model with smooth level-set gluing | Subsequent modification developed by the authors after revisiting the numerical example |

The literal mathematical core of the nonlinear-pendulum stabilization example
is in [`example_2/article_version`](example_2/article_version). It retains the
linear matrix printed in the article and documents its verified discrepancy
with the Jacobian of the displayed nonlinear model. Its reproducible numerical
run uses explicitly labelled current choices for the unreported values of
`K`, `P`, `kappa`, `epsilon`, and the training hyperparameters.

Both Example 1 folders are self-contained so that the equations, training
loop, validation procedure, and saved results can be inspected without
following a framework layer.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r example_1/improved/requirements.txt
```

Run the improved experiment:

```bash
python example_1/improved/example1.py \
  --outdir example_1/improved/results/reference
```

Run the article-version implementation of Example 2:

```bash
python example_2/article_version/example2.py \
  --outdir example_2/article_version/figures/reference
```

Run the implementation tests:

```bash
python -m unittest example_1.original.test_example1
python -m unittest example_1.improved.test_example1
python -m unittest example_2.article_version.test_example2
```

Short smoke runs are available for installation checks:

```bash
python example_1/original/example1.py --quick \
  --outdir example_1/original/results/quick
python example_1/improved/example1.py --quick \
  --outdir example_1/improved/results/quick
```

## Current numerical result

For the recorded improved run, the independent validation set contained
54,140 points outside the compact set `X`. The smallest sampled candidate value
was positive and the largest sampled directional derivative was negative:

```text
min V_smooth                     = +0.012761223688969375
max D V_smooth f                 = -0.04414346562709255
nonpositive-value fraction       = 0
nonnegative-derivative fraction  = 0
```

The corresponding configuration, validation design, and limitations are
documented in [`example_1/improved/AUDIT.md`](example_1/improved/AUDIT.md).

For the recorded Example 2 implementation, the boundary-including `201 x 201`
grid found sparse derivative-sign violations. This grid contains the midpoint
training grid, so it is not an independent sample. The violations are retained
in the published figures and numerical arrays; see
[`example_2/article_version/AUDIT.md`](example_2/article_version/AUDIT.md).

## Implementation and validation notes

The article does not fix every low-level numerical setting needed for a public
software implementation. For this release, the dry-friction law, optimizer
settings, random seed, grid coordinates, and empirical selection of `kappa`
are therefore stated explicitly in the code and saved configuration files.

The `original` version follows the minimum-based construction presented in the
article. Its current validation record includes small decay and dominance
violations on an independent finite grid. We retain and report this result as
part of the numerical audit. The `improved` version replaces the minimum with
a smooth level-set transition and passes the stated finite-grid checks for the
recorded seed.

These computations are numerical evidence on finite domains. They are not a
continuous-domain or unbounded-domain certificate.

## Repository layout

```text
example_1/
  README.md
  original/
    example1.py
    test_example1.py
    README.md
    AUDIT.md
    requirements.txt
  improved/
    example1.py
    test_example1.py
    README.md
    AUDIT.md
    requirements.txt
example_2/
  README.md
  article_version/
    example2.py
    test_example2.py
    README.md
    AUDIT.md
    requirements.txt
  corrected_matrix/
    example2.py
    invariance_audit.py
    switching_audit.py
    boundary_matching.py
    test_example2.py
    README.md
    AUDIT.md
  improved/
    example2.py
    test_example2.py
    README.md
    AUDIT.md
    requirements.txt
```
