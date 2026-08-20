# Neural Lyapunov functions with homogeneous approximation

This repository contains reproducible implementations of the first numerical
example from *A neural network-based stability analysis and stabilization
through homogeneous approximations*.

The code is organized around two versions of the same idea: learn a Lyapunov
candidate for the homogeneous approximation at infinity, learn a second
candidate on the bounded transition region, and unite the two functions.

| Version | Construction | Purpose |
| --- | --- | --- |
| [`example_1/original`](example_1/original) | Minimum-based gluing from equation (16) | Reference implementation of the published construction |
| [`example_1/improved`](example_1/improved) | Positive-definite inner model with smooth level-set gluing | Numerically improved variant used for the main reproducibility experiment |

Both folders are self-contained. The previous multi-stage package was removed
so that the equations, training loop, validation, and saved results can be read
without following a framework layer.

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

Run the implementation tests:

```bash
python -m unittest example_1.original.test_example1
python -m unittest example_1.improved.test_example1
```

Short smoke runs are available for installation checks:

```bash
python example_1/original/example1.py --quick \
  --outdir example_1/original/results/quick
python example_1/improved/example1.py --quick \
  --outdir example_1/improved/results/quick
```

## Main numerical observation

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

## Reproducibility scope

The paper does not report every numerical setting used to produce its figures.
The dry-friction law, optimizer settings, random seed, grid coordinates, and
the empirical selection of `kappa` are therefore stated explicitly in each
implementation instead of being treated as recovered values.

The minimum-based version is kept as the reference baseline. Its validation
record includes small decay and dominance violations on an independent finite
grid. The improved version replaces the minimum with a smooth level-set
transition and passes the stated finite-grid checks for the recorded seed.

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
```
