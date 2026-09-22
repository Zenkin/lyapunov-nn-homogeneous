# Neural Lyapunov functions with homogeneous approximation

Source code accompanying *A neural network-based stability analysis and
stabilization through homogeneous approximations*, maintained by the authors.
The repository contains both numerical examples, their implementation tests,
and recorded numerical audits.

**All maintained implementations are available on `main`.** Start with the
article implementations to inspect the constructions described in the paper;
the corrected and improved variants are explicitly identified below.

## Guide to the experiments

| Example | Implementation | Relationship to the article | Numerical audit |
| --- | --- | --- | --- |
| 1: stability analysis | [`original`](example_1/original) | Minimum-based gluing from equation (16) | [Audit](example_1/original/AUDIT.md) |
| 1: stability analysis | [`improved`](example_1/improved) | Subsequent positive-definite inner model and smooth level-set gluing | [Audit](example_1/improved/AUDIT.md) |
| 2: pendulum stabilization | [`article_version`](example_2/article_version) | Displayed equations, including the matrix printed in the article | [Audit](example_2/article_version/AUDIT.md) |
| 2: pendulum stabilization | [`corrected_matrix`](example_2/corrected_matrix) | Controlled reruns correcting the linearization, with switching-surface diagnostics | [Audit](example_2/corrected_matrix/AUDIT.md) |
| 2: pendulum stabilization | [`improved`](example_2/improved) | Subsequent smooth periodic local-to-neural extension | [Audit](example_2/improved/AUDIT.md) |

The article does not specify every numerical parameter needed to reconstruct
the original runs. The article implementations therefore document their
additional numerical choices; they are not claimed to reproduce the historical
weights or figures exactly. Corrections and later extensions remain separate
from these implementations.

## Installation

Use **Python 3.12** and the pinned CPU dependencies. The saved reference runs
used Python 3.12.13 on Linux; their full environments are recorded in the
[reproducibility summary](publication/reference/reproducibility_summary.md).

```bash
git clone https://github.com/Zenkin/lyapunov-nn-homogeneous.git
cd lyapunov-nn-homogeneous
python -m venv .venv
```

Activate the environment:

```bash
# Linux shell
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Then install the dependencies for all five implementations:

```bash
python -m pip install -r requirements.txt
```

The pinned `torch==2.8.0+cpu` environment targets Linux and Windows.

## Check the installation

Run all implementation tests from the repository root:

```bash
python -m unittest discover -v
```

For short end-to-end checks of the article implementations:

```bash
python -m example_1.original.example1 --quick --outdir example_1/original/results/quick
python -m example_2.article_version.example2 --quick --outdir example_2/article_version/results/quick
```

Quick runs use reduced training and validation settings. Their metrics are
installation diagnostics, not reference results. Use the
[reproduction guide](docs/REPRODUCIBILITY.md) for full runs, all five smoke
commands, and output locations.

## Inspect the recorded results

The committed records and figures can be reviewed without training a model:

| Material | Location |
| --- | --- |
| Example 1, improved: figures, configuration, metrics, and weights | [`example_1/improved/figures/reference`](example_1/improved/figures/reference) |
| Example 2, article implementation: figures and run record | [`example_2/article_version/figures/reference`](example_2/article_version/figures/reference) |
| Example 2, corrected matrix: switching-surface audit and weights | [`example_2/corrected_matrix/figures/reference/invariance_audit`](example_2/corrected_matrix/figures/reference/invariance_audit) |
| Example 2, improved: figures, run record, and weights | [`example_2/improved/figures/reference`](example_2/improved/figures/reference) |
| Improved examples: training protocol, numerical checks, and environments | [`publication/reference`](publication/reference) |

The documented commands write to ignored `results/` directories within each implementation.
The committed `figures/reference/` directories retain the published records.

## Interpretation of the numerical checks

- **Example 1, original:** the recorded finite-grid audit retains small decay
  and dominance violations. A full run exits nonzero when its audit fails,
  after saving the results.
- **Example 1, improved:** the recorded run has positive candidate values and
  negative directional derivatives at all 54,140 independent validation
  points outside `X`. Its maximum sampled derivative is
  `-0.04414346562709255`.
- **Example 2, article implementation:** the printed linear matrix differs
  from the Jacobian of the displayed nonlinear system. The code preserves that
  matrix and reports the discrepancy. Learned and switched derivatives retain
  sparse sign violations on the boundary-including grid, which overlaps the
  training grid.
- **Example 2, improved:** the recorded `401x401` and `801x801` audits have no
  nonnegative derivative samples in the selected origin-connected component
  `V_NN <= 3.436`. All `525/525` declared trajectories reach the target by
  `t=40`; violations outside the selected component remain visible in the
  condition map.

These are fixed-seed, finite-domain numerical results. They are not
continuous-domain or unbounded-domain certificates. Each implementation's
`AUDIT.md` states its assumptions, checks, and limitations.

## Repository layout

```text
example_1/
  original/          # Article construction
  improved/          # Smooth-gluing extension and recorded results
example_2/
  article_version/   # Printed equations and recorded results
  corrected_matrix/  # Linearization controls and switching diagnostics
  improved/          # Smooth periodic extension and recorded results
publication/         # Tables generated from the committed JSON records
docs/                # Reproduction instructions and branch provenance
requirements.txt     # Installation entry point for all experiments
LICENSE              # MIT license
```

See [branch provenance](docs/PROVENANCE.md) for the preserved history of the
consolidation into `main`. The code is distributed under the [MIT license](LICENSE).
