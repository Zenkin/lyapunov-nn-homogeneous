# Neural Lyapunov functions with homogeneous approximation

Code for *A neural network-based stability analysis and stabilization through
homogeneous approximations*.

## Reproduce the figures

On Windows, double-click **START.cmd** and press Enter. It installs Python 3.12
and the pinned dependencies if needed, then recalculates both improved examples
from preserved trained weights. Training is a separate menu option.

On Linux or an existing Python environment, complete the setup below and run:

```bash
python reproduction/reproduce.py --outdir results/reproduced
```

The run checks the numerical results and saves five figures, arrays, and
`verification.json`. See the [figures and protocol](reproduction/README.md),
or open `RESULTS.html` locally for the saved gallery. Short `--quick` runs only
test installation and do not reproduce these results.

## Examples

| Implementation | Method | Validation |
| --- | --- | --- |
| [Example 1: original](example_1/original) | Minimum-based gluing, equation (16) | [Audit](example_1/original/AUDIT.md) |
| [Example 1: improved](example_1/improved) | Positive-definite inner model with smooth gluing | [Audit](example_1/improved/AUDIT.md) |
| [Example 2: article version](example_2/article_version) | Pendulum stabilization using the printed equations | [Audit](example_2/article_version/AUDIT.md) |
| [Example 2: corrected matrix](example_2/corrected_matrix) | Linearization correction and switching diagnostics | [Audit](example_2/corrected_matrix/AUDIT.md) |
| [Example 2: improved](example_2/improved) | Smooth periodic Lyapunov function and feedback | [Audit](example_2/improved/AUDIT.md) |

`original` and `article_version` implement the article's constructions with
numerical choices documented in each folder. They do not reconstruct the
historical training runs. The corrected and improved versions are subsequent
work.

## Setup

Python 3.12, Linux or Windows, CPU. Clone the repository and enter its directory:

```bash
git clone https://github.com/Zenkin/lyapunov-nn-homogeneous.git
cd lyapunov-nn-homogeneous
```

Create and activate a virtual environment on Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies and run the tests:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -v
```

See [Reproducibility](docs/REPRODUCIBILITY.md) for full and short runs of each
example. The documented commands save new output under `results/`.

## Results

Saved figures, records, and available weights are under `figures/reference/`
in each implementation folder. Example 1's original variant has a textual
audit only. [Summary tables](publication/reference/reproducibility_summary.md)
cover the two improved variants, including their execution environments.

Two pendulum checkpoints are preserved: the [reproduced figures](reproduction/README.md)
use `d22b775` (c=3.466; 343/182 trajectories stay/leave the validation domain).
The later `figures/reference/` checkpoint and the summary tables use `d87c896`
(c=3.436; 344/181). Their records describe separate runs.

The recorded runs of the original implementations retain derivative-sign
violations at sampled points. The
improved variants satisfy the reported sign checks on the stated grids and
regions; these finite-sample results do not establish a continuous-domain
certificate. The audits above give the conditions, discrepancies, and
trajectory checks for each version.

[MIT license](LICENSE) · [Version history](docs/PROVENANCE.md)
