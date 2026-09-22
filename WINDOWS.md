# Windows launcher

Double-click `START.cmd` and press Enter. The launcher installs dependencies
if needed, loads the preserved trained networks, recalculates both examples,
checks the numerical results, and opens five figures. No training is needed
to reproduce this set. Each run is saved under a new `results/` directory.

Menu option 3 opens the verified figures already on disk, without setup.
Option 2 trains new networks; these runs can differ from the preserved weights.
Option 7 is only an installation test: its undertrained models are unsuitable
for the paper. See `reproduction/README.md` for checkpoint provenance.

Requirements: Windows x64, internet on first setup, several GB of free disk
space. Python 3.12 is detected automatically; if absent, the launcher installs
Python 3.12.10 for the current user from python.org and checks its signature.
Existing Python installations and the system PATH are left unchanged.

The virtual environment is created outside this folder:

```text
%LOCALAPPDATA%\ArticleEfimov\lyapunov-nn-homogeneous\.env
```

This keeps machine-specific libraries outside the project. Each computer creates
its own environment. Later runs reuse it when all pinned dependencies import
successfully. The PyTorch download is verified by SHA-256.

From a terminal:

```bat
START.cmd setup
START.cmd check
START.cmd reproduce
START.cmd quick all
START.cmd full example1-improved
START.cmd full example2-improved
START.cmd full all
START.cmd figures
START.cmd tables
```

Experiments: `example1-original`, `example1-improved`, `example2-article`,
`example2-corrected`, `example2-improved`, or `all`.

New runs have separate timestamped folders under `results/`, containing logs,
figures, arrays, model outputs, and `run-summary.json`. Existing reference
records are preserved. Full runs can take several minutes per experiment.
The original Example 1 has known numerical audit violations: full mode keeps
its outputs, continues other experiments, and returns exit code 2 for that
reported condition. Exit code 1 indicates a command or setup failure.

`RESULTS.html` displays the five recalculated figures. `REPOSITORY_RESULTS.html`
retains the previous gallery, including intermediate variants and the later
pendulum checkpoint. Their original records remain unchanged.

Checkpoint reproduction checks model hashes before loading and verifies the
numerical results before updating the gallery. If any check fails, the launcher
returns a nonzero code and preserves the previous verified gallery.

`START.cmd tables` rebuilds the later repository checkpoint's tables (c=3.436).
The restored c=3.466 figures have a separate complete `verification.json`.
These records describe different experiments and must not be interchanged.

Python installer documentation: https://docs.python.org/3.12/using/windows.html
