# Reproducibility

Complete the [setup](../README.md#setup), then run the commands below from the
repository root. No external dataset is needed.

## Saved-checkpoint reproduction

```bash
python reproduction/reproduce.py --outdir results/reproduced
```

On Windows, `START.cmd` performs setup and selects this mode by default.
The run verifies checkpoint hashes, recalculates the grids and trajectories,
and exits nonzero if a numerical regression check fails. The output directory
must not already exist. See the [protocol](../reproduction/README.md) for the
two checkpoint sources and plotting domains.

## Tests and short runs

```bash
python -m unittest discover -v
```

The tests check the equations, derivatives, and local designs. Short runs check
training and output generation with reduced grids and iteration counts:

```bash
python -m example_1.original.example1 --quick --outdir example_1/original/results/quick
python -m example_1.improved.example1 --quick --outdir example_1/improved/results/quick
python -m example_2.article_version.example2 --quick --outdir example_2/article_version/results/quick
python -m example_2.corrected_matrix.example2 --quick --outdir example_2/corrected_matrix/results/quick
python -m example_2.improved.example2 --quick --outdir example_2/improved/results/quick
```

CI runs these checks, saved-checkpoint reproduction, and the summary-table
checks on Linux and Windows.
Short-run metrics are not reference results.

## Full runs

```bash
python -m example_1.original.example1 --outdir example_1/original/results/reference
python -m example_1.improved.example1 --outdir example_1/improved/results/reference
python -m example_2.article_version.example2 --outdir example_2/article_version/results/reference
python -m example_2.corrected_matrix.example2 --outdir example_2/corrected_matrix/results/reference
python -m example_2.improved.example2 --outdir example_2/improved/results/reference
```

The corrected-matrix run creates `matrix_only` and `consistent_local_design`
subdirectories. See its [README](../example_2/corrected_matrix/README.md) for
additional switching and boundary checks.

Example 1 saves its results and exits nonzero if a full-run audit fails. The
original variant has [known violations](../example_1/original/AUDIT.md).
Example 2 reports its checks in `run_record.json`; inspect these separately
from the process exit status.

The improved reference runs use seed `20260820` and CPU PyTorch with one
intra-op and one inter-op thread. Versions and timings are in the
[environment table](../publication/reference/execution_environment.csv).
A fixed seed does not guarantee identical results across platforms.

## Output

- `example_*/<version>/figures/reference/`: saved figures, JSON records, and
  available weights. Example 1's original variant has no saved artifacts;
  Example 2's article variant has figures and a run record but no saved weights.
- `example_*/<version>/results/`: local runs, excluded from Git.
- `publication/reference/`: summary tables for the improved variants.

The run commands also generate numerical arrays for pointwise inspection.

## Summary tables

Regenerate the tables from the committed JSON records without training:

```bash
python publication/make_summary.py --outdir publication/results/reproduced
```

Compare the four output files with [`publication/reference`](../publication/reference).
To summarize new runs, pass `--example-1` and `--example-2` with their JSON
paths. Example 1's `config.json` must be beside its `metrics.json`.
