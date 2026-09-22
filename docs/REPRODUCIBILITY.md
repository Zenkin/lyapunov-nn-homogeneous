# Reproducing the experiments

Install the root `requirements.txt` in a Python 3.12 virtual environment as
described in the [README](../README.md). Run every command below from the
repository root. CPU execution is sufficient; no external dataset is required.

## Implementation tests

```bash
python -m unittest discover -v
```

This discovers the test modules in all five implementations, including the
corrected-matrix controls. Tests check the implemented equations, derivatives,
local designs, and related numerical identities. Passing tests do not imply
that a trained candidate satisfies every Lyapunov condition.

## Short installation checks

Each entry point supports `--quick`. Explicit output paths keep quick runs
separate from full runs and from the committed reference records:

```bash
python -m example_1.original.example1 --quick --outdir example_1/original/results/quick
python -m example_1.improved.example1 --quick --outdir example_1/improved/results/quick
python -m example_2.article_version.example2 --quick --outdir example_2/article_version/results/quick
python -m example_2.corrected_matrix.example2 --quick --outdir example_2/corrected_matrix/results/quick
python -m example_2.improved.example2 --quick --outdir example_2/improved/results/quick
```

Quick runs deliberately reduce training and grid sizes; they do not reproduce
the reference metrics. The continuous-integration workflow runs the unit tests
and verifies the publication tables, without retraining the reference models.

## Full runs

Choose the version using the [experiment map](../README.md#guide-to-the-experiments).
These commands use each implementation's recorded default configuration:

```bash
python -m example_1.original.example1 --outdir example_1/original/results/reference
python -m example_1.improved.example1 --outdir example_1/improved/results/reference
python -m example_2.article_version.example2 --outdir example_2/article_version/results/reference
python -m example_2.corrected_matrix.example2 --outdir example_2/corrected_matrix/results/reference
python -m example_2.improved.example2 --outdir example_2/improved/results/reference
```

The corrected-matrix command produces both `matrix_only` and
`consistent_local_design` subdirectories. Its additional switching and
boundary-matching experiments are described in its
[README](../example_2/corrected_matrix/README.md).

Both Example 1 implementations save their output before reporting a failed
full-run audit with a nonzero exit. The original version's known violations
are documented in its [audit](../example_1/original/AUDIT.md). Example 2 records
its numerical checks in `run_record.json`; a successful process exit alone is
not a certificate of stability.

The saved improved reference runs use seed `20260820`, CPU PyTorch, and one
thread. Exact software versions and measured run times are in the
[execution-environment table](../publication/reference/execution_environment.csv).
Floating-point results can differ across platforms. Compare the recorded
configuration and numerical conditions as well as the resulting figures.

## Outputs and committed evidence

| Location | Role |
| --- | --- |
| `example_*/<version>/figures/reference/` | Committed figures, JSON records, and available weights |
| `example_*/<version>/results/` | Ignored local runs, including newly generated arrays and figures |
| `publication/reference/` | Committed tables derived from the improved examples' JSON records |
| `publication/results/` | Ignored local regeneration of those tables |

The repository retains selected reference artifacts rather than every output
of every experiment. In particular, the original Example 1 has a textual
audit, and the article-version Example 2 has figures and a JSON record but no
committed weights. The run commands generate the available detailed numerical
arrays locally. Consult each implementation's README for its precise outputs.

## Regenerate tables without training

This command reads the committed improved-example records using only the
Python standard library:

```bash
python publication/make_summary.py --outdir publication/results/reproduced
```

Compare the four generated files with [`publication/reference`](../publication/reference).
They summarize the **improved** examples, not the literal article constructions.
To compare a new run instead, use `--example-1` and `--example-2` to select its
JSON records; the Example 1 `config.json` must be beside its `metrics.json`.
