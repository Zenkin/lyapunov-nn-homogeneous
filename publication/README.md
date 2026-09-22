# Publication tables

`make_summary.py` converts the committed JSON records of the two improved
examples into compact manuscript tables. It reads saved results only and does
not train, validate, or simulate a model.

From the repository root:

```bash
python publication/make_summary.py --outdir publication/results/reproduced
```

The command writes three CSV files and a Markdown preview to
`publication/results/reproduced`, leaving the committed `publication/reference`
tables available for comparison. Omitting `--outdir` explicitly regenerates the
committed table paths. The tables describe the improved implementations; see
the root README for the article implementations and their separate audits.
