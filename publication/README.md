# Summary tables

`make_summary.py` reads the committed JSON records of the two improved
examples. From the repository root:

```bash
python publication/make_summary.py --outdir publication/results/reproduced
```

The output is three CSV tables and a Markdown summary. Compare them with
[`reference`](reference). Omitting `--outdir` overwrites those reference tables.
No model training is performed.
