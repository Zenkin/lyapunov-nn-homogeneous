# Publication tables

`make_summary.py` converts the committed JSON records of the two improved
examples into compact manuscript tables. It reads saved results only and does
not train, validate, or simulate a model.

From the repository root:

```bash
python publication/make_summary.py
```

The command writes three CSV files and a Markdown preview to
`publication/reference`. Every number in those tables is taken from the JSON
record of the corresponding run.
