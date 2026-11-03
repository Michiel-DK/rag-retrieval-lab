# predictions/

`ragexp run` writes into the committed `results/` only after the run's prediction is committed here.
One file per run name: `predictions/<run name>.md`.

Required: two non-empty level-2 sections.

```markdown
# <run name>

## Prediction
What the number will do, against which baseline, on which metric. Say "unsure" if you are.

## Mechanism
Why. The argument that makes the prediction, written before the result exists.
```

The gate checks that the file is tracked by git and has no uncommitted edits, then records the commit.
A run redirected with `RAGEXP_RESULTS` is exploratory: no gate, and it is not evidence.

The runs from notebooks 01 to 07 predate this gate. Their predictions live in the notebooks.
