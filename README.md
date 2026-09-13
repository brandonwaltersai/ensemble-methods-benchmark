# Ensemble Methods Benchmark

![tests](https://github.com/brandonwaltersai/ensemble-methods-benchmark/actions/workflows/tests.yml/badge.svg)

Two focused experiments on a shared theme: the default settings a
classifier ships with (0.5 decision threshold, raw accuracy as the
headline metric) are convenient defaults, not the correct choice for an
imbalanced problem.

## What's here

### Bagging vs. boosting vs. stacking (`src/ensemble_income_classifier.py`)
Random Forest, Gradient Boosting, and a stacked ensemble (RF + GB + Decision
Tree → Logistic Regression), trained on SMOTE-rebalanced data and evaluated
on the untouched, still-imbalanced UCI Adult Income test set (public
dataset, no auth). **Stacking wins on sensitivity and balanced accuracy**
— the metrics that matter under class imbalance — while accuracy alone
would suggest the three approaches are interchangeable. Full numbers:
[`docs/results.md`](docs/results.md).

### Decision threshold tuning (`src/threshold_tuning.py`)
The default 0.5 classification threshold is arbitrary. This finds the
threshold that actually maximizes balanced accuracy via the ROC curve,
on a noise-injected binary Iris classification task, and shows the real
(if modest) improvement over the default.

## A note on scope

An earlier version of this comparison also benchmarked PyCaret (AutoML)
against the ensemble model above, on a different dataset. I dropped that
piece: the source notebook's "ensemble model" comparison numbers turned
out to be a placeholder that was never replaced with real results — the
code had an unresolved `# TODO: Replace the example values` and the two
columns in the comparison table were, on inspection, identical. Rather
than publish a comparison with fabricated numbers, I kept only the two
pieces here that I verified by actually re-running them.

## Running it

```bash
pip install -r requirements.txt
python -m src.ensemble_income_classifier
python -m src.threshold_tuning
python -m pytest tests/ -v
```

## Project structure

```
src/
  ensemble_income_classifier.py   bagging/boosting/stacking benchmark, UCI Adult Income
  threshold_tuning.py             ROC-based threshold optimization, Iris
tests/
  test_benchmark.py               6 tests, including full end-to-end runs against real data
docs/
  results.md                      full results + reproduction instructions
```

## Stack

Python · scikit-learn · imbalanced-learn (SMOTE)

## Author

Brandon Walters — [LinkedIn](https://www.linkedin.com/in/bw172b29208/) · [GitHub](https://github.com/brandonwaltersai)
