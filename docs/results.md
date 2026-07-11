# Results

## Ensemble benchmark — UCI Adult Income (imbalanced, SMOTE-corrected)

75.9% / 24.1% class split in the raw data. Training set rebalanced 50/50
via SMOTE; test set left untouched (still imbalanced) so metrics reflect
real-world deployment conditions, not an easier resampled test set.

| Model | Sensitivity | Specificity | Accuracy | Balanced Accuracy |
|---|---:|---:|---:|---:|
| Random Forest (bagging) | 0.664 | 0.916 | 0.856 | 0.790 |
| Gradient Boosting (boosting) | 0.697 | 0.903 | 0.853 | 0.800 |
| **Stacking (RF + GB + DT → LR)** | **0.717** | 0.897 | 0.854 | **0.807** |

Stacking wins on both sensitivity and balanced accuracy — the two metrics
that matter most under class imbalance — while all three models land
within half a point of each other on raw accuracy. If you only looked at
accuracy, you'd conclude the three approaches are basically tied; they
aren't, once the minority class is what you actually care about detecting.

## Threshold tuning — Iris (binary: Virginica vs. rest, noise-injected)

| | Threshold | Sensitivity | Specificity | Accuracy | Balanced Accuracy |
|---|---:|---:|---:|---:|---:|
| Default | 0.50 | 0.80 | 0.85 | 0.83 | 0.825 |
| Balanced-accuracy-optimal | 0.50–0.57* | 0.78–0.82 | 0.85–0.91 | 0.84–0.87 | 0.835–0.84 |

AUC: 0.89.

*Exact optimal threshold is seed-dependent (varies slightly run to run
since it's derived from cross-validated probability estimates on a small,
noise-injected dataset) — the consistent finding across runs is that the
optimal threshold is *not* 0.5, and moving it buys a real, if modest,
balanced-accuracy improvement.

## Why this pairing

Two different points about the same underlying problem — a classifier's
default settings (0.5 threshold, raw accuracy as the headline metric)
are convenient, not correct. The ensemble benchmark shows it at the model
level (which algorithm to pick), the threshold tuning shows it at the
decision-boundary level (how to use whichever model you picked).

## Reproducing

```bash
pip install -r requirements.txt
python -m src.ensemble_income_classifier   # downloads UCI Adult (~4MB), takes ~30s
python -m src.threshold_tuning
python -m pytest tests/ -v                  # add -m "not slow" to skip the network-dependent tests
```
