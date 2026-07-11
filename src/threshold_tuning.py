"""Decision-threshold tuning: showing that 0.5 is an arbitrary default, not a rule.

Uses sklearn's bundled Iris dataset (binary: Virginica vs. rest), with added
noise to keep it from being a trivially-separable toy problem.
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve,
)
from sklearn.model_selection import cross_val_predict


def calculate_metrics(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def get_predicted_probabilities(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    iris = load_iris()
    X, y = iris.data, iris.target
    y = (y == 2).astype(int)  # binary: Virginica vs. rest

    rng = np.random.default_rng(random_state)
    X = X + rng.normal(0, 0.5, X.shape)  # noise, so it isn't trivially separable

    rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=random_state)
    y_pred_proba = cross_val_predict(rf, X, y, cv=5, method="predict_proba")[:, 1]
    return y, y_pred_proba


def find_optimal_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray) -> tuple[float, float]:
    """Threshold that maximizes balanced accuracy (== maximizing tpr + tnr).
    Returns (optimal_threshold, auc)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    balanced_acc_per_threshold = (tpr + (1 - fpr)) / 2
    optimal_idx = np.argmax(balanced_acc_per_threshold)
    return float(thresholds[optimal_idx]), float(auc)


def run_comparison(random_state: int = 42) -> dict:
    """Compare metrics at the default 0.5 threshold vs. the balanced-accuracy-optimal one."""
    y, y_pred_proba = get_predicted_probabilities(random_state)

    default_metrics = calculate_metrics(y, (y_pred_proba >= 0.5).astype(int))
    optimal_threshold, auc = find_optimal_threshold(y, y_pred_proba)
    optimal_metrics = calculate_metrics(y, (y_pred_proba >= optimal_threshold).astype(int))

    return {
        "auc": auc,
        "default_threshold": {"threshold": 0.5, **default_metrics},
        "optimal_threshold": {"threshold": optimal_threshold, **optimal_metrics},
    }


if __name__ == "__main__":
    result = run_comparison()
    print(f"AUC: {result['auc']:.4f}\n")
    for label in ("default_threshold", "optimal_threshold"):
        m = result[label]
        print(f"{label} (threshold={m['threshold']:.2f}):")
        for k in ("sensitivity", "specificity", "accuracy", "balanced_accuracy"):
            print(f"  {k}: {m[k]:.4f}")
        print()
