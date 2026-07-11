import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.threshold_tuning import calculate_metrics, find_optimal_threshold, run_comparison
from src.ensemble_income_classifier import evaluate
import numpy as np


def test_ensemble_evaluate_perfect_predictions():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1])
    m = evaluate(y_true, y_pred)
    assert m["sensitivity"] == 1.0
    assert m["balanced_accuracy"] == 1.0


@pytest.mark.slow
def test_run_benchmark_end_to_end():
    """Full pipeline on the real UCI Adult dataset -- downloads ~4MB, network required."""
    from src.ensemble_income_classifier import run_benchmark
    results = run_benchmark(random_state=1)
    assert set(results.keys()) == {"random_forest", "gradient_boosting", "stacking"}
    for metrics in results.values():
        assert 0.0 <= metrics["balanced_accuracy"] <= 1.0


def test_calculate_metrics_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    m = calculate_metrics(y_true, y_pred)
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["accuracy"] == 1.0


def test_calculate_metrics_all_wrong():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    m = calculate_metrics(y_true, y_pred)
    assert m["sensitivity"] == 0.0
    assert m["specificity"] == 0.0


def test_find_optimal_threshold_returns_valid_probability_range():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred_proba = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    threshold, auc = find_optimal_threshold(y_true, y_pred_proba)
    assert 0.0 <= threshold <= 1.0
    assert auc == 1.0  # perfectly separable in this toy example


@pytest.mark.slow
def test_run_comparison_end_to_end():
    """Full pipeline on real Iris data -- slower, but verifies the real thing runs."""
    result = run_comparison(random_state=1)
    assert 0.5 <= result["auc"] <= 1.0
    assert result["optimal_threshold"]["balanced_accuracy"] >= result["default_threshold"]["balanced_accuracy"] - 1e-9
