"""Bagging vs. boosting vs. stacking on the UCI Adult Income dataset (imbalanced, SMOTE-corrected).

Public dataset, no auth: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
           "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
           "hours-per-week", "native-country", "income"]


def load_data(url: str = ADULT_URL) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(url, names=COLUMNS, na_values=" ?", skipinitialspace=True)
    data = data.dropna()
    X = data.drop("income", axis=1)
    y = data["income"].apply(lambda v: 1 if v == ">50K" else 0)
    return pd.get_dummies(X), y


def evaluate(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def run_benchmark(random_state: int = 42) -> dict:
    """Train Random Forest (bagging), Gradient Boosting (boosting), and a stacked
    ensemble on SMOTE-resampled training data; evaluate all three on the untouched
    (still-imbalanced) test set. Returns {model_name: metrics}."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    X_train_res, y_train_res = SMOTE(random_state=random_state).fit_resample(X_train, y_train)

    results = {}

    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf.fit(X_train_res, y_train_res)
    results["random_forest"] = evaluate(y_test, rf.predict(X_test))

    gb = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
    gb.fit(X_train_res, y_train_res)
    results["gradient_boosting"] = evaluate(y_test, gb.predict(X_test))

    stack = StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=50, random_state=random_state)),
            ("gb", GradientBoostingClassifier(n_estimators=50, random_state=random_state)),
            ("dt", DecisionTreeClassifier(random_state=random_state)),
        ],
        final_estimator=LogisticRegression(),
        cv=5,
    )
    stack.fit(X_train_res, y_train_res)
    results["stacking"] = evaluate(y_test, stack.predict(X_test))

    return results


if __name__ == "__main__":
    for name, metrics in run_benchmark().items():
        print(f"\n{name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
