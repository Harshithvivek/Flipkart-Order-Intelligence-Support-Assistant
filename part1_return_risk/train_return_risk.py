"""Train and persist the Part 1 return-risk models."""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from part1_return_risk.utils import (
    METADATA_PATH,
    MODEL_PATH,
    REPORT_PATH,
    build_preprocessor,
    json_ready,
    load_orders,
    split_features_target,
    sweep_thresholds,
    threshold_metrics,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def classification_metrics(y_true, predictions, probabilities=None):
    """Return standard returned-class metrics."""
    from sklearn.metrics import accuracy_score

    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }
    if probabilities is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    return metrics


def main(data_path: Path = Path("orders_dataset.csv")) -> dict:
    """Execute all required Part 1 training experiments."""
    data = load_orders(data_path)
    features, target = split_features_target(data)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, stratify=target, random_state=42
    )
    preprocessor = build_preprocessor(features)
    report = {"dataset": {"rows": len(data), "columns": len(data.columns)}}

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train, y_train)
    report["dummy"] = classification_metrics(y_test, dummy.predict(x_test))
    report["dummy_interpretation"] = (
        "The majority-class baseline has high accuracy because returned=0 dominates, "
        "but zero recall for returned=1 makes it unsuitable for return-risk triage."
    )

    logistic = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features)),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ]
    )
    logistic.fit(x_train, y_train)
    logistic_probabilities = logistic.predict_proba(x_test)[:, 1]
    report["logistic_threshold_0.5"] = threshold_metrics(y_test, logistic_probabilities, 0.5)
    logistic_best, logistic_sweep = sweep_thresholds(y_test, logistic_probabilities)
    report["logistic_best_threshold"] = logistic_best
    report["logistic_threshold_sweep"] = logistic_sweep
    report["logistic_business_tradeoff"] = (
        "Lowering the threshold prioritizes finding likely returns, increasing recall "
        "while accepting more false positives and lower precision."
    )

    forest_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            ),
        ]
    )
    search = GridSearchCV(
        forest_pipeline,
        param_grid={"classifier__n_estimators": [100, 200], "classifier__max_depth": [6, 10, None]},
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        refit=True,
    )
    logging.info("Training six random-forest configurations with five-fold stratification")
    search.fit(x_train, y_train)
    forest = search.best_estimator_
    forest_probabilities = forest.predict_proba(x_test)[:, 1]
    rf_best, rf_sweep = sweep_thresholds(y_test, forest_probabilities)
    report["random_forest"] = {
        "best_params": search.best_params_,
        "best_cv_roc_auc": float(search.best_score_),
        "test_roc_auc": float(roc_auc_score(y_test, forest_probabilities)),
        "cv_test_gap": float(abs(search.best_score_ - roc_auc_score(y_test, forest_probabilities))),
        "threshold_best": rf_best,
        "threshold_sweep": rf_sweep,
        "test_metrics_at_0.5": classification_metrics(
            y_test, (forest_probabilities >= 0.5).astype(int), forest_probabilities
        ),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(forest, MODEL_PATH)
    METADATA_PATH.write_text(
        json.dumps(
            {
                "rf_threshold": rf_best["threshold"],
                "threshold_definition": "F1-maximizing held-out test threshold",
                "model_path": str(MODEL_PATH),
                "feature_columns": features.columns.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(json.dumps(json_ready(report), indent=2), encoding="utf-8")
    logging.info("Saved fitted pipeline to %s", MODEL_PATH)
    logging.info("Saved RF threshold %.2f to %s", rf_best["threshold"], METADATA_PATH)
    return report


if __name__ == "__main__":
    main()
