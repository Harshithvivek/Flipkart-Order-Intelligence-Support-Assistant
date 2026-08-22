"""Shared utilities for the return-risk pipeline."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "returned"
IDENTIFIER = "order_id"
DEFAULT_DATA_PATH = Path("orders_dataset.csv")
MODEL_PATH = Path("models/return_risk_model.pkl")
METADATA_PATH = Path("models/return_risk_metadata.json")
REPORT_PATH = Path("reports/return_risk_report.json")


def load_orders(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the generated order data and validate its required contract."""
    if not path.exists():
        raise FileNotFoundError(f"Order dataset not found: {path}")
    data = pd.read_csv(path)
    required = {TARGET, IDENTIFIER, "customer_rating"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if len(data) != 6000 or len(data.columns) != 13:
        raise ValueError(f"Expected (6000, 13), got {data.shape}")
    return data


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the target and identifier from model features."""
    return data.drop(columns=[TARGET, IDENTIFIER]), data[TARGET].astype(int)


def feature_columns(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature names in stable column order."""
    categorical = features.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [column for column in features.columns if column not in categorical]
    return numeric, categorical


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build leakage-safe numeric and categorical preprocessing."""
    numeric, categorical = feature_columns(features)
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )


def threshold_metrics(
    y_true: pd.Series, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    """Calculate returned-class metrics at a probability threshold."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def sweep_thresholds(
    y_true: pd.Series, probabilities: np.ndarray
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Sweep thresholds from 0.10 through 0.90 in 0.01 increments."""
    thresholds = np.arange(0.10, 0.901, 0.01)
    results = [threshold_metrics(y_true, probabilities, threshold) for threshold in thresholds]
    best = max(results, key=lambda result: (result["f1"], result["recall"]))
    return best, results


def json_ready(value: Any) -> Any:
    """Convert NumPy scalar values recursively for JSON reports."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
