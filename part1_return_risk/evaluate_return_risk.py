"""Evaluate the persisted Part 1 pipeline on a fresh deterministic split."""

import json
from pathlib import Path

import joblib
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from part1_return_risk.utils import load_orders, split_features_target


def main() -> dict:
    """Load the saved pipeline and report held-out metrics."""
    model_path = Path("models/return_risk_model.pkl")
    if not model_path.exists():
        raise FileNotFoundError(f"Saved return-risk model not found: {model_path}")
    data = load_orders()
    features, target = split_features_target(data)
    _, x_test, _, y_test = train_test_split(
        features, target, test_size=0.20, stratify=target, random_state=42
    )
    model = joblib.load(model_path)
    probabilities = model.predict_proba(x_test)[:, 1]
    result = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "classification_report": classification_report(
            y_test, (probabilities >= 0.5).astype(int), output_dict=True, zero_division=0
        ),
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
