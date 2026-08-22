"""Analyze missingness, feature importance, and model subgroups."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

from part1_return_risk.utils import load_orders, split_features_target


def main() -> dict:
    """Produce actual Part 1 analysis outputs from the saved pipeline."""
    data = load_orders()
    features, target = split_features_target(data)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, stratify=target, random_state=42
    )
    model = joblib.load(Path("models/return_risk_model.pkl"))
    classifier = model.named_steps["classifier"]
    transformed_names = model.named_steps["preprocessor"].get_feature_names_out()
    impurity = pd.Series(classifier.feature_importances_, index=transformed_names).sort_values(ascending=False)
    impurity_original = pd.Series(
        {
            feature: impurity[
                impurity.index.to_series().apply(
                    lambda name: name.startswith(f"numeric__{feature}")
                    or name.startswith(f"categorical__{feature}_")
                ).to_numpy()
            ].sum()
            for feature in features.columns
        }
    ).sort_values(ascending=False)
    permutation = permutation_importance(
        model, x_test, y_test, scoring="roc_auc", n_repeats=10, random_state=42, n_jobs=-1
    )
    permutation_series = pd.Series(permutation.importances_mean, index=features.columns).sort_values(ascending=False)
    probabilities = model.predict_proba(x_test)[:, 1]
    threshold = json.loads(Path("models/return_risk_metadata.json").read_text(encoding="utf-8"))["rf_threshold"]
    predictions = (probabilities >= threshold).astype(int)
    subgroup_data = x_test.copy()
    subgroup_data["actual"] = y_test.to_numpy()
    subgroup_data["prediction"] = predictions

    def subgroup_metrics(column: str) -> dict:
        rows = []
        for name, group in subgroup_data.groupby(column):
            rows.append(
                {
                    column: str(name),
                    "count": int(len(group)),
                    "precision": float(precision_score(group.actual, group.prediction, zero_division=0)),
                    "recall": float(recall_score(group.actual, group.prediction, zero_division=0)),
                }
            )
        return rows

    report = {
        "missingness": {
            "classification": "MAR",
            "reason": "customer_rating missingness depends on observed payment_method",
            "cod_rate": float(data.loc[data.payment_method.eq("COD"), "customer_rating"].isna().mean()),
            "non_cod_rate": float(data.loc[~data.payment_method.eq("COD"), "customer_rating"].isna().mean()),
        },
        "impurity_top_5": impurity.head(5).to_dict(),
        "impurity_top_5_original_features": impurity_original.head(5).to_dict(),
        "permutation_top_5_original_features": permutation_series.head(5).to_dict(),
        "importance_comparison": {
            "impurity_features_not_in_permutation_top_5": sorted(
                set(impurity_original.head(5).index) - set(permutation_series.head(5).index)
            ),
            "interpretation": (
                "Impurity importance can overrate noisy continuous variables because tree splits "
                "reuse many candidate cut points; held-out permutation importance measures the "
                "drop in predictive performance after shuffling an original feature."
            ),
        },
        "subgroups": {
            "product_category": subgroup_metrics("product_category"),
            "payment_method": subgroup_metrics("payment_method"),
        },
        "recommended_subgroup_remedy": (
            "Use payment-method-specific probability calibration or thresholds, and add a "
            "historical payment-behavior feature so non-COD cases are not governed by the COD-heavy split."
        ),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/return_risk_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
