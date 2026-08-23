"""Part 1 artifact and dataset contract tests."""

import json
from pathlib import Path

import joblib
import pandas as pd


def test_dataset_contract():
    data = pd.read_csv("orders_dataset.csv")
    assert data.shape == (6000, 13)
    assert data["returned"].isin([0, 1]).all()


def test_saved_return_risk_pipeline_and_threshold():
    model = joblib.load("models/return_risk_model.pkl")
    metadata = json.loads(Path("models/return_risk_metadata.json").read_text())
    assert hasattr(model, "predict_proba")
    assert 0.1 <= metadata["rf_threshold"] <= 0.9
