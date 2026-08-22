"""Agent tools backed by the saved Part 1 and Part 2 artifacts."""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from part2_product_classifier.predict import predict_product_image

MODEL_PATH = Path("models/return_risk_model.pkl")
THRESHOLD_PATH = Path("models/return_risk_metadata.json")


def check_return_risk(order_features: dict) -> dict:
    """Predict return risk using the persisted fitted pipeline and RF threshold."""
    if not isinstance(order_features, dict) or not order_features:
        raise ValueError("order_features must be a non-empty dictionary")
    if not MODEL_PATH.exists() or not THRESHOLD_PATH.exists():
        raise FileNotFoundError("Return-risk model or threshold metadata is missing")
    model = joblib.load(MODEL_PATH)
    threshold = float(json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))["rf_threshold"])
    probability = float(model.predict_proba(pd.DataFrame([order_features]))[0, 1])
    if probability < threshold:
        bucket = "Low"
    elif probability < threshold + 0.15:
        bucket = "Medium"
    else:
        bucket = "High"
    return {"return_probability": probability, "risk_bucket": bucket}


def classify_product_image(image_path: str) -> dict[str, Any]:
    """Call the reusable Part 2 image prediction function on a real PNG."""
    return predict_product_image(image_path)
