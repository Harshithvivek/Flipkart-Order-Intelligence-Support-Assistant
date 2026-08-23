"""Part 2 model and sample artifact tests."""

import json
from pathlib import Path


def test_part2_artifacts_exist():
    assert Path("models/product_classifier.pt").exists()
    training = json.loads(Path("reports/product_classifier_training.json").read_text())
    evaluation = json.loads(Path("reports/product_classifier_evaluation.json").read_text())
    assert training["train_size"] == 55000
    assert training["validation_size"] == 5000
    assert evaluation["test_size"] == 10000
    assert len(evaluation["confusion_matrix"]) == 10
    assert all(len(row) == 10 for row in evaluation["confusion_matrix"])


def test_real_sample_images_exist():
    assert len(list(Path("data/sample_images").glob("*.png"))) >= 5
