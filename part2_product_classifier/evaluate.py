"""Evaluate the saved classifier once on the untouched Fashion-MNIST test split."""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets

from part2_product_classifier.predict import CLASSES, TRANSFORM, load_model
from part2_product_classifier.train_classifier import ROOT

MODEL_PATH = Path("models/product_classifier.pt")
REPORT_PATH = Path("reports/product_classifier_evaluation.json")
SAMPLE_DIR = Path("data/sample_images")


def main() -> dict:
    """Evaluate all 10,000 untouched test images and report actual predictions."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved product classifier not found: {MODEL_PATH}")
    test_set = datasets.FashionMNIST(ROOT, train=False, download=True, transform=TRANSFORM)
    raw_test_set = datasets.FashionMNIST(ROOT, train=False, download=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(str(MODEL_PATH), device=device)
    loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)
    predictions, labels = [], []
    with torch.inference_mode():
        for images, batch_labels in loader:
            predictions.extend(model(images.to(device)).argmax(1).cpu().tolist())
            labels.extend(batch_labels.tolist())
    matrix = confusion_matrix(labels, predictions, labels=list(range(10)))
    class_report = classification_report(
        labels, predictions, target_names=CLASSES, output_dict=True, zero_division=0
    )
    off_diagonal = [
        (int(matrix[actual, predicted]), CLASSES[actual], CLASSES[predicted])
        for actual in range(10)
        for predicted in range(10)
        if actual != predicted
    ]
    off_diagonal.sort(reverse=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    exported = []
    seen = set()
    for index, label in enumerate(raw_test_set.targets.tolist()):
        if label in seen:
            continue
        safe_name = CLASSES[label].replace("/", "-").replace(" ", "_").lower()
        path = SAMPLE_DIR / f"{index:05d}_{safe_name}.png"
        raw_test_set[index][0].save(path)
        exported.append(str(path))
        seen.add(label)
        if len(seen) == len(CLASSES):
            break
    result = {
        "test_size": len(test_set),
        "accuracy": float(accuracy_score(labels, predictions)),
        "confusion_matrix": matrix.tolist(),
        "per_class": {name: class_report[name] for name in CLASSES},
        "highest_confusion_pairs": [
            {"count": count, "actual": actual, "predicted": predicted}
            for count, actual, predicted in off_diagonal[:5]
        ],
        "sample_images": exported,
    }
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
