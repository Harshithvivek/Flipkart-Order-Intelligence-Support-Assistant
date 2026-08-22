"""Evaluate the saved classifier once on untouched Fashion-MNIST test data."""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from part2_product_classifier.utils import CLASS_NAMES, MODEL_PATH, SAMPLE_DIR, load_test_dataset, load_product_classifier


def main(batch_size: int = 128) -> dict:
    """Evaluate test data and export five real test images."""
    dataset = load_test_dataset()
    network, _ = load_product_classifier(MODEL_PATH)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    actual, predicted = [], []
    with torch.inference_mode():
        for images, labels in loader:
            actual.extend(labels.tolist())
            predicted.extend(network(images).argmax(1).tolist())
    matrix = confusion_matrix(actual, predicted, labels=list(range(10)))
    report = classification_report(actual, predicted, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    pairs = []
    for row in range(10):
        for column in range(10):
            if row != column:
                pairs.append((int(matrix[row, column]), CLASS_NAMES[row], CLASS_NAMES[column]))
    pairs.sort(reverse=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for index in [3, 5, 7, 11, 13]:
        image, label = dataset[index]
        # Re-open the original dataset image to export a genuine 28x28 PNG.
        raw_image = load_test_dataset().data[index].numpy()
        from PIL import Image
        Image.fromarray(raw_image).save(SAMPLE_DIR / f"{index:02d}_{CLASS_NAMES[int(label)].replace('/', '_').replace(' ', '_').lower()}.png")
    result = {
        "test_size": len(dataset),
        "accuracy": float(report["accuracy"]),
        "confusion_matrix": matrix.tolist(),
        "per_class": {name: report[name] for name in CLASS_NAMES},
        "highest_confusion_pairs": [
            {"count": count, "actual": actual_name, "predicted": predicted_name}
            for count, actual_name, predicted_name in pairs[:5]
        ],
        "sample_images": [str(path) for path in sorted(SAMPLE_DIR.glob("*.png"))],
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/product_classifier_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
