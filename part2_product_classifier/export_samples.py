"""Export one real PNG from the Fashion-MNIST test split for every class."""

from pathlib import Path

import torchvision
from PIL import Image

from part2_product_classifier.train_classifier import CLASSES, ROOT

OUTPUT_DIR = Path("data/sample_images")


def main() -> list[str]:
    """Write genuine untouched test images and return their paths."""
    dataset = torchvision.datasets.FashionMNIST(ROOT, train=False, download=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    found = {}
    for index in range(len(dataset)):
        image, label = dataset[index]
        if label in found:
            continue
        safe_name = CLASSES[label].replace("/", "-").replace(" ", "_").lower()
        path = OUTPUT_DIR / f"{index:05d}_{safe_name}.png"
        image.save(path)
        found[label] = str(path)
        if len(found) == len(CLASSES):
            break
    print(f"exported={len(found)} sample_images={OUTPUT_DIR}")
    return list(found.values())


if __name__ == "__main__":
    main()
