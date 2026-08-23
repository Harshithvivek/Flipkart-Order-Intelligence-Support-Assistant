"""Verify the required Fashion-MNIST split sizes without touching test labels."""

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import FashionMNIST

from part2_product_classifier.train_classifier import ROOT, SEED


if __name__ == "__main__":
    train = FashionMNIST(ROOT, train=True, download=True)
    test = FashionMNIST(ROOT, train=False, download=True)
    indices = np.arange(len(train))
    train_indices, validation_indices = train_test_split(
        indices, test_size=5000, stratify=np.asarray(train.targets), random_state=SEED
    )
    print(f"full_train={len(train)} train={len(train_indices)} validation={len(validation_indices)} test={len(test)}")
    assert (len(train), len(train_indices), len(validation_indices), len(test)) == (60000, 55000, 5000, 10000)
    print("split_check=PASS")
