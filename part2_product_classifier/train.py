"""Train a CPU-friendly transfer-learning Fashion-MNIST classifier."""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from part2_product_classifier.utils import (
    ARTIFACT_DIR,
    CLASS_NAMES,
    MODEL_PATH,
    build_resnet18,
    load_train_dataset,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def cache_features(network, dataset, indices, cache_path, batch_size, device):
    """Extract frozen-backbone features for a subset and persist them."""
    if cache_path.exists():
        logging.info("Loading feature cache %s", cache_path)
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0)
    network.eval()
    feature_batches, label_batches = [], []
    with torch.inference_mode():
        for images, labels in loader:
            feature_batches.append(network(images.to(device)).flatten(1).cpu())
            label_batches.append(labels.cpu())
    cached = {"features": torch.cat(feature_batches), "labels": torch.cat(label_batches)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cached, cache_path)
    return cached


def train_head(cached, epochs, learning_rate, batch_size, device):
    """Train a small classifier over cached 512-dimensional backbone vectors."""
    head = nn.Linear(cached["features"].shape[1], len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(cached["features"], cached["labels"]), batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        head.train()
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_function(head(features), labels)
            loss.backward()
            optimizer.step()
        logging.info("head epoch=%d/%d loss=%.4f", epoch + 1, epochs, float(loss))
    return head.cpu()


def accuracy(network, dataset, indices, batch_size):
    """Measure accuracy on a train or validation subset."""
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0)
    correct = total = 0
    network.eval()
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(next(network.parameters()).device), labels.to(next(network.parameters()).device)
            correct += int((network(images).argmax(1) == labels).sum())
            total += len(labels)
    return correct / total


def main(epochs: int = 5, batch_size: int = 128, learning_rate: float = 1e-3) -> dict:
    """Train on train/validation only and save the reusable model checkpoint."""
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("training device=%s", device)
    dataset = load_train_dataset()
    all_indices = np.arange(len(dataset))
    labels = np.asarray(dataset.targets)
    train_indices, validation_indices = train_test_split(
        all_indices, test_size=5000, stratify=labels, random_state=42
    )
    logging.info("Fashion-MNIST train=%d validation=%d test=10000 untouched", len(train_indices), len(validation_indices))

    network = build_resnet18(pretrained=True).to(device)
    backbone = nn.Sequential(*list(network.children())[:-1]).to(device)
    train_cache = cache_features(backbone, dataset, train_indices, ARTIFACT_DIR / "fashion_train_features.pt", batch_size, device)
    validation_cache = cache_features(backbone, dataset, validation_indices, ARTIFACT_DIR / "fashion_validation_features.pt", batch_size, device)
    head = train_head(train_cache, epochs, learning_rate, batch_size, device)

    with torch.inference_mode():
        validation_predictions = head(validation_cache["features"]).argmax(1)
    before_accuracy = float((validation_predictions == validation_cache["labels"]).float().mean())
    logging.info("feature extraction validation accuracy=%.4f", before_accuracy)

    fine_tuned = False
    after_accuracy = before_accuracy
    if before_accuracy < 0.80:
        logging.info("Validation accuracy below 0.80; fine-tuning late ResNet layers")
        network.fc = head.to(device)
        for parameter in network.layer4.parameters():
            parameter.requires_grad = True
        optimizer = torch.optim.Adam(
            [parameter for parameter in network.parameters() if parameter.requires_grad], lr=1e-4
        )
        loss_function = nn.CrossEntropyLoss()
        loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True, num_workers=0)
        for epoch in range(max(1, epochs // 2)):
            network.train()
            for images, labels_batch in loader:
                images, labels_batch = images.to(device), labels_batch.to(device)
                optimizer.zero_grad()
                loss = loss_function(network(images), labels_batch)
                loss.backward()
                optimizer.step()
            logging.info("fine-tune epoch=%d loss=%.4f", epoch + 1, float(loss))
        after_accuracy = float(accuracy(network, dataset, validation_indices, batch_size))
        fine_tuned = True
    else:
        network.fc = head.to(device)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": network.state_dict(),
            "class_names": CLASS_NAMES,
            "architecture": "resnet18",
            "pretrained_backbone": True,
            "image_size": 224,
            "imagenet_mean": [0.485, 0.456, 0.406],
            "imagenet_std": [0.229, 0.224, 0.225],
        },
        MODEL_PATH,
    )
    result = {
        "train_size": len(train_indices),
        "validation_size": len(validation_indices),
        "test_size": 10000,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss",
        "device": str(device),
        "frozen_backbone": not fine_tuned,
        "fine_tuned_late_layers": fine_tuned,
        "validation_accuracy_before_fine_tuning": before_accuracy,
        "validation_accuracy_after_fine_tuning": after_accuracy,
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/product_classifier_training.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logging.info("Saved classifier to %s", MODEL_PATH)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    arguments = parser.parse_args()
    main(arguments.epochs, arguments.batch_size, arguments.learning_rate)
