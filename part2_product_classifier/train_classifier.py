"""Part 2 Fashion-MNIST transfer learning pipeline.

Stage A caches frozen ResNet-18 features and trains a small head. Stage B
fine-tunes only late layers when Stage A validation accuracy is below 0.90.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torchvision
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

SEED = 42
IMAGE_SIZE = 224
TARGET_VALIDATION_ACCURACY = 0.90
CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
ROOT = Path("data/fashion_mnist")
MODEL_PATH = Path("models/product_classifier.pt")
ARTIFACT_DIR = Path("artifacts")
REPORT_PATH = Path("reports/product_classifier_training.json")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def plain_transform() -> transforms.Compose:
    """Convert grayscale images to normalized 3-channel ResNet inputs."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def augmented_transform() -> transforms.Compose:
    """Apply light augmentation for the optional late-layer fine-tuning stage."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def build_backbone(pretrained: bool = True) -> nn.Module:
    """Build ResNet-18 with a 512-dimensional feature output."""
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    return model


@torch.inference_mode()
def cache_features(dataset, indices, cache_path, device):
    """Cache frozen backbone features and labels for a dataset subset."""
    if cache_path.exists():
        logging.info("Loading feature cache: %s", cache_path)
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    backbone = build_backbone(pretrained=True).to(device).eval()
    loader = DataLoader(Subset(dataset, indices), batch_size=512, shuffle=False, num_workers=0)
    features, labels = [], []
    for images, batch_labels in loader:
        features.append(backbone(images.to(device)).cpu())
        labels.append(batch_labels)
    cached = {"features": torch.cat(features), "labels": torch.cat(labels)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cached, cache_path)
    return cached


def evaluate_head(head, cached, device):
    """Evaluate a classifier head over cached validation features."""
    head.eval()
    with torch.inference_mode():
        predictions = head(cached["features"].to(device)).argmax(1).cpu().numpy()
    return float(accuracy_score(cached["labels"].numpy(), predictions))


def evaluate_model(model, dataset, indices, device):
    """Evaluate a full model on a specified split."""
    loader = DataLoader(Subset(dataset, indices) if indices is not None else dataset, batch_size=512, shuffle=False, num_workers=0)
    predictions, labels = [], []
    model.eval()
    with torch.inference_mode():
        for images, batch_labels in loader:
            predictions.extend(model(images.to(device)).argmax(1).cpu().tolist())
            labels.extend(batch_labels.tolist())
    return np.asarray(predictions), np.asarray(labels)


def main() -> dict:
    """Train, optionally fine-tune, and save the classifier state dictionary."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quick = os.environ.get("QUICK_TEST") == "1"
    logging.info("device=%s quick_test=%s", device, quick)

    train_full = torchvision.datasets.FashionMNIST(ROOT, train=True, download=True, transform=plain_transform())
    test_set = torchvision.datasets.FashionMNIST(ROOT, train=False, download=True, transform=plain_transform())
    labels = np.asarray(train_full.targets)
    indices = np.arange(len(train_full))
    train_indices, validation_indices = train_test_split(indices, test_size=5000, stratify=labels, random_state=SEED)
    test_indices = np.arange(len(test_set))
    if quick:
        rng = np.random.default_rng(SEED)
        train_indices = rng.choice(train_indices, 4000, replace=False)
        validation_indices = rng.choice(validation_indices, 1500, replace=False)
        test_indices = rng.choice(test_indices, 2000, replace=False)
    logging.info("splits train=%d validation=%d test=%d", len(train_indices), len(validation_indices), len(test_indices))

    train_cache = cache_features(train_full, train_indices, ARTIFACT_DIR / "fashion_train_features.pt", device)
    validation_cache = cache_features(train_full, validation_indices, ARTIFACT_DIR / "fashion_validation_features.pt", device)
    head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 10)).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(train_cache["features"], train_cache["labels"]), batch_size=512, shuffle=True)
    head_epochs = 8 if quick else 25
    for epoch in range(head_epochs):
        head.train()
        for features, batch_labels in loader:
            optimizer.zero_grad()
            loss = loss_function(head(features.to(device)), batch_labels.to(device))
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0 or epoch == head_epochs - 1:
            logging.info("stage_a epoch=%d/%d val_accuracy=%.4f", epoch + 1, head_epochs, evaluate_head(head, validation_cache, device))
    stage_a_accuracy = evaluate_head(head, validation_cache, device)

    # The head was trained on features from the pretrained backbone, so the
    # saved frozen-backbone model must use the same pretrained weights.
    final_model = build_backbone(pretrained=True)
    final_model.fc = head.cpu()
    fine_tuned = False
    stage_b_accuracy = None
    if stage_a_accuracy < TARGET_VALIDATION_ACCURACY:
        fine_tuned = True
        logging.info("stage_a below %.2f; fine-tuning layer3, layer4, and classifier", TARGET_VALIDATION_ACCURACY)
        final_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        final_model.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 10))
        for parameter in final_model.parameters():
            parameter.requires_grad = False
        for name, parameter in final_model.named_parameters():
            if name.startswith(("layer3", "layer4", "fc")):
                parameter.requires_grad = True
        final_model = final_model.to(device)
        augmented_train = torchvision.datasets.FashionMNIST(ROOT, train=True, download=False, transform=augmented_transform())
        fine_loader = DataLoader(Subset(augmented_train, train_indices.tolist()), batch_size=128, shuffle=True, num_workers=0)
        optimizer = torch.optim.Adam([p for p in final_model.parameters() if p.requires_grad], lr=1e-4)
        fine_epochs = 2 if quick else 6
        for epoch in range(fine_epochs):
            final_model.train()
            for images, batch_labels in fine_loader:
                optimizer.zero_grad()
                loss = loss_function(final_model(images.to(device)), batch_labels.to(device))
                loss.backward()
                optimizer.step()
            predictions, actual = evaluate_model(final_model, train_full, validation_indices, device)
            stage_b_accuracy = float(accuracy_score(actual, predictions))
            logging.info("stage_b epoch=%d/%d val_accuracy=%.4f", epoch + 1, fine_epochs, stage_b_accuracy)
    else:
        final_model = final_model.to(device)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_model.cpu().state_dict(), MODEL_PATH)
    result = {
        "device": str(device), "quick_test": quick,
        "train_size": int(len(train_indices)), "validation_size": int(len(validation_indices)),
        "test_size": int(len(test_indices)), "full_test_size": 10000,
        "stage_a_validation_accuracy": stage_a_accuracy,
        "stage_b_fine_tuned": fine_tuned,
        "stage_b_validation_accuracy": stage_b_accuracy,
        "head_epochs": head_epochs,
        "optimizer": "Adam", "head_learning_rate": 1e-3,
        "fine_tune_learning_rate": 1e-4,
        "frozen_layers": "backbone during Stage A; layer1/layer2 during Stage B",
        "cache_files": [str(ARTIFACT_DIR / "fashion_train_features.pt"), str(ARTIFACT_DIR / "fashion_validation_features.pt")],
        "checkpoint": str(MODEL_PATH),
    }
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logging.info("saved=%s", MODEL_PATH)
    return result


if __name__ == "__main__":
    main()
