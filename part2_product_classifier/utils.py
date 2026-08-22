"""Shared Fashion-MNIST and ResNet-18 utilities."""

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

DATA_DIR = Path("data/fashion_mnist")
MODEL_PATH = Path("models/product_classifier.pt")
ARTIFACT_DIR = Path("artifacts")
SAMPLE_DIR = Path("data/sample_images")
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def image_transform() -> transforms.Compose:
    """Convert grayscale Fashion-MNIST images into normalized ResNet inputs."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def raw_image_transform() -> transforms.Compose:
    """Transform used only for exporting original test samples."""
    return transforms.Compose([transforms.ToTensor()])


def load_train_dataset(download: bool = True) -> datasets.FashionMNIST:
    """Load the 60,000-image Fashion-MNIST training split."""
    return datasets.FashionMNIST(DATA_DIR, train=True, transform=image_transform(), download=download)


def load_test_dataset(download: bool = True) -> datasets.FashionMNIST:
    """Load the untouched 10,000-image Fashion-MNIST test split."""
    return datasets.FashionMNIST(DATA_DIR, train=False, transform=image_transform(), download=download)


def build_resnet18(pretrained: bool = True) -> models.ResNet:
    """Build ResNet-18 and replace its classifier with ten Fashion-MNIST classes."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    network = models.resnet18(weights=weights)
    for parameter in network.parameters():
        parameter.requires_grad = False
    network.fc = nn.Linear(network.fc.in_features, len(CLASS_NAMES))
    return network


def load_product_classifier(model_path: Path = MODEL_PATH) -> tuple[nn.Module, dict[str, Any]]:
    """Load the reusable classifier checkpoint and its configuration."""
    if not model_path.exists():
        raise FileNotFoundError(f"Product classifier not found: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    network = build_resnet18(pretrained=False)
    network.load_state_dict(checkpoint["model_state"])
    network.eval()
    return network, checkpoint


def predict_product_image(image_path: str, model_path: Path = MODEL_PATH) -> dict[str, Any]:
    """Predict a Fashion-MNIST-style PNG using the saved classifier."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Product image not found: {path}")
    network, checkpoint = load_product_classifier(model_path)
    image = Image.open(path).convert("L")
    tensor = image_transform()(image).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(network(tensor), dim=1)[0]
    index = int(probabilities.argmax())
    return {
        "category": checkpoint["class_names"][index],
        "confidence": float(probabilities[index]),
    }
