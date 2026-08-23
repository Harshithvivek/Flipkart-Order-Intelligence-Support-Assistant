"""Offline-safe loading and single-image prediction for Part 2."""

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _build_model() -> nn.Module:
    """Build the exact saved architecture without downloading pretrained weights."""
    model = resnet18(weights=None)
    model.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 10))
    return model


def load_model(model_path: str = "models/product_classifier.pt", device: str = "cpu") -> nn.Module:
    """Load the trained state dictionary with no network access."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Product classifier not found: {path}")
    model = _build_model()
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.eval().to(device)


def predict_image(image_path: str, model: nn.Module, device: str = "cpu") -> dict:
    """Return the predicted Fashion-MNIST class and confidence."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Product image not found: {path}")
    image = Image.open(path).convert("L")
    with torch.inference_mode():
        probabilities = torch.softmax(model(TRANSFORM(image).unsqueeze(0).to(device)), dim=1)[0]
    index = int(probabilities.argmax())
    return {"label": CLASSES[index], "confidence": round(float(probabilities[index]), 4)}


def load_product_classifier(model_path: str = "models/product_classifier.pt") -> nn.Module:
    """Compatibility loader used by the support agent."""
    return load_model(model_path)


def predict_product_image(image_path: str, model_path: str = "models/product_classifier.pt") -> dict:
    """Compatibility prediction API used by Part 3."""
    result = predict_image(image_path, load_model(model_path))
    return {"category": result["label"], "confidence": result["confidence"]}
