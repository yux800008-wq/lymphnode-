from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize, ToTensor


ROOT = Path(__file__).resolve().parents[1]
CDFI_CHECKPOINT = ROOT / "deploy_models" / "cdfi_resnet18_best.pt"
VASCULARITY_CHECKPOINT = ROOT / "deploy_models" / "vascularity_resnet18_best.pt"
VASCULARITY_LABELS = ["avascular", "mixed", "peripheral", "hilar"]


def _square_resize(image_rgb: np.ndarray) -> Image.Image:
    height, width = image_rgb.shape[:2]
    size = max(height, width)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y = (size - height) // 2
    x = (size - width) // 2
    canvas[y:y + height, x:x + width] = image_rgb
    return Image.fromarray(cv2.resize(canvas, (224, 224), interpolation=cv2.INTER_AREA))


class ResNetClassifier:
    def __init__(self, checkpoint: Path, labels: list[str], device: torch.device):
        self.labels = labels
        self.device = device
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
        model.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
        self.model = model.to(device).eval()
        self.transform = Compose([
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(self, image_rgb: np.ndarray) -> dict:
        tensor = self.transform(_square_resize(image_rgb)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()
        class_id = int(np.argmax(probabilities))
        return {
            "predicted_class": class_id,
            "label": self.labels[class_id],
            "confidence": float(probabilities[class_id]),
            "probabilities": {label: float(prob) for label, prob in zip(self.labels, probabilities)},
        }


class ModalityClassifier(ResNetClassifier):
    def __init__(self, device: torch.device):
        super().__init__(CDFI_CHECKPOINT, ["B-mode", "CDFI"], device)

    def predict_bgr(self, image_bgr: np.ndarray) -> dict:
        return self.predict(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


class VascularityClassifier(ResNetClassifier):
    def __init__(self, device: torch.device):
        super().__init__(VASCULARITY_CHECKPOINT, VASCULARITY_LABELS, device)

    def predict_bgr(self, image_bgr: np.ndarray) -> dict:
        return self.predict(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
