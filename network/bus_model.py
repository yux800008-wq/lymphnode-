from typing import Dict, Optional

import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from torchvision.models import resnet34
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


HEAD_OUTPUT_DIMS = {
    "posterior": 1,
    "liquefaction": 1,
    "hilum": 1,
    "margin": 3,
    "shape": 3,
}


class ResNet34Encoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = resnet34(weights="DEFAULT" if pretrained else None)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class MultiTaskHeads(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.heads = nn.ModuleDict({
            name: nn.Linear(feature_dim, out_dim)
            for name, out_dim in HEAD_OUTPUT_DIMS.items()
        })

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: head(features) for name, head in self.heads.items()}


class ConcatFusionModel(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.context_encoder = ResNet34Encoder(pretrained=pretrained)
        self.lesion_encoder = ResNet34Encoder(pretrained=pretrained)
        feature_dim = self.context_encoder.feature_dim
        if self.lesion_encoder.feature_dim != feature_dim:
            raise RuntimeError("Context and Lesion feature dimensions must match.")
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.heads = MultiTaskHeads(feature_dim=feature_dim)

    def forward(self, context: torch.Tensor, lesion: torch.Tensor) -> Dict[str, torch.Tensor]:
        fused = torch.cat([
            self.context_encoder(context),
            self.lesion_encoder(lesion),
        ], dim=1)
        return self.heads(self.fusion(fused))


def letterbox_square(image: Image.Image, size: int = 224) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size
    scale = min(size / width, size / height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = image.resize(new_size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), color=(0, 0, 0))
    canvas.paste(resized, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
    return canvas


def to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    return TF.normalize(
        TF.to_tensor(image),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )


class PairedUltrasoundTransform:
    def __init__(self, output_size: int = 224, train: bool = True):
        self.output_size = output_size
        self.train = train

    def __call__(
        self,
        context: Image.Image,
        lesion: Optional[Image.Image] = None,
    ):
        context = letterbox_square(context, self.output_size)
        context_tensor = to_normalized_tensor(context)
        lesion_tensor = None
        if lesion is not None:
            lesion_tensor = to_normalized_tensor(
                letterbox_square(lesion, self.output_size)
            )
        return context_tensor, lesion_tensor
