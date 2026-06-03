"""Model registry with singleton lifecycle for CPU-safe reuse."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import json

import torch

try:
    import segmentation_models_pytorch as smp
except Exception:
    smp = None

try:
    import timm
except Exception:
    timm = None


ISIC_IDX_TO_CLASS = {
    0: "AKIEC",
    1: "BCC",
    2: "BKL",
    3: "DF",
    4: "MEL",
    5: "NV",
    6: "VASC",
}


@dataclass(frozen=True)
class NormalizationConfig:
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]


@dataclass(frozen=True)
class ModelConfig:
    seg_input_size: int = 256
    cls_input_size: int = 224
    seg_norm: NormalizationConfig = NormalizationConfig((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    cls_norm: NormalizationConfig = NormalizationConfig((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


class ModelRegistry:
    """Singleton model registry to prevent repeated CPU loads."""

    _instance: Optional["ModelRegistry"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, base_dir: Optional[Path] = None, device: Optional[torch.device] = None):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.base_dir = Path(base_dir or Path.cwd())
        self.device = device or torch.device("cpu")
        self.config = ModelConfig()
        self._seg_model = None
        self._cls_model = None
        self._idx_to_class: Dict[int, str] = dict(ISIC_IDX_TO_CLASS)

    @classmethod
    def get_instance(cls, base_dir: Optional[Path] = None, device: Optional[torch.device] = None) -> "ModelRegistry":
        return cls(base_dir=base_dir, device=device)

    def load_all(self) -> None:
        self.load_segmentation_model()
        self.load_classification_model()

    def load_segmentation_model(self):
        if self._seg_model is not None:
            return self._seg_model
        if smp is None:
            raise RuntimeError("segmentation_models_pytorch is not available")

        ckpt_path = self._resolve_best_model(
            self.base_dir / "3_Checkpoints" / "03_deeplabv3plus_complete.json",
            fallback=self.base_dir / "4_Models" / "deeplabv3plus" / "deeplabv3plus_best.pth",
        )

        model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )
        state = torch.load(str(ckpt_path), map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model = model.to(self.device)
        model.eval()
        self._seg_model = model
        return model

    def load_classification_model(self):
        if self._cls_model is not None:
            return self._cls_model
        if timm is None:
            raise RuntimeError("timm is not available")

        ckpt_path = self._resolve_best_model(
            self.base_dir / "3_Checkpoints" / "06_classification_complete.json",
            fallback=self.base_dir / "4_Models" / "classification" / "efficientnet_attention_best.pth",
        )
        idx_to_class = dict(ISIC_IDX_TO_CLASS)
        self._idx_to_class = idx_to_class

        class ChannelAttention(torch.nn.Module):
            def __init__(self, in_channels, reduction=16):
                super().__init__()
                self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
                self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
                self.fc = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
                )
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x):
                avg_out = self.fc(self.avg_pool(x))
                max_out = self.fc(self.max_pool(x))
                return self.sigmoid(avg_out + max_out)

        class SpatialAttention(torch.nn.Module):
            def __init__(self, kernel_size=7):
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x):
                avg_out = torch.mean(x, dim=1, keepdim=True)
                max_out, _ = torch.max(x, dim=1, keepdim=True)
                x = torch.cat([avg_out, max_out], dim=1)
                return self.sigmoid(self.conv(x))

        class CBAM(torch.nn.Module):
            def __init__(self, in_channels, reduction=16):
                super().__init__()
                self.channel_att = ChannelAttention(in_channels, reduction)
                self.spatial_att = SpatialAttention()

            def forward(self, x):
                x = x * self.channel_att(x)
                x = x * self.spatial_att(x)
                return x

        class EfficientNetWithAttention(torch.nn.Module):
            def __init__(self, num_classes, pretrained=True):
                super().__init__()
                # Use EfficientNet-B1 to match checkpoint block structure and feature dims.
                self.backbone = timm.create_model("efficientnet_b1", pretrained=pretrained, num_classes=0)
                self.feature_dim = self.backbone.num_features
                self.attention = CBAM(self.feature_dim, reduction=16)
                self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
                # Match checkpoint: Dropout -> Linear(1280 -> num_classes)
                self.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(self.feature_dim, num_classes),
                )

            def forward(self, x):
                features = self.backbone.forward_features(x)
                features_att = self.attention(features)
                pooled = self.global_pool(features_att).flatten(1)
                return self.classifier(pooled)

        num_classes = len(idx_to_class) if idx_to_class else 7
        model = EfficientNetWithAttention(num_classes=num_classes, pretrained=False)
        state = torch.load(str(ckpt_path), map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        # Remap DataParallel prefix if present
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        model.load_state_dict(state)
        model = model.to(self.device)
        model.eval()
        self._cls_model = model
        return model

    def get_segmentation_model(self):
        return self._seg_model

    def get_classification_model(self):
        return self._cls_model

    def get_class_labels(self) -> Dict[int, str]:
        return dict(self._idx_to_class)

    def clear(self) -> None:
        self._seg_model = None
        self._cls_model = None
        self._idx_to_class = {}

    @staticmethod
    def _resolve_best_model(checkpoint_json: Path, fallback: Path) -> Path:
        if checkpoint_json.exists():
            try:
                with checkpoint_json.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                path = data.get("paths", {}).get("best_model")
                if path:
                    return Path(path)
            except Exception:
                pass
        return Path(fallback)

    @staticmethod
    def _read_class_labels(checkpoint_json: Path) -> Dict[int, str]:
        if checkpoint_json.exists():
            try:
                with checkpoint_json.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                mapping = data.get("config", {}).get("idx_to_class", {})
                if mapping:
                    return {int(k): v for k, v in mapping.items()}
            except Exception:
                return {}
        return {}