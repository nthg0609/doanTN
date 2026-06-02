"""Isolated classification sanity check for the dermatology EfficientNet model.

This script bypasses the unified pipeline, segmentation mask, ROI cropping,
and safety gate. It loads the classification weights directly and runs a plain
RGB resize -> tensor -> normalize -> softmax inference on either a single image
or a full directory (Batch Testing).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional
from collections import defaultdict

import numpy as np
import torch
import timm
from PIL import Image
from tqdm.auto import tqdm


IDX_TO_CLASS: Dict[int, str] = {
    0: "AKIEC",
    1: "BCC",
    2: "BKL",
    3: "DF",
    4: "MEL",
    5: "NV",
    6: "VASC",
}


class ChannelAttention(torch.nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        hidden_channels = max(1, in_channels // reduction)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden_channels, in_channels, 1, bias=False),
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(torch.nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = torch.nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class CBAM(torch.nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


class EfficientNetWithAttention(torch.nn.Module):
    def __init__(self, num_classes: int = 7, pretrained: bool = False):
        super().__init__()
        # EfficientNet-B1 matches the trained checkpoint structure.
        self.backbone = timm.create_model("efficientnet_b1", pretrained=pretrained, num_classes=0)
        self.feature_dim = self.backbone.num_features
        self.attention = CBAM(self.feature_dim, reduction=16)
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(self.feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        features = self.attention(features)
        pooled = self.global_pool(features).flatten(1)
        return self.classifier(pooled)


def _candidate_weight_paths(repo_root: Path) -> Iterable[Path]:
    yield repo_root / "4_Models" / "classification" / "efficientnet_attention_best.pth"
    yield repo_root / "3_Checkpoints" / "06_classification_finetuned.pth"
    yield repo_root / "4_Models" / "classification" / "efficientnet_attention_last.pth"
    yield repo_root / "3_Checkpoints" / "classification" / "efficientnet_attention_best.pth"


def _resolve_weights_path(repo_root: Path, explicit_path: Optional[str]) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_absolute():
            path = repo_root / path
        if not path.exists():
            raise FileNotFoundError(f"Weights file not found: {path}")
        return path

    for candidate in _candidate_weight_paths(repo_root):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No classification weights file found in the known locations.")


def _load_state_dict(weights_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    state = torch.load(str(weights_path), map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if any(key.startswith("module.") for key in state.keys()):
        state = {key.replace("module.", "", 1): value for key, value in state.items()}
    return state


def _preprocess_image(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224), resample=Image.Resampling.BILINEAR)
    array = np.asarray(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


def _print_probabilities(probabilities: torch.Tensor) -> None:
    for index, probability in enumerate(probabilities.tolist()):
        label = IDX_TO_CLASS.get(index, str(index))
        print(f"{index}: {label} = {probability:.6f}")


def run_single_image(model: torch.nn.Module, image_path: Path, device: torch.device, weights_path: Path) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with torch.inference_mode():
        tensor = _preprocess_image(image_path).to(device)
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu()
        predicted_index = int(torch.argmax(probabilities).item())
        predicted_label = IDX_TO_CLASS[predicted_index]

    print(f"Image: {image_path}")
    print(f"Weights: {weights_path}")
    print("Probabilities:")
    _print_probabilities(probabilities)
    print(f"Predicted index: {predicted_index}")
    print(f"Predicted label: {predicted_label}")


def run_batch_directory(model: torch.nn.Module, dir_path: Path, device: torch.device) -> None:
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in dir_path.rglob("*") if p.suffix.lower() in exts]
    
    if not images:
        print(f"No images found in {dir_path}")
        return

    total = 0
    correct = 0
    preds_count = defaultdict(int)

    print(f"\nBắt đầu quét {len(images)} ảnh trong thư mục: {dir_path}")
    
    for img_path in tqdm(images, desc="Sanity Check"):
        # Lấy nhãn thực tế từ tên thư mục cha
        true_label = img_path.parent.name.strip().upper()
        
        with torch.inference_mode():
            tensor = _preprocess_image(img_path).to(device)
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu()
            predicted_index = int(torch.argmax(probabilities).item())
            predicted_label = IDX_TO_CLASS.get(predicted_index, "UNKNOWN")

        total += 1
        if predicted_label == true_label:
            correct += 1
        preds_count[predicted_label] += 1

    accuracy = (correct / total) * 100 if total > 0 else 0

    print("\n" + "="*40)
    print(" BÁO CÁO SANITY CHECK (BATCH TESTING) ")
    print("="*40)
    print(f"Tổng số ảnh đã test : {total}")
    print(f"Số ảnh đoán ĐÚNG    : {correct}")
    print(f"Độ chính xác (Acc)  : {accuracy:.2f}%")
    print("-" * 40)
    print("Chi tiết phân bổ dự đoán của Model:")
    for label in IDX_TO_CLASS.values():
        count = preds_count.get(label, 0)
        print(f"  - {label}: {count} ảnh")
    print("="*40)


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated sanity check for classification weights.")
    parser.add_argument("--image", default=None, help="Path to a single input image.")
    parser.add_argument("--dir", default=None, help="Path to a directory of images for batch testing (e.g., Data_test/BKL).")
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional path to the classification .pth file. If omitted, known locations are searched.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to use, for example cpu or cuda:0.",
    )
    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.error("Bạn phải cung cấp ít nhất một tham số: --image (test 1 ảnh) hoặc --dir (test cả thư mục).")

    repo_root = Path(__file__).resolve().parents[1]
    weights_path = _resolve_weights_path(repo_root, args.weights)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print(f"Đang nạp mô hình từ: {weights_path}")
    print(f"Thiết bị chạy: {device}")
    
    model = EfficientNetWithAttention(num_classes=len(IDX_TO_CLASS), pretrained=False)
    state_dict = _load_state_dict(weights_path, device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    if args.image:
        image_path = Path(args.image)
        if not image_path.is_absolute():
            image_path = repo_root / image_path
        run_single_image(model, image_path, device, weights_path)

    if args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_absolute():
            dir_path = repo_root / dir_path
        run_batch_directory(model, dir_path, device)


if __name__ == "__main__":
    main()