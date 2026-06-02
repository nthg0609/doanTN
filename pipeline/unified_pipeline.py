from __future__ import annotations
from pathlib import Path
"""Unified dermatology pipeline with a single inference contract."""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np
import cv2
import torch
from PIL import Image

from .model_registry import ModelRegistry
from .safety_gate import SafetyGate, SafetyGateConfig


@dataclass
class InferenceResult:
    status: str
    image_path: str
    triage_reason: Optional[str]
    preprocess: Dict[str, Any]
    segmentation: Dict[str, Any]
    metrics: Dict[str, Any]
    classification: Optional[Dict[str, Any]]
    report: str


class UnifiedDermatologyPipeline:
    """Parallel contract: raw image -> segmentation branch + classification branch -> decision."""

    def __init__(
        self,
        base_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
        load_models: bool = True,
        safety_config: Optional[SafetyGateConfig] = None,
        safety_config_path: Optional[str] = None,
        safety_overrides: Optional[Dict[str, Any]] = None,
        seg_threshold: float = 0.3,
        min_area_px: int = 64,
        mode: str = "classification",
    ):
        self.registry = ModelRegistry.get_instance(base_dir=base_dir, device=device)
        if safety_config_path:
            safety_config = SafetyGateConfig.from_json(safety_config_path)
        if safety_overrides:
            base_cfg = asdict(safety_config) if safety_config else {}
            base_cfg.update(safety_overrides)
            safety_config = SafetyGateConfig.from_dict(base_cfg)
        self.safety_gate = SafetyGate(safety_config)
        self.seg_threshold = float(seg_threshold)
        self.min_area_px = int(min_area_px)
        self.mode = mode
        if load_models:
            self.registry.load_all()

    def run(self, image_path: str, question: Optional[str] = None, return_mask: bool = False) -> Dict[str, Any]:
        try:
            img_rgb, resolved = self._safe_load_rgb(image_path)
        except Exception as e:
            return self._triage_result(
                image_path=image_path,
                reason="image_load_failed",
                report=f"Triage: cannot read image ({e}).",
            )

        img_type = self._detect_image_type(img_rgb, resolved)

        # Nhánh 1: segmentation, metrics, và mask hiển thị chỉ đọc ảnh gốc RGB.
        seg_mask, seg_info = self._segment(img_rgb)
        metrics = self._get_lesion_metrics(seg_mask)

        # Nhánh 2: classification chạy ĐỘC LẬP trên chính ảnh gốc, không nhân mask.
        cls_result = None
        cls_confidence = None
        if self.mode in ("classification", "both"):
            cls_result = self._classify(img_rgb)
            cls_confidence = None if cls_result is None else cls_result.get("confidence")

        gate = self.safety_gate.evaluate(metrics, cls_confidence)
        if not gate.accept:
            report = self._safe_fallback_report(metrics, gate.reason)
            result = {
                "status": "triage",
                "image_path": resolved,
                "triage_reason": gate.reason,
                "preprocess": {"image_type": img_type, "preset": "raw_rgb"},
                "segmentation": seg_info,
                "metrics": metrics,
                "classification": cls_result,
                "report": report,
            }
            if return_mask:
                result["segmentation_mask"] = seg_mask
            return result

        report = self._clinical_report(metrics, cls_result)
        result = {
            "status": "ok",
            "image_path": resolved,
            "triage_reason": None,
            "preprocess": {"image_type": img_type, "preset": "raw_rgb"},
            "segmentation": seg_info,
            "metrics": metrics,
            "classification": cls_result,
            "report": report,
        }
        if return_mask:
            result["segmentation_mask"] = seg_mask
        return result

    def _segment(self, img_rgb: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
        seg_model = self.registry.get_segmentation_model()
        if seg_model is None:
            return np.zeros(img_rgb.shape[:2], dtype=np.uint8), {"method": "deeplab", "error": "model_unavailable"}

        cfg = self.registry.config
        mean = np.array(cfg.seg_norm.mean, dtype=np.float32)
        std = np.array(cfg.seg_norm.std, dtype=np.float32)

        h, w = img_rgb.shape[:2]
        resized = cv2.resize(img_rgb, (cfg.seg_input_size, cfg.seg_input_size), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        arr = (arr - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(self.registry.device)

        with torch.inference_mode():
            logits = seg_model(tensor)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            if prob.ndim == 3:
                prob = prob[0]

        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (prob >= self.seg_threshold).astype(np.uint8)
        mask = self._postprocess_mask(mask)
        seg_info = {
            "method": "deeplab",
            "threshold": float(self.seg_threshold),
            "lesion_found": int(mask.sum() > 0),
        }

        if mask.sum() == 0:
            fallback_mask, fb_info = self._classical_fallback_mask(img_rgb)
            if fb_info.get("accepted", False):
                mask = fallback_mask
                seg_info.update({"method": "classical_fallback", **fb_info})
        return mask, seg_info

    def _classify(self, img_rgb: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Luồng Classification độc lập: Đồng bộ 100% với sanity_check_cls.py
        """
        cls_model = self.registry.get_classification_model()
        if cls_model is None:
            return None

        # 1. Ép dùng PIL để Resize giống hệt sanity_check_cls.py để tránh lệch phép nội suy
        pil_img = Image.fromarray(img_rgb)
        pil_img = pil_img.resize((224, 224), resample=Image.Resampling.BILINEAR)
        arr = np.asarray(pil_img).astype(np.float32) / 255.0

        # 2. Hardcode chuẩn ImageNet
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)

        # 3. Tạo Tensor
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(self.registry.device)

        with torch.inference_mode():
            logits = cls_model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        # 4. Hardcode từ điển nhãn (Chống lệch do JSON file của Registry)
        idx_to_class = {0: "AKIEC", 1: "BCC", 2: "BKL", 3: "DF", 4: "MEL", 5: "NV", 6: "VASC"}
        label = idx_to_class.get(pred_idx, str(pred_idx))

        return {
            "prediction": label,
            "confidence": confidence,
            "probabilities": {idx_to_class.get(i, str(i)): float(p) for i, p in enumerate(probs)},
        }

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        # keep largest connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        if num_labels <= 1:
            return np.zeros_like(mask)
        best = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        out = np.zeros_like(mask)
        out[labels == best] = 1
        return out

    def _classical_fallback_mask(self, img_rgb: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.medianBlur(gray, 5)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
        num_labels, labels = cv2.connectedComponents(closed)
        if num_labels <= 1:
            return np.zeros_like(gray), {"accepted": False, "reason": "no_component"}

        max_area = 0
        best = None
        for lbl in range(1, num_labels):
            comp = (labels == lbl).astype(np.uint8)
            a = int(cv2.countNonZero(comp))
            if a > max_area:
                max_area = a
                best = comp

        if best is None or max_area == 0:
            return np.zeros_like(gray), {"accepted": False, "reason": "empty_component"}

        contours, _ = cv2.findContours((best * 255).astype(np.uint8).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(gray), {"accepted": False, "reason": "no_contour"}
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        bbox_aspect = float(w) / max(1.0, float(h))
        lesion_area = int(cv2.countNonZero(best))
        img_h, img_w = gray.shape[:2]
        image_area = img_h * img_w

        hull = cv2.convexHull(cnt)
        hull_area = int(cv2.contourArea(hull)) if hull is not None else 0
        solidity = float(lesion_area) / max(1.0, float(hull_area)) if hull_area > 0 else 0.0

        M = cv2.moments(cnt)
        if M.get("m00", 0) != 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
        else:
            cx, cy = float(x + w / 2.0), float(y + h / 2.0)
        center_x, center_y = img_w / 2.0, img_h / 2.0
        dx = cx - center_x
        dy = cy - center_y
        diag = np.sqrt(img_w**2 + img_h**2)
        center_dist_norm = np.sqrt(dx * dx + dy * dy) / (diag / 2.0 + 1e-8)

        accepted = (
            lesion_area >= self.min_area_px
            and 0.25 <= bbox_aspect <= 4.0
            and solidity >= 0.35
            and center_dist_norm <= 0.7
        )

        info = {
            "accepted": bool(accepted),
            "lesion_area": int(lesion_area),
            "image_area": int(image_area),
            "bbox_aspect": float(bbox_aspect),
            "solidity": float(solidity),
            "center_dist_norm": float(center_dist_norm),
        }
        return best.astype(np.uint8), info

    def _get_lesion_metrics(self, mask: np.ndarray) -> Dict[str, Any]:
        mask = (np.asarray(mask) > 0).astype(np.uint8)
        h, w = mask.shape[:2]
        img_area = max(int(h * w), 1)
        lesion_area = int(cv2.countNonZero(mask))
        if lesion_area < self.min_area_px:
            return {
                "area_ratio": 0.0,
                "border_complexity": 0.0,
                "lesion_area": lesion_area,
                "image_area": img_area,
                "low_confidence": True,
            }
        contours, _ = cv2.findContours((mask * 255).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {
                "area_ratio": 0.0,
                "border_complexity": 0.0,
                "lesion_area": lesion_area,
                "image_area": img_area,
                "low_confidence": True,
            }
        largest = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(largest, True))
        area_ratio = float(lesion_area) / float(img_area)
        border_complexity = perimeter / max(np.sqrt(float(lesion_area)), 1.0)
        return {
            "area_ratio": float(area_ratio),
            "border_complexity": float(border_complexity),
            "lesion_area": lesion_area,
            "image_area": img_area,
            "low_confidence": False,
        }

    @staticmethod
    def _safe_load_rgb(path: str) -> tuple[np.ndarray, str]:
        if not path:
            raise FileNotFoundError("empty path")
        if not Path(path).exists():
            raise FileNotFoundError(path)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            # OpenCV returns BGR; convert to RGB for model consistency.
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return rgb, path
        img = Image.open(path).convert("RGB")
        return np.array(img), path

    @staticmethod
    def _detect_image_type(img_rgb: np.ndarray, path: Optional[str] = None) -> str:
        h, w = img_rgb.shape[:2]
        aspect = float(max(h, w)) / max(1.0, float(min(h, w)))
        filename = Path(path or "").name.lower()
        if aspect > 2.0 or filename.startswith("img") or filename.startswith("dcim") or max(h, w) > 1200:
            return "phone"
        return "dermoscopy"

    @staticmethod
    def _clinical_report(metrics: Dict[str, Any], cls_result: Optional[Dict[str, Any]]) -> str:
        area = metrics.get("area_ratio", 0)
        complexity = metrics.get("border_complexity", 0)
        if area > 0.08 or complexity > 5.5:
            risk = "HIGH RISK"
        elif area > 0.03 or complexity > 3.5:
            risk = "MODERATE RISK"
        else:
            risk = "LOW RISK"
        diagnosis = cls_result.get("prediction") if cls_result else "N/A"
        conf = cls_result.get("confidence") if cls_result else 0.0
        return (
            f"Dermatology Report\n"
            f"- Risk level: {risk}\n"
            f"- Area ratio: {area:.4f}\n"
            f"- Border complexity: {complexity:.4f}\n"
            f"- Classification: {diagnosis} (conf={conf:.2f})\n"
            "Recommendation: confirm with dermatologist."
        )

    @staticmethod
    def _safe_fallback_report(metrics: Dict[str, Any], reason: str) -> str:
        return (
            "Triage Mode: prediction rejected due to safety gate.\n"
            f"Reason: {reason}.\n"
            "Recommendation: retake image or consult dermatologist."
        )

    @staticmethod
    def _triage_result(image_path: str, reason: str, report: str) -> Dict[str, Any]:
        return {
            "status": "triage",
            "image_path": image_path,
            "triage_reason": reason,
            "preprocess": {},
            "segmentation": {},
            "metrics": {},
            "classification": None,
            "report": report,
        }