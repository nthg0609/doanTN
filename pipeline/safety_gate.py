"""Safety gate for selective prediction and triage mode.

Hỗ trợ ngưỡng động (adaptive thresholds) theo loại ảnh:
  - 'dermoscopy': ảnh chụp từ máy dermoscope, phân phối chuẩn ISIC.
  - 'phone': ảnh chụp điện thoại thực tế, background đa dạng hơn.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json


@dataclass(frozen=True)
class SafetyGateConfig:
    # ── Ngưỡng chung (dermoscopy) ──────────────────────────────────────────────
    min_mask_area_px: int = 64
    min_area_ratio: float = 0.001
    max_area_ratio: float = 0.75
    max_border_complexity: float = 8.0
    min_class_confidence: float = 0.60

    # ── Ngưỡng riêng cho ảnh điện thoại (phone) ───────────────────────────────
    # Ảnh phone thường có cận cảnh hơn hoặc bối cảnh rộng hơn → cần nới lỏng.
    phone_min_area_ratio: float = 0.0005
    phone_max_area_ratio: float = 0.92
    phone_max_border_complexity: float = 14.0

    # ── Ngưỡng cảnh báo lâm sàng ác tính ─────────────────────────────────────
    # Khi xác suất bất kỳ lớp ác tính (MEL, BCC, AKIEC) vượt ngưỡng này mà
    # nhãn dự đoán chính lại là lành tính → hiển thị cảnh báo y khoa.
    malignant_alert_threshold: float = 0.15

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SafetyGateConfig":
        base = cls()
        merged = {
            "min_mask_area_px":           data.get("min_mask_area_px",           base.min_mask_area_px),
            "min_area_ratio":             data.get("min_area_ratio",             base.min_area_ratio),
            "max_area_ratio":             data.get("max_area_ratio",             base.max_area_ratio),
            "max_border_complexity":      data.get("max_border_complexity",      base.max_border_complexity),
            "min_class_confidence":       data.get("min_class_confidence",       base.min_class_confidence),
            "phone_min_area_ratio":       data.get("phone_min_area_ratio",       base.phone_min_area_ratio),
            "phone_max_area_ratio":       data.get("phone_max_area_ratio",       base.phone_max_area_ratio),
            "phone_max_border_complexity": data.get("phone_max_border_complexity", base.phone_max_border_complexity),
            "malignant_alert_threshold":  data.get("malignant_alert_threshold",  base.malignant_alert_threshold),
        }
        return cls(**merged)

    @classmethod
    def from_json(cls, path: str | Path) -> "SafetyGateConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass(frozen=True)
class SafetyGateResult:
    accept: bool
    reason: str
    details: Dict[str, float]


class SafetyGate:
    def __init__(self, config: Optional[SafetyGateConfig] = None):
        self.config = config or SafetyGateConfig()

    def evaluate(
        self,
        metrics: Dict[str, float],
        cls_confidence: Optional[float],
        image_type: str = "dermoscopy",
    ) -> SafetyGateResult:
        """Đánh giá xem kết quả có đủ tin cậy để đưa ra phân tích lâm sàng không.

        Args:
            metrics: Chỉ số hình học từ segmentation (area_ratio, border_complexity...).
            cls_confidence: Độ tin cậy dự đoán từ mô hình phân loại.
            image_type: Loại ảnh — 'dermoscopy' hoặc 'phone'.
        """
        area_ratio       = float(metrics.get("area_ratio", 0.0))
        lesion_area      = int(metrics.get("lesion_area", 0))
        border_complexity = float(metrics.get("border_complexity", 0.0))
        low_conf         = bool(metrics.get("low_confidence", False))

        # ── Chọn ngưỡng theo loại ảnh ─────────────────────────────────────────
        cfg = self.config
        is_phone = (image_type == "phone")
        eff_min_area_ratio  = cfg.phone_min_area_ratio       if is_phone else cfg.min_area_ratio
        eff_max_area_ratio  = cfg.phone_max_area_ratio       if is_phone else cfg.max_area_ratio
        eff_max_border      = cfg.phone_max_border_complexity if is_phone else cfg.max_border_complexity

        # ── Bước 1: Kiểm tra mask tổn thương ──────────────────────────────────
        if low_conf or lesion_area < cfg.min_mask_area_px:
            return SafetyGateResult(False, "empty_or_low_confidence_mask", {
                "lesion_area": lesion_area,
                "area_ratio":  area_ratio,
            })

        # ── Bước 2: Kiểm tra tỉ lệ diện tích (theo loại ảnh) ─────────────────
        if area_ratio < eff_min_area_ratio or area_ratio > eff_max_area_ratio:
            return SafetyGateResult(False, "area_ratio_out_of_bounds", {
                "area_ratio":     area_ratio,
                "eff_min":        eff_min_area_ratio,
                "eff_max":        eff_max_area_ratio,
                "image_type":     image_type,
            })

        # ── Bước 3: Kiểm tra độ phức tạp bờ (theo loại ảnh) ──────────────────
        if border_complexity > eff_max_border:
            return SafetyGateResult(False, "border_complexity_out_of_bounds", {
                "border_complexity": border_complexity,
                "eff_max":           eff_max_border,
                "image_type":        image_type,
            })

        # ── Bước 4: Kiểm tra độ tin cậy phân loại ────────────────────────────
        if cls_confidence is None:
            return SafetyGateResult(False, "classification_unavailable", {})

        if cls_confidence < cfg.min_class_confidence:
            return SafetyGateResult(False, "low_classification_confidence", {
                "cls_confidence":       float(cls_confidence),
                "min_class_confidence": cfg.min_class_confidence,
            })

        return SafetyGateResult(True, "accepted", {
            "cls_confidence": float(cls_confidence),
            "area_ratio":     area_ratio,
            "image_type":     image_type,
        })
