"""Safety gate for selective prediction and triage mode."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json


@dataclass(frozen=True)
class SafetyGateConfig:
    min_mask_area_px: int = 64
    min_area_ratio: float = 0.001
    max_area_ratio: float = 0.75
    max_border_complexity: float = 8.0
    min_class_confidence: float = 0.60

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SafetyGateConfig":
        base = cls()
        merged = {
            "min_mask_area_px": data.get("min_mask_area_px", base.min_mask_area_px),
            "min_area_ratio": data.get("min_area_ratio", base.min_area_ratio),
            "max_area_ratio": data.get("max_area_ratio", base.max_area_ratio),
            "max_border_complexity": data.get("max_border_complexity", base.max_border_complexity),
            "min_class_confidence": data.get("min_class_confidence", base.min_class_confidence),
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

    def evaluate(self, metrics: Dict[str, float], cls_confidence: Optional[float]) -> SafetyGateResult:
        area_ratio = float(metrics.get("area_ratio", 0.0))
        lesion_area = int(metrics.get("lesion_area", 0))
        border_complexity = float(metrics.get("border_complexity", 0.0))
        low_conf = bool(metrics.get("low_confidence", False))

        if low_conf or lesion_area < self.config.min_mask_area_px:
            return SafetyGateResult(False, "empty_or_low_confidence_mask", {
                "lesion_area": lesion_area,
                "area_ratio": area_ratio,
            })

        if area_ratio < self.config.min_area_ratio or area_ratio > self.config.max_area_ratio:
            return SafetyGateResult(False, "area_ratio_out_of_bounds", {
                "area_ratio": area_ratio,
            })

        if border_complexity > self.config.max_border_complexity:
            return SafetyGateResult(False, "border_complexity_out_of_bounds", {
                "border_complexity": border_complexity,
            })

        if cls_confidence is None:
            return SafetyGateResult(False, "classification_unavailable", {})

        if cls_confidence < self.config.min_class_confidence:
            return SafetyGateResult(False, "low_classification_confidence", {
                "cls_confidence": float(cls_confidence),
            })

        return SafetyGateResult(True, "accepted", {
            "cls_confidence": float(cls_confidence),
            "area_ratio": area_ratio,
        })
