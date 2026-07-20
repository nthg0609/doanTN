from __future__ import annotations
from pathlib import Path
"""Unified dermatology pipeline with a single inference contract.

Nâng cấp P1-1: Adaptive Safety Gate — pass img_type vào SafetyGate.evaluate()
Nâng cấp P1-2: TTA Segmentation — dùng multiscale inference cho ảnh điện thoại
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np
import cv2
import torch
from PIL import Image

from .model_registry import ModelRegistry
from .safety_gate import SafetyGate, SafetyGateConfig
from .multimodal_fusion import MultimodalBayesianFusion
from .interactive_sam import InteractiveSegmenter

try:
    import sys
    import os
    _base = str(Path(__file__).resolve().parent.parent)
    if _base not in sys.path:
        sys.path.insert(0, _base)
    from derma_inference_utils import multiscale_segment_from_rgb
    _TTA_AVAILABLE = True
except ImportError:
    _TTA_AVAILABLE = False


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
        use_tta: bool = True,  # P1-2: Kích hoạt TTA cho ảnh phone
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
        self.use_tta = use_tta and _TTA_AVAILABLE  # P1-2: TTA chỉ hoạt động khi module có sẵn
        self._interactive_segmenter: Optional[InteractiveSegmenter] = None  # lazy singleton, tránh nạp lại SAM mỗi lần click
        if load_models:
            self.registry.load_all()

    def _get_interactive_segmenter(self) -> InteractiveSegmenter:
        """Tạo (hoặc tái sử dụng) InteractiveSegmenter, nạp checkpoint MobileSAM nếu có sẵn.

        Nếu checkpoint không tồn tại (hoặc thư viện mobile_sam/segment_anything chưa cài),
        InteractiveSegmenter tự động fallback về GrabCut — hành vi cũ không đổi.
        """
        if self._interactive_segmenter is None:
            sam_checkpoint = self.registry.base_dir / "4_Models" / "sam" / "mobile_sam.pt"
            self._interactive_segmenter = InteractiveSegmenter(
                checkpoint_path=str(sam_checkpoint) if sam_checkpoint.exists() else None
            )
        return self._interactive_segmenter

    def run(
        self,
        image_path: str,
        question: Optional[str] = None,
        return_mask: bool = False,
        age: Optional[float] = None,
        gender: Optional[str] = None,
        body_location: Optional[str] = None,
        lambda_val: Optional[float] = None,
        interactive_point: Optional[tuple[int, int]] = None,
        custom_mask: Optional[np.ndarray] = None,
        malignant_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
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
        if custom_mask is not None:
            seg_mask = custom_mask
            seg_info = {"method": "custom_drawn_canvas"}
        # Nếu có điểm nhấp tương tác (SAM/GrabCut), chạy phân đoạn tương tác.
        elif interactive_point is not None:
            pt_x, pt_y = interactive_point
            segmenter = self._get_interactive_segmenter()
            seg_mask, seg_info = segmenter.segment_by_point(img_rgb, pt_x, pt_y)
            # Nếu mặt nạ tương tác quá nhỏ (do lỗi phân đoạn hoặc click nhầm vùng da lành), dùng phân đoạn tự động
            if int(seg_mask.sum()) < 100:
                fallback_mask, fb_info = self._segment(img_rgb, image_type=img_type)
                seg_mask = fallback_mask
                seg_info = {
                    "method": "deeplab_fallback_interactive",
                    "reason": "interactive_mask_too_small",
                    "original_interactive_info": seg_info
                }
        else:
            # P1-2: Truyền img_type để kích hoạt TTA cho ảnh phone.
            seg_mask, seg_info = self._segment(img_rgb, image_type=img_type)
            
        metrics = self._get_lesion_metrics(seg_mask, img_rgb=img_rgb)

        # Nhánh 2: classification chạy ĐỘC LẬP trên chính ảnh gốc, không nhân mask.
        cls_result = None
        cls_confidence = None
        if self.mode in ("classification", "both"):
            cls_result = self._classify(
                img_rgb,
                seg_mask=seg_mask,
                lesion_metrics=metrics,
                age=age,
                gender=gender,
                body_location=body_location,
                lambda_val=lambda_val
            )
            cls_confidence = None if cls_result is None else cls_result.get("confidence")

        # P1-1: Pass img_type vào Safety Gate để áp dụng ngưỡng động theo loại ảnh.
        gate = self.safety_gate.evaluate(metrics, cls_confidence, image_type=img_type, malignant_threshold=malignant_threshold)
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

        # Tính toán Grad-CAM nếu phân loại thành công
        gradcam_img = None
        if return_mask and cls_result and cls_result.get("prediction") != "N/A":
            label_to_idx = {"AKIEC": 0, "BCC": 1, "BKL": 2, "DF": 3, "MEL": 4, "NV": 5, "VASC": 6}
            pred_idx = label_to_idx.get(cls_result.get("prediction"), 0)
            cam_heatmap = self._run_gradcam(img_rgb, pred_idx)
            if cam_heatmap is not None:
                gradcam_img = self._generate_gradcam_overlay(img_rgb, cam_heatmap)

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
            if gradcam_img is not None:
                result["gradcam_image"] = gradcam_img
        return result

    def _segment(self, img_rgb: np.ndarray, image_type: str = "dermoscopy") -> tuple[np.ndarray, Dict[str, Any]]:
        """Phân vùng tổn thương.

        P1-2: Nếu use_tta=True và image_type='phone', dùng Multi-Scale TTA
        (multiscale_segment_from_rgb) để tăng robustness trên ảnh điện thoại.
        """
        seg_model = self.registry.get_segmentation_model()
        if seg_model is None:
            return np.zeros(img_rgb.shape[:2], dtype=np.uint8), {"method": "deeplab", "error": "model_unavailable"}

        cfg = self.registry.config
        mean = np.array(cfg.seg_norm.mean, dtype=np.float32)
        std  = np.array(cfg.seg_norm.std,  dtype=np.float32)

        # ── P1-2: TTA cho ảnh phone ─────────────────────────────────────────
        if self.use_tta and image_type == "phone":
            tta_mask, _prob_map, tta_info = multiscale_segment_from_rgb(
                img_rgb,
                model=seg_model,
                device=self.registry.device,
                scales=(1.0, 0.75, 0.5),
                input_size=cfg.seg_input_size,
                threshold=self.seg_threshold,
                min_area_px=self.min_area_px,
                mean=mean,
                std=std,
                morph_kernel=5,
            )
            tta_mask = self._postprocess_mask(tta_mask)
            tta_info["method"] = "deeplab_tta"
            if tta_mask.sum() == 0:
                fallback_mask, fb_info = self._classical_fallback_mask(img_rgb)
                if fb_info.get("accepted", False):
                    tta_mask = fallback_mask
                    tta_info.update({"method": "classical_fallback_after_tta", **fb_info})
            return tta_mask, tta_info

        # ── Standard single-pass segmentation ──────────────────────────────
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

    def _classify(
        self,
        img_rgb: np.ndarray,
        seg_mask: Optional[np.ndarray] = None,
        lesion_metrics: Optional[Dict[str, Any]] = None,
        age: Optional[float] = None,
        gender: Optional[str] = None,
        body_location: Optional[str] = None,
        lambda_val: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Luồng Classification: dùng ảnh đã cắt ROI theo mask khi mask hợp lệ (khớp đúng
        cách dữ liệu huấn luyện classifier được tạo — xem
        2_Notebooks/05_ROI_Extraction.ipynb::extract_roi, bounding box của contour lớn
        nhất + padding=10px). Nếu mask không hợp lệ/rỗng, fallback về ảnh gốc để giữ
        tính độc lập/robust giữa 2 nhánh (không để lỗi segmentation làm hỏng classification).
        Tích hợp Hợp nhất Bayes Đa phương thức (Multimodal Fusion).
        """
        cls_model = self.registry.get_classification_model()
        if cls_model is None:
            return None

        # 0. Chọn ảnh đầu vào: ROI-cropped nếu mask hợp lệ, ngược lại ảnh gốc.
        classify_input = img_rgb
        is_valid_mask = (
            seg_mask is not None
            and lesion_metrics is not None
            and not lesion_metrics.get("low_confidence", True)
            and int(lesion_metrics.get("lesion_area", 0)) >= self.min_area_px
        )
        if is_valid_mask:
            cropped = self._crop_to_roi(img_rgb, seg_mask, padding=10)
            if cropped is not None and cropped.size > 0:
                classify_input = cropped

        # 1. Ép dùng PIL để Resize giống hệt sanity_check_cls.py để tránh lệch phép nội suy
        pil_img = Image.fromarray(classify_input)
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

        # 4. Hardcode từ điển nhãn (Chống lệch do JSON file của Registry)
        idx_to_class = {0: "AKIEC", 1: "BCC", 2: "BKL", 3: "DF", 4: "MEL", 5: "NV", 6: "VASC"}
        raw_probs = {idx_to_class.get(i, str(i)): float(p) for i, p in enumerate(probs)}

        # Hợp nhất Đa phương thức (Multimodal Late Fusion)
        fusion = MultimodalBayesianFusion()
        fused_probs = fusion.fuse(
            raw_probs,
            age=age,
            gender=gender,
            body_location=body_location,
            lambda_val=lambda_val
        )

        pred_label = max(fused_probs, key=fused_probs.get)
        confidence = fused_probs[pred_label]

        return {
            "prediction": pred_label,
            "confidence": confidence,
            "probabilities": fused_probs,
            "raw_probabilities": raw_probs,
        }

    @staticmethod
    def _crop_to_roi(img_rgb: np.ndarray, mask: np.ndarray, padding: int = 10) -> Optional[np.ndarray]:
        """Cắt ảnh theo bounding box của contour lớn nhất trong mask + padding cố định.

        Khớp đúng công thức đã dùng để tạo dữ liệu huấn luyện classifier
        (xem 2_Notebooks/05_ROI_Extraction.ipynb::extract_roi, padding=10px), nhằm
        đồng bộ phân phối đầu vào giữa huấn luyện và suy luận trực tuyến.
        """
        m = (np.asarray(mask) > 0).astype(np.uint8)
        img_h, img_w = img_rgb.shape[:2]
        if m.shape[:2] != (img_h, img_w):
            m = cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        contours, _ = cv2.findContours((m * 255).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_w, x + w + padding)
        y2 = min(img_h, y + h + padding)
        if x2 <= x1 or y2 <= y1:
            return None
        return img_rgb[y1:y2, x1:x2]

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
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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

    def _get_lesion_metrics(self, mask: np.ndarray, img_rgb: Optional[np.ndarray] = None) -> Dict[str, Any]:
        mask = (np.asarray(mask) > 0).astype(np.uint8)
        h, w = mask.shape[:2]
        img_area = max(int(h * w), 1)
        lesion_area = int(cv2.countNonZero(mask))
        _empty = {
            "asymmetry": 0.0,
            "border_complexity": 0.0,
            "color_variation": 0.0,
            "diameter_px": 0.0,
            "area_ratio": 0.0,
            "circularity": 0.0,
            "lesion_area": lesion_area,
            "image_area": img_area,
            "low_confidence": True,
        }
        if lesion_area < self.min_area_px:
            return _empty
        contours, _ = cv2.findContours((mask * 255).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return _empty
        largest = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(largest, True))
        area_ratio = float(lesion_area) / float(img_area)

        # --- B (Border): Độ phức tạp bờ = Chu vi / √Diện tích ---
        border_complexity = perimeter / max(np.sqrt(float(lesion_area)), 1.0)

        # --- A (Asymmetry): Chia mask theo centroid (cả 2 trục ngang + dọc) ---
        M = cv2.moments(largest)
        if M.get("m00", 0) > 0:
            cx = int(round(M["m10"] / M["m00"]))
            cy = int(round(M["m01"] / M["m00"]))
        else:
            cx, cy = w // 2, h // 2

        # Trục ngang (horizontal split tại cy)
        top_half = mask[:cy, :]
        bot_half = mask[cy:, :]
        max_rows = max(top_half.shape[0], bot_half.shape[0])
        top_padded = np.pad(top_half, ((0, max_rows - top_half.shape[0]), (0, 0)))
        bot_flipped = np.pad(np.flipud(bot_half), ((0, max_rows - bot_half.shape[0]), (0, 0)))
        asym_h = float(np.sum(np.abs(top_padded.astype(np.int32) - bot_flipped.astype(np.int32))))

        # Trục dọc (vertical split tại cx)
        left_half = mask[:, :cx]
        right_half = mask[:, cx:]
        max_cols = max(left_half.shape[1], right_half.shape[1])
        left_padded = np.pad(left_half, ((0, 0), (0, max_cols - left_half.shape[1])))
        right_flipped = np.pad(np.fliplr(right_half), ((0, 0), (0, max_cols - right_half.shape[1])))
        asym_v = float(np.sum(np.abs(left_padded.astype(np.int32) - right_flipped.astype(np.int32))))

        # Normalize: Asymmetry score ∈ [0, 1] — 0 = hoàn toàn đối xứng, 1 = bất đối xứng tối đa
        asymmetry = float(np.clip((asym_h + asym_v) / (2.0 * max(lesion_area, 1)), 0.0, 1.0))

        # --- C (Color): Độ lệch chuẩn RGB trung bình trong vùng tổn thương ---
        color_variation = 0.0
        if img_rgb is not None:
            # Đảm bảo mask cùng kích thước ảnh gốc
            img_h, img_w = img_rgb.shape[:2]
            if (img_h, img_w) != (h, w):
                mask_resized = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = mask
            lesion_pixels = img_rgb[mask_resized > 0]  # shape: (N, 3)
            if len(lesion_pixels) > 0:
                # Tính std dev từng kênh R, G, B rồi lấy trung bình, chuẩn hóa về [0, 1]
                std_per_channel = np.std(lesion_pixels.astype(np.float64), axis=0)  # shape: (3,)
                # max std của một kênh 8-bit là 127.5, chuẩn hóa
                color_variation = float(np.clip(np.mean(std_per_channel) / 127.5, 0.0, 1.0))

        # --- D (Diameter): Đường kính tương đương (equivalent diameter) tính bằng pixel ---
        diameter_px = float(2.0 * np.sqrt(float(lesion_area) / np.pi))

        # --- Circularity (chỉ số phụ): 4π·Area / Perimeter² ---
        circularity = (4.0 * np.pi * float(lesion_area)) / max(perimeter ** 2, 1e-6)
        circularity = float(np.clip(circularity, 0.0, 1.0))

        return {
            "asymmetry": float(asymmetry),
            "border_complexity": float(border_complexity),
            "color_variation": float(color_variation),
            "diameter_px": float(diameter_px),
            "area_ratio": float(area_ratio),
            "circularity": float(circularity),
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

    def _run_gradcam(self, img_rgb: np.ndarray, target_class_idx: int) -> Optional[np.ndarray]:
        """Tính toán bản đồ kích hoạt Grad-CAM trên lớp attention CBAM."""
        cls_model = self.registry.get_classification_model()
        if cls_model is None:
            return None
            
        pil_img = Image.fromarray(img_rgb)
        pil_img = pil_img.resize((224, 224), resample=Image.Resampling.BILINEAR)
        arr = np.asarray(pil_img).astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
        
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(self.registry.device)
        tensor.requires_grad = True
        
        activations = []
        gradients = []
        
        def save_activation(module, input, output):
            activations.append(output.detach())
            
        def save_gradient(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())
            
        target_layer = cls_model.attention
        h_f = target_layer.register_forward_hook(save_activation)
        h_b = target_layer.register_full_backward_hook(save_gradient)
        
        try:
            with torch.enable_grad():
                logits = cls_model(tensor)
                loss = logits[0, target_class_idx]
                cls_model.zero_grad()
                loss.backward()
                
            if not activations or not gradients:
                return None
                
            act = activations[0]
            grad = gradients[0]
            
            pooled_grad = torch.mean(grad, dim=[2, 3], keepdim=True)
            cam = torch.sum(act * pooled_grad, dim=1).squeeze(0)
            cam = torch.relu(cam)
            
            cam_max = torch.max(cam)
            if cam_max > 0:
                cam = cam / cam_max
                
            return cam.cpu().numpy()
            
        except Exception as e:
            print(f"Grad-CAM error: {e}")
            return None
        finally:
            h_f.remove()
            h_b.remove()

    def _generate_gradcam_overlay(self, img_rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        """Chồng bản đồ nhiệt Grad-CAM lên ảnh gốc sử dụng hệ màu Jet."""
        heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        heatmap_scaled = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_scaled, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        alpha = 0.45
        overlay = cv2.addWeighted(heatmap_colored, alpha, img_rgb, 1 - alpha, 0)
        return overlay