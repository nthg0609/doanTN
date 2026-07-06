"""Interactive Segmentation module for Dermatology.
Supports deep learning MobileSAM (if checkpoint and library are available)
with a robust, zero-dependency GrabCut/region-growing fallback for CPU/offline demo stability.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2

# Try to import segment-anything / mobile-sam
try:
    from mobile_sam import sam_model_registry, SamPredictor
    _MOBILE_SAM_AVAILABLE = True
except ImportError:
    try:
        from segment_anything import sam_model_registry, SamPredictor
        _MOBILE_SAM_AVAILABLE = True
    except ImportError:
        _MOBILE_SAM_AVAILABLE = False


class InteractiveSegmenter:
    """Handles click-based interactive segmentation using SAM or GrabCut fallback."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.predictor = None
        self.sam_available = _MOBILE_SAM_AVAILABLE and checkpoint_path is not None and os.path.exists(checkpoint_path)

        if self.sam_available:
            try:
                # Default is vit_t for MobileSAM, vit_b/h for standard SAM
                model_type = "vit_t" if "mobile_sam" in str(checkpoint_path).lower() else "vit_h"
                sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
                sam.to(device=device)
                self.predictor = SamPredictor(sam)
            except Exception as e:
                print(f"[InteractiveSegmenter] Failed to load SAM checkpoint: {e}. Falling back to GrabCut.")
                self.sam_available = False

    def segment_by_point(
        self,
        img_rgb: np.ndarray,
        point_x: int,
        point_y: int,
        label: int = 1  # 1 = foreground point, 0 = background
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Segments the image based on a single point prompt (click)."""
        h, w = img_rgb.shape[:2]
        
        # Boundary check for clicked point
        point_x = max(0, min(point_x, w - 1))
        point_y = max(0, min(point_y, h - 1))

        # --- Case A: Deep Learning MobileSAM ---
        if self.sam_available and self.predictor is not None:
            try:
                self.predictor.set_image(img_rgb)
                input_point = np.array([[point_x, point_y]])
                input_label = np.array([label])
                
                masks, scores, _ = self.predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )
                # Select the mask with the highest confidence score
                best_idx = np.argmax(scores)
                mask = masks[best_idx].astype(np.uint8)
                return mask, {"method": "mobile_sam", "score": float(scores[best_idx])}
            except Exception as e:
                print(f"[InteractiveSegmenter] SAM prediction failed: {e}. Fallback to GrabCut.")

        # --- Case B: GrabCut Fallback (Zero-Dependency & Fast CPU performance) ---
        # Initialize GrabCut mask
        gc_mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
        
        # Define a bounding box centered around the click point as a probable foreground region
        box_w, box_h = int(w * 0.35), int(h * 0.35)
        x_min = max(0, point_x - box_w // 2)
        y_min = max(0, point_y - box_h // 2)
        x_max = min(w, point_x + box_w // 2)
        y_max = min(h, point_y + box_h // 2)
        
        rect = (x_min, y_min, x_max - x_min, y_max - y_min)
        
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        
        # Run GrabCut
        try:
            cv2.grabCut(img_rgb, gc_mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
            # 0, 2 are background; 1, 3 are foreground
            mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
            
            # Postprocess to ensure the clicked point is connected
            # Keep the component that contains the clicked point
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num_labels > 1:
                target_label = labels[point_y, point_x]
                if target_label > 0:
                    mask = (labels == target_label).astype(np.uint8)
                else:
                    # If click lands on background label in GrabCut, choose the largest component
                    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    mask = (labels == largest_idx).astype(np.uint8)

            if int(mask.sum()) < 100:
                raise ValueError("GrabCut mask is too small or empty")

            return mask, {"method": "grabcut_interactive_fallback", "rect": rect}
        except Exception as e:
            # Absolute fallback: floodFill starting from the seed point
            mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
            # Create a thresholded image to help floodFill
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            # Find adaptive local range
            seed_val = int(gray[point_y, point_x])
            lo_diff = min(30, max(10, seed_val // 4))
            hi_diff = min(30, max(10, (255 - seed_val) // 4))
            
            h_f, w_f = gray.shape
            ff_mask = np.zeros((h_f + 2, w_f + 2), dtype=np.uint8)
            try:
                cv2.floodFill(gray, ff_mask, (point_x, point_y), 1, loDiff=lo_diff, upDiff=hi_diff, flags=4 | (1 << 8) | cv2.FLOODFILL_MASK_ONLY)
                mask = ff_mask[1:-1, 1:-1]
                return mask, {"method": "floodfill_interactive_fallback", "lo_diff": lo_diff, "hi_diff": hi_diff}
            except Exception as e_ff:
                # Return a circle of radius 40 around the point if everything else fails
                cv2.circle(mask, (point_x, point_y), 40, 1, -1)
                return mask, {"method": "circle_fallback", "error": str(e_ff)}
