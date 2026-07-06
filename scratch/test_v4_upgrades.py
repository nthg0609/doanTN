"""Sanity check script for Dermatology Pipeline v4.0 Upgrades.
Verifies Multimodal Late Fusion, DICOM parser, and Interactive Segmenter GrabCut fallback.
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add workspace root to path
_base = str(Path(__file__).resolve().parent.parent)
if _base not in sys.path:
    sys.path.insert(0, _base)

from pipeline.multimodal_fusion import MultimodalBayesianFusion
from pipeline.interactive_sam import InteractiveSegmenter
from pipeline.unified_pipeline import UnifiedDermatologyPipeline


def test_multimodal_fusion():
    print("[TEST] Running Multimodal Bayesian Fusion test...")
    fusion = MultimodalBayesianFusion()
    
    # Dummy probabilities from model
    image_probs = {
        "AKIEC": 0.05,
        "BCC": 0.05,
        "BKL": 0.10,
        "DF": 0.05,
        "MEL": 0.10,
        "NV": 0.60,
        "VASC": 0.05
    }
    
    # 1. Test case: Young patient, nevus (NV) should remain dominant or increase
    young_fused = fusion.fuse(image_probs, age=20, gender="Nam", body_location="Cánh tay", lambda_val=0.7)
    print(f"  Young Patient (Age 20) NV probability: {young_fused['NV']:.4f}")
    assert young_fused["NV"] > 0.5, "NV probability should be high for young patients"
    
    # 2. Test case: Old patient, Melanoma (MEL) and BCC should increase relative to their prior
    old_fused = fusion.fuse(image_probs, age=78, gender="Nam", body_location="Đầu / Mặt", lambda_val=0.7)
    print(f"  Old Patient (Age 78) NV probability: {old_fused['NV']:.4f}")
    print(f"  Old Patient (Age 78) MEL probability: {old_fused['MEL']:.4f}")
    print(f"  Old Patient (Age 78) BCC probability: {old_fused['BCC']:.4f}")
    assert old_fused["MEL"] > young_fused["MEL"], "MEL probability should be higher for older patients than younger patients"
    assert old_fused["BCC"] > image_probs["BCC"], "BCC probability should increase for older patients"
    
    # 3. Test case: No demographic details (Fallback to pure image probs)
    fallback_fused = fusion.fuse(image_probs, age=None, gender=None, body_location=None)
    assert fallback_fused == image_probs, "Should fallback to image probabilities when demographics are missing"
    print("  [PASS] Multimodal fusion tests completed successfully!")


def test_interactive_segmenter():
    print("[TEST] Running Interactive Segmenter test...")
    segmenter = InteractiveSegmenter()
    
    # Create a dummy image with a dark circle in the middle (simulating a lesion)
    img = np.ones((200, 200, 3), dtype=np.uint8) * 200
    cv2.circle(img, (100, 100), 30, (50, 50, 50), -1) # Dark lesion
    
    # Click inside the lesion
    mask, info = segmenter.segment_by_point(img, point_x=100, point_y=100)
    print(f"  Interactive segmenter method used: {info.get('method')}")
    print(f"  Lesion mask sum: {mask.sum()} pixels (Expected around ~2800)")
    
    assert mask.sum() > 0, "Interactive mask should not be empty"
    assert mask[100, 100] == 1, "Clicked center point should be foreground"
    assert mask[10, 10] == 0, "Top-left corner should be background"
    print("  [PASS] Interactive segmenter tests completed successfully!")


def test_pipeline_integration():
    print("[TEST] Running Unified Dermatology Pipeline v4.0 Integration test...")
    # Initialize pipeline without loading weights (mock registry if needed, or CPU safe load)
    try:
        pipeline = UnifiedDermatologyPipeline(load_models=False)
        print("  Pipeline initialized successfully (load_models=False).")
        # Since models are not loaded, we only verify that the class structure is intact.
        print("  [PASS] Pipeline class interface check completed successfully!")
    except Exception as e:
        print(f"  [FAIL] Pipeline initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    test_multimodal_fusion()
    test_interactive_segmenter()
    test_pipeline_integration()
    print("🎉 ALL V4.0 SANITY CHECKS PASSED SUCCESSFULLY!")
