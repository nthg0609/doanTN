import os
import sys
import time
import torch
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.unified_pipeline import UnifiedDermatologyPipeline
from pipeline.model_registry import ModelRegistry

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = UnifiedDermatologyPipeline(mode="both", load_models=True)

# Run benchmark on a few images
test_dir = "d:/DoAn_DaLieu/1_Data/processed/roi_data/test"
# Find first image
img_path = None
for root, dirs, files in os.walk(test_dir):
    for f in files:
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(root, f)
            break
    if img_path:
        break

if not img_path:
    print("No test images found")
    sys.exit(1)

print(f"Benchmarking with image: {img_path}")

# Warmup
for _ in range(5):
    _ = pipeline.run(img_path, return_mask=True)

# Benchmark runs
num_runs = 20
times = {
    "load_preprocess": [],
    "segmentation": [],
    "metrics_roi": [],
    "classification": [],
    "total": []
}

for _ in range(num_runs):
    t0 = time.perf_counter()
    
    # 1. Preprocessing & Load
    t_load_start = time.perf_counter()
    img_rgb, resolved = pipeline._safe_load_rgb(img_path)
    img_type = pipeline._detect_image_type(img_rgb, resolved)
    t_load_end = time.perf_counter()
    times["load_preprocess"].append((t_load_end - t_load_start) * 1000)
    
    # 2. Segmentation
    t_seg_start = time.perf_counter()
    seg_mask, seg_info = pipeline._segment(img_rgb, image_type=img_type)
    t_seg_end = time.perf_counter()
    times["segmentation"].append((t_seg_end - t_seg_start) * 1000)
    
    # 3. Metrics/ROI extraction
    t_roi_start = time.perf_counter()
    metrics = pipeline._get_lesion_metrics(seg_mask)
    t_roi_end = time.perf_counter()
    times["metrics_roi"].append((t_roi_end - t_roi_start) * 1000)
    
    # 4. Classification
    t_cls_start = time.perf_counter()
    cls_result = pipeline._classify(img_rgb)
    t_cls_end = time.perf_counter()
    times["classification"].append((t_cls_end - t_cls_start) * 1000)
    
    t_total_end = time.perf_counter()
    times["total"].append((t_total_end - t0) * 1000)

print("\n=== PIPELINE LATENCY BREAKDOWN (ms) ===")
for k, v in times.items():
    print(f"{k}: {np.mean(v):.2f} ms ± {np.std(v):.2f} ms")

mean_total = np.mean(times["total"])
print(f"\nTime degradation ratios:")
for k in ["load_preprocess", "segmentation", "metrics_roi", "classification"]:
    ratio = np.mean(times[k]) / mean_total * 100
    print(f"  {k}: {ratio:.2f}%")
