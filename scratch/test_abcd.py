import sys
import os
from pathlib import Path

# Add root folder to sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, root_dir)

from pipeline.unified_pipeline import UnifiedDermatologyPipeline
import torch

device = torch.device("cpu")
pipeline = UnifiedDermatologyPipeline(device=device, load_models=True)

img_path = "anh/media__1783268147920.png"
print("Running pipeline on", img_path)
res = pipeline.run(img_path, return_mask=True)
print("Status:", res.get("status"))
print("Metrics:", res.get("metrics"))
