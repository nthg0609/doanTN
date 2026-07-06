import os
import sys
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, accuracy_score

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.model_registry import ModelRegistry

# Load classification model using ModelRegistry
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
registry = ModelRegistry(device=device)

# Load classification model
print("Loading classification model...")
cls_model = registry.load_classification_model()
idx_to_class = registry.get_class_labels()
print("Class mapping:", idx_to_class)

# Prep transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
test_dir = "d:/DoAn_DaLieu/1_Data/processed/roi_data/test"
dataset = ImageFolder(test_dir, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

print(f"Loaded {len(dataset)} images in test set from {test_dir}")

# Predict
all_preds = []
all_labels = []

cls_model.eval()
with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        outputs = cls_model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Get report
target_names = [idx_to_class[i] for i in range(len(idx_to_class))]
report = classification_report(all_labels, all_preds, target_names=target_names, digits=6)
print("\n=== CLASSIFICATION REPORT ===")
print(report)

acc = accuracy_score(all_labels, all_preds)
print(f"Overall Accuracy: {acc:.6f}")
