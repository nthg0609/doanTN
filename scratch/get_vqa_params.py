import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from scripts.train_vqa_joint import VisionBackbone, CPUMedicalVQAModel, PEFT_AVAILABLE

if PEFT_AVAILABLE:
    from peft import LoraConfig, get_peft_model
else:
    print("PEFT not available")

device = torch.device("cpu")
vision_backbone = VisionBackbone().to(device)
model = CPUMedicalVQAModel(vision_backbone).to(device)

# Freeze main vision but unfreeze CBAM
for p in model.vision_backbone.parameters():
    p.requires_grad = False
for p in model.vision_backbone.attention.parameters():
    p.requires_grad = True

# PEFT LoRA
if PEFT_AVAILABLE:
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model.llm = get_peft_model(model.llm, peft_config)

# Enable projection layer
for p in model.projection.parameters():
    p.requires_grad = True

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Trainable percentage: {trainable_params / total_params * 100:.4f}%")
