"""
Kịch bản huấn luyện đồng thời (Joint Fine-tuning với LoRA) cho mô hình VQA y tế da liễu.
Các thành phần huấn luyện:
1. Nhánh Vision: Mở băng khối CBAM Attention (Spatial + Channel), đóng băng EfficientNet-B1.
2. Nhánh Projection: Huấn luyện hoàn toàn để dịch chuyển không gian đặc trưng.
3. Nhánh Language: Cấu hình LoRA (PEFT) cho DistilGPT-2 (target c_attn).
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import AutoModelForCausalLM, AutoTokenizer
import timm
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Hỗ trợ import peft linh hoạt
try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Đường dẫn mặc định
BASE_DIR = r"d:\DoAn_DaLieu"
DATASET_DIR = os.path.join(BASE_DIR, "9_VQA", "dermavqa_dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
VQA_MODEL_DIR = os.path.join(BASE_DIR, "9_VQA", "models")
CLASS_MODEL_PATH = os.path.join(BASE_DIR, "4_Models", "efficientnet_attention_best.pth")
DEFAULT_CHECKPOINT_OUT = os.path.join(VQA_MODEL_DIR, "dermavqa_gpt2_joint_best.pth")

# ==============================================================================
# 1. Định nghĩa Kiến trúc Mô hình (CBAM + VisionBackbone + CPUMedicalVQAModel)
# ==============================================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        return x * self.spatial_att(x * self.channel_att(x))

class VisionBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b1", pretrained=False, num_classes=0)
        self.attention = CBAM(self.backbone.num_features, reduction=16)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        features = self.attention(self.backbone.forward_features(x))
        return self.global_pool(features).flatten(1)

class CPUMedicalVQAModel(nn.Module):
    def __init__(self, vision_backbone):
        super().__init__()
        self.vision_backbone = vision_backbone
        self.llm = AutoModelForCausalLM.from_pretrained("distilgpt2")

        # Khởi tạo Projection Layer
        self.projection = nn.Sequential(
            nn.Linear(1280, 768),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(768, 768),
            nn.Dropout(0.3),
        )

    def forward(self, images, input_ids, attention_mask, labels=None):
        img_embeds = self.projection(self.vision_backbone(images)).unsqueeze(1)
        # Sử dụng base transformer wte
        if hasattr(self.llm, "transformer"):
            text_embeds = self.llm.transformer.wte(input_ids)
        else: # PEFT wraps llm
            text_embeds = self.llm.base_model.model.transformer.wte(input_ids)

        inputs_embeds = torch.cat([img_embeds, text_embeds], dim=1)

        img_mask = torch.ones((images.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat([img_mask, attention_mask], dim=1)

        full_labels = None
        if labels is not None:
            img_labels = torch.full((images.size(0), 1), -100, dtype=labels.dtype, device=labels.device)
            full_labels = torch.cat([img_labels, labels], dim=1)

        return self.llm(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=full_labels)

# ==============================================================================
# 2. Tập dữ liệu VQA Dataset
# ==============================================================================

class DermaVQADataset(Dataset):
    def __init__(self, data, image_size=224, augment=False):
        self.data = data
        self.image_size = image_size
        self.augment = augment

        base_ops = [transforms.Resize((image_size, image_size))]
        aug_ops = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
        ]
        norm_ops = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        if self.augment:
            self.transform = transforms.Compose(base_ops + aug_ops + norm_ops)
        else:
            self.transform = transforms.Compose(base_ops + norm_ops)

    def __len__(self):
        return len(self.data)

    def _resolve_img_path(self, item):
        image_path = item.get("image_path", "")
        if os.path.isabs(image_path):
            return image_path
        candidates = [
            os.path.join(BASE_DIR, image_path),
            os.path.join(DATASET_DIR, image_path),
            os.path.join(IMAGES_DIR, os.path.basename(image_path)),
        ]
        for p in candidates:
            if p and os.path.exists(p):
                return p
        return None

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = self._resolve_img_path(item)

        if img_path and os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
        else:
            # Fallback tạo ảnh ngẫu nhiên nếu mất file
            synthetic = np.random.randint(100, 180, (self.image_size, self.image_size, 3), dtype=np.uint8)
            image = Image.fromarray(synthetic)

        return {
            "image": self.transform(image),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
        }

# ==============================================================================
# 3. Tiến trình Huấn luyện
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Joint Fine-tuning VQA Model with LoRA")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr_vision", type=float, default=2e-5, help="Learning rate for CBAM Attention")
    parser.add_argument("--lr_llm", type=float, default=5e-5, help="Learning rate for LLM / Projection")
    parser.add_argument("--sanity_check", action="store_true", help="Run 1 epoch on 2 samples to verify code flow")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}")

    # Đọc dữ liệu
    qa_path = os.path.join(DATASET_DIR, "QA_pairs.json")
    if not os.path.exists(qa_path):
        print(f"[ERROR] QA Dataset not found at: {qa_path}")
        return 1

    with open(qa_path, "r", encoding="utf-8") as f:
        dataset_json = json.load(f)

    if args.sanity_check:
        print("--- SANITY CHECK MODE ACTIVATED ---")
        train_data = dataset_json[:4]
        val_data = dataset_json[4:6]
        args.epochs = 1
        args.batch_size = 2
    else:
        train_data, val_data = train_test_split(dataset_json, test_size=0.15, random_state=42)

    print(f"Dataset summary: Train={len(train_data)} | Val={len(val_data)}")

    # Khởi tạo Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Khởi tạo Model
    vision_backbone = VisionBackbone().to(device)
    
    # Nạp trọng số phân loại nếu có
    if os.path.exists(CLASS_MODEL_PATH):
        try:
            cls_ckpt = torch.load(CLASS_MODEL_PATH, map_location=device)
            cls_state = cls_ckpt.get("model_state_dict", cls_ckpt)
            bb_state = {k.replace("backbone.", ""): v for k, v in cls_state.items() if k.startswith("backbone.")}
            att_state = {k.replace("attention.", ""): v for k, v in cls_state.items() if k.startswith("attention.")}
            vision_backbone.backbone.load_state_dict(bb_state, strict=False)
            vision_backbone.attention.load_state_dict(att_state, strict=False)
            print("Successfully preloaded vision weights from classification checkpoint.")
        except Exception as e:
            print(f"Warning: Could not load classification weights: {e}. Training from scratch.")
    
    model = CPUMedicalVQAModel(vision_backbone).to(device)

    # Đóng băng xương sống Vision chính nhưng mở khóa CBAM Attention
    for p in model.vision_backbone.parameters():
        p.requires_grad = False
    for p in model.vision_backbone.attention.parameters():
        p.requires_grad = True

    # Cấu hình LoRA trên LLM (GPT-2)
    if PEFT_AVAILABLE:
        print("PEFT library available. Injecting LoRA adapter to DistilGPT-2...")
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn"], # Trực quan hóa các lớp Attention của GPT-2
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model.llm = get_peft_model(model.llm, peft_config)
        model.llm.print_trainable_parameters()
    else:
        print("PEFT not installed. Falling back to fine-tuning the last 2 Transformer layers...")
        for p in model.llm.parameters():
            p.requires_grad = False
        for p in model.llm.transformer.h[-2:].parameters():
            p.requires_grad = True
        for p in model.llm.lm_head.parameters():
            p.requires_grad = True

    # Đảm bảo Projection Layer luôn được huấn luyện
    for p in model.projection.parameters():
        p.requires_grad = True

    # Gom nhóm Optimizer với Learning Rate khác nhau
    vision_params = list(model.vision_backbone.attention.parameters())
    llm_proj_params = [p for p in model.parameters() if p.requires_grad and id(p) not in [id(vp) for vp in vision_params]]

    optimizer = torch.optim.AdamW([
        {"params": vision_params, "lr": args.lr_vision},
        {"params": llm_proj_params, "lr": args.lr_llm}
    ], weight_decay=0.05)

    # Loader
    train_dataset = DermaVQADataset(train_data, augment=True)
    val_dataset = DermaVQADataset(val_data, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    best_val_loss = float("inf")

    # Vòng lặp huấn luyện chính
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in loop:
            images = batch["image"].to(device)
            prompts = [f"Question: {q} Answer: {a}{tokenizer.eos_token}" for q, a in zip(batch["question"], batch["answer"])]
            
            tokens = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=96,
                return_tensors="pt"
            ).to(device)
            
            labels = tokens["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100

            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass
            outputs = model(images, tokens["input_ids"], tokens["attention_mask"], labels=labels)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Đánh giá trên tập Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                prompts = [f"Question: {q} Answer: {a}{tokenizer.eos_token}" for q, a in zip(batch["question"], batch["answer"])]
                tokens = tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    max_length=96,
                    return_tensors="pt"
                ).to(device)
                labels = tokens["input_ids"].clone()
                labels[labels == tokenizer.pad_token_id] = -100
                
                outputs = model(images, tokens["input_ids"], tokens["attention_mask"], labels=labels)
                val_loss += outputs.loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} finished. Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Lưu checkpoint tốt nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(DEFAULT_CHECKPOINT_OUT), exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "epoch": epoch + 1
            }, DEFAULT_CHECKPOINT_OUT)
            print(f"  Saved best checkpoint at epoch {epoch+1}: {DEFAULT_CHECKPOINT_OUT}")

    print("Joint Fine-tuning process completed successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
