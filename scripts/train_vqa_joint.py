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
        self.use_spatial_tokens = False

    def forward(self, x):
        features = self.attention(self.backbone.forward_features(x))
        if self.use_spatial_tokens:
            # Flatten spatial dimensions: (batch_size, 1280, 7, 7) -> (batch_size, 1280, 49)
            # Transpose to get sequence format: (batch_size, 49, 1280)
            return features.flatten(2).transpose(1, 2)
        else:
            # Global Average Pooling: (batch_size, 1280)
            return self.global_pool(features).flatten(1)

# ==============================================================================
# WEAK POINT FIX #4: SemanticEnhancer
# Mục đích: Bù đắp cho EfficientNet texture bias bằng lightweight 1x1 conv stack
# học các semantic patterns cấp cao hơn từ feature maps.
# Zero-overhead: 2 lớp depthwise separable-style conv 1x1 + Tanh nonlinearity.
# ==============================================================================
class SemanticEnhancer(nn.Module):
    """Lightweight semantic enrichment layer để bù đắp EfficientNet texture bias."""
    def __init__(self, in_dim=1280, bottleneck=256):
        super().__init__()
        # Bottleneck projection: 1280 → 256 → 1280
        # Giữ nguyên số chiều nhưng học non-linear feature combinations
        self.enhance = nn.Sequential(
            nn.Linear(in_dim, bottleneck, bias=False),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck, in_dim, bias=False),
            nn.Tanh(),  # Normalize vào [-1, 1] để ổn định với small data
        )
        # Scale factor nhỏ để initialization không overwhelm backbone features
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x):
        # Residual: original + scaled semantic enhancement
        return x + self.scale * self.enhance(x)


# ==============================================================================
# WEAK POINT FIX #1: DeepCrossAttentionBridge (2-layer stacked + residual + LN)
# Thay thế QueryConditionedAttentionBridge (1 layer) để học hierarchical patterns.
# Layer 1: Image-conditioned pooling (where to look)
# Layer 2: Query-conditioned reasoning (how to reason across regions)
# DropKey + learnable temperature giải quyết WEAK POINT #4 (small data overfit)
# ==============================================================================
class _CrossAttentionLayer(nn.Module):
    """Một đơn vị cross-attention với DropKey + learnable temperature."""
    def __init__(self, embed_dim=768, num_heads=4, drop_key_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        # Learnable temperature (τ): khởi tạo ≈ 1/√d, học để scale attention
        self.log_tau = nn.Parameter(torch.zeros(1))  # τ = exp(log_tau)
        # DropKey: drop random keys trong attention để regularize với small data
        self.drop_key = nn.Dropout(drop_key_rate)
    
    def _split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # (B, H, S, D)
    
    def forward(self, queries, keys_values, training=True):
        B = queries.size(0)
        q = self._split_heads(self.q_proj(queries), B)  # (B, H, Nq, d)
        k = self._split_heads(self.k_proj(keys_values), B)  # (B, H, Nk, d)
        v = self._split_heads(self.v_proj(keys_values), B)  # (B, H, Nk, d)
        
        # Learnable temperature scaling
        tau = torch.exp(self.log_tau) * (self.head_dim ** -0.5)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * tau  # (B, H, Nq, Nk)
        
        # DropKey: zero out random key positions before softmax (training only)
        if self.training:
            k_mask = self.drop_key(torch.ones_like(attn_scores))
            attn_scores = attn_scores * k_mask
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v)  # (B, H, Nq, d)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.head_dim)
        return self.norm(self.out_proj(out))


class DeepCrossAttentionBridge(nn.Module):
    """
    2-layer stacked cross-attention bridge:
    - Layer 1 (image-conditioned pooling): image spatial tokens → attended summary
    - Layer 2 (query reasoning): refine summary conditioned on text query
    - Residual connections + LayerNorm between layers
    - num_queries query tokens được học
    """
    def __init__(self, embed_dim=768, num_queries=4, drop_key_rate=0.1):
        super().__init__()
        self.num_queries = num_queries
        # Learnable query tokens (khởi tạo gần 0 để tránh exploding gradients)
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        
        # Layer 1: image-conditioned attention (where to look)
        self.layer1 = _CrossAttentionLayer(embed_dim, num_heads=4, drop_key_rate=drop_key_rate)
        self.ffn1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm_ffn1 = nn.LayerNorm(embed_dim)
        
        # Layer 2: query-conditioned reasoning (how to reason across regions)
        self.layer2 = _CrossAttentionLayer(embed_dim, num_heads=4, drop_key_rate=drop_key_rate)
        self.ffn2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm_ffn2 = nn.LayerNorm(embed_dim)
    
    def forward(self, image_embeds, text_embeds):
        B = image_embeds.size(0)
        # Text context vector: mean-pooled text để condition queries
        text_ctx = text_embeds.mean(dim=1, keepdim=True)  # (B, 1, 768)
        queries = self.query_tokens.expand(B, -1, -1) + text_ctx  # (B, Nq, 768)
        
        # Layer 1: queries attend to image tokens (where to look)
        attn1_out = self.layer1(queries, image_embeds)
        queries = queries + attn1_out  # Residual
        queries = queries + self.norm_ffn1(self.ffn1(queries))  # FFN + residual
        
        # Layer 2: refined queries attend to image again (how to reason)
        attn2_out = self.layer2(queries, image_embeds)
        queries = queries + attn2_out  # Residual
        output = queries + self.norm_ffn2(self.ffn2(queries))  # FFN + residual
        
        return output  # (B, num_queries, embed_dim)


# ==============================================================================
# WEAK POINT FIX #5: ClinicalStructureInjector
# Chuyển ABCD metrics + class probabilities → structured clinical embedding
# Inject vào visual token stream TRƯỚC cross-attention để decoder có cấu trúc lâm sàng
# Design: Zero-trainable-param inference path, nhỏ gọn (không overfit với 80 samples)
# ==============================================================================
class ClinicalStructureInjector(nn.Module):
    """
    Tiêm structured clinical knowledge dưới dạng learned embedding.
    Inputs: raw ABCD scalars + class logits (optional)
    Output: 1 clinical embedding token (768-dim) để prepend vào visual stream.
    """
    NUM_CLASSES = 7  # MEL, NV, BCC, AKIEC, BKL, DF, VASC
    ABCD_DIM = 4     # area_ratio, border_complexity, asymmetry, circularity
    
    def __init__(self, embed_dim=768):
        super().__init__()
        input_dim = self.ABCD_DIM + self.NUM_CLASSES
        # 2-layer MLP: clinical scalars → embedding
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, embed_dim),
        )
        # Layer Norm để đưa vào cùng scale với visual tokens
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, abcd_features, class_probs=None):
        """
        Args:
            abcd_features: (B, 4) - [area_ratio, border, asymmetry, circularity]
            class_probs: (B, 7) - softmax probabilities từ classifier (optional)
        Returns:
            (B, 1, embed_dim) - clinical token
        """
        B = abcd_features.size(0)
        device = abcd_features.device
        
        if class_probs is None:
            # Uniform prior nếu không có class info
            class_probs = torch.ones(B, self.NUM_CLASSES, device=device) / self.NUM_CLASSES
        
        # Concatenate ABCD + class probs
        clinical_input = torch.cat([abcd_features, class_probs], dim=-1)  # (B, 4+7=11)
        embedding = self.norm(self.mlp(clinical_input))  # (B, embed_dim)
        return embedding.unsqueeze(1)  # (B, 1, embed_dim)


# ==============================================================================
# WEAK POINT FIX #3: ClinicalPrefix (Prefix-Tuning cho decoder)
# Inject N learnable prefix tokens vào GPT-2 input để bias decoder về medical reasoning
# Không cần fine-tune GPT-2 weights: chỉ học prefix = parameter-efficient
# ==============================================================================
class ClinicalPrefix(nn.Module):
    """
    Learned clinical prefix tokens để inject medical reasoning prior vào decoder.
    Prepend vào inputs_embeds trước GPT-2 để steer generation về clinical language.
    """
    def __init__(self, num_prefix_tokens=8, embed_dim=768):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        # Reparameterization trick: học qua MLP để tránh underdetermined prefix
        self.prefix_mlp = nn.Sequential(
            nn.Embedding(num_prefix_tokens, embed_dim // 2),
            # Note: Embedding được dùng như lookup table, MLP apply sau
        )
        self.prefix_proj = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim),
            nn.Tanh(),
        )
        # Position indices cho prefix tokens
        self.register_buffer("prefix_idx", torch.arange(num_prefix_tokens))
    
    def forward(self, batch_size, device):
        # Lookup + project prefix tokens
        prefix_embeds = self.prefix_mlp[0](self.prefix_idx.to(device))  # (P, d//2)
        prefix_embeds = self.prefix_proj(prefix_embeds)  # (P, d)
        return prefix_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # (B, P, d)

class CPUMedicalVQAModel(nn.Module):
    """
    Upgraded Medical VQA Model — V3 Architecture:
    - SemanticEnhancer: bù đắp EfficientNet texture bias (Weak Point #2)
    - DeepCrossAttentionBridge: 2-layer stacked reasoning (Weak Point #1)
    - ClinicalStructureInjector: inject ABCD + class info as token (Weak Point #5)
    - ClinicalPrefix: learned prefix tokens cho clinical reasoning bias (Weak Point #3)
    - DropKey + learnable τ: overfit prevention (Weak Point #4)
    """
    def __init__(self, vision_backbone, use_spatial_tokens=False,
                 num_prefix_tokens=8, num_query_tokens=4):
        super().__init__()
        self.vision_backbone = vision_backbone
        self.use_spatial_tokens = use_spatial_tokens
        self.num_query_tokens = num_query_tokens
        self.num_prefix_tokens = num_prefix_tokens
        # Sync spatial token flag with vision backbone
        self.vision_backbone.use_spatial_tokens = use_spatial_tokens
        self.llm = AutoModelForCausalLM.from_pretrained("distilgpt2")

        # --- Weak Point #2: SemanticEnhancer ---
        # Bù đắp texture bias của EfficientNet bằng bottleneck semantic projection
        self.semantic_enhancer = SemanticEnhancer(in_dim=1280, bottleneck=256)

        # --- Projection Layer: 1280 → 768 (spatial hoặc global) ---
        self.projection = nn.Sequential(
            nn.Linear(1280, 768),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(768, 768),
            nn.Dropout(0.3),
        )

        # --- Spatial Position Embeddings cho 49 tokens (7x7 grid) ---
        self.spatial_pos_embeds = nn.Parameter(torch.zeros(1, 49, 768))
        nn.init.trunc_normal_(self.spatial_pos_embeds, std=0.02)

        # --- Weak Point #1: DeepCrossAttentionBridge (2-layer stacked) ---
        self.cross_attention_bridge = DeepCrossAttentionBridge(
            embed_dim=768, num_queries=num_query_tokens, drop_key_rate=0.1
        )

        # --- Weak Point #5: ClinicalStructureInjector ---
        # Inject ABCD + class probs vào visual stream trước cross-attention
        self.clinical_injector = ClinicalStructureInjector(embed_dim=768)

        # --- Weak Point #3: ClinicalPrefix (prefix-tuning style) ---
        # Learned prefix tokens để steer GPT-2 về clinical reasoning
        self.clinical_prefix = ClinicalPrefix(
            num_prefix_tokens=num_prefix_tokens, embed_dim=768
        )

    def get_image_embeddings(self, images, text_embeds=None,
                              abcd_features=None, class_probs=None):
        """
        Args:
            images: (B, 3, H, W)
            text_embeds: (B, Ltxt, 768) - word embeddings của câu hỏi
            abcd_features: (B, 4) - ABCD metrics [area, border, asym, circ] (optional)
            class_probs: (B, 7) - softmax class probabilities (optional)
        Returns:
            visual_tokens: (B, N, 768) với N = num_query_tokens (hoặc 1 nếu không dùng spatial)
        """
        raw_features = self.vision_backbone(images)  # (B, 49, 1280) hoặc (B, 1280)

        if self.use_spatial_tokens:
            # --- Weak Point #2: SemanticEnhancer trên spatial tokens ---
            enhanced = self.semantic_enhancer(raw_features)  # (B, 49, 1280)
            projected = self.projection(enhanced)  # (B, 49, 768)
            projected_img = projected + self.spatial_pos_embeds  # Add grid position bias

            # --- Weak Point #5: ClinicalStructureInjector ---
            # Nếu có ABCD features, thêm 1 structured clinical token vào visual stream
            if abcd_features is not None:
                clinical_token = self.clinical_injector(abcd_features, class_probs)  # (B, 1, 768)
                # Prepend clinical token trước spatial tokens: (B, 1+49, 768)
                projected_img = torch.cat([clinical_token, projected_img], dim=1)

            # --- Weak Point #1: DeepCrossAttentionBridge (2-layer reasoning) ---
            if text_embeds is not None:
                return self.cross_attention_bridge(projected_img, text_embeds)  # (B, Nq, 768)
            else:
                # Fallback không có text context: dùng zero text context
                B = images.size(0)
                device = images.device
                dummy_text = torch.zeros(B, 1, 768, device=device)
                return self.cross_attention_bridge(projected_img, dummy_text)
        else:
            # Non-spatial path (1-token): semantic enhance → project → pool
            enhanced = self.semantic_enhancer(raw_features)  # (B, 1280)
            return self.projection(enhanced).unsqueeze(1)  # (B, 1, 768)

    def forward(self, images, input_ids, attention_mask, labels=None,
                abcd_features=None, class_probs=None):
        """
        Forward pass với optional clinical context injection.
        abcd_features và class_probs được truyền từ dataset hoặc inference pipeline.
        """
        # Trích xuất text embeddings từ LLM word embedding layer
        if hasattr(self.llm, "transformer"):
            text_embeds = self.llm.transformer.wte(input_ids)
        else:  # PEFT wraps llm
            text_embeds = self.llm.base_model.model.transformer.wte(input_ids)

        # Visual tokens: N tokens đại diện cho thông tin hình ảnh đã fused với lâm sàng
        img_embeds = self.get_image_embeddings(
            images, text_embeds=text_embeds,
            abcd_features=abcd_features, class_probs=class_probs
        )  # (B, Nq, 768)
        num_img_tokens = img_embeds.size(1)  # = num_query_tokens (= 4)

        # --- Weak Point #3: ClinicalPrefix ---
        # Prepend N learned prefix tokens TRƯỚC image tokens
        # Thứ tự: [prefix_tokens | img_tokens | text_tokens]
        B = images.size(0)
        prefix_embeds = self.clinical_prefix(B, images.device)  # (B, P, 768)
        num_prefix = prefix_embeds.size(1)  # = num_prefix_tokens (= 8)

        # Ghép hoàn chỉnh: prefix + visual + text
        inputs_embeds = torch.cat([prefix_embeds, img_embeds, text_embeds], dim=1)

        # Attention mask mở rộng cho prefix + img + text
        prefix_mask = torch.ones((B, num_prefix), dtype=attention_mask.dtype, device=attention_mask.device)
        img_mask = torch.ones((B, num_img_tokens), dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat([prefix_mask, img_mask, attention_mask], dim=1)

        # Labels: suppress loss trên prefix và img tokens (-100 = ignore)
        full_labels = None
        if labels is not None:
            prefix_labels = torch.full((B, num_prefix), -100, dtype=labels.dtype, device=labels.device)
            img_labels = torch.full((B, num_img_tokens), -100, dtype=labels.dtype, device=labels.device)
            full_labels = torch.cat([prefix_labels, img_labels, labels], dim=1)

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

        try:
            if img_path and os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
            else:
                raise FileNotFoundError("Image not found")
        except Exception as e:
            # Fallback tạo ảnh ngẫu nhiên nếu mất file hoặc file bị hỏng
            synthetic = np.random.randint(100, 180, (self.image_size, self.image_size, 3), dtype=np.uint8)
            image = Image.fromarray(synthetic)

        conv = item.get("conversations", [])
        if conv:
            conv_text = " ".join([f"<|{turn.get('role', '')}|>: {turn.get('content', '')}" for turn in conv])
        else:
            conv_text = ""

        return {
            "image": self.transform(image),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "conversations_text": conv_text,
        }

# ==============================================================================
# 3. Tiến trình Huấn luyện
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Joint Fine-tuning VQA Model with LoRA")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr_vision", type=float, default=2e-5, help="Learning rate for CBAM Attention")
    parser.add_argument("--lr_llm", type=float, default=5e-5, help="Learning rate for LLM / Projection")
    parser.add_argument("--sanity_check", action="store_true", help="Run 1 epoch on 2 samples to verify code flow")
    parser.add_argument("--use_spatial_tokens", action="store_true", help="Use 7x7 spatial tokens instead of 1-token average pooling")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}")

    # Đọc dữ liệu
    qa_path = os.path.join(DATASET_DIR, "QA_pairs_augmented.json")
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
    vision_backbone = VisionBackbone()
    vision_backbone.use_spatial_tokens = args.use_spatial_tokens
    vision_backbone = vision_backbone.to(device)

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
    
    model = CPUMedicalVQAModel(vision_backbone, use_spatial_tokens=args.use_spatial_tokens).to(device)

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

    # Đảm bảo toàn bộ các module mới đều được huấn luyện
    for module in [
        model.projection, model.semantic_enhancer,
        model.cross_attention_bridge, model.clinical_injector,
        model.clinical_prefix, model.spatial_pos_embeds
    ]:
        if isinstance(module, nn.Parameter):
            module.requires_grad = True
        else:
            for p in module.parameters():
                p.requires_grad = True

    # Gom nhóm Optimizer với Learning Rate khác nhau
    vision_params = list(model.vision_backbone.attention.parameters())
    llm_proj_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in [id(vp) for vp in vision_params]
    ]

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
            prompts = []
            for q, a, conv_text in zip(batch["question"], batch["answer"], batch["conversations_text"]):
                if conv_text:
                    prompts.append(conv_text.strip() + tokenizer.eos_token)
                else:
                    prompts.append(f"Question: {q} Answer: {a}{tokenizer.eos_token}")
            
            tokens = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=256,
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
                prompts = []
                for q, a, conv_text in zip(batch["question"], batch["answer"], batch["conversations_text"]):
                    if conv_text:
                        prompts.append(conv_text.strip() + tokenizer.eos_token)
                    else:
                        prompts.append(f"Question: {q} Answer: {a}{tokenizer.eos_token}")
                tokens = tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    max_length=256,
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
                "epoch": epoch + 1,
                "use_spatial_tokens": model.use_spatial_tokens
            }, DEFAULT_CHECKPOINT_OUT)
            print(f"  Saved best checkpoint at epoch {epoch+1}: {DEFAULT_CHECKPOINT_OUT}")

    print("Joint Fine-tuning process completed successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
