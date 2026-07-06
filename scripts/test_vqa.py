import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer

BASE_DIR = Path("d:/DoAn_DaLieu")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.train_vqa_joint import VisionBackbone, CPUMedicalVQAModel
from PIL import Image
import torchvision.transforms as transforms

def test_vqa():
    model_path = r"d:\DoAn_DaLieu\9_VQA\models\dermavqa_gpt2_joint_best.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading best checkpoint from {model_path} onto {device}...")
    
    # Init architecture
    vision_backbone = VisionBackbone().to(device)
    model = CPUMedicalVQAModel(vision_backbone).to(device)
    
    # Wrap LLM with PEFT LoRA so state_dict keys match
    try:
        from peft import LoraConfig, get_peft_model
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model.llm = get_peft_model(model.llm, peft_config)
    except ImportError:
        pass

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    use_spatial_tokens = checkpoint.get("use_spatial_tokens", False)
    model.use_spatial_tokens = use_spatial_tokens
    model.vision_backbone.use_spatial_tokens = use_spatial_tokens
    has_prefix = any(k.startswith("clinical_prefix.") for k in checkpoint["model_state_dict"].keys())
    model.has_prefix = has_prefix
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Model loaded successfully!")
    
    # Lấy thử 1 ảnh từ thư mục dataset mới
    import glob
    import os
    img_files = glob.glob(r"d:\DoAn_DaLieu\9_VQA\dermavqa_dataset\images\*.jpg")
    if not img_files:
        print("No images found for testing.")
        return
        
    test_img_path = img_files[0]
    print(f"\nTesting with image: {os.path.basename(test_img_path)}")
    
    image = Image.open(test_img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Prompt
    prompt = "<|user|>: Doctor, I've been experiencing some unusual symptoms. I have a systemic rash and I've also been told I have pleural effusion. Can you help me understand what's going on? <|doctor|>:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print("\nGenerating response...")
    with torch.no_grad():
        if hasattr(model.llm, "transformer"):
            text_embeds = model.llm.transformer.wte(inputs["input_ids"])
        elif hasattr(model.llm, "base_model"):
            text_embeds = model.llm.base_model.model.transformer.wte(inputs["input_ids"])
        else:
            text_embeds = None

        if hasattr(model, "get_image_embeddings"):
            img_embeds = model.get_image_embeddings(img_tensor, text_embeds=text_embeds)
        else:
            img_embeds = model.projection(model.vision_backbone(img_tensor)).unsqueeze(1)

        # --- V3: Prepend ClinicalPrefix tokens nếu model có ---
        if hasattr(model, "clinical_prefix") and getattr(model, "has_prefix", True):
            prefix_embeds = model.clinical_prefix(img_tensor.size(0), img_tensor.device)
            inputs_embeds = torch.cat([prefix_embeds, img_embeds, text_embeds], dim=1)
        else:
            inputs_embeds = torch.cat([img_embeds, text_embeds], dim=1)
        
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long)
        
        generated_ids = model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=100,
            repetition_penalty=1.2,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
        print("Generated IDs:", generated_ids)
        
    # The output includes the dummy img token at position 0, so we slice
    # Actually, generating from inputs_embeds returns the sequence.
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"\n[Generated Answer]: {output_text}")

if __name__ == "__main__":
    test_vqa()
