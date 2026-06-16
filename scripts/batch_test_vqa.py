import sys
import os
import json
from pathlib import Path
import glob
import torch
from transformers import AutoTokenizer
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

BASE_DIR = Path("d:/DoAn_DaLieu")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.train_vqa_joint import VisionBackbone, CPUMedicalVQAModel

def batch_test_vqa():
    model_path = r"d:\DoAn_DaLieu\9_VQA\models\dermavqa_gpt2_joint_best.pth"
    output_file = r"d:\DoAn_DaLieu\9_VQA\inference_results.json"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading best checkpoint from {model_path} onto {device}...")
    
    vision_backbone = VisionBackbone().to(device)
    model = CPUMedicalVQAModel(vision_backbone).to(device)
    
    try:
        from peft import LoraConfig, get_peft_model
        peft_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["c_attn"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )
        model.llm = get_peft_model(model.llm, peft_config)
    except ImportError:
        pass

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Get test images
    img_files = glob.glob(r"d:\DoAn_DaLieu\9_VQA\dermavqa_dataset\images\*.jpg")
    img_files = img_files[:20]  # Just test on 20 images to save time
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Prompt yêu cầu chẩn đoán
    prompt = "<|user|>: What is the diagnosis for this skin condition? <|doctor|>:"
    
    results = []
    
    print(f"\nRunning batch inference on {len(img_files)} images...")
    for img_path in tqdm(img_files):
        try:
            image = Image.open(img_path).convert("RGB")
            img_tensor = transform(image).unsqueeze(0).to(device)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                img_embeds = model.projection(model.vision_backbone(img_tensor)).unsqueeze(1)
                text_embeds = model.llm.base_model.model.transformer.wte(inputs["input_ids"])
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
            
            output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            results.append({
                "image": os.path.basename(img_path),
                "generated_answer": output_text
            })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"\n[SUCCESS] Saved {len(results)} inference results to {output_file}")

if __name__ == "__main__":
    batch_test_vqa()
