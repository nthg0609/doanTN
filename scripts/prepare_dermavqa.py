import os
import json
from pathlib import Path
from tqdm import tqdm

def main():
    base_dir = Path(r"d:\DoAn_DaLieu\9_VQA\dermavqa_dataset")
    iiyi_dir = base_dir / "data" / "iiyi"
    images_dir = base_dir / "images"
    
    # We will process train and valid JSONs
    json_files = [
        iiyi_dir / "train.json",
        iiyi_dir / "valid_ht.json",
        iiyi_dir / "test_ht.json"
    ]
    
    out_data = []
    
    for jpath in json_files:
        if not jpath.exists():
            print(f"Skipping {jpath}, file not found.")
            continue
            
        print(f"Processing {jpath.name}...")
        with open(jpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for item in tqdm(data):
            query = item.get("query_content_en", "").strip()
            if not query:
                query = item.get("query_title_en", "").strip()
                
            if not query:
                continue
                
            responses = item.get("responses", [])
            if not responses:
                continue
                
            # Take the first available English response
            answer = ""
            for resp in responses:
                ans = resp.get("content_en", "").strip()
                if ans:
                    answer = ans
                    break
                    
            if not answer:
                continue
                
            image_ids = item.get("image_ids", [])
            for img_id in image_ids:
                # Some JSONs have just the ID without .jpg, some have .jpg
                if not img_id.endswith(".jpg") and not img_id.endswith(".png"):
                    img_id += ".jpg"
                    
                img_path_rel = f"images/{img_id}"
                img_path_abs = images_dir / img_id
                
                # We can optionally check if image exists, but since we are unzipping async, 
                # we just assume they will be there.
                
                out_data.append({
                    "image_path": img_path_rel,
                    "question": query,
                    "answer": answer
                })
                
    out_path = base_dir / "QA_pairs.json"
    print(f"\nTotal extracted QA pairs: {len(out_data)}")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, ensure_ascii=False, indent=4)
    print(f"Saved Unified QA to {out_path}")

if __name__ == "__main__":
    main()
