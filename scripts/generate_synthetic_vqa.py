import os
import sys
import json
import time
import argparse
from tqdm import tqdm

try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[ERROR] Thư viện openai hoặc python-dotenv chưa được cài đặt.")
    print("Vui lòng chạy lệnh: pip install openai python-dotenv")
    sys.exit(1)

# Lấy API Key từ biến môi trường
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("[ERROR] Không tìm thấy OPENAI_API_KEY trong biến môi trường.")
    print('Hãy chạy lệnh này trong Terminal: $env:OPENAI_API_KEY="sk-..."')
    sys.exit(1)

client = OpenAI(api_key=api_key)

PROMPT_TEMPLATE = """
You are an expert dermatologist talking to a patient.
I will give you a short question and a short answer regarding a skin condition.
Your task is to rewrite this single-turn Q&A into a natural, multi-turn conversation between a "user" (patient) and a "doctor" (dermatologist).
The conversation must cover the facts in the short answer, explain the medical reasoning clearly but in a friendly tone, use diverse vocabulary, and sound like a real clinical consultation.

Input Question: "{question}"
Input Answer: "{answer}"

Output ONLY a raw JSON array of objects, with no markdown formatting, no ```json tags.
Format exactly like this:
[
  {{"role": "user", "content": "Doctor, I have..."}},
  {{"role": "doctor", "content": "Hello! Looking at this..."}},
  {{"role": "user", "content": "Is it serious?"}},
  {{"role": "doctor", "content": "Not to worry..."}}
]
"""

def generate_multi_turn(question, answer):
    prompt = PROMPT_TEMPLATE.format(question=question, answer=answer)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that only outputs raw JSON arrays without markdown blocks."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        text = response.choices[0].message.content.strip()
        
        # Clean up markdown if model mistakenly added it
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        # Parse JSON
        conversation = json.loads(text.strip())
        return conversation
    except Exception as e:
        print(f"\n[API Error] Lỗi khi sinh dữ liệu: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Sinh dữ liệu VQA nhân tạo bằng GPT-4o-mini")
    parser.add_argument("--input", default=r"d:\DoAn_DaLieu\9_VQA\dermavqa_dataset\QA_pairs.json", help="File QA gốc")
    parser.add_argument("--output", default=r"d:\DoAn_DaLieu\9_VQA\dermavqa_dataset\QA_pairs_augmented.json", help="File đích")
    parser.add_argument("--limit", type=int, default=10, help="Số lượng mẫu muốn sinh (để test API). Đặt 0 để chạy toàn bộ.")
    args = parser.parse_args()

    # Tải file input
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Tải file output nếu đã có (để resume)
    augmented_data = []
    processed_paths = set()
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            try:
                augmented_data = json.load(f)
                processed_paths = {item["image_path"] for item in augmented_data}
                print(f"[INFO] Resume từ {len(augmented_data)} mẫu đã lưu.")
            except json.JSONDecodeError:
                print("[WARNING] File output cũ bị lỗi JSON. Tạo mới từ đầu.")

    # Lọc những mẫu chưa xử lý
    remaining_data = [item for item in data if item["image_path"] not in processed_paths]
    print(f"[INFO] Tổng mẫu: {len(data)}. Đã xử lý: {len(processed_paths)}. Còn lại: {len(remaining_data)}.")

    if args.limit > 0:
        remaining_data = remaining_data[:args.limit]
        print(f"[INFO] Chế độ TEST: Sẽ chạy {args.limit} mẫu.")

    if not remaining_data:
        print("Đã hoàn thành toàn bộ dữ liệu!")
        return

    # Tiến hành xử lý
    for item in tqdm(remaining_data, desc="Đang sinh dữ liệu"):
        q = item.get("question", "")
        a = item.get("answer", "")
        img_path = item.get("image_path", "")

        conversation = generate_multi_turn(q, a)
        
        if conversation:
            new_item = {
                "image_path": img_path,
                "conversations": conversation
            }
            augmented_data.append(new_item)
            
            # Save checkpoint liên tục
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(augmented_data, f, ensure_ascii=False, indent=2)
                
            # Nghỉ 0.5s để tránh vượt quá Rate Limit 
            time.sleep(0.5)
        else:
            # Nếu API lỗi (Quota exceeded), dừng lại để bảo toàn dữ liệu
            print("\n[INFO] Dừng chạy do lỗi API. Bạn có thể chạy lại lệnh này sau để resume.")
            break

    print(f"\n[SUCCESS] Đã lưu {len(augmented_data)} mẫu vào {args.output}")

if __name__ == "__main__":
    main()
