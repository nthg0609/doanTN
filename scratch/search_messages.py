import os
import json

def search_messages():
    path = r"C:\Users\nguye\.gemini\antigravity-ide\brain\34de1d54-81e4-4fd5-b165-4b09b6313356\.system_generated\messages"
    for file in os.listdir(path):
        if not file.endswith(".json") or file in ["cursor.json", "read.json"]:
            continue
        full = os.path.join(path, file)
        try:
            with open(full, "r", encoding="utf-8") as f:
                data = json.load(f)
                text = json.dumps(data, ensure_ascii=False)
                if "None" in text or "none" in text or "tải báo cáo" in text.lower():
                    print(f"File {file}:")
                    # print role and content if available
                    if isinstance(data, dict):
                        print("  keys:", list(data.keys()))
                        if "role" in data:
                            print(f"  role: {data['role']}")
                        if "content" in data:
                            print(f"  content: {data['content'][:500]}")
                    print("-" * 50)
        except Exception as e:
            print("Error reading:", file, e)

if __name__ == "__main__":
    search_messages()
