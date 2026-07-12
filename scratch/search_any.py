import json

def search_any():
    path = r"C:\Users\nguye\.gemini\antigravity-ide\brain\34de1d54-81e4-4fd5-b165-4b09b6313356\.system_generated\logs\transcript.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if "None" in line or "none" in line or "tải" in line:
                try:
                    obj = json.loads(line)
                    print(f"Line {idx}, step: {obj.get('step_index')}, type: {obj.get('type')}")
                    # print some keys
                    for k, v in obj.items():
                        if isinstance(v, str) and ("None" in v or "none" in v or "tải" in v):
                            print(f"  {k}: {v[:300]}")
                except Exception:
                    pass

if __name__ == "__main__":
    search_any()
