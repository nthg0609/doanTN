import json

def search_transcript():
    path = r"C:\Users\nguye\.gemini\antigravity-ide\brain\34de1d54-81e4-4fd5-b165-4b09b6313356\.system_generated\logs\transcript.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                obj = json.loads(line)
                content = obj.get("content", "")
                if "None" in content or "none" in content or "tải báo cáo" in content.lower():
                    print(f"Step {obj.get('step_index')}, type: {obj.get('type')}:")
                    print(content[:500])
                    print("-" * 50)
            except Exception as e:
                pass

if __name__ == "__main__":
    search_transcript()
