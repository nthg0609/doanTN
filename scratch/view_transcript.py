def view_lines():
    path = r"C:\Users\nguye\.gemini\antigravity-ide\brain\34de1d54-81e4-4fd5-b165-4b09b6313356\.system_generated\logs\transcript.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        for i in range(5):
            line = f.readline()
            if not line:
                break
            print(f"Line {i}: {line[:300]}")

if __name__ == "__main__":
    view_lines()
