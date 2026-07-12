import os
import json

def print_all_messages():
    path = r"C:\Users\nguye\.gemini\antigravity-ide\brain\34de1d54-81e4-4fd5-b165-4b09b6313356\.system_generated\messages"
    files = sorted(os.listdir(path))
    for file in files:
        if not file.endswith(".json") or file in ["cursor.json", "read.json"]:
            continue
        full = os.path.join(path, file)
        try:
            with open(full, "r", encoding="utf-8") as f:
                data = json.load(f)
                sender = data.get("sender")
                recipient = data.get("recipient")
                content = data.get("content", "")
                print(f"File: {file} | Sender: {sender} | Recipient: {recipient}")
                if isinstance(content, str):
                    print(f"  Content: {content[:300]}")
                else:
                    print(f"  Content: (non-str: {type(content)}) {str(content)[:100]}")
                print("-" * 50)
        except Exception as e:
            pass

if __name__ == "__main__":
    print_all_messages()
