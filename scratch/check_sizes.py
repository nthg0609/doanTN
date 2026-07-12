import os

def check_sizes():
    path = r"C:\Users\nguye\.gemini\antigravity-ide\brain\34de1d54-81e4-4fd5-b165-4b09b6313356\.system_generated"
    for root, dirs, files in os.walk(path):
        for file in files:
            full = os.path.join(root, file)
            size = os.path.getsize(full)
            if size > 0:
                print(f"{os.path.relpath(full, path)}: {size} bytes")

if __name__ == "__main__":
    check_sizes()
