import os

def list_brain_dir():
    path = r"C:\Users\nguye\.gemini\antigravity-ide\brain\34de1d54-81e4-4fd5-b165-4b09b6313356"
    if not os.path.exists(path):
        print("Path does not exist:", path)
        return
    for root, dirs, files in os.walk(path):
        for file in files:
            full = os.path.join(root, file)
            print(os.path.relpath(full, path))

if __name__ == "__main__":
    list_brain_dir()
