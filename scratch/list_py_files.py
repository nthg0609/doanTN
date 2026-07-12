import os

def list_py_files():
    for root, dirs, files in os.walk("."):
        if ".venv" in root or ".git" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                print(os.path.join(root, file))

if __name__ == "__main__":
    list_py_files()
