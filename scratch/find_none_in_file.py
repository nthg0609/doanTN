import re

def search_none():
    with open("app_streamlit.py", "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    for idx, line in enumerate(lines):
        if "None" in line:
            # check if it's not just a type hint or standard comparison
            # e.g., print lines that are assignments to None or st. calls with None
            print(f"Line {idx+1}: {line.strip()}")

if __name__ == "__main__":
    search_none()
