import py_compile

try:
    py_compile.compile("app_streamlit.py", doraise=True)
    print("Syntax verification passed! No compilation errors.")
except Exception as e:
    print("Compilation failed:")
    print(e)
