import marshal
import pathlib

pyc_path = pathlib.Path(r"D:\DoAn_DaLieu\__pycache__\app_streamlit.cpython-313.pyc")
data = pyc_path.read_bytes()
code_obj = marshal.loads(data[16:])

out_consts_path = pathlib.Path(r"D:\DoAn_DaLieu\scratch\decompiled_consts_313.txt")

def dump_consts_recursive(co, f, prefix=""):
    f.write(f"\n=========================================\n")
    f.write(f"Code object: {prefix}{co.co_name}\n")
    f.write(f"=========================================\n")
    
    # In ra names và consts
    f.write(f"Names: {list(co.co_names)}\n\n")
    
    for i, const in enumerate(co.co_consts):
        # Nếu const là code object, đệ quy tiếp
        if type(const).__name__ == 'code':
            f.write(f"Const {i} (CodeObject): {const.co_name}\n")
            dump_consts_recursive(const, f, prefix + f"{co.co_name} -> ")
        else:
            # Ghi ra kiểu và giá trị hằng số
            f.write(f"Const {i} ({type(const).__name__}): {repr(const)}\n")

with open(out_consts_path, "w", encoding="utf-8") as f:
    dump_consts_recursive(code_obj, f)

print(f"Dumped all constants recursively to {out_consts_path}")
