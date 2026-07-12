import ast

def find_st_calls():
    with open("app_streamlit.py", "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    class CallVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "st":
                method = node.func.attr
                # Get the first argument or keyword args if any
                args_repr = []
                for arg in node.args:
                    try:
                        args_repr.append(ast.unparse(arg))
                    except Exception:
                        args_repr.append("?")
                for kw in node.keywords:
                    try:
                        args_repr.append(f"{kw.arg}={ast.unparse(kw.value)}")
                    except Exception:
                        args_repr.append(f"{kw.arg}=?")
                
                print(f"[Line {node.lineno}] st.{method}({', '.join(args_repr)})")
            self.generic_visit(node)

    CallVisitor().visit(tree)

if __name__ == "__main__":
    find_st_calls()
