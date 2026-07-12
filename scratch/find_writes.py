import ast

def find_writes():
    with open("app_streamlit.py", "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    display_methods = ["write", "markdown", "text", "caption", "latex", "code", "json", "html"]
    
    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node):
            # check if it is st.something or col.something or container.something
            is_display = False
            method_name = ""
            if isinstance(node.func, ast.Attribute):
                method_name = node.func.attr
                if method_name in display_methods:
                    is_display = True
            
            if is_display:
                try:
                    args = [ast.unparse(a) for a in node.args]
                except Exception:
                    args = ["?"]
                print(f"[Line {node.lineno}]: {ast.unparse(node.func)}({', '.join(args)})")
            self.generic_visit(node)

    Visitor().visit(tree)

if __name__ == "__main__":
    find_writes()
