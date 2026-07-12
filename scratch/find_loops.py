import ast

def find_all_loops_with_writes():
    with open("app_streamlit.py", "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    class LoopVisitor(ast.NodeVisitor):
        def visit_For(self, node):
            writes = []
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                    method_name = child.func.attr
                    if method_name in ["write", "markdown", "text", "caption", "info", "warning", "error", "success", "button", "columns"]:
                        writes.append(method_name)
            
            if writes:
                try:
                    source_segment = ast.get_source_segment(open("app_streamlit.py", "r", encoding="utf-8").read(), node)
                    first_line = source_segment.split('\n')[0] if source_segment else ""
                except Exception:
                    first_line = "Line " + str(node.lineno)
                print(f"[Loop at line {node.lineno}]: {first_line}")
                print(f"  Write calls: {writes}")
            self.generic_visit(node)

    LoopVisitor().visit(tree)

if __name__ == "__main__":
    find_all_loops_with_writes()
