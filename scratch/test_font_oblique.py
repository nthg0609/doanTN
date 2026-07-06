from fpdf import FPDF

def test():
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("DejaVu", "", "fonts/DejaVuSans.ttf")
        print("Regular added successfully")
        pdf.add_font("DejaVu", "B", "fonts/DejaVuSans-Bold.ttf")
        print("Bold added successfully")
        pdf.add_font("DejaVu", "I", "fonts/DejaVuSans-Oblique.ttf")
        print("Italic added successfully")
        pdf.set_font("DejaVu", "I", 11)
        pdf.cell(0, 10, "Thử nghiệm chữ nghiêng Tiếng Việt", new_x="LMARGIN", new_y="NEXT")
        out = pdf.output()
        print("Output success, size:", len(out))
    except Exception as e:
        print("Failed during font registration test:", e)

if __name__ == "__main__":
    test()
