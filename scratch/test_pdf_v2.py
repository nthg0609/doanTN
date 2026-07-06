import os
import sys

def test():
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        
        # Load Unicode font
        font_path = r"d:\DoAn_DaLieu\fonts\DejaVuSans.ttf"
        font_bold_path = r"d:\DoAn_DaLieu\fonts\DejaVuSans-Bold.ttf"
        
        pdf.add_font('DejaVu', '', font_path)
        pdf.add_font('DejaVu', 'B', font_bold_path)
        
        pdf.set_font('DejaVu', 'B', 16)
        pdf.cell(0, 10, "BÁO CÁO CHẨN ĐOÁN DA LIỄU AI", new_x="LMARGIN", new_y="NEXT", align="C")
        
        pdf.set_font('DejaVu', '', 11)
        pdf.cell(0, 7, "Bệnh nhân: Nguyễn Văn A", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 7, "Quê quán: Hà Nội, Việt Nam", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 7, "Vị trí tổn thương: Khuỷu tay", new_x="LMARGIN", new_y="NEXT")
        
        # Output directly as bytes without dest="S"
        out = pdf.output()
        print("Output type:", type(out))
        print("Output size:", len(out))
        
        # Save to file to manually inspect if needed
        with open("scratch/test_vietnamese.pdf", "wb") as f:
            f.write(out)
        print("Saved test PDF successfully!")
            
    except Exception as e:
        print("Error during PDF test:", e)

if __name__ == "__main__":
    test()
