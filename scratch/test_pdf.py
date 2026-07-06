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
        
        pdf.add_font('DejaVu', '', font_path, uni=True)
        pdf.add_font('DejaVu', 'B', font_bold_path, uni=True)
        
        pdf.set_font('DejaVu', 'B', 16)
        pdf.cell(0, 10, "BÁO CÁO CHẨN ĐOÁN DA LIỄU AI", ln=True, align="C")
        
        pdf.set_font('DejaVu', '', 11)
        pdf.cell(0, 10, "Bệnh nhân: Nguyễn Văn A", ln=True)
        pdf.cell(0, 10, "Quê quán: Hà Nội, Việt Nam", ln=True)
        pdf.cell(0, 10, "Vị trí tổn thương: Khuỷu tay", ln=True)
        
        out = pdf.output(dest="S")
        print("Output type:", type(out))
        if isinstance(out, str):
            print("Output string length:", len(out))
            # Try encoding to see if it raises error
            try:
                b = out.encode("latin-1")
                print("Encoded to latin-1 successfully, size:", len(b))
            except Exception as e:
                print("Encoding to latin-1 failed:", e)
            try:
                b = out.encode("utf-8")
                print("Encoded to utf-8 successfully, size:", len(b))
            except Exception as e:
                print("Encoding to utf-8 failed:", e)
        elif isinstance(out, bytes) or isinstance(out, bytearray):
            print("Output bytes size:", len(out))
            
    except Exception as e:
        print("Error during PDF test:", e)

if __name__ == "__main__":
    test()
