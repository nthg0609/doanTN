from fpdf import FPDF
import datetime

def test():
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Load Unicode font
        font_path = "fonts/DejaVuSans.ttf"
        font_bold_path = "fonts/DejaVuSans-Bold.ttf"
        font_italic_path = "fonts/DejaVuSans-Oblique.ttf"
        
        pdf.add_font("DejaVu", "", font_path)
        pdf.add_font("DejaVu", "B", font_bold_path)
        pdf.add_font("DejaVu", "I", font_italic_path)
        
        # 1. Vẽ đường viền trang trí ở trên cùng (Medical Blue Accent)
        pdf.set_fill_color(37, 99, 235)  # #2563eb
        pdf.rect(15, 10, 180, 3, "F")
        
        pdf.ln(5)
        # 2. Tiêu đề
        pdf.set_font("DejaVu", "B", 15)
        pdf.set_text_color(30, 58, 138)  # #1e3a8a (Deep Medical Blue)
        pdf.cell(0, 10, "BỆNH VIỆN ĐA KHOA QUỐC TẾ AI-DERMA", new_x="LMARGIN", new_y="NEXT", align="C")
        
        pdf.set_font("DejaVu", "", 9)
        pdf.set_text_color(100, 116, 139)  # Slate
        pdf.cell(0, 5, "Hệ thống Bệnh án Điện tử EHR — Trung tâm Phân tích Lâm sàng Da liễu", new_x="LMARGIN", new_y="NEXT", align="C")
        
        pdf.ln(5)
        pdf.set_draw_color(226, 232, 240)  # #e2e8f0
        pdf.line(15, 38, 195, 38)
        
        # Tiêu đề báo cáo
        pdf.ln(5)
        pdf.set_font("DejaVu", "B", 13)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 8, "PHIẾU KẾT QUẢ CHẨN ĐOÁN LÂM SÀNG DA LIỄU", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(4)
        
        # --- PHẦN I: THÔNG TIN HÀNH CHÍNH ---
        pdf.set_font("DejaVu", "B", 10)
        pdf.set_text_color(37, 99, 235)
        pdf.cell(0, 6, "I. THÔNG TIN BỆNH NHÂN", new_x="LMARGIN", new_y="NEXT")
        
        # Thiết lập bảng thông tin
        pdf.set_font("DejaVu", "", 9)
        pdf.set_text_color(30, 41, 59)
        pdf.set_draw_color(203, 213, 225)
        pdf.set_fill_color(248, 250, 252)  # Nền nhạt
        
        # Dòng 1: Họ tên + Giới tính
        pdf.cell(30, 7, " Họ tên bệnh nhân", border=1, fill=True)
        pdf.set_font("DejaVu", "B", 9)
        pdf.cell(70, 7, " NGUYỄN VĂN A", border=1)
        pdf.set_font("DejaVu", "", 9)
        pdf.cell(30, 7, " Giới tính / Tuổi", border=1, fill=True)
        pdf.cell(50, 7, " Nam / 30 tuổi", border=1)
        pdf.ln(7)
        
        # Dòng 2: Quê quán + Vị trí tổn thương
        pdf.cell(30, 7, " Quê quán", border=1, fill=True)
        pdf.cell(70, 7, " Hà Nội, Việt Nam", border=1)
        pdf.cell(30, 7, " Vị trí tổn thương", border=1, fill=True)
        pdf.cell(50, 7, " Khuỷu tay phải", border=1)
        pdf.ln(10)
        
        # --- PHẦN II: PHÂN TÍCH ĐỊNH LƯỢNG AI ---
        pdf.set_font("DejaVu", "B", 10)
        pdf.set_text_color(37, 99, 235)
        pdf.cell(0, 6, "II. KẾT QUẢ ĐỊNH LƯỢNG HÌNH ẢNH (AI METRICS)", new_x="LMARGIN", new_y="NEXT")
        
        # Header bảng AI
        pdf.set_font("DejaVu", "B", 9)
        pdf.set_text_color(255, 255, 255)
        pdf.set_fill_color(30, 58, 138)  # Deep Blue Header
        pdf.cell(70, 7, " Chỉ số đánh giá", border=1, fill=True)
        pdf.cell(40, 7, " Giá trị phân tích", border=1, fill=True, align="C")
        pdf.cell(70, 7, " Đánh giá lâm sàng", border=1, fill=True)
        pdf.ln(7)
        
        pdf.set_text_color(30, 41, 59)
        # Các chỉ số
        metrics = [
            ("Chẩn đoán bệnh lý", "Ung thư biểu mô tế bào đáy (BCC)", "Độ tin cậy: 92.5%"),
            ("Tỉ lệ diện tích (Area ratio)", "0.1452", "Chiếm 14.52% diện tích vùng ảnh"),
            ("Độ phức tạp viền (Border complexity)", "1.8904", "Viền gồ ghề, cấu trúc không đều"),
            ("Độ bất đối xứng (Asymmetry)", "0.6512", "Mức độ bất đối xứng tương đối cao"),
            ("Độ tròn hình học (Circularity)", "0.4521", "Méo mó so với dạng hình tròn chuẩn")
        ]
        
        for idx, (label, val, comment) in enumerate(metrics):
            pdf.set_fill_color(248, 250, 252) if idx % 2 == 0 else pdf.set_fill_color(255, 255, 255)
            pdf.cell(70, 7, f" {label}", border=1, fill=True)
            pdf.cell(40, 7, f" {val}", border=1, fill=True, align="C")
            pdf.cell(70, 7, f" {comment}", border=1, fill=True)
            pdf.ln(7)
            
        pdf.ln(5)
        
        # --- PHẦN III: KHUYẾN NGHỊ LÂM SÀNG ---
        pdf.set_font("DejaVu", "B", 10)
        pdf.set_text_color(37, 99, 235)
        pdf.cell(0, 6, "III. KHUYẾN NGHỊ & ĐỊNH HƯỚNG LÂM SÀNG", new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("DejaVu", "", 9.5)
        pdf.set_text_color(30, 41, 59)
        pdf.set_fill_color(239, 246, 255) # Light blue box
        pdf.set_draw_color(191, 219, 254)
        
        adv = (
            "Kết quả phân tích hình ảnh cho thấy tổn thương có dấu hiệu bất đối xứng và đường viền phức tạp. "
            "Khuyến nghị bác sĩ thực hiện soi da kỹ lưỡng hoặc chỉ định sinh thiết khẩn cấp để chẩn đoán mô bệnh học xác định. "
            "Bệnh nhân nên hạn chế cọ xát vùng tổn thương và tránh tiếp xúc trực tiếp với tia UV."
        )
        pdf.multi_cell(180, 5, adv, border=1, fill=True)
        pdf.ln(5)
        
        # --- CHỮ KÝ BÁC SĨ ---
        pdf.ln(5)
        pdf.set_font("DejaVu", "I", 9.5)
        now_str = datetime.datetime.now().strftime("Hà Nội, ngày %d tháng %m năm %Y")
        pdf.cell(180, 5, now_str, new_x="LMARGIN", new_y="NEXT", align="R")
        pdf.ln(2)
        pdf.set_font("DejaVu", "B", 9.5)
        pdf.cell(180, 5, "Bác sĩ chẩn đoán hình ảnh", new_x="LMARGIN", new_y="NEXT", align="R")
        pdf.set_font("DejaVu", "", 8.5)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(180, 5, "(Ký và ghi rõ họ tên)", new_x="LMARGIN", new_y="NEXT", align="R")
        
        # Disclaimer ở dưới cùng
        pdf.set_y(-25)
        pdf.set_font("DejaVu", "I", 8)
        pdf.set_text_color(239, 68, 68)  # Red warning
        dis = (
            "* TUYÊN BỐ MIỄN TRỪ: Hệ thống AI này chỉ đóng vai trò hỗ trợ sàng lọc lâm sàng sơ bộ dựa trên học máy. "
            "Kết quả phân tích không thể thay thế quyết định chẩn đoán y khoa chuyên môn của bác sĩ da liễu có thẩm quyền."
        )
        pdf.multi_cell(180, 4, dis, align="C")
        
        out = pdf.output()
        with open("scratch/test_beautiful.pdf", "wb") as f:
            f.write(out)
        print("Beautiful PDF saved successfully!")
    except Exception as e:
        print("Error during beautiful PDF test:", e)

if __name__ == "__main__":
    test()
