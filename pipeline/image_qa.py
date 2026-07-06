import cv2
import numpy as np
from typing import Dict, Any

def check_image_quality(img_rgb: np.ndarray) -> Dict[str, Any]:
    """
    Kiểm tra chất lượng ảnh đầu vào:
    1. Độ mờ (Blurry) bằng phương sai Laplacian.
    2. Độ sáng (Poor lighting / Overexposed) bằng trung bình cường độ sáng của ảnh xám.
    """
    # Chuyển sang ảnh xám để phân tích cường độ
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # 1. Tính toán độ mờ (Laplacian variance)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # 2. Tính toán độ sáng trung bình (Mean gray intensity)
    mean_brightness = float(np.mean(gray))
    
    # Các ngưỡng tiêu chuẩn lâm sàng đề xuất
    BLURRY_THRESHOLD = 80.0
    DARK_THRESHOLD = 50.0
    BRIGHT_THRESHOLD = 210.0
    
    # Kỹ thuật giảm định kiến thuật toán (Algorithmic Bias Mitigation):
    # Nếu ảnh rất nét (laplacian_var >= 100.0) nhưng tối, có thể do tông da sẫm tự nhiên (Fitzpatrick Type V, VI).
    # Ta tự động hạ ngưỡng tối tối thiểu xuống 30.0 thay vì 50.0 để tránh phân biệt đối xử lâm sàng.
    if laplacian_var >= 100.0:
        DARK_THRESHOLD = 30.0
        
    is_blurry = laplacian_var < BLURRY_THRESHOLD
    is_dark = mean_brightness < DARK_THRESHOLD
    is_bright = mean_brightness > BRIGHT_THRESHOLD
    
    issues = []
    if is_blurry:
        issues.append("ảnh bị mờ/out-focus (độ nét thấp)")
    if is_dark:
        issues.append("ảnh thiếu sáng (quá tối)")
    if is_bright:
        issues.append("ảnh quá sáng (bị lóa/overexposed)")
        
    status = "ok" if not issues else "warning"
    
    # Tạo khuyến nghị ban đầu
    recommendations = []
    if is_blurry:
        recommendations.append("Hãy giữ chắc tay và chụp lại cận cảnh tổn thương để ảnh rõ nét, không bị rung mờ.")
    if is_dark:
        recommendations.append("Hãy chọn nơi có ánh sáng tự nhiên hoặc bật đèn hỗ trợ để tổn thương da hiển thị rõ màu sắc thực tế.")
    if is_bright:
        recommendations.append("Hãy tắt flash hoặc điều chỉnh góc chụp để tránh chói lóa sáng làm mờ chi tiết vùng da u.")
        
    return {
        "status": status,
        "blur_score": laplacian_var,
        "brightness_score": mean_brightness,
        "is_blurry": is_blurry,
        "is_poor_lighting": is_dark or is_bright,
        "issues": issues,
        "recommendations": recommendations
    }
