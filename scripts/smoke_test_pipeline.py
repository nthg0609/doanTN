"""Minimal smoke test without loading heavy models - Absolute Package Import."""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# 1. Ép hệ thống định vị duy nhất thư mục gốc của Đồ án (Không nạp thư mục con)
BASE_DIR = Path("d:/DoAn_DaLieu")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# 2. Nạp module theo đúng cấu trúc Package Cha (Parent Package)
try:
    # Gọi thông qua package cha 'pipeline' để bảo toàn logic relative import bên trong
    from pipeline.unified_pipeline import UnifiedDermatologyPipeline
    print("[OK] Hệ thống đã nhận diện Package 'pipeline' và nạp UnifiedDermatologyPipeline thành công.")
except ImportError as e:
    print(f"[ERR] Không thể nạp module từ Package.")
    print(f"Chi tiết lỗi hệ thống: {e}")
    print("\nMẹo kiểm tra: Hãy chắc chắn thư mục d:/DoAn_DaLieu/pipeline có tồn tại file __init__.py và unified_pipeline.py")
    sys.exit(1)


def main() -> int:
    out_dir = Path("d:/DoAn_DaLieu/5_Results")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / "smoke_test_image.jpg"

    # Tạo ảnh giả lập để chạy qua luồng xử lý dữ liệu đầu vào
    img = np.full((224, 224, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(tmp_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Khởi chạy pipeline kiểm thử logic (không load trọng số nặng)
    pipeline = UnifiedDermatologyPipeline(load_models=False)
    result = pipeline.run(str(tmp_path))

    out_path = out_dir / "smoke_test_unified_pipeline.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n--- KẾT QUẢ KIỂM THỬ THẬT (SMOKE TEST RESULT) ---")
    print("Smoke test status:", result.get("status"))
    print("Output JSON:", out_path)
    
    if tmp_path.exists():
        tmp_path.unlink()
        
    return 0


if __name__ == "__main__":
    raise SystemExit(main())