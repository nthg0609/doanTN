"""CLI runner for UnifiedDermatologyPipeline - Absolute Package Import."""

import argparse
import json
import sys
from pathlib import Path

# 1. Ép hệ thống định vị duy nhất thư mục gốc của Đồ án (Bypass môi trường ảo)
BASE_DIR = Path("d:/DoAn_DaLieu")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# 2. Nạp module thông qua package cha 'pipeline' để bảo toàn relative import
try:
    from pipeline.unified_pipeline import UnifiedDermatologyPipeline
    print("[OK] Đã nạp thành công UnifiedDermatologyPipeline cho phiên chạy thực tế.")
except ImportError as e:
    print(f"[ERR] Không thể nạp module từ Package.")
    print(f"Chi tiết lỗi hệ thống: {e}")
    sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified dermatology inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--mode", default="classification", choices=["classification", "both"], help="Pipeline mode")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument("--safety_config", default=None, help="Path to safety gate JSON config")
    parser.add_argument("--min_conf", type=float, default=None, help="Override classification confidence threshold")
    args = parser.parse_args()

    # Khởi chạy pipeline thực tế (Mặc định load_models=True sẽ được gọi bên trong class)
    overrides = {}
    if args.min_conf is not None:
        overrides["min_class_confidence"] = args.min_conf
    pipeline = UnifiedDermatologyPipeline(
        mode=args.mode,
        safety_config_path=args.safety_config,
        safety_overrides=overrides or None,
    )
    result = pipeline.run(args.image)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
    print("\n--- KẾT QUẢ INFERENCE THỰC TẾ (REAL MODEL OUTPUT) ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())