import sys
import os
sys.path.append("d:/DoAn_DaLieu")

from app_streamlit import generate_pdf_report

patient_info = {
    "name": "Nguyen Van A",
    "age": 30,
    "hometown": "Hanoi"
}

visit_data = {
    "created_at": "2026/06/17 12:00:00",
    "location": "Tay",
    "ai_extracted_metrics": {
        "prediction": "MEL",
        "confidence": 0.8543,
        "status": "ok",
        "probabilities": {
            "MEL": 0.8543,
            "NV": 0.1021,
            "BCC": 0.0436
        },
        "area_ratio": 0.1234,
        "border_complexity": 1.4567,
        "asymmetry": 0.7890,
        "circularity": 0.5432
    },
    "image_url": "https://i.ibb.co/ykZ89w8/clinical.jpg",  # Just a random image url to test downloading if any
    "mask_url": None,
    "gradcam_url": None,
    "vqa_conversations": [
        {"role": "user", "content": "Day la gi?"},
        {"role": "assistant", "content": "Day la anh ton thuong da lieu."}
    ]
}

try:
    pdf_bytes = generate_pdf_report(patient_info, visit_data)
    print("PDF generation succeeded!")
    print(f"PDF Size: {len(pdf_bytes)} bytes")
    with open("d:/DoAn_DaLieu/scratch/test_output.pdf", "wb") as f:
        f.write(pdf_bytes)
    print("PDF saved to d:/DoAn_DaLieu/scratch/test_output.pdf")
except Exception as e:
    import traceback
    print("PDF generation failed:")
    traceback.print_exc()
