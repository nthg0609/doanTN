import os
import sys
from unittest.mock import MagicMock

sys.path.append(os.path.abspath("."))

# Mock streamlit and submodules
class MockModule(MagicMock):
    @property
    def __path__(self):
        return []

streamlit_mock = MockModule()
sys.modules['streamlit'] = streamlit_mock
sys.modules['streamlit.components'] = MockModule()
sys.modules['streamlit.components.v1'] = MockModule()
sys.modules['streamlit_drawable_canvas'] = MockModule()

import app_streamlit

def test_pdf():
    pat_info = {
        "name": "Nguyen Van A",
        "age": "25",
        "gender": "Nam",
        "hometown": "Hà Nội",
        "location": "Lưng",
    }
    v_pdf = {
        "ai_extracted_metrics": {
            "prediction": "MEL",
            "confidence": 0.85,
            "area_ratio": 0.123,
            "border_complexity": 2.34,
            "asymmetry": 0.45,
            "circularity": 0.67,
        }
    }
    
    print("Calling generate_pdf_report...")
    pdf_bytes = app_streamlit.generate_pdf_report(pat_info, v_pdf)
    print("Result type:", type(pdf_bytes))
    print("Result size:", len(pdf_bytes) if pdf_bytes else "None")

if __name__ == "__main__":
    test_pdf()
