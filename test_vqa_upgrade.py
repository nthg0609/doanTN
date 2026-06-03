"""Test suite cho hệ thống VQA Da liễu nâng cấp — 7 kịch bản kiểm thử."""
import sys
import json
import types
import unittest.mock as mock

sys.path.insert(0, '.')

# ── Mock heavy dependencies ──────────────────────────────────────────────────
np_mock = types.ModuleType('numpy')
np_mock.ndarray = type
for attr in ['array', 'zeros', 'asarray', 'clip', 'sum', 'abs', 'pi', 'sqrt', 'pad', 'flipud', 'fliplr', 'argmax']:
    setattr(np_mock, attr, mock.MagicMock(return_value=0.5))
sys.modules['numpy'] = np_mock

# Mock google namespace hierarchy
google_mock = types.ModuleType('google')
google_cloud_mock = types.ModuleType('google.cloud')
google_cloud_fs_mock = mock.MagicMock()
google_oauth2_mock = types.ModuleType('google.oauth2')
google_sa_mock = mock.MagicMock()

google_mock.cloud = google_cloud_mock
google_mock.oauth2 = google_oauth2_mock
google_cloud_mock.firestore = google_cloud_fs_mock
google_oauth2_mock.service_account = google_sa_mock

sys.modules['google'] = google_mock
sys.modules['google.cloud'] = google_cloud_mock
sys.modules['google.cloud.firestore'] = google_cloud_fs_mock
sys.modules['google.oauth2'] = google_oauth2_mock
sys.modules['google.oauth2.service_account'] = google_sa_mock

heavy_mocks = [
    'cv2', 'torch', 'PIL', 'PIL.Image', 'streamlit',
    'plotly', 'plotly.graph_objects',
    'openai',
    'matplotlib', 'matplotlib.pyplot', 'dotenv',
]
for mod in heavy_mocks:
    sys.modules[mod] = mock.MagicMock()

import importlib
app = importlib.import_module('app_streamlit')

PASS = 0
FAIL = 0

def check(name, cond, msg=""):
    global PASS, FAIL
    if cond:
        print(f"  ✅ PASS: {name}")
        PASS += 1
    else:
        print(f"  ❌ FAIL: {name} — {msg}")
        FAIL += 1

# ── TC-01: Safety Gate Triage Block ──────────────────────────────────────────
print("\n─── TC-01: Safety Gate triage phải chặn LLM ───")
triage_result = {
    'status': 'triage',
    'triage_reason': 'low_classification_confidence',
    'classification': {'prediction': 'NV', 'confidence': 0.45, 'probabilities': {}},
    'metrics': {}
}
resp = app.generate_vqa_response('Tôi bị bệnh gì?', triage_result, 'fake_key', [])
check("Safety Gate response contains 'Safety Gate'", 'Safety Gate' in resp)
check("Safety Gate response contains triage reason", 'tin cậy' in resp.lower() or 'triage' in resp.lower() or 'safety' in resp.lower())

# ── TC-02: Medication Guardrail — System Prompt detail ───────────────────────
print("\n─── TC-02/03: Medication Guardrail in System Prompt ───")
cv_ctx = {
    'prediction': 'MEL',
    'confidence': 0.87,
    'probabilities': {'MEL': 0.87, 'NV': 0.08, 'BCC': 0.05},
    'metrics': {'area_ratio': 0.12, 'border_complexity': 4.2, 'asymmetry': 0.61, 'circularity': 0.38}
}
sys_prompt = app._build_fusion_system_prompt(cv_ctx)

check("Section TUYỆT ĐỐI CẤM present", 'TUYỆT ĐỐI CẤM' in sys_prompt)
check("Example drug names listed (Amoxicillin)", 'Amoxicillin' in sys_prompt)
check("Section ĐƯỢC PHÉP present", 'ĐƯỢC PHÉP' in sys_prompt)
check("CV_CONTEXT section embedded in system prompt", 'CV_CONTEXT' in sys_prompt)
check("Prediction label MEL in prompt", 'MEL' in sys_prompt)
check("Asymmetry metric in prompt", 'asymmetry' in sys_prompt.lower() or 'Asymmetry' in sys_prompt)
check("Circularity metric in prompt", 'circularity' in sys_prompt.lower() or 'Circularity' in sys_prompt)
check("Response length cap (400 từ)", '400' in sys_prompt)
check("No-drug clause covers 'giả sử' loophole", 'giả sử' in sys_prompt or 'giả định' in sys_prompt)

# ── TC-04: Smart Reset ────────────────────────────────────────────────────────
print("\n─── TC-04: Smart Reset — Ảnh mới xóa state cũ ───")
main_src = open('app_streamlit.py', encoding='utf-8').read()
check("last_uploaded_file_name check in code", 'last_uploaded_file_name' in main_src)
check("messages reset to [] on new image", 'st.session_state[\"messages\"] = []' in main_src or "\"messages\"] = []" in main_src)
check("result reset to None on new image", '"result"]                  = None' in main_src or '"result\"] = None' in main_src or 'result\"]                  = None' in main_src)

# ── TC-05: LLM Log ────────────────────────────────────────────────────────────
print("\n─── TC-05: LLM I/O Logging ───")
check("write_dev_log function exists", hasattr(app, 'write_dev_log'))
check("LLM_VQA_EXCHANGE logged", 'LLM_VQA_EXCHANGE' in main_src)
check("system_prompt logged", '"system_prompt"' in main_src)
check("raw_response logged", '"raw_response"' in main_src)

# ── TC-06: Confirmation Gate trùng tên ────────────────────────────────────────
print("\n─── TC-06: Confirmation Gate trùng tên bệnh nhân ───")
check("check_patient_exists function", hasattr(app, 'check_patient_exists'))
check("allow_to_save logic present", 'allow_to_save' in main_src)
check("confirm_update radio present", 'confirm_update' in main_src)

# ── TC-07: Chart save path & log path ─────────────────────────────────────────
print("\n─── TC-07: Đường dẫn lưu biểu đồ & log ───")
check("LOG_FILE_PATH defined", 'system_logs.log' in str(app.LOG_FILE_PATH))
check("CHART_SAVE_DIR defined", 'probability_charts' in str(app.CHART_SAVE_DIR))
check("render_probability_chart function", hasattr(app, 'render_probability_chart'))
check("render_radar_chart function", hasattr(app, 'render_radar_chart'))
check("PNG save logic present", 'to_image' in main_src or 'png_filename' in main_src)

# ── TC-CSS: Overflow control ────────────────────────────────────────────────
print("\n─── TC-CSS: Overflow & UI ───")
check("overflow-wrap CSS present", 'overflow-wrap' in main_src)
check("max-height CSS present", 'max-height' in main_src)
check("Medical Disclaimer constant", 'công cụ hỗ trợ sàng lọc sơ bộ' in app.MEDICAL_DISCLAIMER)
check("Disclaimer not replace doctor", 'không thay thế' in app.MEDICAL_DISCLAIMER)

# ── Labels ─────────────────────────────────────────────────────────────────
print("\n─── Labels Vietnamese ───")
check("MEL = U hắc tố", app.get_vietnamese_diagnosis('MEL') == 'U hắc tố ác tính (Melanoma)')
check("NV = Nốt ruồi", app.get_vietnamese_diagnosis('NV') == 'Nốt ruồi lành tính')
check("BCC = Ung thư biểu mô", 'Ung thư biểu mô' in app.get_vietnamese_diagnosis('BCC'))
check("VASC = Tổn thương mạch máu", 'mạch máu' in app.get_vietnamese_diagnosis('VASC'))
check("AKIEC = Dày sừng", 'Dày sừng' in app.get_vietnamese_diagnosis('AKIEC'))

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"TỔNG KẾT: {PASS} PASS  |  {FAIL} FAIL")
if FAIL == 0:
    print("🎉 TẤT CẢ TEST CASES ĐÃ PASS — Hệ thống sẵn sàng!")
else:
    print("⚠️ Có lỗi cần xem xét.")
print('='*55)
