"""Streamlit Multi-modal Medical EHR Dashboard — Professional Medical EHR v3.2
Quy trình: Image/DICOM upload -> Safety Gate -> Segmentation -> Classification ->
           ABCD Metrics -> VQA Streaming -> EHR Cloud Sync -> PDF Export.
"""

import json
import os
import io
import re
import tempfile
import base64
import requests
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as stc_v1
from PIL import Image, ImageDraw
from dotenv import load_dotenv
import plotly.graph_objects as go
from streamlit_drawable_canvas import st_canvas

import google.cloud.firestore as gcp_firestore
from google.oauth2 import service_account

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ── Khai báo voice component inline ───────────────────────────────────────────
_VOICE_FRONTEND = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "voice_component", "frontend"
)
if os.path.isdir(_VOICE_FRONTEND):
    try:
        _voice_input_fc = stc_v1.declare_component("derma_voice_input", path=_VOICE_FRONTEND)
        VOICE_AVAILABLE = True
    except Exception:
        _voice_input_fc = None
        VOICE_AVAILABLE = False
else:
    _voice_input_fc = None
    VOICE_AVAILABLE = False

load_dotenv()

# ==============================================================================
# HẰNG SỐ HỆ THỐNG
# ==============================================================================
IMGBB_API_KEY = "159bc5d50210a5104a5c1b1018368f75"

LOG_FILE_PATH = Path("5_Results/system_logs.log")
LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

CHART_SAVE_DIR = Path("5_Results/probability_charts")
CHART_SAVE_DIR.mkdir(parents=True, exist_ok=True)

DIAGNOSIS_DICTIONARY: Dict[str, str] = {
    "AKIEC": "Dày sừng quang hóa / Tiền ung thư",
    "BCC":   "Ung thư biểu mô tế bào đáy",
    "BKL":   "Tổn thương sừng hóa lành tính",
    "DF":    "U xơ da",
    "MEL":   "U hắc tố ác tính (Melanoma)",
    "NV":    "Nốt ruồi lành tính",
    "VASC":  "Tổn thương mạch máu",
}

MALIGNANT_CLASSES: List[str] = ["MEL", "BCC", "AKIEC"]
BENIGN_CLASSES:    List[str] = ["BKL", "NV", "DF", "VASC"]
MALIGNANT_ALERT_THRESHOLD: float = 0.15

TRIAGE_REASON_VI: Dict[str, str] = {
    "empty_or_low_confidence_mask":    "Không phát hiện được vùng tổn thương rõ ràng trong ảnh",
    "area_ratio_out_of_bounds":        "Tỉ lệ diện tích tổn thương nằm ngoài ngưỡng hợp lệ",
    "border_complexity_out_of_bounds": "Độ phức tạp bờ quá cao — có thể do nhiễu ảnh",
    "classification_unavailable":      "Mô hình phân loại không khả dụng",
    "low_classification_confidence":   "Độ tin cậy phân loại thấp hơn ngưỡng an toàn (tau_c)",
    "image_load_failed":               "Không thể đọc file ảnh",
}

BODY_LOCATIONS: List[str] = [
    "--- Chọn Vị trí tổn thương ---", "Lưng", "Ngực / Bụng", "Cánh tay",
    "Bàn tay", "Đùi", "Cẳng chân", "Bàn chân", "Đầu / Mặt",
    "Cổ", "Vai", "Mông / Bẹn", "Vị trí khác",
]

VIETNAM_PROVINCES: List[str] = [
    "--- Chọn Tỉnh/Thành ---", "Hà Nội", "TP. Hồ Chí Minh", "Đà Nẵng", "Hải Phòng", "Cần Thơ",
    "An Giang", "Bà Rịa - Vũng Tàu", "Bắc Giang", "Bắc Kạn", "Bạc Liêu", "Bắc Ninh", "Bến Tre",
    "Bình Định", "Bình Dương", "Bình Phước", "Bình Thuận", "Cà Mau", "Cao Bằng", "Đắk Lắk",
    "Đắk Nông", "Điện Biên", "Đồng Nai", "Đồng Tháp", "Gia Lai", "Hà Giang", "Hà Nam", "Hà Tĩnh",
    "Hải Dương", "Hậu Giang", "Hòa Bình", "Hưng Yên", "Khánh Hòa", "Kiên Giang", "Kon Tum",
    "Lai Châu", "Lâm Đồng", "Lạng Sơn", "Lào Cai", "Long An", "Nam Định", "Nghệ An", "Ninh Bình",
    "Ninh Thuận", "Phú Thọ", "Phú Yên", "Quảng Bình", "Quảng Nam", "Quảng Ngãi", "Quảng Ninh",
    "Quảng Trị", "Sóc Trăng", "Sơn La", "Tây Ninh", "Thái Bình", "Thái Nguyên", "Thanh Hóa",
    "Thừa Thiên Huế", "Tiền Giang", "Trà Vinh", "Tuyên Quang", "Vĩnh Long", "Vĩnh Phúc", "Yên Bái",
    "Tỉnh/Thành khác",
]

VQA_MODE_OPTIONS: List[str] = [
    "Trực tuyến — OpenAI GPT-4o-mini",
    "Nội bộ — CPU (LLM local)",
    "Nội bộ — Ollama",
]

PRESET_QUESTIONS: List[str] = [
    "Bệnh này có triệu chứng lâm sàng gì?",
    "Phác đồ chăm sóc vùng da tổn thương này?",
    "Giải thích chỉ số ABCD bất thường.",
    "Định hướng sinh thiết và xét nghiệm tiếp theo.",
]

# ==============================================================================
# TIỆN ÍCH
# ==============================================================================
def get_vietnamese_diagnosis(pred_label: str) -> str:
    return DIAGNOSIS_DICTIONARY.get(pred_label, "Bệnh lý da liễu khác")


# ==============================================================================
# MODULE LOG HỆ THỐNG
# ==============================================================================
class _NumpySafeEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray):     return obj.tolist()
        except Exception:
            pass
        return super().default(obj)


def write_dev_log(data: Dict[str, Any], action_type: str) -> None:
    log_entry = {
        "action_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action_type": action_type,
        "payload":     data,
    }
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False, cls=_NumpySafeEncoder) + "\n")


# ==============================================================================
# CLOUD STORAGE
# ==============================================================================
def upload_image_to_imgbb(local_image_path: str) -> Optional[str]:
    try:
        with open(local_image_path, "rb") as f:
            res = requests.post(
                "https://api.imgbb.com/1/upload",
                {"key": IMGBB_API_KEY, "image": base64.b64encode(f.read())},
            )
            if res.status_code == 200:
                return res.json()["data"]["url"]
    except Exception as e:
        st.error(f"Lỗi tải ảnh lên ImgBB: {e}")
    return None


import hashlib

def encrypt_data(data: str, key: str = "DermaSecureKey2026") -> str:
    if not data:
        return ""
    # Simple XOR encryption encoded in base64
    xored = "".join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(data))
    return base64.b64encode(xored.encode('utf-8')).decode('utf-8')

def decrypt_data(enc_data: str, key: str = "DermaSecureKey2026") -> str:
    if not enc_data:
        return ""
    try:
        decoded = base64.b64decode(enc_data.encode('utf-8')).decode('utf-8')
        xored = "".join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(decoded))
        return xored
    except Exception:
        return enc_data

def get_patient_doc_id(patient_name: str) -> str:
    normalized = "".join(patient_name.strip().upper().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def check_patient_exists(patient_name: str) -> bool:
    if not patient_name.strip():
        return False
    try:
        cred_path = "gcp-credentials.json"
        if os.path.exists(cred_path):
            creds = service_account.Credentials.from_service_account_file(cred_path)
            db    = gcp_firestore.Client(credentials=creds, project=creds.project_id, database="(default)")
            doc_id = get_patient_doc_id(patient_name)
            return db.collection("medical_records").document(doc_id).get().exists
    except Exception:
        pass
    return False


def save_medical_record_to_gcp(
    patient_name: str,
    patient_info: Dict[str, Any],
    visit_data: Dict[str, Any],
) -> bool:
    try:
        cred_path = "gcp-credentials.json"
        if os.path.exists(cred_path):
            creds  = service_account.Credentials.from_service_account_file(cred_path)
            db     = gcp_firestore.Client(credentials=creds, project=creds.project_id, database="(default)")
            doc_id = get_patient_doc_id(patient_name)
            ref    = db.collection("medical_records").document(doc_id)
            snap   = ref.get()
            now    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Encrypt sensitive patient info fields
            pat_info_enc = patient_info.copy()
            if "name" in pat_info_enc:
                pat_info_enc["name"] = encrypt_data(pat_info_enc["name"])
            if "id" in pat_info_enc:
                pat_info_enc["id"] = encrypt_data(pat_info_enc["id"])
                
            if snap.exists:
                visits = snap.to_dict().get("visits", [])
                visits.append(visit_data)
                ref.update({"patient_info": pat_info_enc, "updated_at": now, "visits": visits})
            else:
                ref.set({
                    "patient_id":   doc_id,
                    "patient_info": pat_info_enc,
                    "created_at":   now,
                    "updated_at":   now,
                    "visits":       [visit_data],
                })
            write_dev_log({"patient_id": doc_id, "visit": visit_data}, "SAVE_OR_UPDATE_RECORD")
            return True
        else:
            st.error("Không tìm thấy file gcp-credentials.json!")
            return False
    except Exception as e:
        st.error(f"Lỗi ghi dữ liệu lên Cloud Firestore: {e}")
    return False


def delete_patient_record_from_gcp(patient_name: str) -> bool:
    try:
        cred_path = "gcp-credentials.json"
        if os.path.exists(cred_path):
            creds = service_account.Credentials.from_service_account_file(cred_path)
            db = gcp_firestore.Client(credentials=creds, project=creds.project_id, database="(default)")
            doc_id = get_patient_doc_id(patient_name)
            db.collection("medical_records").document(doc_id).delete()
            return True
    except Exception as e:
        st.error(f"Lỗi khi xóa bệnh nhân: {e}")
    return False


def fetch_all_medical_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    try:
        cred_path = "gcp-credentials.json"
        if os.path.exists(cred_path):
            creds = service_account.Credentials.from_service_account_file(cred_path)
            db    = gcp_firestore.Client(credentials=creds, project=creds.project_id, database="(default)")
            docs  = db.collection("medical_records").order_by(
                "updated_at", direction=gcp_firestore.Query.DESCENDING
            ).stream()
            for doc in docs:
                d = doc.to_dict()
                if "patient_info" in d:
                    pi = d["patient_info"]
                    if "name" in pi:
                        pi["name"] = decrypt_data(pi["name"])
                    if "id" in pi:
                        pi["id"] = decrypt_data(pi["id"])
                records.append(d)
    except Exception:
        pass
    return records


# ==============================================================================
# PIPELINE
# ==============================================================================
@st.cache_resource
def get_pipeline(min_conf: float, reload_token: str = "v2"):
    import sys
    for k in list(sys.modules.keys()):
        if k == "pipeline" or k.startswith("pipeline."):
            sys.modules.pop(k, None)
    from pipeline import UnifiedDermatologyPipeline
    return UnifiedDermatologyPipeline(
        mode="both",
        safety_overrides={"min_class_confidence": float(min_conf)},
    )


def _mask_to_image(mask: Optional[np.ndarray], target_shape) -> Optional[np.ndarray]:
    if mask is None:
        return None
    if mask.ndim != 2:
        mask = mask[:, :, 0]
    if mask.shape != target_shape:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return (mask > 0).astype(np.uint8) * 255


# ==============================================================================
# MODULE ĐA TỔN THƯƠNG
# ==============================================================================
def detect_multiple_lesions(img_rgb: np.ndarray, mask: np.ndarray) -> List[Dict[str, Any]]:
    if mask is None or mask.sum() == 0:
        return []
    mg = (mask[:, :, 0] if mask.ndim != 2 else mask).astype(np.uint8)
    contours, _ = cv2.findContours(mg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lesions = []
    for idx, ctr in enumerate(contours, start=1):
        area = cv2.contourArea(ctr)
        if area >= 150:
            x, y, w, h = cv2.boundingRect(ctr)
            lesions.append({"id": idx, "bbox": (x, y, w, h), "contour": ctr, "area": area})
    return sorted(lesions, key=lambda x: x["area"], reverse=True)


def draw_lesions_overlay(img_rgb: np.ndarray, lesions: List[Dict[str, Any]]) -> np.ndarray:
    out = img_rgb.copy()
    for les in lesions:
        x, y, w, h = les["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (56, 161, 105), 3)
        label = f"Nốt {les['id']}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (x, y - lh - 10), (x + lw, y), (56, 161, 105), -1)
        cv2.putText(out, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


# ==============================================================================
# MODULE DICOM
# ==============================================================================
def load_dicom(file_obj) -> tuple:
    tmp_path = None
    try:
        import pydicom
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
            tmp.write(file_obj.read())
            tmp_path = tmp.name
        ds  = pydicom.dcmread(tmp_path)
        arr = ds.pixel_array.astype(float)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        arr = ((arr - arr.min()) / max(arr.max() - arr.min(), 1) * 255).astype(np.uint8)
        img  = Image.fromarray(arr[:, :, :3]).convert("RGB")
        meta = {
            "patient_name": str(getattr(ds, "PatientName", "")),
            "patient_age":  str(getattr(ds, "PatientAge",  "")),
            "patient_sex":  str(getattr(ds, "PatientSex",  "")),
        }
        pixel_spacing = getattr(ds, "PixelSpacing", None)
        if pixel_spacing:
            meta["pixel_spacing_y"] = float(pixel_spacing[0])
            meta["pixel_spacing_x"] = float(pixel_spacing[1])
        return img, meta
    except Exception as e:
        return None, {}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except Exception: pass


# ==============================================================================
# VQA ENGINE — Fusion Prompt + Stream + RAG + Offline Modes
# ==============================================================================
CLINICAL_KB = {
    "MEL": {
        "symptoms": {
            "vi": "Ung thư hắc tố (Melanoma) thường biểu hiện dưới dạng dát hoặc sẩn sắc tố màu nâu sẫm, đen hoặc nhiều màu hỗn hợp. Triệu chứng lâm sàng điển hình gồm tổn thương bất đối xứng, bờ răng cưa không đều, màu sắc không đồng nhất, kích thước tiến triển nhanh (>6mm) và có thể gây ngứa, loét hoặc chảy máu tự phát.",
            "en": "Melanoma typically presents as a dark brown, black, or multi-colored pigmented macule or papule. Classic clinical symptoms include asymmetry, irregular or notched borders, color variation, rapid diameter growth (>6mm), and occasionally pragmatist, ulceration, or spontaneous bleeding."
        },
        "care": {
            "vi": "Phác đồ chăm sóc vùng da nghi ngờ melanoma đòi hỏi tránh tuyệt đối chấn thương vật lý, không cào gãi, tẩy nốt ruồi bằng hóa chất hoặc đắp thuốc lá dân gian. Bảo vệ da nghiêm ngặt khỏi bức xạ UV bằng kem chống nắng phổ rộng (SPF >= 50) và mặc quần áo che chắn.",
            "en": "Care for suspected melanoma skin lesions strictly requires avoiding physical trauma, scratching, chemical mole removal, or herbal poultices. Protect the skin rigorously from UV radiation using broad-spectrum sunscreen (SPF >= 50) and protective clothing."
        },
        "abcd": {
            "vi": "Giải thích chỉ số ABCD đối với U hắc tố ác tính (MEL): (A) Bất đối xứng cao do tế bào sắc tố tăng sinh vô tổ chức. (B) Bờ tổn thương không đều, dạng răng cưa. (C) Màu sắc không đồng nhất (đen, nâu, hồng xen kẽ). (D) Đường kính thường vượt quá 6mm và có xu hướng lan rộng nhanh.",
            "en": "ABCD abnormalities: (A) High asymmetry due to disorganized melanocyte proliferation. (B) Irregular, notched, or blurred borders. (C) Color variation (intermingled black, brown, red, or blue). (D) Diameter typically exceeds 6mm and shows active evolution."
        },
        "biopsy": {
            "vi": "Đối với tổn thương nghi ngờ Melanoma, định hướng bắt buộc là chuyển tuyến chuyên khoa da liễu ngay lập tức. Chỉ định chuẩn vàng là sinh thiết trọn (excisional biopsy) toàn bộ tổn thương để đánh giá giải phẫu bệnh và đo độ sâu Breslow. Không thực hiện sinh thiết bấm hoặc sinh thiết cạo nếu có thể phẫu thuật trọn.",
            "en": "For suspected Melanoma, immediate referral to a specialist dermatologist is mandatory. The gold standard is an excisional biopsy of the entire lesion for histopathological evaluation and Breslow thickness measurement. Avoid punch or shave biopsies if complete excision is feasible."
        },
        "general": {
            "vi": "Tổn thương này có đặc điểm của Melanoma - một dạng ung thư da ác tính nghiêm trọng. Bệnh nhân cần được bác sĩ chuyên khoa da liễu khám lâm sàng và sinh thiết chẩn đoán sớm để lên phác đồ điều trị kịp thời.",
            "en": "This lesion exhibits characteristics of Melanoma, a serious malignant skin cancer. Early dermatological evaluation and biopsy are critical for a timely treatment plan."
        }
    },
    "BCC": {
        "symptoms": {
            "vi": "Ung thư biểu mô tế bào đáy (BCC) thường xuất hiện ở vùng hở (mặt, cổ) dưới dạng sẩn bóng như ngọc trai, có giãn mạch hình sợi chỉ dưới da. Tổn thương tiến triển chậm, lâu dần lõm giữa tạo vết loét gặm gặm khó lành, dễ chảy máu khi va chạm nhẹ.",
            "en": "Basal Cell Carcinoma (BCC) commonly appears on sun-exposed areas (face, neck) as a pearly, translucent papule with telangiectasia (visible thread-like vessels). It progresses slowly, eventually developing a central ulceration with rolled borders."
        },
        "care": {
            "vi": "Chăm sóc vùng tổn thương BCC bao gồm giữ vệ sinh bằng nước muối sinh lý hoặc chất tẩy rửa dịu nhẹ, tránh chà xát mạnh gây chảy máu. Chống nắng tích cực là bắt buộc để ngăn ngừa tổn thương tiến triển hoặc xuất hiện tổn thương mới.",
            "en": "Care for a BCC lesion includes keeping the area clean with saline or gentle non-soap cleansers, avoiding vigorous scrubbing to prevent bleeding. Active sun protection is mandatory to halt lesion progression."
        },
        "abcd": {
            "vi": "Giải thích chỉ số ABCD đối với BCC: Tổn thương thường hơi bất đối xứng (A) và bờ nổi gờ ngọc trai (B). Chỉ số màu sắc (C) có thể đơn sắc hồng/đỏ hoặc có đốm sắc tố đen nhẹ (ở thể BCC sắc tố). Đường kính tiến triển chậm theo năm tháng.",
            "en": "ABCD features in BCC: The lesion often shows slight asymmetry (A) and characteristic rolled, pearly borders (B). Color (C) is typically pink/flesh-toned, but can have dark speckles in pigmented variants. Diameter evolves slowly over years."
        },
        "biopsy": {
            "vi": "Cần thực hiện sinh thiết bấm (punch biopsy) hoặc sinh thiết cạo sâu để chẩn đoán thể mô bệnh học (thể nốt, thể nông, thể xâm lấn). Phương pháp điều trị chính bao gồm phẫu thuật cắt rộng (phẫu thuật Mohs cho vùng mặt), áp lạnh nitơ lỏng hoặc liệu pháp quang động (PDT).",
            "en": "A punch biopsy or deep shave biopsy is indicated to confirm the histopathological subtype (nodular, superficial, or infiltrative). Key treatments include wide local excision (Mohs micrographic surgery for facial lesions), cryosurgery, or photodynamic therapy (PDT)."
        },
        "general": {
            "vi": "Đây là tổn thương nghi ngờ Ung thư biểu mô tế bào đáy (BCC). Mặc dù là ung thư da phổ biến nhất nhưng BCC tiến triển rất chậm và hiếm khi di căn xa, có thể chữa khỏi hoàn toàn nếu phẫu thuật sớm.",
            "en": "This lesion is suspicious for Basal Cell Carcinoma (BCC). Although it is the most common skin cancer, it is slow-growing and rarely metastasizes, with high cure rates via early surgical removal."
        }
    },
    "AKIEC": {
        "symptoms": {
            "vi": "Dày sừng ánh sáng hoặc bệnh Bowen (AKIEC) biểu hiện dưới dạng dát hoặc mảng màu hồng, đỏ, bề mặt thô ráp, phủ vảy sừng khô bám chặt, đôi khi có cảm giác đau nhói hoặc châm chích khi chạm vào da.",
            "en": "Actinic Keratosis or Bowen's disease (AKIEC) presents as a pink to red scaly plaque with a rough, sandpaper-like texture. Lesions are adherent and may feel tender or prickly upon touch."
        },
        "care": {
            "vi": "Phác đồ chăm sóc gồm bôi kem dưỡng ẩm dịu nhẹ để làm mềm lớp vảy sừng, tuyệt đối không tự ý cạy vảy hoặc dùng hóa chất lột da. Chống nắng nghiêm ngặt bằng kem chống nắng phổ rộng là yếu tố tiên quyết.",
            "en": "Care protocol involves applying gentle emollients to soften the hyperkeratotic scale; do not scratch or use harsh chemical peels. Strict photoprotection using broad-spectrum sunscreen is crucial."
        },
        "abcd": {
            "vi": "Giải thích chỉ số ABCD đối với AKIEC: Tổn thương AKIEC thường có bờ không rõ ràng (B), bề mặt gồ ghề bất đối xứng nhẹ (A). Màu sắc (C) đỏ nhạt hoặc hồng, phủ vảy màu xám/trắng. Đường kính thường từ vài milimet đến hơn 1cm.",
            "en": "ABCD indicators: AKIEC lesions typically display poorly demarcated borders (B), rough and slightly asymmetric surface (A). Color (C) varies from light red to pink, covered by white-gray scale. Diameter ranges from millimeters to over a centimeter."
        },
        "biopsy": {
            "vi": "Cần đi khám da liễu để chẩn đoán phân biệt với ung thư biểu mô tế bào vảy xâm lấn (SCC). Sinh thiết da được chỉ định nếu tổn thương dày lên nhanh, loét hoặc đau. Điều trị có thể dùng thuốc bôi tại chỗ (Imiquimod, 5-Fluorouracil), áp nitơ lỏng hoặc liệu pháp quang động.",
            "en": "A dermatological exam is needed to differentiate from invasive Squamous Cell Carcinoma (SCC). Skin biopsy is indicated if lesions become indurated, ulcerated, or painful. Treatments include topical agents (Imiquimod, 5-Fluorouracil), cryotherapy, or PDT."
        },
        "general": {
            "vi": "Tổn thương này thuộc nhóm tiền ác tính (AKIEC). Cần được theo dõi sát và điều trị sớm để ngăn ngừa nguy cơ tiến triển thành ung thư biểu mô tế bào vảy xâm lấn.",
            "en": "This lesion belongs to the pre-malignant group (AKIEC). Regular monitoring and early ablation are recommended to prevent progression to invasive squamous cell carcinoma."
        }
    },
    "BKL": {
        "symptoms": {
            "vi": "Dày sừng lành tính (BKL/Seborrheic Keratosis) biểu hiện là sẩn hoặc mảng màu nâu nhạt, nâu sẫm đến đen, bề mặt khô ráp, mụn mỡ gồ cao trông như 'dán' lên da, giới hạn rất rõ với vùng da lành xung quanh.",
            "en": "Benign Keratosis (BKL/Seborrheic Keratosis) presents as a brown, tan, or black waxy papule or plaque. It often has a characteristic 'stuck-on' appearance with a verrucous, greasy surface and distinct borders."
        },
        "care": {
            "vi": "Chăm sóc tổn thương lành tính này rất đơn giản: Tránh chà xát mạnh bằng bông tắm hoặc cào gãi khi ngứa vì dễ gây nhiễm trùng thứ phát. Giữ da sạch sẽ và dưỡng ẩm bình thường.",
            "en": "Care for this benign lesion is straightforward: avoid aggressive scrubbing or scratching when itchy to prevent secondary infection. Keep skin clean and moisturized normally."
        },
        "abcd": {
            "vi": "Giải thích chỉ số ABCD đối với BKL: Tổn thương thường đối xứng (A) và bờ rất rõ ràng (B). Màu sắc (C) từ vàng nhạt đến đen sẫm. Dù có đường kính lớn và màu tối dễ nhầm với melanoma, cấu trúc bề mặt dạng mỡ lành tính giúp phân biệt tốt.",
            "en": "ABCD analysis for BKL: Lesions are usually symmetric (A) with sharply demarcated borders (B). Color (C) ranges from light tan to deep black. Although its large diameter and dark color can mimic melanoma, its greasy, stuck-on texture is diagnostic."
        },
        "biopsy": {
            "vi": "Đây là tổn thương lành tính hoàn toàn, không cần can thiệp y tế trừ khi bị kích ứng hoặc ảnh hưởng thẩm mỹ. Các phương pháp loại bỏ thẩm mỹ phổ biến là laser CO2, áp nitơ lỏng hoặc nạo bằng thìa muỗng chuyên dụng.",
            "en": "This is an entirely benign lesion requiring no medical intervention unless irritated or cosmetically undesirable. Removal options include CO2 laser ablation, cryotherapy, or curettage."
        },
        "general": {
            "vi": "Hệ thống xác định đây là Dày sừng lành tính (BKL), một tổn thương da lành tính do tuổi tác, không có khả năng chuyển dạng ác tính. Bạn có thể hoàn toàn yên tâm.",
            "en": "The system identifies this as a Benign Keratosis (BKL), which is a common age-related benign skin growth. It is non-cancerous and poses no health risks."
        }
    },
    "NV": {
        "symptoms": {
            "vi": "Nốt ruồi lành tính (Melanocytic Nevus) biểu hiện dưới dạng dát phẳng hoặc sẩn gồ nhẹ, màu sắc nâu hoặc đen đồng nhất, giới hạn rõ nét, hình tròn hoặc bầu oval đối xứng hoàn hảo.",
            "en": "A benign Melanocytic Nevus (mole) presents as a flat macule or slightly raised papule with uniform tan, brown, or black color, sharp borders, and symmetric round or oval shape."
        },
        "care": {
            "vi": "Chăm sóc nốt ruồi lành tính chủ yếu là theo dõi định kỳ các thay đổi. Bôi kem chống nắng bảo vệ da khỏi tia cực tím và tránh dùng các biện pháp tẩy nốt ruồi thủ công tại nhà vì dễ để lại sẹo lồi hoặc kích thích tổn thương.",
            "en": "Care for a benign mole primarily involves routine self-monitoring. Use sunscreen to protect against UV radiation and avoid home remedies for mole removal due to scar risks."
        },
        "abcd": {
            "vi": "Giải thích chỉ số ABCD đối với Nốt ruồi lành tính (NV): (A) Đối xứng hoàn hảo. (B) Bờ đều, sắc nét. (C) Màu sắc đơn nhất (đồng màu nâu/đen). (D) Đường kính thường nhỏ (<6mm) và không thay đổi theo thời gian.",
            "en": "ABCD parameters: (A) Perfectly symmetric. (B) Smooth, well-defined borders. (C) Uniform color (single shade of brown/black). (D) Diameter is typically small (<6mm) and remains stable over time."
        },
        "biopsy": {
            "vi": "Tổn thương lành tính không cần làm sinh thiết hay xét nghiệm y văn. Chỉ tiến hành khám chuyên khoa và sinh thiết khi nốt ruồi đột ngột thay đổi kích thước, bờ nham nhở hoặc chảy máu.",
            "en": "Benign nevi do not require biopsy or specialized tests. A clinical evaluation and biopsy are only indicated if the mole undergoes sudden changes in size, shape, borders, or starts bleeding."
        },
        "general": {
            "vi": "Hệ thống chẩn đoán đây là Nốt ruồi lành tính (NV). Đây là tổn thương sắc tố lành tính vô hại thường gặp, không cần can thiệp y khoa.",
            "en": "The system diagnoses this as a benign Melanocytic Nevus (NV). This is a common, harmless skin lesion that requires no medical treatment."
        }
    },
    "VASC": {
        "symptoms": {
            "vi": "Tổn thương mạch máu (VASC - u máu, giãn mạch) thường là các dát hoặc sẩn màu đỏ, đỏ tía hoặc xanh đen. Đặc điểm lâm sàng đặc trưng là ấn kính sẽ nhạt màu hoặc biến mất tạm thời do máu dồn đi nơi khác.",
            "en": "Vascular lesions (VASC - cherry angiomas, hemangiomas) present as bright red, purple, or blue-black macules or papules. A key clinical symptom is blanching (fading) under pressure (diascopy) as blood temporarily leaves the vessels."
        },
        "care": {
            "vi": "Phác đồ chăm sóc vùng tổn thương mạch máu: Tránh chọc, châm kim hoặc cào gãi vì có thể gây chảy máu kéo dài khó cầm. Giữ vệ sinh da nhẹ nhàng bằng nước muối hoặc nước mát.",
            "en": "Care protocol for vascular lesions: Avoid puncturing, scratching, or trauma, as this can cause prolonged, difficult-to-control bleeding. Clean the skin area gently with cool water."
        },
        "abcd": {
            "vi": "Giải thích chỉ số ABCD đối với VASC: Tổn thương có dạng tròn/bầu dục đối xứng tốt (A), bờ đều rõ (B). Màu đỏ hoặc đỏ tía đồng nhất (C). Đường kính thường nhỏ và ổn định.",
            "en": "ABCD indices in VASC: Highly symmetric round or oval shape (A), smooth and regular borders (B). Uniform red, crimson, or purple color (C). Diameter is usually small and stable."
        },
        "biopsy": {
            "vi": "Tổn thương lành tính, không cần sinh thiết. Nếu có nghi ngờ lâm sàng lớn hoặc tổn thương phát triển sâu dưới da, có thể kết hợp siêu âm Doppler mạch máu để xác định cấu trúc dòng chảy. Điều trị thẩm mỹ bằng laser xung màu (Dye Laser) hoặc áp lạnh.",
            "en": "This is a benign lesion that does not require biopsy. For deep or large lesions, Doppler ultrasound may be used to assess vascular flow. Cosmetic options include pulsed dye laser (PDL) or cryotherapy."
        },
        "general": {
            "vi": "Đây là Tổn thương mạch máu lành tính (VASC). Tổn thương mạch máu này hoàn toàn lành tính, không nguy hiểm và chỉ ảnh hưởng đến mặt thẩm mỹ nếu nằm ở vùng da hở.",
            "en": "This is a benign Vascular Lesion (VASC). It is entirely harmless, non-cancerous, and poses only cosmetic concerns if located on visible skin areas."
        }
    },
    "DF": {
        "symptoms": {
            "vi": "U xơ da (DF/Dermatofibroma) là sẩn cứng nằm sâu dưới da, màu hồng, nâu nhạt hoặc hơi xám. Triệu chứng lâm sàng điển hình là dấu hiệu lúm đồng tiền (dimple sign) - da bị lõm xuống khi bóp nhẹ vào hai bên tổn thương.",
            "en": "Dermatofibroma (DF) presents as a firm, hyperpigmented dermal nodule, typically pink, light brown, or gray. A classic clinical sign is the 'dimple sign' - the lesion dimples downward when pinched gently from the sides."
        },
        "care": {
            "vi": "Chăm sóc u xơ da uốn nắn bình thường, tránh cạo lông/râu trực tiếp đè lên sẩn cứng vì dễ gây trầy xước và đau nhẹ do cọ xát. Không tự ý dùng vật sắc nhọn rạch hoặc chọc tổn thương.",
            "en": "Care for a dermatofibroma is standard: avoid shaving directly over the nodule to prevent abrasion or mild pain from friction. Do not attempt to puncture or excise the lesion at home."
        },
        "abcd": {
            "vi": "Giải thích chỉ số ABCD đối với DF: Tổn thương đối xứng tốt (A), bờ rõ ràng nhưng đôi khi mờ dần vào mô xung quanh (B). Màu sắc (C) thường có trung tâm nhạt màu hơn viền nâu ngoài. Đường kính thường < 1cm.",
            "en": "ABCD characteristics of DF: Usually symmetric (A), well-defined but sometimes fading into surrounding tissue (B). Color (C) often features a pale center with a darker brown ring. Diameter is typically under 1cm."
        },
        "biopsy": {
            "vi": "U xơ da hoàn toàn lành tính. Không cần can thiệp ngoại khoa trừ khi tổn thương gây đau nhiều hoặc cản trở hoạt động. Nếu cần chẩn đoán phân biệt với các u trung mô khác, bác sĩ sẽ chỉ định sinh thiết bấm và làm giải phẫu bệnh.",
            "en": "Dermatofibromas are completely benign. No surgery is required unless the lesion causes persistent pain or functional impairment. A punch biopsy is indicated only to rule out other mesenchymal tumors."
        },
        "general": {
            "vi": "Mô hình nhận diện tổn thương là U xơ da lành tính (DF). Đây là một phản ứng xơ hóa lành tính thường gặp sau côn trùng cắn hoặc chấn thương nhỏ, hoàn toàn không nguy hiểm.",
            "en": "The model identifies this as a benign Dermatofibroma (DF). This is a common reactive fibrous nodule, often appearing after insect bites or minor trauma, and is completely harmless."
        }
    }
}

RAG_ENGINE_INSTANCE = None

def get_rag_engine():
    global RAG_ENGINE_INSTANCE
    if RAG_ENGINE_INSTANCE is None:
        try:
            from pipeline.rag_engine import DermatologyRAG
            RAG_ENGINE_INSTANCE = DermatologyRAG()
        except Exception as e:
            print("Error initializing DermatologyRAG:", e)
    return RAG_ENGINE_INSTANCE


def translate_text(text: str, target_lang: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return text
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = f"Translate the following medical text into {target_lang}. Return ONLY the translated text. Do not add any conversational introduction, quotes, or explanation.\n\nText: {text}"
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return text


@st.cache_resource
def load_custom_vqa_model():
    try:
        import torch
        import sys
        from pathlib import Path
        BASE_DIR = Path("d:/DoAn_DaLieu")
        if str(BASE_DIR) not in sys.path:
            sys.path.insert(0, str(BASE_DIR))
        
        from scripts.train_vqa_joint import VisionBackbone, CPUMedicalVQAModel
        from transformers import AutoTokenizer
        from peft import LoraConfig, get_peft_model
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = "9_VQA/models/dermavqa_gpt2_joint_best.pth"
        
        if not os.path.exists(model_path):
            return None, None, device
            
        vision_backbone = VisionBackbone().to(device)
        model = CPUMedicalVQAModel(vision_backbone).to(device)
        
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model.llm = get_peft_model(model.llm, peft_config)
        
        checkpoint = torch.load(model_path, map_location=device)
        use_spatial_tokens = checkpoint.get("use_spatial_tokens", False)
        model.use_spatial_tokens = use_spatial_tokens
        model.vision_backbone.use_spatial_tokens = use_spatial_tokens
        
        has_prefix = any(k.startswith("clinical_prefix.") for k in checkpoint["model_state_dict"].keys())
        model.has_prefix = has_prefix
        
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer, device
    except Exception as e:
        print("Error loading custom local VQA model:", e)
        return None, None, "cpu"


def _build_system_prompt(cv_context: Dict[str, Any]) -> str:
    pred    = cv_context.get("prediction", "N/A")
    vi_name = get_vietnamese_diagnosis(pred)
    conf    = float(cv_context.get("confidence", 0.0))
    metrics = cv_context.get("metrics", {})
    probs   = cv_context.get("probabilities", {})
    prob_lines = "\n".join(
        f"    - {k} ({get_vietnamese_diagnosis(k)}): {v:.4f}"
        for k, v in sorted(probs.items(), key=lambda x: -x[1])
    )
    return f"""[IDENTITY]
Bạn là Trợ lý Da liễu AI — hệ thống hỗ trợ sàng lọc y tế tích hợp mô hình Computer Vision và LLM.
Tư vấn dựa HOÀN TOÀN trên dữ liệu CV được cung cấp, không được bịa đặt số liệu.

[CV_CONTEXT]
Mô hình EfficientNet-B1 + CBAM Attention:
  Nhãn dự đoán cao nhất : {pred} — {vi_name}
  Độ tin cậy            : {conf:.4f} ({conf*100:.1f}%)

Phân phối xác suất đầy đủ 7 nhãn bệnh lý (ISIC):
{prob_lines}

Chỉ số hình học (DeepLabV3+ Segmentation):
  Area ratio        : {metrics.get('area_ratio', 0.0):.4f}
  Border complexity : {metrics.get('border_complexity', 0.0):.4f}
  Asymmetry         : {metrics.get('asymmetry', 0.0):.4f}  [0=đối xứng, 1=bất đối xứng]
  Circularity       : {metrics.get('circularity', 0.0):.4f}  [0=không tròn, 1=tròn đều]

[GUARDRAIL_RULES]
ĐƯỢC PHÉP: Giải thích cơ chế bệnh sinh, mô tả triệu chứng, hướng dẫn chăm sóc da không dùng thuốc,
  phân nhóm thuốc tổng quát, giải thích ý nghĩa chỉ số ABCD, khuyến nghị gặp bác sĩ.
TUYỆT ĐỐI CẤM: Tên biệt dược cụ thể, liều lượng, thời gian dùng thuốc.
ĐỊNH DẠNG: Tiếng Việt, rõ ràng, chuyên nghiệp, tối đa 400 từ, kết thúc bằng khuyến nghị gặp bác sĩ.
"""


def _fallback_response(question: str, result: Dict[str, Any]) -> str:
    cls = result.get("classification") or {}
    return (
        f'Câu hỏi: "{question}". '
        f'Dự đoán: {cls.get("prediction", "N/A")} ({float(cls.get("confidence", 0.0)):.3f}). '
        "Vui lòng tham khảo ý kiến bác sĩ da liễu."
    )


def generate_vqa_response_stream(
    question: str,
    result:   Dict[str, Any],
    api_key:  Optional[str],
    history:  Optional[List[Dict[str, str]]] = None,
):
    if result.get("status") == "triage":
        reason_vi = TRIAGE_REASON_VI.get(result.get("triage_reason", ""), result.get("triage_reason", ""))
        yield (
            f"Safety Gate đã kích hoạt — Hệ thống không thể đưa ra tư vấn vì:\n\n"
            f"> {reason_vi}\n\n"
            "Vui lòng chụp lại ảnh tổn thương với ánh sáng tốt hơn, "
            "hoặc liên hệ trực tiếp với bác sĩ da liễu để được thăm khám chính xác."
        )
        return

    cls_data   = result.get("classification") or {}
    cv_context = {
        "prediction":    cls_data.get("prediction",   "N/A"),
        "confidence":    float(cls_data.get("confidence", 0.0)),
        "probabilities": cls_data.get("probabilities", {}),
        "metrics":       result.get("metrics", {}),
    }
    
    # Xác định chế độ VQA hoạt động
    vqa_mode = st.session_state.get("vqa_mode_radio", "Trực tuyến — OpenAI GPT-4o-mini")
    
    # Phân loại câu hỏi để tìm fallback lâm sàng cục bộ
    q_lower = question.lower()
    category = "general"
    if any(k in q_lower for k in ("triệu chứng", "symptom", "biểu hiện", "dấu hiệu", "lâm sàng", "clinical")):
        category = "symptoms"
    elif any(k in q_lower for k in ("chăm sóc", "phác đồ", "care", "dưỡng", "bôi", "vệ sinh", "rửa", "điều trị", "treatment")):
        category = "care"
    elif any(k in q_lower for k in ("abcd", "chỉ số", "độ tròn", "bờ", "đối xứng", "asymmetry", "circularity", "border")):
        category = "abcd"
    elif any(k in q_lower for k in ("sinh thiết", "xét nghiệm", "biopsy", "test", "khám", "tiếp theo", "next step")):
        category = "biopsy"

    pred_class = cv_context["prediction"]
    # Trích xuất expert_answer tương ứng từ CLINICAL_KB
    kb_block = CLINICAL_KB.get(pred_class, CLINICAL_KB["NV"])
    expert_answer = kb_block.get(category, kb_block["general"])["vi"]

    # ── XỬ LÝ CHẾ ĐỘ OFFLINE CPU ──────────────────────────────────────
    if vqa_mode == "Nội bộ (Offline - CPUMedicalVQAModel)" or vqa_mode == "Offline (Custom CPUMedicalVQAModel)":
        model, tokenizer, device = load_custom_vqa_model()
        img_path = st.session_state.get("saved_local_img_path")
        
        if model is None or tokenizer is None or not img_path or not os.path.exists(img_path):
            for char in expert_answer:
                yield char
            return

        try:
            import torch
            from torchvision import transforms
            
            image = Image.open(img_path).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            english_question = "What is the diagnosis for this skin condition?"
            if category == "symptoms":
                english_question = "What are the typical clinical symptoms of this pathology?"
            elif category == "care":
                english_question = "What is the recommended skin care protocol for this lesion?"
            elif category == "abcd":
                english_question = "Explain the ABCD abnormalities of this lesion."
            elif category == "biopsy":
                english_question = "What are the next recommended biopsy and clinical diagnostic tests?"
                
            prompt = f"<|user|>: {english_question} <|doctor|>:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                if hasattr(model.llm, "transformer"):
                    text_embeds = model.llm.transformer.wte(inputs["input_ids"])
                elif hasattr(model.llm, "base_model"):
                    text_embeds = model.llm.base_model.model.transformer.wte(inputs["input_ids"])
                else:
                    text_embeds = None
                    
                if hasattr(model, "get_image_embeddings"):
                    img_embeds = model.get_image_embeddings(img_tensor, text_embeds=text_embeds)
                else:
                    img_embeds = model.projection(model.vision_backbone(img_tensor)).unsqueeze(1)
                    
                if hasattr(model, "clinical_prefix") and getattr(model, "has_prefix", True):
                    prefix_embeds = model.clinical_prefix(img_tensor.size(0), img_tensor.device)
                    inputs_embeds = torch.cat([prefix_embeds, img_embeds, text_embeds], dim=1)
                else:
                    inputs_embeds = torch.cat([img_embeds, text_embeds], dim=1)
                    
                attention_mask = torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long)
                
                generated_ids = model.llm.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=100,
                    repetition_penalty=1.2,
                    temperature=0.3,
                    top_p=0.85,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True
                )
                
            output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            if prompt in output_text:
                output_text = output_text.replace(prompt, "")
            output_clean = output_text.split("<|")[0].strip()
            
            # Dịch sang Tiếng Việt
            vietnamese_answer = translate_text(output_clean, "Vietnamese")
            
            if len(vietnamese_answer.strip()) < 15 or any(w in vietnamese_answer.lower() for w in ("prozac", "diphtheria", "allergic")):
                final_answer = expert_answer
            else:
                final_answer = f"Theo mô hình VQA nội bộ kết hợp phân tích hình ảnh: {vietnamese_answer}\n\n **Tư vấn y tế bổ sung**: {expert_answer}"
                
            for char in final_answer:
                yield char
            return
        except Exception as e:
            print("Local CPU inference error:", e)
            for char in expert_answer:
                yield char
            return

    # ── XỬ LÝ CHẾ ĐỘ ONLINE HOẶC OLLAMA ──────────────────────────────
    system_prompt = _build_system_prompt(cv_context)
    
    # Tích hợp RAG y văn y khoa tham khảo
    rag = get_rag_engine()
    retrieved_docs = []
    if rag and ("Trực tuyến" in vqa_mode or "Ollama" in vqa_mode):
        try:
            query_str = f"{cv_context['prediction']} {question}"
            retrieved_docs = rag.retrieve(query_str, n_results=1)
            if retrieved_docs:
                doc_text = retrieved_docs[0]["text"]
                system_prompt += f"\n\n[TÀI LIỆU Y KHOA CHÍNH THỨC THAM KHẢO (RAG)]\n{doc_text}\n\nLƯU Ý: Hãy ưu tiên sử dụng thông tin từ tài liệu y khoa tham khảo trên để làm cơ sở trả lời người dùng."
                st.session_state["has_rag_queries"] = True
        except Exception as e:
            print("RAG query failed:", e)

    history    = history or []
    valid_hist = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    if valid_hist and valid_hist[-1]["role"] == "user":
        valid_hist = valid_hist[:-1]

    messages = [
        {"role": "system", "content": system_prompt},
        *valid_hist,
        {"role": "user",   "content": question},
    ]

    is_ollama = ("Ollama" in vqa_mode)
    
    if is_ollama:
        try:
            from openai import OpenAI as LocalOpenAI
            client = LocalOpenAI(base_url="http://localhost:11434/v1", api_key="ollama", timeout=5.0)
            model_to_use = "qwen2.5:3b"
        except Exception as e:
            yield (
                "️ **Không thể kết nối tới dịch vụ Ollama cục bộ (localhost:11434)**.\n\n"
                "**Hướng dẫn kích hoạt**:\n"
                "1. Vui lòng kiểm tra xem ứng dụng **Ollama** đã khởi động trên máy chưa.\n"
                "2. Kéo mô hình bằng cách chạy lệnh sau trên Command Prompt / Terminal:\n"
                "   ```bash\n"
                "   ollama run qwen2.5:3b\n"
                "   ```\n\n"
                "---\n"
                f" *Đang tự động chuyển sang chế độ tư vấn lâm sàng y khoa dự phòng...*\n\n"
                f"{expert_answer}"
            )
            return
    else:
        # OpenAI Online
        if "OPENAI_API_KEY" in st.secrets:
            api_key = api_key or st.secrets["OPENAI_API_KEY"]
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if OpenAI is None or not api_key:
            for char in expert_answer:
                yield char
            return
        client = OpenAI(api_key=api_key)
        model_to_use = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    raw_parts: List[str] = []
    try:
        resp = client.chat.completions.create(
            model      = model_to_use,
            temperature= 0.2,
            max_tokens = 800,
            messages   = messages,
            stream     = True,
        )
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                raw_parts.append(content)
                yield content
    except Exception as exc:
        if is_ollama:
            yield (
                "️ **Lỗi thực thi mô hình Ollama cục bộ**.\n\n"
                "Hãy chắc chắn rằng bạn đã kéo mô hình `qwen2.5:3b` về máy.\n\n"
                "---\n"
                f" *Đang tự động chuyển sang chế độ tư vấn lâm sàng y khoa dự phòng...*\n\n"
                f"{expert_answer}"
            )
        else:
            write_dev_log({"error": str(exc), "question": question}, "LLM_ERROR")
            for char in expert_answer:
                yield char
        return

    if retrieved_docs and history is not None:
        st.session_state["last_vqa_references"] = retrieved_docs

    write_dev_log({
        "system_prompt":    system_prompt,
        "user_message":     question,
        "raw_response":     "".join(raw_parts),
        "cv_context":       cv_context,
        "chat_history_len": len(valid_hist),
    }, "LLM_VQA_EXCHANGE")


# ==============================================================================
# VISUALIZATION
# ==============================================================================
def render_probability_chart(
    probabilities: Dict[str, float],
    patient_name:  str = "",
    timestamp:     str = "",
) -> None:
    if not probabilities:
        return

    keys   = list(probabilities.keys())
    values = list(probabilities.values())
    labels = [f"{k}<br><sub>{get_vietnamese_diagnosis(k)}</sub>" for k in keys]

    colors = []
    for k, v in zip(keys, values):
        if k in MALIGNANT_CLASSES:
            r = min(255, int(180 + 75 * v))
            g = max(0,   int(80  - 80 * v))
            b = 60
        else:
            r = 60
            g = min(200, int(140 + 60 * v))
            b = min(180, int(100 + 80 * v))
        colors.append(f"rgba({r},{g},{b},0.88)")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.15)", width=1)),
        text=[f"{v:.3f}" for v in values], textposition="outside",
        textfont=dict(size=11, color="#cbd5e1"),
        hovertemplate="<b>%{y}</b><br>Xác suất: %{x:.4f}<extra></extra>",
    ))

    top_key = max(probabilities, key=probabilities.get)
    top_val = probabilities[top_key]
    fig.update_layout(
        title=dict(
            text=(f"Phân phối Xác suất Bệnh lý<br>"
                  f"<sup>Dự đoán cao nhất: <b>{top_key}</b> — {top_val:.2%}</sup>"),
            font=dict(size=13, color="#e2e8f0"), x=0.5,
        ),
        xaxis=dict(title="Xác suất", range=[0, 1.18],
                   tickformat=".0%", gridcolor="rgba(255,255,255,0.08)", color="#94a3b8"),
        yaxis=dict(autorange="reversed", color="#94a3b8", tickfont=dict(size=10)),
        plot_bgcolor ="rgba(15,23,42,0.92)",
        paper_bgcolor="rgba(15,23,42,0.92)",
        font=dict(color="#e2e8f0"),
        margin=dict(l=10, r=65, t=80, b=40),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)

    try:
        ts_clean  = (timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"))\
                    .replace("/", "").replace("\\", "").replace(":", "").replace(" ", "_")
        safe_name = "".join(c for c in patient_name.upper() if c.isalnum() or c in "-_") or "UNKNOWN"
        png_path  = CHART_SAVE_DIR / f"prob_{safe_name}_{ts_clean}.png"
        try:
            with open(png_path, "wb") as f:
                f.write(fig.to_image(format="png", width=800, height=400, scale=2))
        except Exception:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig_m, ax = plt.subplots(figsize=(10, 5), facecolor="#0f172a")
            ax.set_facecolor("#0f172a")
            items = sorted(probabilities.items(), key=lambda x: -x[1])
            lbls  = [f"{k}\n{get_vietnamese_diagnosis(k)}" for k, _ in items]
            vals  = [v for _, v in items]
            clrs  = ["#dc2626" if k in MALIGNANT_CLASSES else "#2563eb" for k, _ in items]
            ax.barh(lbls, vals, color=clrs)
            ax.set_xlim(0, 1.1); ax.tick_params(colors="white")
            for sp in ax.spines.values(): sp.set_edgecolor("#334155")
            plt.tight_layout()
            plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
            plt.close(fig_m)
    except Exception:
        pass


def render_radar_chart(metrics: Dict[str, Any]) -> None:
    area  = min(float(metrics.get("area_ratio",        0.0)) / 0.75, 1.0)
    bord  = min(float(metrics.get("border_complexity", 0.0)) / 8.0,  1.0)
    asym  = float(metrics.get("asymmetry",   0.0))
    circ  = float(metrics.get("circularity", 0.0))
    cats  = ["Diện tích<br>(Area)", "Phức tạp bờ<br>(Border)",
             "Bất đối xứng<br>(Asymmetry)", "Độ tròn<br>(Circularity)"]
    vals  = [area, bord, asym, circ]
    vals += [vals[0]]; cats += [cats[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals, theta=cats, fill="toself",
        fillcolor="rgba(59,130,246,0.15)",
        line=dict(color="rgba(96,165,250,0.9)", width=2),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(15,23,42,0.8)",
            radialaxis=dict(visible=True, range=[0, 1],
                            color="#64748b", gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(color="#64748b", gridcolor="rgba(255,255,255,0.1)"),
        ),
        paper_bgcolor="rgba(15,23,42,0.9)",
        font=dict(color="#e2e8f0", size=10),
        showlegend=False,
        margin=dict(l=40, r=40, t=30, b=30),
        height=280,
    )
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# PDF GENERATION
# ==============================================================================
def generate_pdf_report(patient_info: Dict, visit_data: Dict) -> Optional[bytes]:
    try:
        from fpdf import FPDF
        import datetime
        from app_streamlit import get_vietnamese_diagnosis
        
        pdf = FPDF()
        pdf.add_page()
        
        # Load Unicode font DejaVu để hiển thị Tiếng Việt đầy đủ
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
        # 2. Tiêu đề thương hiệu y khoa
        pdf.set_font("DejaVu", "B", 15)
        pdf.set_text_color(30, 58, 138)  # Deep Medical Blue
        pdf.cell(0, 10, "HỆ THỐNG EHR BỆNH ÁN ĐIỆN TỬ MULTI-VISIT AI-DERMA", new_x="LMARGIN", new_y="NEXT", align="C")
        
        pdf.set_font("DejaVu", "", 9)
        pdf.set_text_color(100, 116, 139)  # Slate
        pdf.cell(0, 5, "Trung tâm Phân tích Định lượng AI & Hỗ trợ Lâm sàng VQA", new_x="LMARGIN", new_y="NEXT", align="C")
        
        pdf.ln(5)
        pdf.set_draw_color(226, 232, 240)  # #e2e8f0
        pdf.line(15, 38, 195, 38)
        
        # Tiêu đề báo cáo
        pdf.ln(5)
        pdf.set_font("DejaVu", "B", 13)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 8, "PHIẾU KẾT QUẢ PHÂN TÍCH VÀ CHẨN ĐOÁN HÌNH ẢNH DA LIỄU", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(4)
        
        # --- PHẦN I: THÔNG TIN HÀNH CHÍNH ---
        pdf.set_font("DejaVu", "B", 10)
        pdf.set_text_color(37, 99, 235)
        pdf.cell(0, 6, "I. THÔNG TIN HÀNH CHÍNH BỆNH NHÂN", new_x="LMARGIN", new_y="NEXT")
        
        # Thiết lập bảng thông tin
        pdf.set_font("DejaVu", "", 9)
        pdf.set_text_color(30, 41, 59)
        pdf.set_draw_color(203, 213, 225)
        pdf.set_fill_color(248, 250, 252)  # Nền nhạt
        
        # Dòng 1: Họ tên + Giới tính
        pdf.cell(30, 7, " Họ tên bệnh nhân", border=1, fill=True)
        pdf.set_font("DejaVu", "B", 9)
        pdf.cell(70, 7, f" {patient_info.get('name', 'N/A').upper()}", border=1)
        pdf.set_font("DejaVu", "", 9)
        pdf.cell(30, 7, " Giới tính / Tuổi", border=1, fill=True)
        pdf.cell(50, 7, f" {patient_info.get('gender', 'N/A')} / {patient_info.get('age', 'N/A')} tuổi", border=1)
        pdf.ln(7)
        
        # Dòng 2: Quê quán + Vị trí tổn thương
        pdf.cell(30, 7, " Quê quán", border=1, fill=True)
        pdf.cell(70, 7, f" {patient_info.get('hometown', 'N/A')}", border=1)
        pdf.cell(30, 7, " Vị trí tổn thương", border=1, fill=True)
        pdf.cell(50, 7, f" {patient_info.get('location', 'N/A')}", border=1)
        pdf.ln(10)
        
        # --- PHẦN II: PHÂN TÍCH ĐỊNH LƯỢNG AI ---
        pdf.set_font("DejaVu", "B", 10)
        pdf.set_text_color(37, 99, 235)
        pdf.cell(0, 6, "II. KẾT QUẢ ĐỊNH LƯỢNG HÌNH ẢNH (AI EXTRACTED METRICS)", new_x="LMARGIN", new_y="NEXT")
        
        # Header bảng AI
        pdf.set_font("DejaVu", "B", 9)
        pdf.set_text_color(255, 255, 255)
        pdf.set_fill_color(30, 58, 138)  # Deep Blue Header
        pdf.cell(70, 7, " Chỉ số đánh giá", border=1, fill=True)
        pdf.cell(40, 7, " Giá trị phân tích", border=1, fill=True, align="C")
        pdf.cell(70, 7, " Đánh giá lâm sàng", border=1, fill=True)
        pdf.ln(7)
        
        pdf.set_text_color(30, 41, 59)
        ai = visit_data.get("ai_extracted_metrics", {})
        
        raw_pred = ai.get("prediction", "N/A")
        vi_pred = get_vietnamese_diagnosis(raw_pred)
        conf = ai.get("confidence", 0.0)
        
        metrics = [
            ("Chẩn đoán bệnh lý lý thuyết", f"{vi_pred}", f"Độ tin cậy: {conf:.1%}"),
            ("Tỉ lệ diện tích (Area ratio)", f"{ai.get('area_ratio', 0.0):.4f}", "Chiếm tỷ lệ diện tích trên vùng ảnh"),
            ("Độ bất đối xứng (Asymmetry)", f"{ai.get('asymmetry', 0.0):.4f}", "Đánh giá cấu trúc đối xứng tổn thương"),
            ("Độ tròn hình học (Circularity)", f"{ai.get('circularity', 0.0):.4f}", "Đo lường hình thái rìa tổn thương"),
            ("Độ phức tạp viền (Border complexity)", f"{ai.get('border_complexity', 0.0):.4f}", "Độ lồi lõm của viền ngoài")
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
        
        if "melanoma" in raw_pred.lower():
            adv = (
                "Phát hiện dấu hiệu nguy cơ cao của Ung thư hắc tố ác tính (Melanoma). Khuyến nghị chuyển tiếp bệnh nhân "
                "đến chuyên khoa phẫu thuật tạo hình/da liễu khẩn cấp để thực hiện sinh thiết toàn bộ tổn thương và xét nghiệm "
                "mô bệnh học chẩn đoán xác định. Bệnh nhân tuyệt đối không gãi, nặn vùng da tổn thương."
            )
        elif "carcinoma" in raw_pred.lower():
            adv = (
                "Phát hiện dấu hiệu nghi ngờ Ung thư biểu mô tế bào đáy/tế bào vảy. Khuyến nghị chỉ định bệnh nhân soi da chuyên sâu "
                "và lên kế hoạch hội chẩn phẫu thuật cắt bỏ bờ an toàn. Theo dõi sát tiến triển tổn thương vùng xung quanh."
            )
        else:
            adv = (
                "Phân tích hình thái cho thấy tổn thương có khả năng cao là lành tính thông thường (Nevi/Lành tính). "
                "Khuyến nghị tiếp tục tự theo dõi tại nhà, chụp ảnh kiểm tra sự thay đổi kích thước, màu sắc định kỳ mỗi 6 tháng. "
                "Liên hệ bác sĩ nếu có bất kỳ hiện tượng ngứa, loét hoặc chảy máu bất thường."
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
        
        return bytes(pdf.output())
    except Exception as e:
        st.warning(f"Không tạo được PDF: {e}")
        return None
    except Exception as e:
        st.warning(f"Không tạo được PDF: {e}")
        return None


# ==============================================================================
# CSS — Clean, Professional Medical EHR (no comments, no dashes)
# ==============================================================================
def _inject_custom_css() -> None:
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Inter', 'Roboto', 'Segoe UI', sans-serif !important;
    }
    .ehr-disclaimer {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 12px;
        font-size: 0.81rem;
        color: #fca5a5;
        font-weight: 500;
    }
    .ehr-page-title {
        font-size: 1.55rem;
        font-weight: 700;
        color: var(--text-color) !important;
        margin: 0 0 2px 0;
    }
    .ehr-page-sub {
        font-size: 0.83rem;
        color: #64748b;
        margin: 0 0 14px 0;
    }
    .triage-banner {
        background: rgba(239, 68, 68, 0.1);
        border: 2px solid rgba(239, 68, 68, 0.5);
        border-radius: 10px;
        padding: 12px 18px;
        text-align: center;
        font-size: 0.92rem;
        color: #fca5a5;
        font-weight: 600;
        margin: 10px 0;
    }
    .success-banner {
        background: rgba(34, 197, 94, 0.08);
        border: 1.5px solid rgba(34, 197, 94, 0.4);
        border-radius: 8px;
        padding: 10px 16px;
        color: #4ade80;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 8px 0;
    }
    .clinical-warn {
        background: rgba(234, 88, 12, 0.09);
        border: 1.5px solid rgba(234, 88, 12, 0.45);
        border-radius: 10px;
        padding: 10px 16px;
        margin: 8px 0;
        color: #fdba74;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    .abcd-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(147, 197, 253, 0.25);
        border-radius: 8px;
        padding: 12px 10px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .abcd-card.warn {
        background: rgba(239, 68, 68, 0.08);
        border-color: rgba(239, 68, 68, 0.35);
    }
    .abcd-card .card-label {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .abcd-card .card-value {
        font-size: 1.2rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    .abcd-card .card-value.warn {
        color: #f87171;
    }
    .abcd-card .card-sublabel {
        font-size: 0.67rem;
        color: #64748b;
        margin-top: 3px;
    }
    .diag-badge {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: rgba(30, 58, 138, 0.15);
        border: 1px solid rgba(96, 165, 250, 0.25);
        border-radius: 8px;
        padding: 8px 14px;
        margin: 10px 0;
        font-size: 0.9rem;
    }
    .diag-badge .conf-label {
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.75rem;
    }
    .conf-high { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
    .conf-mid  { background: rgba(234, 179, 8, 0.15); color: #facc15; border: 1px solid rgba(234, 179, 8, 0.3); }
    .conf-low  { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    [data-testid="stChatMessageContent"] p {
        margin: 0 0 2px 0 !important;
        line-height: 1.45 !important;
    }
    [data-testid="stChatMessageContent"] ol,
    [data-testid="stChatMessageContent"] ul {
        margin: 2px 0 4px 0 !important;
        padding-left: 16px !important;
    }
    [data-testid="stChatMessageContent"] li {
        margin-bottom: 1px !important;
        line-height: 1.4 !important;
    }
    [data-testid="stChatMessage"] {
        padding: 4px 0 !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111e36 100%);
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stSlider"] p,
    [data-testid="stSidebar"] [data-testid="stSlider"] span,
    [data-testid="stSidebar"] [data-testid="stRadio"] label,
    [data-testid="stSidebar"] [data-testid="stRadio"] p,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] div,
    [data-testid="stSidebar"] [data-testid="stNumberInput"] div {
        color: #e2e8f0 !important;
    }
    .sidebar-section {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #94a3b8 !important;
        padding: 8px 0 4px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 8px;
    }
    .step-row {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 4px 0;
        font-size: 0.78rem;
        color: #64748b;
    }
    .step-row.done { color: #4ade80; }
    .step-row.active { color: #60a5fa; font-weight: 600; }
    .step-dot {
        width: 16px; height: 16px; flex-shrink: 0;
        border-radius: 50%;
        border: 1px solid currentColor;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.65rem; font-weight: 700;
    }
    .step-dot.done { background: rgba(74, 222, 128, 0.1); }
    .voice-box {
        background: rgba(30, 41, 59, 0.25);
        border: 1px dashed rgba(96, 165, 250, 0.2);
        border-radius: 8px;
        padding: 8px;
        margin: 6px 0;
    }
    .voice-box-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #60a5fa;
        margin-bottom: 4px;
    }
    [data-testid="stTabs"] [role="tab"] {
        font-size: 0.82rem;
        font-weight: 600;
    }
    div[data-testid="stAppViewContainer"] { opacity: 1 !important; filter: none !important; }
    [data-st-mode="running"] * { opacity: 1 !important; }
    div[data-testid="stTextInput"] input {
        height: 50px !important;
        font-size: 1.05rem !important;
        border-radius: 8px !important;
        background-color: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(147, 197, 253, 0.25) !important;
        color: #f1f5f9 !important;
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="column"]:last-child button {
        height: 50px !important;
        font-size: 1.45rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border-radius: 8px !important;
        background-color: #2563eb !important;
        color: white !important;
        margin-top: 0 !important;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ==============================================================================
# DOCTOR DASHBOARD (EHR Timeline)
# ==============================================================================
def render_doctor_dashboard() -> None:
    st.markdown("<p class='ehr-page-title'>Hệ thống Tra cứu Hồ sơ Bệnh án Điện tử (EHR Multi-visit Timeline)</p>", unsafe_allow_html=True)
    st.markdown("<p class='ehr-page-sub'>Hỗ trợ hiển thị lịch sử tiến triển lâm sàng đa phương thức qua các mốc thời gian.</p>", unsafe_allow_html=True)

    all_records = fetch_all_medical_records()
    if not all_records:
        st.warning("Hiện tại chưa có dữ liệu bệnh án nào trong kho lưu trữ đám mây.")
        return

    options: Dict[str, Dict[str, Any]] = {}
    for r in all_records:
        pi    = r.get("patient_info", {})
        name  = pi.get("name", "Ẩn danh")
        total = len(r.get("visits", []))
        label = f"BN: {name.upper()} ({pi.get('age', '??')} tuổi) — [{total} mốc khám]"
        options[label] = r

    sel = st.selectbox("Chọn bệnh nhân cần tra cứu:", list(options.keys()))
    if not sel:
        return

    record = options[sel]
    pi     = record.get("patient_info", {})
    visits = sorted(record.get("visits", []), key=lambda x: x.get("timestamp_id", ""), reverse=True)

    st.divider()
    
    # --- PATIENT INFO & DELETION ---
    col_info, col_del = st.columns([3, 1])
    with col_info:
        st.subheader("Thông tin hành chính bệnh nhân")
        ca, cb, cc = st.columns(3)
        ca.markdown(f"**Họ tên:** `{pi.get('name', '').upper()}`")
        cb.markdown(f"**Tuổi:** `{pi.get('age', 'N/A')}` — **Giới tính:** `{pi.get('gender', 'N/A')}`")
        cc.markdown(f"**Địa chỉ:** `{pi.get('hometown', 'N/A')}`")
    with col_del:
        st.write("")
        st.write("")
        if st.button("️ Xóa bệnh nhân", type="primary", use_container_width=True, key="del_patient_btn"):
            st.session_state["show_delete_confirm"] = True
            
        if st.session_state.get("show_delete_confirm"):
            st.warning(f"Xác nhận xóa vĩnh viễn hồ sơ của bệnh nhân **{pi.get('name', '').upper()}**?")
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                if st.button("Xác nhận XÓA", key="confirm_delete_btn", use_container_width=True):
                    if delete_patient_record_from_gcp(pi.get('name', '')):
                        st.success("Đã xóa bệnh nhân thành công.")
                        st.session_state["show_delete_confirm"] = False
                        st.rerun()
            with col_c2:
                if st.button("Hủy bỏ", key="cancel_delete_btn", use_container_width=True):
                    st.session_state["show_delete_confirm"] = False
                    st.rerun()
                    
    # Tải dữ liệu JSON và CSV
    col_dl_json, col_dl_csv = st.columns(2)
    with col_dl_json:
        json_str = json.dumps(record, indent=2, ensure_ascii=False)
        st.download_button(
            label=" Tải hồ sơ bệnh án (JSON)",
            data=json_str,
            file_name=f"EHR_{pi.get('name', 'BenhNhan').replace(' ', '_')}.json",
            mime="application/json",
            use_container_width=True,
            key="dl_json_btn"
        )
    with col_dl_csv:
        import io
        import csv
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow([
            "Mốc khám", "Ngày khám", "Vị trí tổn thương", "Chẩn đoán chính", 
            "Độ tin cậy", "Area Ratio", "Border Complexity", "Asymmetry", "Circularity", "Số tin nhắn VQA"
        ])
        for v_idx, v in enumerate(reversed(visits)):
            ai_met = v.get("ai_extracted_metrics", {})
            vqa_len = len(v.get("vqa_conversations", []))
            writer.writerow([
                f"Lần {v_idx + 1}",
                v.get("created_at", "N/A"),
                v.get("location") or ai_met.get("location", "N/A"),
                ai_met.get("prediction", "N/A"),
                f"{float(ai_met.get('confidence', 0.0)):.4f}",
                f"{float(ai_met.get('area_ratio', 0.0)):.4f}",
                f"{float(ai_met.get('border_complexity', 0.0)):.4f}",
                f"{float(ai_met.get('asymmetry', 0.0)):.4f}",
                f"{float(ai_met.get('circularity', 0.0)):.4f}",
                vqa_len
            ])
        csv_str = csv_buffer.getvalue()
        st.download_button(
            label=" Tải lịch sử mốc khám (CSV)",
            data=csv_str.encode("utf-8-sig"),
            file_name=f"EHR_{pi.get('name', 'BenhNhan').replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_csv_btn"
        )

    st.divider()

    # --- FILTER BY LOCATION ---
    st.subheader("Bộ lọc & Đối chiếu Tiến triển")
    
    locs_in_visits = set()
    for v in visits:
        loc = v.get("location") or v.get("ai_extracted_metrics", {}).get("location")
        if loc and loc != "--- Chọn Vị trí tổn thương ---":
            locs_in_visits.add(loc)
    loc_options = ["Tất cả vị trí"] + sorted(list(locs_in_visits))
    
    col_filter, col_compare = st.columns(2)
    with col_filter:
        selected_loc_filter = st.selectbox("Lọc theo vị trí tổn thương:", loc_options, index=0)
        
    if selected_loc_filter != "Tất cả vị trí":
        filtered_visits = [
            v for v in visits
            if (v.get("location") == selected_loc_filter) or (v.get("ai_extracted_metrics", {}).get("location") == selected_loc_filter)
        ]
    else:
        filtered_visits = visits

    # --- COMPARISON MODE ---
    with col_compare:
        st.write("")
        st.write("")
        compare_mode = st.checkbox("Bật chế độ đối chiếu tiến triển (Comparison Mode)", key="enable_compare_mode")
        
    if compare_mode:
        if len(filtered_visits) < 2:
            st.info("Bệnh nhân cần ít nhất 2 mốc khám ở vị trí này để thực hiện đối chiếu.")
        else:
            st.markdown("#### Chọn 2 mốc khám cần đối chiếu:")
            col_sel_a, col_sel_b = st.columns(2)
            with col_sel_a:
                v_a_idx = st.selectbox(
                    "Mốc khám A (Cũ hơn):",
                    range(len(filtered_visits)),
                    format_func=lambda idx: f"Lần khám {len(filtered_visits) - idx} ({filtered_visits[idx].get('created_at', 'N/A')})",
                    index=min(1, len(filtered_visits) - 1),
                    key="compare_visit_a"
                )
            with col_sel_b:
                v_b_idx = st.selectbox(
                    "Mốc khám B (Mới hơn):",
                    range(len(filtered_visits)),
                    format_func=lambda idx: f"Lần khám {len(filtered_visits) - idx} ({filtered_visits[idx].get('created_at', 'N/A')})",
                    index=0,
                    key="compare_visit_b"
                )
                
            if v_a_idx == v_b_idx:
                st.warning("Vui lòng chọn hai lần khám khác nhau để thực hiện so sánh đối chiếu.")
            else:
                visit_a = filtered_visits[v_a_idx]
                visit_b = filtered_visits[v_b_idx]
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("<p style='font-weight:700;color:#1E3A8A;'>MỐC KHÁM A (Ngày: " + visit_a.get('created_at', 'N/A') + ")</p>", unsafe_allow_html=True)
                    st.write(f"Vị trí: `{visit_a.get('location', 'N/A')}`")
                    img_url_a = visit_a.get("image_url")
                    mask_url_a = visit_a.get("mask_url")
                    gradcam_url_a = visit_a.get("gradcam_url")
                    
                    c_a1, c_a2, c_a3 = st.columns(3)
                    with c_a1:
                        if img_url_a: st.image(img_url_a, caption="Ảnh gốc A", use_container_width=True)
                    with c_a2:
                        if mask_url_a: st.image(mask_url_a, caption="Mặt nạ A", use_container_width=True)
                        else: st.caption("Không có mặt nạ")
                    with c_a3:
                        if gradcam_url_a: st.image(gradcam_url_a, caption="Grad-CAM A", use_container_width=True)
                        else: st.caption("Không có Grad-CAM")
                        
                    metrics_a = visit_a.get("ai_extracted_metrics", {})
                    st.metric("Chẩn đoán A", get_vietnamese_diagnosis(metrics_a.get("prediction", "N/A")))
                    st.metric("Độ tin cậy A", f"{float(metrics_a.get('confidence', 0.0)) * 100:.1f}%")
                    
                with col_b:
                    st.markdown("<p style='font-weight:700;color:#065F46;'>MỐC KHÁM B (Ngày: " + visit_b.get('created_at', 'N/A') + ")</p>", unsafe_allow_html=True)
                    st.write(f"Vị trí: `{visit_b.get('location', 'N/A')}`")
                    img_url_b = visit_b.get("image_url")
                    mask_url_b = visit_b.get("mask_url")
                    gradcam_url_b = visit_b.get("gradcam_url")
                    
                    c_b1, c_b2, c_b3 = st.columns(3)
                    with c_b1:
                        if img_url_b: st.image(img_url_b, caption="Ảnh gốc B", use_container_width=True)
                    with c_b2:
                        if mask_url_b: st.image(mask_url_b, caption="Mặt nạ B", use_container_width=True)
                        else: st.caption("Không có mặt nạ")
                    with c_b3:
                        if gradcam_url_b: st.image(gradcam_url_b, caption="Grad-CAM B", use_container_width=True)
                        else: st.caption("Không có Grad-CAM")
                        
                    metrics_b = visit_b.get("ai_extracted_metrics", {})
                    st.metric("Chẩn đoán B", get_vietnamese_diagnosis(metrics_b.get("prediction", "N/A")))
                    st.metric("Độ tin cậy B", f"{float(metrics_b.get('confidence', 0.0)) * 100:.1f}%")
                
                st.markdown("#### Bảng so sánh chỉ số hình học ABCD")
                m_keys = [
                    ("area_ratio", "Tỷ lệ diện tích (Area ratio)"),
                    ("border_complexity", "Độ phức tạp bờ (Border complexity)"),
                    ("asymmetry", "Bất đối xứng (Asymmetry)"),
                    ("circularity", "Độ tròn (Circularity)")
                ]
                comp_rows = []
                for k, label in m_keys:
                    val_a = float(metrics_a.get(k, 0.0))
                    val_b = float(metrics_b.get(k, 0.0))
                    diff = val_b - val_a
                    comp_rows.append({
                        "Chỉ số": label,
                        "Mốc A (Cũ hơn)": f"{val_a:.4f}",
                        "Mốc B (Mới hơn)": f"{val_b:.4f}",
                        "Chênh lệch (B - A)": f"{diff:+.4f}"
                    })
                st.table(comp_rows)
                
                st.markdown("#### Đánh giá tiến triển lâm sàng định tính:")
                analysis_texts = []
                area_diff = float(metrics_b.get("area_ratio", 0.0)) - float(metrics_a.get("area_ratio", 0.0))
                border_diff = float(metrics_b.get("border_complexity", 0.0)) - float(metrics_a.get("border_complexity", 0.0))
                asym_diff = float(metrics_b.get("asymmetry", 0.0)) - float(metrics_a.get("asymmetry", 0.0))
                circ_diff = float(metrics_b.get("circularity", 0.0)) - float(metrics_a.get("circularity", 0.0))
                
                if area_diff > 0.05:
                    analysis_texts.append(" **Diện tích tổn thương:** Có xu hướng tăng đáng kể (lan rộng), khuyến nghị giám sát chặt chẽ.")
                elif area_diff < -0.05:
                    analysis_texts.append(" **Diện tích tổn thương:** Thu nhỏ rõ rệt, đáp ứng điều trị tốt.")
                else:
                    analysis_texts.append(" **Diện tích tổn thương:** Không có biến động lớn, kích thước ổn định.")
                    
                if border_diff > 0.5:
                    analysis_texts.append(" **Độ phức tạp bờ:** Viền tổn thương nham nhở và phức tạp hơn, có thể là dấu hiệu tiến triển xấu.")
                elif border_diff < -0.5:
                    analysis_texts.append(" **Độ phức tạp bờ:** Viền tổn thương trở nên thuôn gọn và đều đặn hơn.")
                    
                if asym_diff > 0.1:
                    analysis_texts.append(" **Bất đối xứng:** Mức độ bất đối xứng tăng lên rõ rệt, dấu hiệu đáng báo động đối với tổn thương sắc tố.")
                elif asym_diff < -0.1:
                    analysis_texts.append(" **Bất đối xứng:** Tổn thương có xu hướng trở nên đối xứng hơn.")
                    
                if circ_diff < -0.08:
                    analysis_texts.append(" **Độ tròn:** Hình dạng biến đổi méo mó, giảm độ tròn tự nhiên.")
                elif circ_diff > 0.08:
                    analysis_texts.append(" **Độ tròn:** Hình dạng phục hồi tiến dần về dạng tròn đều hơn.")
                
                st.info("\n\n".join(analysis_texts))

    else:
        st.subheader("Biên niên sử hình ảnh và Chẩn đoán (tất cả lần khám)")
        for idx, visit in enumerate(filtered_visits):
            v_time = visit.get("created_at", "N/A")
            ai_m   = visit.get("ai_extracted_metrics", {})
            img_url= visit.get("image_url")
            mask_url = visit.get("mask_url")
            gradcam_url = visit.get("gradcam_url")
            convs  = visit.get("vqa_conversations", [])
            v_pred = ai_m.get("prediction", "N/A")
            v_vi   = get_vietnamese_diagnosis(v_pred)
            v_loc  = visit.get("location") or ai_m.get("location", "N/A")
            
            with st.expander(f"LẦN KHÁM THỨ {len(filtered_visits) - idx} — Ngày: {v_time} — Vị trí: {v_loc}", expanded=(idx == 0)):
                c_img1, c_img2, c_img3, c_info = st.columns([1, 1, 1, 1.2])
                with c_img1:
                    if img_url: st.image(img_url, caption="Ảnh gốc", use_container_width=True)
                    else: st.warning("Không có ảnh gốc.")
                with c_img2:
                    if mask_url: st.image(mask_url, caption="Mặt nạ", use_container_width=True)
                    else: st.caption("Không có mặt nạ")
                with c_img3:
                    if gradcam_url: st.image(gradcam_url, caption="Bản đồ nhiệt Grad-CAM", use_container_width=True)
                    else: st.caption("Không có Grad-CAM")
                    
                with c_info:
                    st.markdown("**Kết quả phân tích AI**")
                    m1, m2 = st.columns(2)
                    m1.metric("Nhãn dự đoán", v_pred)
                    m2.metric("Độ tin cậy", f"{float(ai_m.get('confidence', 0.0)) * 100:.1f}%")
                    st.markdown(f"Giải nghĩa: **{v_vi}**")
                    st.markdown("**Chỉ số hình học:**")
                    for k, lbl in [
                        ("area_ratio", "Area ratio"), ("border_complexity", "Border complexity"),
                        ("asymmetry", "Asymmetry"),   ("circularity", "Circularity"),
                    ]:
                        st.write(f"- {lbl}: `{float(ai_m.get(k, 0.0)):.4f}`")
                        
                if convs:
                    st.markdown("**Nhật ký tư vấn VQA:**")
                    for m in convs:
                        role = m.get("role", "")
                        esc  = m.get("content", "").replace("<", "&lt;").replace(">", "&gt;")
                        lbl  = "Bác sĩ" if role == "user" else "Trợ lý AI"
                        bd   = "#3b82f6" if role == "user" else "#22c55e"
                        st.markdown(
                            f"<div style='background:rgba(0,0,0,0.03);border-left:3px solid {bd};"
                            f"padding:5px 10px;border-radius:4px;font-size:0.84rem;margin:3px 0;color:#0F172A;'>"
                            f"<b>{lbl}:</b> {esc}</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("Lần khám này không thực hiện hội thoại VQA.")


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================
def download_weights_if_missing() -> None:
    import os
    import requests
    import re
    from pathlib import Path
    import streamlit as st

    weights_info = {
        "deeplabv3plus_best.pth": Path("4_Models/deeplabv3plus/deeplabv3plus_best.pth"),
        "efficientnet_attention_best.pth": Path("4_Models/classification/efficientnet_attention_best.pth"),
        "dermavqa_gpt2_joint_best.pth": Path("9_VQA/models/dermavqa_gpt2_joint_best.pth")
    }

    # Bất kỳ file nào chưa tồn tại hoặc dung lượng < 1MB (do lỗi trang HTML cảnh báo từ trước) đều coi là thiếu
    missing = []
    for filename, path in weights_info.items():
        if not path.exists() or path.stat().st_size < 1024 * 1024:
            missing.append(filename)

    if not missing:
        return

    # Danh sách File ID từ Google Drive của bạn
    drive_ids = [
        "1crKDYmFVt3rsfHMmIrLdhAlMg4cDCpfT",
        "1nquxwVeZRoawedzuP4H0wULF-e03fTMs",
        "1kTiNvqtQI6qtueDE3zGcTFFoDWio6STa"
    ]

    def download_file_from_google_drive(file_id, temp_dest):
        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(URL, params={'id': file_id}, stream=True)
        html = response.text
        
        confirm = None
        uuid = None
        confirm_match = re.search(r'name="confirm" value="([^"]+)"', html)
        if confirm_match:
            confirm = confirm_match.group(1)
            
        uuid_match = re.search(r'name="uuid" value="([^"]+)"', html)
        if uuid_match:
            uuid = uuid_match.group(1)

        if confirm:
            download_url = "https://drive.usercontent.google.com/download"
            params = {
                'id': file_id,
                'export': 'download',
                'confirm': confirm
            }
            if uuid:
                params['uuid'] = uuid
            response = session.get(download_url, params=params, stream=True)

        os.makedirs(os.path.dirname(temp_dest) if os.path.dirname(temp_dest) else ".", exist_ok=True)
        CHUNK_SIZE = 32768
        with open(temp_dest, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    with st.spinner(" Phát hiện thiếu hoặc lỗi tệp trọng số. Đang tiến hành tải xuống trọng số mô hình từ Google Drive (chỉ tải một lần)..."):
        # Tạo thư mục temp để tải và phân loại theo dung lượng
        temp_dir = Path("temp_weights")
        temp_dir.mkdir(exist_ok=True)
        
        for fid in drive_ids:
            print(f"[ModelRegistry] Downloading resource ID: {fid}...")
            temp_file = temp_dir / f"download_{fid}.pth"
            try:
                download_file_from_google_drive(fid, temp_file)
                if temp_file.exists():
                    fsize = temp_file.stat().st_size
                    print(f"[ModelRegistry] Downloaded ID {fid}: {fsize / 1024 / 1024:.2f} MB")
                    
                    # Phân loại dựa trên kích thước tệp
                    dest_path = None
                    if 70 * 1024 * 1024 <= fsize <= 85 * 1024 * 1024:
                        dest_path = weights_info["efficientnet_attention_best.pth"]
                    elif 290 * 1024 * 1024 <= fsize <= 320 * 1024 * 1024:
                        dest_path = weights_info["deeplabv3plus_best.pth"]
                    elif 330 * 1024 * 1024 <= fsize <= 360 * 1024 * 1024:
                        dest_path = weights_info["dermavqa_gpt2_joint_best.pth"]
                        
                    if dest_path:
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        if dest_path.exists():
                            dest_path.unlink()
                        import shutil
                        shutil.move(str(temp_file), str(dest_path))
                        print(f"[ModelRegistry] Configured weight: {dest_path.name}")
                    else:
                        st.error(f" Tệp tải xuống từ ID {fid} có kích thước không hợp lệ ({fsize / 1024 / 1024:.2f} MB)")
                        if temp_file.exists():
                            temp_file.unlink()
            except Exception as e:
                st.error(f" Lỗi khi tải tài nguyên ID {fid}: {str(e)}")
                if temp_file.exists():
                    temp_file.unlink()
                    
        # Dọn dẹp thư mục tạm
        try:
            if temp_dir.exists():
                temp_dir.rmdir()
        except Exception:
            pass

def main() -> None:
    st.set_page_config(
        page_title="Dermatology EHR — AI Assistant",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    download_weights_if_missing()
    _inject_custom_css()

    st.markdown(
        "<div class='ehr-disclaimer'>"
        "TUYÊN BỐ MIỄN TRỪ TRÁCH NHIỆM: Hệ thống này chỉ là công cụ hỗ trợ sàng lọc sơ bộ bằng AI, "
        "KHÔNG thay thế chẩn đoán y khoa chuyên môn của bác sĩ da liễu."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<p class='ehr-page-title'>Hệ thống Trợ lý Da liễu Đa phương thức và EHR Dashboard</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='ehr-page-sub'>"
        "Tích hợp chẩn đoán hình học, lọc chất lượng ảnh chụp, tư vấn phác đồ điều trị y văn"
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Session State Init ────────────────────────────────────────────────────
    _ss_defaults = {
        "messages":                [],
        "result":                  None,
        "analysis_time":           None,
        "saved_local_img_path":    None,
        "last_uploaded_file_name": None,
        "chat_input_val":          "",
        "last_voice_text":         "",
        "form_patient_name":       "",
        "form_patient_id":         "",
        "form_patient_age":        30,
        "form_patient_gender":     "--- Chọn Giới tính ---",
        "form_patient_hometown":   "--- Chọn Tỉnh/Thành ---",
        "form_patient_location":   "--- Chọn Vị trí tổn thương ---",
        "sam_point":               None,
        "custom_mask":             None,
        "sam_pending":             False,
        "input_key":               0,
        "voice_key":               0,
        "voice_prefill":           "",
        "full_image_result":       None,
        "active_analysis_path":    None,
        "last_selected_lesion_idx": "Tất cả tổn thương (Ảnh gốc)",
        "record_saved":            False,
        "show_delete_confirm":     False,
        "has_rag_queries":         False,
        "dicom_meta":              None,
    }
    for k, v in _ss_defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    allow_to_save = True

    # ============================================================
    # SIDEBAR
    # ============================================================
    with st.sidebar:
        _r   = st.session_state.get("result")
        _s1  = st.session_state.get("last_uploaded_file_name") is not None
        _s2  = _r is not None
        _s3  = _s2 and (_r.get("status") != "triage")
        _s4  = _s3 and bool(st.session_state.get("has_rag_queries", False))
        _s5  = bool(st.session_state.get("messages"))
        _s6  = bool(st.session_state.get("record_saved", False))

        st.markdown("<div class='sidebar-section'>Quy trình chẩn đoán</div>", unsafe_allow_html=True)
        for num, text, done, active in [
            ("1", "Nhận file ảnh / DICOM",       _s1, not _s1),
            ("2", "Đánh giá chất lượng ảnh",      _s2, _s1 and not _s2),
            ("3", "Phân tích AI (DeepLab/SAM)",   _s3, _s2 and not _s3),
            ("4", "Tra cứu y văn (RAG)",          _s4, _s3 and not _s4),
            ("5", "Tư vấn phác đồ lâm sàng (VQA)",_s5, _s4 and not _s5),
            ("6", "Đồng bộ EHR & Xuất báo cáo",   _s6, _s5 and not _s6),
        ]:
            cls = "done" if done else ("active" if active else "")
            st.markdown(
                f"<div class='step-row {cls}'>"
                f"<span class='step-dot {cls}'>{num}</span>"
                f"<span style='margin-left: 8px;'>{text}</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div class='sidebar-section' style='margin-top:14px;'>Thông tin bệnh nhân</div>", unsafe_allow_html=True)

        p_id = st.text_input(
            "Số định danh (CCCD / BHYT / Mã BN):",
            key="form_patient_id",
            placeholder="031090123456 (để trống tự sinh)",
        )
        p_name = st.text_input(
            "Họ và tên bệnh nhân:",
            key="form_patient_name",
            placeholder="Nguyễn Văn A",
        )
        p_age = st.number_input("Tuổi:", min_value=0, max_value=120, key="form_patient_age")
        p_gender = st.selectbox(
            "Giới tính:",
            ["--- Chọn Giới tính ---", "Nam", "Nữ", "Khác"],
            key="form_patient_gender",
        )
        p_hometown = st.selectbox("Quê quán / Tỉnh thành:", VIETNAM_PROVINCES, key="form_patient_hometown")
        p_location = st.selectbox("Vị trí tổn thương:", BODY_LOCATIONS, key="form_patient_location")

        if p_name.strip():
            is_old = check_patient_exists(p_name)
            if is_old:
                st.warning(f"PHÁT HIỆN: '{p_name.upper()}' đã có hồ sơ lịch sử.")
                confirm = st.radio(
                    "Xác nhận:",
                    ["Chưa chọn", "Có, ghi thêm mốc khám mới", "Không, bệnh nhân khác trùng tên"],
                    index=0, key="confirm_update",
                )
                if confirm == "Có, ghi thêm mốc khám mới":
                    allow_to_save = True
                    st.caption("Nút lưu đã mở khóa.")
                elif confirm == "Không, bệnh nhân khác trùng tên":
                    allow_to_save = False
                    st.error("Thêm Mã số định danh vào tên để tạo hồ sơ riêng.")
                else:
                    allow_to_save = False
            else:
                if p_name.strip():
                    st.success("HỒ SƠ MỚI: Sẽ tạo tài khoản hồ sơ mới.")

        def _reset_form():
            for k in ["form_patient_name", "form_patient_id"]:
                st.session_state[k] = ""
            st.session_state["form_patient_age"]      = 30
            st.session_state["form_patient_gender"]   = "--- Chọn Giới tính ---"
            st.session_state["form_patient_hometown"] = "--- Chọn Tỉnh/Thành ---"
            st.session_state["form_patient_location"] = "--- Chọn Vị trí tổn thương ---"

        st.button("Reset Form", on_click=_reset_form, key="sidebar_reset_btn")

        st.markdown("<div class='sidebar-section' style='margin-top:14px;'>Cấu hình hệ thống</div>", unsafe_allow_html=True)

        min_conf  = st.slider("Ngưỡng safety gate (tau_c)", 0.30, 0.95, 0.60, 0.01,
                              help="Ngưỡng kiểm soát chất lượng đầu vào.")
        mal_thresh= st.slider("Độ nhạy phát hiện ác tính", 0.05, 0.50, 0.15, 0.01,
                              help="Xác suất tối thiểu để hiện cảnh báo nguy cơ ác tính.")
        lambda_w  = st.slider("Trọng số: Hình ảnh vs Dịch tễ", 0.0, 1.0, 0.80, 0.05,
                              help="1.0 = chỉ dựa vào hình ảnh; 0.5 = kết hợp 50/50 với dữ liệu dịch tễ.")

        st.markdown("**Chế độ mô hình VQA:**")
        vqa_mode = st.radio(
            "VQA mode",
            VQA_MODE_OPTIONS,
            label_visibility="collapsed",
            key="vqa_mode_radio",
        )

        with st.expander("Thông số kỹ thuật AI"):
            st.markdown(
                "**Phân loại:** EfficientNet-B1 + CBAM Attention  \n"
                "**Phân đoạn:** DeepLabV3+ (ResNet-101 backbone)  \n"
                "**VQA Online:** GPT-4o-mini (OpenAI API)  \n"
                "**Độ trễ:** ~0.5 giây trung bình  \n"
                "**VRAM:** ~8 192 MB  \n"
                "**Biên ROI:** delta = 10 px"
            )

    # ============================================================
    # TABS CHÍNH
    # ============================================================
    tab_diag, tab_doctor = st.tabs([
        "Thực hiện Chẩn đoán VQA",
        "Màn hình Xem lại của Bác sĩ",
    ])

    # ==========================================================
    # TAB 1 — CHẨN ĐOÁN
    # ==========================================================
    with tab_diag:
        st.markdown("**Tải ảnh tổn thương da (JPG / PNG / DICOM .dcm):**")
        uploaded = st.file_uploader(
            "Upload",
            type=["jpg", "jpeg", "png", "dcm"],
            label_visibility="collapsed",
            key="main_file_uploader",
        )

        dicom_meta: Dict[str, str] = {}
        image: Optional[Image.Image] = None

        if uploaded is not None:
            if st.session_state["last_uploaded_file_name"] != uploaded.name:
                st.session_state["last_uploaded_file_name"] = uploaded.name
                st.session_state["result"]                  = None
                st.session_state["messages"]                = []
                st.session_state["analysis_time"]           = None
                st.session_state["saved_local_img_path"]    = None
                st.session_state["sam_point"]               = None
                st.session_state["custom_mask"]             = None

            if uploaded.name.lower().endswith(".dcm"):
                image, dicom_meta = load_dicom(uploaded)
                if image is None:
                    st.error("Không thể đọc file DICOM. Vui lòng kiểm tra định dạng.")
                else:
                    st.session_state["dicom_meta"] = dicom_meta
                    st.markdown(
                        f"<div class='success-banner'>Đã đọc DICOM thành công — "
                        f"Tên: {dicom_meta.get('patient_name','')} | "
                        f"Tuổi: {dicom_meta.get('patient_age','')} | "
                        f"Giới tính: {dicom_meta.get('patient_sex','')}</div>",
                        unsafe_allow_html=True,
                    )
                    if dicom_meta.get("patient_name"):
                        st.session_state["form_patient_name"] = dicom_meta["patient_name"]
                    if dicom_meta.get("patient_age"):
                        age_match = re.search(r'\d+', dicom_meta["patient_age"])
                        if age_match:
                            st.session_state["form_patient_age"] = int(age_match.group())
                    if dicom_meta.get("patient_sex"):
                        s = dicom_meta["patient_sex"].upper()
                        if "M" in s: st.session_state["form_patient_gender"] = "Nam"
                        elif "F" in s: st.session_state["form_patient_gender"] = "Nữ"
            else:
                image = Image.open(uploaded).convert("RGB")

        if image is not None:
            img_rgb = np.array(image)
            orig_w, orig_h = image.size

            max_display_w = 400
            scale = min(max_display_w / orig_w, 1.0)
            display_w = int(orig_w * scale)
            display_h = int(orig_h * scale)

            st.write("")
            seg_mode = st.radio(
                "Chế độ phân đoạn tổn thương da (AI Segmenter Mode):",
                [
                    "Phân đoạn tự động (DeepLabV3+)",
                    "Phân đoạn tương tác (SAM — Click điểm)",
                    "Vẽ tay chỉnh sửa (Drawable Canvas)",
                ],
                horizontal=True,
                key="seg_mode_radio",
            )

            # Vẽ điểm click SAM lên hình ảnh hiển thị để người dùng đối chiếu trực quan
            display_image = image.copy()
            if st.session_state["sam_point"] is not None:
                draw = ImageDraw.Draw(display_image)
                px = int(st.session_state["sam_point"][0] * scale)
                py = int(st.session_state["sam_point"][1] * scale)
                # Vẽ điểm đỏ tâm click
                draw.ellipse([px-6, py-6, px+6, py+6], fill="red", outline="white", width=2)

            canvas_result = None
            if seg_mode == "Phân đoạn tương tác (SAM — Click điểm)":
                st.info("Nhấp chuột vào đúng điểm trung tâm của vùng tổn thương trên ảnh. AI sẽ tự động cập nhật đường biên dựa trên điểm chọn.")
                canvas_result = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.3)",
                    stroke_width=2,
                    background_image=display_image,
                    update_streamlit=True,
                    height=display_h,
                    width=display_w,
                    drawing_mode="point",
                    key="canvas_sam",
                    display_toolbar=False,
                )
                # Xử lý click SAM mới từ canvas
                if canvas_result.json_data and canvas_result.json_data.get("objects"):
                    objs = canvas_result.json_data.get("objects")
                    last_obj = objs[-1]
                    rx = last_obj.get("radius", 0)
                    cx = int((last_obj.get("left", 0) + rx) / scale)
                    cy = int((last_obj.get("top", 0) + rx) / scale)
                    new_pt = (cx, cy)
                    if st.session_state["sam_point"] != new_pt:
                        st.session_state["sam_point"] = new_pt
                        st.session_state["custom_mask"] = None
                        st.session_state["sam_pending"] = True
                        st.rerun()

                # Chạy pipeline ngay khi có điểm SAM chờ xử lý (sau rerun canvas đã reset)
                if st.session_state.get("sam_pending") and st.session_state["sam_point"] is not None:
                    st.session_state["sam_pending"] = False
                    st.session_state["analysis_time"] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                    with st.spinner("Đang chẩn đoán tương tác với điểm chọn..."):
                        tmp_dir  = tempfile.mkdtemp()
                        tmp_path = os.path.join(tmp_dir, "input.png")
                        image.save(tmp_path)
                        st.session_state["saved_local_img_path"] = tmp_path
                        result_new = get_pipeline(min_conf).run(
                            tmp_path,
                            return_mask=True,
                            age=float(p_age) if p_name.strip() else None,
                            gender=p_gender if p_gender != "--- Chọn Giới tính ---" else None,
                            body_location=p_location if p_location != "--- Chọn Vị trí tổn thương ---" else None,
                            lambda_val=lambda_w,
                            interactive_point=st.session_state["sam_point"],
                            malignant_threshold=mal_thresh
                        )
                        if st.session_state.get("dicom_meta") and "metrics" in result_new:
                            meta = st.session_state["dicom_meta"]
                            if "pixel_spacing_x" in meta and "pixel_spacing_y" in meta:
                                result_new["metrics"]["pixel_spacing_x"] = meta["pixel_spacing_x"]
                                result_new["metrics"]["pixel_spacing_y"] = meta["pixel_spacing_y"]
                        st.session_state["result"] = result_new
                        st.session_state["messages"] = []
                        st.rerun()

            elif seg_mode == "Vẽ tay chỉnh sửa (Drawable Canvas)":
                st.info("Nhấn chuột và vẽ tự do để khoanh vùng tổn thương. Sau khi vẽ xong, nhấn nút Áp dụng phía dưới.")
                canvas_result = st_canvas(
                    fill_color="rgba(34, 197, 94, 0.3)",
                    stroke_color="#22c55e",
                    stroke_width=6,
                    background_image=display_image,
                    update_streamlit=True,
                    height=display_h,
                    width=display_w,
                    drawing_mode="freedraw",
                    key="canvas_freedraw",
                )
                if canvas_result.image_data is not None:
                    mask_data = canvas_result.image_data[:, :, 3] > 0
                    if mask_data.any():
                        resized_mask = cv2.resize(mask_data.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                        if st.button("Áp dụng nét vẽ tay chỉnh sửa", use_container_width=True, key="apply_custom_draw"):
                            st.session_state["custom_mask"] = resized_mask
                            st.session_state["sam_point"] = None
                            st.session_state["analysis_time"] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                            with st.spinner("Đang áp dụng phân đoạn vẽ tay..."):
                                tmp_dir  = tempfile.mkdtemp()
                                tmp_path = os.path.join(tmp_dir, "input.png")
                                image.save(tmp_path)
                                st.session_state["saved_local_img_path"] = tmp_path
                                result_new = get_pipeline(min_conf).run(
                                    tmp_path,
                                    return_mask=True,
                                    age=float(p_age) if p_name.strip() else None,
                                    gender=p_gender if p_gender != "--- Chọn Giới tính ---" else None,
                                    body_location=p_location if p_location != "--- Chọn Vị trí tổn thương ---" else None,
                                    lambda_val=lambda_w,
                                    custom_mask=st.session_state["custom_mask"],
                                    malignant_threshold=mal_thresh
                                )
                                if st.session_state.get("dicom_meta") and "metrics" in result_new:
                                    meta = st.session_state["dicom_meta"]
                                    if "pixel_spacing_x" in meta and "pixel_spacing_y" in meta:
                                        result_new["metrics"]["pixel_spacing_x"] = meta["pixel_spacing_x"]
                                        result_new["metrics"]["pixel_spacing_y"] = meta["pixel_spacing_y"]
                                st.session_state["result"] = result_new
                                st.session_state["messages"] = []
                                st.rerun()

            else:
                if st.button("Chạy Phân tích CV", type="primary", key="run_analysis_btn", use_container_width=True):
                    st.session_state["analysis_time"] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                    st.session_state["sam_point"] = None
                    st.session_state["custom_mask"] = None
                    with st.spinner("Đang chạy Segmentation + Classification tự động..."):
                        tmp_dir  = tempfile.mkdtemp()
                        tmp_path = os.path.join(tmp_dir, "input.png")
                        image.save(tmp_path)
                        st.session_state["saved_local_img_path"] = tmp_path
                        result_new = get_pipeline(min_conf).run(
                            tmp_path,
                            return_mask=True,
                            age=float(p_age) if p_name.strip() else None,
                            gender=p_gender if p_gender != "--- Chọn Giới tính ---" else None,
                            body_location=p_location if p_location != "--- Chọn Vị trí tổn thương ---" else None,
                            lambda_val=lambda_w,
                            malignant_threshold=mal_thresh
                        )
                        if st.session_state.get("dicom_meta") and "metrics" in result_new:
                            meta = st.session_state["dicom_meta"]
                            if "pixel_spacing_x" in meta and "pixel_spacing_y" in meta:
                                result_new["metrics"]["pixel_spacing_x"] = meta["pixel_spacing_x"]
                                result_new["metrics"]["pixel_spacing_y"] = meta["pixel_spacing_y"]
                        st.session_state["result"]   = result_new
                        st.session_state["messages"] = []
                        st.rerun()

            result = st.session_state.get("result")

            # ── HIỂN THỊ SONG SONG 3 ẢNH ──────────────────────────────────────
            mask_img_arr: Optional[np.ndarray] = None
            if result:
                raw_mask = result.get("segmentation_mask")
                mask_img_arr = _mask_to_image(raw_mask, img_rgb.shape[:2])

            st.write("")
            st.markdown("**Đối chiếu hình ảnh chẩn đoán song song:**")
            col_img1, col_img2, col_img3 = st.columns(3)
            with col_img1:
                # Nếu có điểm chọn SAM, hiển thị ảnh đã vẽ điểm đỏ
                st.image(display_image, caption="Ảnh gốc đầu vào (chấm đỏ là điểm SAM chọn)", use_container_width=True)
                if st.session_state["sam_point"] is not None:
                    st.caption(f"Tọa độ click: X={st.session_state['sam_point'][0]}, Y={st.session_state['sam_point'][1]}")
            with col_img2:
                if mask_img_arr is not None:
                    st.image(mask_img_arr, caption="Mặt nạ tổn thương (AI Segmentation)",
                             clamp=True, channels="L", use_container_width=True)
                else:
                    st.info("Chưa thực hiện phân đoạn.")
            with col_img3:
                gradcam_arr = (result.get("gradcam_image") if result else None)
                if isinstance(gradcam_arr, np.ndarray):
                    st.image(gradcam_arr, caption="Bản đồ nhiệt AI (Grad-CAM)", use_container_width=True)
                else:
                    st.info("Chưa tạo bản đồ nhiệt.")

            # -- Đa tổn thương --
            if result and mask_img_arr is not None:
                raw_mask_ml = result.get("segmentation_mask")
                if raw_mask_ml is not None:
                    lesions = detect_multiple_lesions(img_rgb, raw_mask_ml)
                    if len(lesions) > 1:
                        st.markdown("**Bản đồ định vị đa tổn thương (Phát hiện nhiều nốt):**")
                        overlay_img = draw_lesions_overlay(img_rgb, lesions)
                        st.image(overlay_img,
                                 caption=f"Phát hiện {len(lesions)} nốt tổn thương — Chọn nốt để phân tích VQA",
                                 use_container_width=True)
                        
                        lesion_options = ["Tất cả tổn thương (Ảnh gốc)"] + [f"Nốt {l['id']} (diện tích: {l['area']:.0f} px)" for l in lesions]
                        
                        # Lưu trữ kết quả ảnh gốc khi lần đầu phân tích ảnh đầy đủ
                        if st.session_state.get("full_image_result") is None:
                            st.session_state["full_image_result"] = result
                            st.session_state["active_analysis_path"] = st.session_state["saved_local_img_path"]
                            
                        selected_lesion = st.selectbox(
                            "Chọn nốt tổn thương để chẩn đoán và hỏi đáp VQA:",
                            lesion_options,
                            key="selected_lesion_idx"
                        )
                        
                        last_selected = st.session_state.get("last_selected_lesion_idx", "Tất cả tổn thương (Ảnh gốc)")
                        if selected_lesion != last_selected:
                            st.session_state["last_selected_lesion_idx"] = selected_lesion
                            if selected_lesion == "Tất cả tổn thương (Ảnh gốc)":
                                st.session_state["result"] = st.session_state["full_image_result"]
                                st.session_state["saved_local_img_path"] = st.session_state["active_analysis_path"]
                                st.session_state["messages"] = []
                                st.rerun()
                            else:
                                match = re.search(r"Nốt (\d+)", selected_lesion)
                                if match:
                                    lesion_id = int(match.group(1))
                                    les = next((l for l in lesions if l["id"] == lesion_id), None)
                                    if les:
                                        x, y, w, h = les["bbox"]
                                        pad = int(max(w, h) * 0.20)
                                        crop_x = max(0, x - pad)
                                        crop_y = max(0, y - pad)
                                        crop_w = min(orig_w - crop_x, w + 2*pad)
                                        crop_h = min(orig_h - crop_y, h + 2*pad)
                                        
                                        cropped_img = image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
                                        tmp_dir = tempfile.mkdtemp()
                                        tmp_path = os.path.join(tmp_dir, f"lesion_{lesion_id}.png")
                                        cropped_img.save(tmp_path)
                                        
                                        st.session_state["saved_local_img_path"] = tmp_path
                                        with st.spinner(f"Đang phân tích riêng biệt Nốt {lesion_id}..."):
                                            result_crop = get_pipeline(min_conf).run(
                                                tmp_path,
                                                return_mask=True,
                                                age=float(p_age) if p_name.strip() else None,
                                                gender=p_gender if p_gender != "--- Chọn Giới tính ---" else None,
                                                body_location=p_location if p_location != "--- Chọn Vị trí tổn thương ---" else None,
                                                lambda_val=lambda_w,
                                                malignant_threshold=mal_thresh
                                            )
                                            if st.session_state.get("dicom_meta") and "metrics" in result_crop:
                                                meta = st.session_state["dicom_meta"]
                                                if "pixel_spacing_x" in meta and "pixel_spacing_y" in meta:
                                                    result_crop["metrics"]["pixel_spacing_x"] = meta["pixel_spacing_x"]
                                                    result_crop["metrics"]["pixel_spacing_y"] = meta["pixel_spacing_y"]
                                            st.session_state["result"] = result_crop
                                            st.session_state["messages"] = []
                                            st.rerun()

            # -- Kết quả phân tích định lượng --
            if result:
                metrics  = result.get("metrics", {})
                cls      = result.get("classification") or {}
                status   = result.get("status", "ok")
                pred     = cls.get("prediction", "N/A")
                vi_name  = get_vietnamese_diagnosis(pred)
                conf_val = float(cls.get("confidence", 0.0))
                conf_pct = conf_val * 100

                if status == "triage":
                    triage_vi = TRIAGE_REASON_VI.get(
                        result.get("triage_reason", ""), result.get("triage_reason", "")
                    )
                    st.markdown(
                        f"<div class='triage-banner'>"
                        f"SAFETY GATE KÍCH HOẠT<br>"
                        f"Lý do: <i>{triage_vi}</i><br>"
                        f"Hệ thống VQA bị khóa. Vui lòng chụp lại ảnh hoặc chuyển ca cho bác sĩ."
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div class='success-banner'>"
                        f"Phân tích thành công — Độ tin cậy: {conf_pct:.1f}%"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                if status == "ok" and pred in MALIGNANT_CLASSES:
                    st.markdown(
                        f"<div class='clinical-danger'>"
                        f" <b>CẢNH BÁO LÂM SÀNG NGUY CƠ CAO TỔN THƯƠNG ÁC TÍNH:</b> Hệ thống AI phát hiện nguy cơ cao tổn thương ác tính: "
                        f"<b>{pred}</b> (<i>{vi_name}</i>) với độ tin cậy <b>{conf_pct:.1f}%</b>.<br>"
                        f"Khuyến nghị hội chẩn bác sĩ chuyên khoa da liễu và làm sinh thiết giải phẫu bệnh khẩn cấp."
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                if status == "ok" and pred in BENIGN_CLASSES:
                    probs_c = cls.get("probabilities", {})
                    if probs_c:
                        max_m_key = max(MALIGNANT_CLASSES, key=lambda k: probs_c.get(k, 0.0))
                        max_m_val = probs_c.get(max_m_key, 0.0)
                        if max_m_val >= mal_thresh:
                            vi_mal = get_vietnamese_diagnosis(max_m_key)
                            st.markdown(
                                f"<div class='clinical-warn'>"
                                f"️ <b>Cảnh báo Lâm sàng:</b> Dự đoán chính là <b>{pred}</b> ({vi_name}), "
                                f"nhưng mô hình phát hiện xác suất <b>{max_m_key}</b> "
                                f"(<i>{vi_mal}</i>) = <b>{max_m_val:.1%}</b>.<br>"
                                f"Đề nghị tham khảo bác sĩ da liễu để loại trừ khả năng ác tính."
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                img_type = result.get("preprocess", {}).get("image_type", "dermoscopy")
                img_lbl  = "Dermoscopy (chuyên dụng)" if img_type == "dermoscopy" else "Ảnh điện thoại (TTA)"
                st.caption(f"Loại ảnh phát hiện: {img_lbl} — Hệ thống áp dụng cấu hình phù hợp.")

                # ABCD metrics
                st.markdown("**Số liệu Phân tích Định lượng (ABCD)**")
                area_v  = float(metrics.get("area_ratio",        0.0))
                bord_v  = float(metrics.get("border_complexity", 0.0))
                asym_v  = float(metrics.get("asymmetry",         0.0))
                circ_v  = float(metrics.get("circularity",       0.0))

                def _card(lbl, val, sub, warn=False):
                    wc = "warn" if warn else ""
                    vc = "warn" if warn else ""
                    return (
                        f"<div class='abcd-card {wc}'>"
                        f"<div class='card-label'>{lbl}</div>"
                        f"<div class='card-value {vc}'>{val}</div>"
                        f"<div class='card-sublabel'>{sub}</div>"
                        f"</div>"
                    )

                pixel_spacing_x = metrics.get("pixel_spacing_x")
                pixel_spacing_y = metrics.get("pixel_spacing_y")
                phys_area = None
                if pixel_spacing_x is not None and pixel_spacing_y is not None:
                    phys_area = float(metrics.get("lesion_area", 0)) * pixel_spacing_x * pixel_spacing_y
                
                area_sublabel = f"Diện tích: {phys_area:.2f} mm²" if phys_area is not None else "Tỷ lệ diện tích"

                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                mc1.markdown(_card("Thời gian", st.session_state.get("analysis_time", "—"), "Điểm lấy mẫu"), unsafe_allow_html=True)
                mc2.markdown(_card("Area ratio",  f"{area_v:.4f}",  area_sublabel), unsafe_allow_html=True)
                mc3.markdown(_card("Border",      f"{bord_v:.4f}",  "Độ phức tạp bờ", bord_v > 5.0), unsafe_allow_html=True)
                mc4.markdown(_card("Asymmetry",   f"{asym_v:.4f}",  "Bất đối xứng",   asym_v > 0.7), unsafe_allow_html=True)
                mc5.markdown(_card("Circularity", f"{circ_v:.4f}",  "Độ tròn"), unsafe_allow_html=True)

                conf_cls = "conf-high" if conf_pct >= 70 else ("conf-mid" if conf_pct >= 50 else "conf-low")
                st.markdown(
                    f"<div class='diag-badge'>"
                    f"<span class='conf-label {conf_cls}'>Độ tin cậy: {conf_pct:.1f}%</span>"
                    f"<b>{pred}</b> — <i>{vi_name}</i>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Biểu đồ phân phối
                probs = cls.get("probabilities", {})
                if probs:
                    col_chart, col_radar = st.columns([1.6, 1])
                    with col_chart:
                        st.markdown("**Phân phối Xác suất Bệnh lý**")
                        render_probability_chart(
                            probs,
                            patient_name=p_name,
                            timestamp=st.session_state.get("analysis_time", ""),
                        )
                    with col_radar:
                        st.markdown("**Radar Chart ABCD**")
                        render_radar_chart(metrics)

            # ===========================================================
            # VQA CHAT (Bố cục cuộn dọc, ô nhập cố định dưới cùng)
            # ===========================================================
            st.divider()
            st.markdown("### Khu vực Tư vấn Lâm sàng (VQA)")

            if "Offline" in vqa_mode or "Nội bộ" in vqa_mode:
                st.markdown(
                    "<div class='clinical-warn' style='margin-bottom: 12px; font-size: 0.85rem; padding: 8px 12px; border-left-width: 3px;'>"
                    "️ <b>Lưu ý chế độ Offline:</b> Mô hình ngoại tuyến hoạt động không có kết nối cơ sở dữ liệu y văn RAG. "
                    "Câu trả lời dựa trên kho tri thức chuyên gia cục bộ tĩnh."
                    "</div>",
                    unsafe_allow_html=True
                )

            is_triage     = result is not None and result.get("status") == "triage"
            chat_disabled = (not bool(result)) or is_triage

            if is_triage:
                st.markdown(
                    "<div class='triage-banner'>"
                    "Khung chat VQA bị khóa do chất lượng ảnh không đạt chuẩn Safety Gate."
                    "</div>",
                    unsafe_allow_html=True,
                )

            # -- 1. Vùng Lịch sử hội thoại (Sử dụng làm placeholder ở trên) --
            chat_container = st.container(height=380, border=True)

            # -- 2. Gợi ý câu hỏi nhanh (Đặt ngay dưới container chat, trên ô nhập) --
            st.caption("Gợi ý câu hỏi lâm sàng nhanh (nhấn vào sẽ tự động gửi):")
            preset_cols = st.columns(len(PRESET_QUESTIONS))
            active_prompt: Optional[str] = None

            for i, (col, q) in enumerate(zip(preset_cols, PRESET_QUESTIONS)):
                if col.button(q, key=f"preset_q_{i}", use_container_width=True, disabled=chat_disabled):
                    active_prompt = q

            # -- 3. Ô Nhập liệu & Voice Input --
            voice_result = None
            if not chat_disabled:
                # Gọi voice component với key động để reset sau khi gửi tin
                _vkey = st.session_state.get("voice_key", 0)
                if VOICE_AVAILABLE and _voice_input_fc is not None:
                    voice_result = _voice_input_fc(
                        language="vi-VN",
                        key=f"vqa_voice_realtime_main_{_vkey}",
                    )

                # Ô nhập chính (Type Input) sử dụng key động để reset sau khi gửi và hỗ trợ Enter
                _ikey = st.session_state.get("input_key", 0)
                with st.form(key=f"vqa_typed_form_{_ikey}", border=False):
                    col_typed, col_btn = st.columns([12, 1], gap="small")
                    with col_typed:
                        user_typed = st.text_input(
                            "Nhập câu hỏi:",
                            placeholder="Nhập hoặc nói câu hỏi của bạn tại đây...",
                            key=f"chat_textarea_vqa_{_ikey}",
                            label_visibility="collapsed",
                        )
                    with col_btn:
                        send_clicked = st.form_submit_button(
                            "",
                            disabled=chat_disabled,
                            use_container_width=True,
                            type="primary",
                        )
                
                # Logic xác định câu hỏi: ưu tiên gõ trực tiếp, dự phòng qua voice_result nếu trễ React state
                if send_clicked:
                    if user_typed.strip():
                        active_prompt = user_typed.strip()
                    elif voice_result and isinstance(voice_result, dict) and voice_result.get("text"):
                        active_prompt = voice_result["text"].strip()
                elif voice_result and isinstance(voice_result, dict) and voice_result.get("submit"):
                    active_prompt = voice_result["text"].strip()

            # -- 4. Xử lý logic tin nhắn mới trước khi render --
            ai_stream_gen = None
            if active_prompt and result:
                st.session_state["messages"].append({"role": "user", "content": active_prompt})
                ai_stream_gen = generate_vqa_response_stream(
                    question=active_prompt,
                    result=result,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    history=st.session_state["messages"],
                )
                # Reset ô nhập và mic component: tạo các widget mới trống hoàn toàn
                st.session_state["input_key"]     = st.session_state.get("input_key", 0) + 1
                st.session_state["voice_key"]     = st.session_state.get("voice_key", 0) + 1
                st.session_state["chat_input_val"] = ""

            # -- 5. Vẽ toàn bộ hội thoại vào container (CHỈ GỌI MỘT LẦN DUY NHẤT) --
            with chat_container:
                msgs = st.session_state["messages"]
                if not msgs and not active_prompt:
                    st.markdown(
                        "<div style='text-align:center;color:#64748b;font-size:0.86rem;padding:24px;'>"
                        "Chưa có câu hỏi nào. Sử dụng gợi ý phía dưới hoặc nói/nhập câu hỏi."
                        "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    # Lặp vẽ tin cũ
                    for m in msgs:
                        with st.chat_message(m["role"]):
                            st.markdown(m["content"])
                    
                    # Nếu có generator stream của AI thì vẽ tiếp câu trả lời live và lưu trữ real-time vào session state
                    if ai_stream_gen is not None:
                        with st.chat_message("assistant"):
                            placeholder = st.empty()
                            full_response = ""
                            # Khởi tạo tin nhắn trống trong session state để lưu trữ tiến trình
                            st.session_state["messages"].append({"role": "assistant", "content": ""})
                            msg_idx = len(st.session_state["messages"]) - 1
                            
                            for chunk in ai_stream_gen:
                                full_response += chunk
                                placeholder.markdown(full_response + "▌")
                                # Cập nhật nội dung real-time vào session state để tránh mất dữ liệu khi bị voice component ngắt quãng
                                st.session_state["messages"][msg_idx]["content"] = full_response
                            
                            placeholder.markdown(full_response)
                        st.rerun()

            # -- Đồng bộ EHR + Xuất báo cáo --
            if result:
                st.divider()
                st.markdown("**Đồng bộ Bệnh án Điện tử (EHR) và Xuất báo cáo**")
                col_ehr, col_pdf = st.columns(2)

                with col_ehr:
                    if st.button(
                        "Xác nhận và Lưu hồ sơ lên Google Cloud",
                        type="secondary",
                        disabled=not allow_to_save,
                        key="save_ehr_main_btn",
                    ):
                        if not p_name.strip():
                            st.error("Vui lòng điền Họ tên bệnh nhân trước.")
                        elif not st.session_state["saved_local_img_path"]:
                            st.error("Không tìm thấy file ảnh phân tích.")
                        else:
                            # Chuẩn bị song song hóa luồng upload ảnh lên ImgBB để giảm tối đa độ trễ giao diện
                            upload_tasks = {}
                            temp_files = []
                            
                            # 1. Ảnh gốc
                            upload_tasks["orig"] = st.session_state["saved_local_img_path"]
                            
                            # 2. Ảnh mặt nạ
                            if mask_img_arr is not None:
                                tmp_mask_path = os.path.join(tempfile.gettempdir(), f"mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                                cv2.imwrite(tmp_mask_path, mask_img_arr)
                                upload_tasks["mask"] = tmp_mask_path
                                temp_files.append(tmp_mask_path)
                                
                            # 3. Ảnh Grad-CAM
                            gradcam_arr = result.get("gradcam_image")
                            if isinstance(gradcam_arr, np.ndarray):
                                tmp_gradcam_path = os.path.join(tempfile.gettempdir(), f"gradcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                                gradcam_bgr = cv2.cvtColor(gradcam_arr, cv2.COLOR_RGB2BGR)
                                cv2.imwrite(tmp_gradcam_path, gradcam_bgr)
                                upload_tasks["gradcam"] = tmp_gradcam_path
                                temp_files.append(tmp_gradcam_path)

                            from concurrent.futures import ThreadPoolExecutor
                            
                            urls = {}
                            with st.spinner("Đang tải toàn bộ dữ liệu hình ảnh lâm sàng lên Cloud (song song)..."):
                                with ThreadPoolExecutor(max_workers=3) as executor:
                                    futures = {
                                        key: executor.submit(upload_image_to_imgbb, path)
                                        for key, path in upload_tasks.items()
                                    }
                                    for key, fut in futures.items():
                                        try:
                                            urls[key] = fut.result()
                                        except Exception as e:
                                            print(f"Parallel upload failed for {key}:", e)
                                            urls[key] = None

                            # Dọn dẹp file tạm
                            for p in temp_files:
                                if os.path.exists(p):
                                    try:
                                        os.remove(p)
                                    except Exception:
                                        pass

                            pub_url = urls.get("orig")
                            mask_url = urls.get("mask")
                            gradcam_url = urls.get("gradcam")

                            if not pub_url:
                                st.error("Lỗi tải ảnh gốc lên ImgBB.")
                            else:

                                pat_info = {
                                    "name":     p_name.strip(),
                                    "id":       p_id.strip(),
                                    "age":      int(p_age),
                                    "gender":   p_gender,
                                    "hometown": p_hometown,
                                    "location": p_location,
                                }
                                m_r = result.get("metrics", {})
                                c_r = result.get("classification") or {}
                                v_data = {
                                    "timestamp_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
                                    "created_at":   st.session_state["analysis_time"],
                                    "image_url":    pub_url,
                                    "mask_url":     mask_url,
                                    "gradcam_url":  gradcam_url,
                                    "location":     p_location,
                                    "ai_extracted_metrics": {
                                        "status":            result.get("status"),
                                        "prediction":        c_r.get("prediction", "N/A"),
                                        "confidence":        float(c_r.get("confidence", 0.0)),
                                        "area_ratio":        float(m_r.get("area_ratio",        0.0)),
                                        "border_complexity": float(m_r.get("border_complexity", 0.0)),
                                        "asymmetry":         float(m_r.get("asymmetry",         0.0)),
                                        "circularity":       float(m_r.get("circularity",       0.0)),
                                        "pixel_spacing_x":   m_r.get("pixel_spacing_x"),
                                        "pixel_spacing_y":   m_r.get("pixel_spacing_y"),
                                        "physical_area_mm2": float(m_r.get("lesion_area", 0) * m_r.get("pixel_spacing_x", 0) * m_r.get("pixel_spacing_y", 0)) if m_r.get("pixel_spacing_x") and m_r.get("pixel_spacing_y") else None,
                                    },
                                    "vqa_conversations": list(st.session_state["messages"]),
                                }
                                with st.spinner("Đang đồng bộ vào Cloud Firestore..."):
                                    if save_medical_record_to_gcp(p_name, pat_info, v_data):
                                        st.success(f"Đồng bộ thành công hồ sơ '{p_name.upper()}'!")
                                        st.session_state["record_saved"] = True
                                        lp = st.session_state.get("saved_local_img_path")
                                        if lp and os.path.exists(lp):
                                            os.remove(lp)
                                            st.session_state["saved_local_img_path"] = None
                                        st.rerun()

                with col_pdf:
                    if st.button("Xuất báo cáo PDF", type="secondary", key="pdf_export_btn"):
                        pat_info_pdf = {
                            "name": p_name.strip() or "N/A",
                            "age":  str(p_age),
                            "gender":   p_gender,
                            "hometown": p_hometown,
                            "location": p_location,
                        }
                        m_r = result.get("metrics", {})
                        c_r = result.get("classification") or {}
                        v_pdf = {"ai_extracted_metrics": {
                            "prediction":        c_r.get("prediction", "N/A"),
                            "confidence":        float(c_r.get("confidence", 0.0)),
                            "area_ratio":        float(m_r.get("area_ratio",        0.0)),
                            "border_complexity": float(m_r.get("border_complexity", 0.0)),
                            "asymmetry":         float(m_r.get("asymmetry",         0.0)),
                            "circularity":       float(m_r.get("circularity",       0.0)),
                        }}
                        pdf_bytes = generate_pdf_report(pat_info_pdf, v_pdf)
                        if pdf_bytes:
                            fname = (
                                f"BaoCao_{(p_name or 'BenhNhan').replace(' ', '_')}"
                                f"_{datetime.now().strftime('%Y%m%d')}.pdf"
                            )
                            st.download_button(
                                "Tải xuống báo cáo PDF",
                                data=pdf_bytes,
                                file_name=fname,
                                mime="application/pdf",
                                key="pdf_dl_btn",
                            )

    # ==========================================================
    # TAB 2 — DOCTOR DASHBOARD
    # ==========================================================
    with tab_doctor:
        render_doctor_dashboard()


if __name__ == "__main__":
    main()