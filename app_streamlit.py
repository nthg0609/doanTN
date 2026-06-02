"""Streamlit Multi-modal Medical EHR Dashboard with Cloud Firestore & ImgBB."""

import json
import os
import tempfile
import base64
import requests
from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

import google.cloud.firestore as gcp_firestore
from google.oauth2 import service_account

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()

# API Key ImgBB trực tuyến
IMGBB_API_KEY = "159bc5d50210a5104a5c1b1018368f75"

# Đường dẫn file Log hệ thống (Dev-only)
LOG_FILE_PATH = Path("5_Results/system_logs.log")
LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Từ điển chẩn đoán y khoa phục vụ việc dịch nghĩa trên giao diện
DIAGNOSIS_DICTIONARY = {
    "AKIEC": "Dày sừng quang hóa / Tiền ung thư",
    "BCC": "Ung thư biểu mô tế bào đáy",
    "BKL": "Tổn thương sừng hóa lành tính",
    "DF": "U xơ da",
    "MEL": "U hắc tố ác tính",
    "NV": "Nốt ruồi bình thường",
    "VASC": "Tổn thương mạch máu"
}

def get_vietnamese_diagnosis(pred_label: str) -> str:
    """Chuyển đổi nhãn mô hình viết tắt sang tên bệnh Tiếng Việt tường minh"""
    return DIAGNOSIS_DICTIONARY.get(pred_label, "Bệnh lý da liễu khác")


# ==============================================================================
# MODULE FIRESTORE UTILS
# ==============================================================================
def write_dev_log(data: Dict[str, Any], action_type: str):
    """Ghi cấu trúc gói dữ liệu JSON thô vào file log cục bộ, ẩn hoàn toàn khỏi giao diện UI"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "action_time": timestamp,
        "action_type": action_type,
        "payload": data
    }
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

def upload_image_to_imgbb(local_image_path: str) -> Optional[str]:
    """Tải ảnh lên máy chủ đám mây ImgBB và trả về URL trực tuyến"""
    try:
        with open(local_image_path, "rb") as file:
            url = "https://api.imgbb.com/1/upload"
            payload = {"key": IMGBB_API_KEY, "image": base64.b64encode(file.read())}
            res = requests.post(url, payload)
            if res.status_code == 200:
                return res.json()["data"]["url"]
    except Exception as e:
        st.error(f"❌ Lỗi tải ảnh lên ImgBB: {e}")
    return None

def check_patient_exists(patient_name: str) -> bool:
    """Kiểm tra nhanh xem ID bệnh nhân đã tồn tại trên Firestore chưa"""
    if not patient_name.strip():
        return False
    try:
        cred_path = "gcp-credentials.json"
        if os.path.exists(cred_path):
            creds = service_account.Credentials.from_service_account_file(cred_path)
            db = gcp_firestore.Client(credentials=creds, project=creds.project_id, database="(default)")
            doc_id = "".join(patient_name.strip().upper().split())
            return db.collection("medical_records").document(doc_id).get().exists
    except Exception:
        pass
    return False

def save_medical_record_to_gcp(patient_name: str, patient_info: Dict[str, Any], visit_data: Dict[str, Any]) -> bool:
    """Lưu trữ hồ sơ: Dùng .update() cho tài liệu cũ và .set() cho tài liệu mới"""
    try:
        cred_path = "gcp-credentials.json"
        if os.path.exists(cred_path):
            creds = service_account.Credentials.from_service_account_file(cred_path)
            db = gcp_firestore.Client(credentials=creds, project=creds.project_id, database="(default)")
            
            doc_id = "".join(patient_name.strip().upper().split())
            doc_ref = db.collection("medical_records").document(doc_id)
            
            doc_snap = doc_ref.get()
            if doc_snap.exists:
                current_data = doc_snap.to_dict()
                existing_visits = current_data.get("visits", [])
                existing_visits.append(visit_data)
                
                doc_ref.update({
                    "patient_info": patient_info,
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "visits": existing_visits
                })
            else:
                doc_ref.set({
                    "patient_id": doc_id,
                    "patient_info": patient_info,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "visits": [visit_data]
                })
            
            write_dev_log({"patient_id": doc_id, "visit": visit_data}, action_type="SAVE_OR_UPDATE_RECORD")
            return True
        else:
            st.error("❌ Không tìm thấy file gcp-credentials.json trong thư mục gốc!")
            return False
    except Exception as e:
        st.error(f"❌ Lỗi ghi dữ liệu lên Cloud Firestore: {e}")
    return False

def fetch_all_medical_records() -> list[Dict[str, Any]]:
    """Tải toàn bộ danh sách bệnh nhân từ Firestore về"""
    records = []
    try:
        cred_path = "gcp-credentials.json"
        if os.path.exists(cred_path):
            creds = service_account.Credentials.from_service_account_file(cred_path)
            db = gcp_firestore.Client(credentials=creds, project=creds.project_id, database="(default)")
            docs = db.collection("medical_records").order_by("updated_at", direction=gcp_firestore.Query.DESCENDING).stream()
            for doc in docs:
                records.append(doc.to_dict())
    except Exception:
        pass
    return records


# ==============================================================================
# PIPELINE AND PREPROCESSING UTILS
# ==============================================================================
@st.cache_resource
def get_pipeline(min_conf: float):
    from pipeline import UnifiedDermatologyPipeline
    return UnifiedDermatologyPipeline(mode="both", safety_overrides={"min_class_confidence": float(min_conf)})

def _mask_to_image(mask: Optional[np.ndarray], target_shape) -> Optional[np.ndarray]:
    if mask is None: return None
    if mask.ndim != 2: mask = mask[:, :, 0]
    if mask.shape != target_shape:
        mask = cv2.resize(mask.astype(np.uint8), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8) * 255

def _build_system_prompt() -> str:
    return (
        "Bạn là Bác sĩ Da liễu AI. Đầu vào của bạn là số liệu JSON.\n"
        "- QUY TẮC 1: Đọc tên bệnh dựa trên TỪ ĐIỂN: AKIEC (Dày sừng quang hóa), BCC (Ung thư biểu mô tế bào đáy), BKL (Tổn thương sừng hóa lành tính), DF (U xơ da), MEL (U hắc tố ác tính), NV (Nốt ruồi bình thường), VASC (Tổn thương mạch máu).\n"
        "- QUY TẮC 2: Được phép phân tích triệu chứng lâm sàng dựa trên hình học biên, nhưng KHÔNG ĐƯỢC KÊ ĐƠN THUỐC.\n"
        "- QUY TẮC 3: Luôn nhắc nhở đi khám thực tế."
    )

def _fallback_response(question: str, result: Dict[str, Any]) -> str:
    cls = result.get("classification", {}) or {}
    return f"Bạn hỏi: \"{question}\". Mô hình dự đoán: {cls.get('prediction', 'N/A')} ({cls.get('confidence', 0.0):.3f})."

def llm_postprocess(question: str, result: Dict[str, Any], api_key: Optional[str], history: Optional[list[Dict[str, str]]] = None) -> str:
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if OpenAI is None or not api_key: return _fallback_response(question, result)
    client = OpenAI(api_key=api_key)
    probabilities = (result.get("classification", {}) or {}).get("probabilities", {})
    prob_list = [{"label": k, "probability": float(v)} for k, v in probabilities.items()] if isinstance(probabilities, dict) else []
    payload = json.dumps({"metrics": result.get("metrics", {}), "classification": {"prediction": (result.get("classification", {}) or {}).get("prediction"), "confidence": (result.get("classification", {}) or {}).get("confidence"), "probabilities": prob_list}}, ensure_ascii=False)
    
    history_messages = [{"role": msg["role"], "content": msg["content"]} for msg in (history or []) if msg.get("role") in ("user", "assistant") and msg.get("content")]
    if history_messages and history_messages[-1]["role"] == "user": history_messages = history_messages[:-1]
    try:
        resp = client.chat.completions.create(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.2, messages=[{"role": "system", "content": _build_system_prompt()}, *history_messages, {"role": "user", "content": f"Câu hỏi: {question}\nJSON:\n{payload}"}])
        return resp.choices[0].message.content
    except Exception: return _fallback_response(question, result)


# ==============================================================================
# GIAO DIỆN TRỰC QUAN CHO BÁC SĨ (DASHBOARD TIMELINE ĐA ẢNH)
# ==============================================================================
def render_doctor_dashboard():
    st.header("📂 Hệ thống Tra cứu Hồ sơ Bệnh án Điện tử (Đa mốc thời gian)")
    st.info("Hỗ trợ hiển thị lịch sử tiến triển lâm sàng đa phương thức (nhiều ảnh ở các thời điểm khác nhau) của bệnh nhân.")
    
    all_records = fetch_all_medical_records()
    if not all_records:
        st.warning("📭 Hiện tại kho lưu trữ đám mây chưa có dữ liệu bệnh án nào hợp lệ.")
        return
        
    patient_options = {}
    for r in all_records:
        p_info = r.get("patient_info", {})
        p_name = p_info.get("name", "Ẩn danh")
        p_age = p_info.get("age", "??")
        total_visits = len(r.get("visits", []))
        option_label = f"👤 BN: {p_name} ({p_age} tuổi) - [{total_visits} mốc ảnh bệnh án]"
        patient_options[option_label] = r

    selected_patient_key = st.selectbox("🔍 Chọn bệnh nhân cần tra cứu lịch sử bệnh lý:", list(patient_options.keys()))
    
    if selected_patient_key:
        record = patient_options[selected_patient_key]
        p_info = record.get("patient_info", {})
        visits = record.get("visits", [])
        
        st.markdown("---")
        st.subheader("👤 Thông tin hành chính bệnh nhân")
        cc1, cc2, cc3 = st.columns(3)
        cc1.markdown(f"**Họ và tên:** `{p_info.get('name', '').upper()}`")
        cc2.markdown(f"**Tuổi lâm sàng:** `{p_info.get('age', 'N/A')}`")
        cc3.markdown(f"**Địa chỉ thường trú:** `{p_info.get('hometown', 'N/A')}`")
        st.markdown("---")
        
        visits = sorted(visits, key=lambda x: x.get("timestamp_id", ""), reverse=True)
        
        st.subheader("📅 Biên niên sử hình ảnh & Chẩn đoán qua các thời kỳ")
        for idx, visit in enumerate(visits):
            v_time = visit.get("created_at", "N/A")
            ai_metrics = visit.get("ai_extracted_metrics", {})
            image_url = visit.get("image_url")
            conversations = visit.get("vqa_conversations", [])
            
            v_pred = ai_metrics.get("prediction", "N/A")
            v_vi_name = get_vietnamese_diagnosis(v_pred)
            
            with st.expander(f"📸 LẦN KHÁM THỨ {len(visits) - idx} - Ngày tiếp nhận: {v_time}", expanded=(idx == 0)):
                col_img, col_data = st.columns([1, 1.2])
                with col_img:
                    if image_url:
                        st.image(image_url, caption=f"Ảnh tổn thương da tại thời điểm: {v_time}", use_container_width=True)
                    else:
                        st.warning("Không có tệp ảnh cho lần khám này.")
                        
                with col_data:
                    st.markdown("#### 🩺 Kết quả phân tích AI")
                    st.metric("Nhãn dự đoán gốc", v_pred)
                    conf_val = ai_metrics.get("confidence", 0.0)
                    st.metric("Độ tin cậy thuật toán", f"{conf_val * 100:.2f}%" if conf_val <= 1.0 else f"{conf_val:.2f}%")
                    
                    st.markdown(f"**Giải nghĩa y khoa:** <span style='color:#1E88E5; font-weight:bold;'>{v_vi_name}</span>", unsafe_allow_html=True)
                    
                    st.markdown("**Chỉ số đo đạc hình học:**")
                    st.write(f"- Tỉ lệ diện tích (Area ratio): `{ai_metrics.get('area_ratio', 0.0):.4f}`")
                    st.write(f"- Độ phức tạp bờ (Border complexity): `{ai_metrics.get('border_complexity', 0.0):.4f}`")
                
                st.markdown("##### 💬 Nhật ký tư vấn y khoa (VQA Chat History)")
                if not conversations:
                    st.caption("Lần khám này không thực hiện hội thoại phụ.")
                else:
                    for msg in conversations:
                        if msg.get("role") == "user":
                            st.markdown(f"👉 **Hỏi:** {msg.get('content')}")
                        else:
                            st.markdown(f"🤖 **Đáp:** {msg.get('content')}")


# ==============================================================================
# MAIN APPLICATION INTERFACE
# ==============================================================================
def main():
    st.set_page_config(page_title="Dermatology VQA Chat", page_icon="🔬", layout="wide")
    st.title("🔬 Hệ thống Trợ lý Da liễu Đa phương thức & EHR Dashboard")

    tab_diagnosis, tab_doctor = st.tabs(["🔬 Thực hiện Chẩn đoán VQA", "📂 Màn hình Xem lại của Bác sĩ"])

    if "messages" not in st.session_state: st.session_state["messages"] = []
    if "result" not in st.session_state: st.session_state["result"] = None
    if "analysis_time" not in st.session_state: st.session_state["analysis_time"] = None
    if "saved_local_img_path" not in st.session_state: st.session_state["saved_local_img_path"] = None
    if "last_uploaded_file_name" not in st.session_state: st.session_state["last_uploaded_file_name"] = None
    
    if "form_patient_name" not in st.session_state: st.session_state["form_patient_name"] = ""
    if "form_patient_age" not in st.session_state: st.session_state["form_patient_age"] = 25
    if "form_patient_hometown" not in st.session_state: st.session_state["form_patient_hometown"] = ""

    # --- TAB 1: THỰC HIỆN CHẨN ĐOÁN VQA ---
    with tab_diagnosis:
        with st.sidebar:
            st.header("📋 THÔNG TIN BỆNH NHÂN")
            
            p_name = st.text_input("Họ và tên bệnh nhân:", key="form_patient_name", placeholder="Nguyễn Văn A")
            p_age = st.number_input("Tuổi:", min_value=0, max_value=120, key="form_patient_age")
            p_hometown = st.text_input("Quê quán / Địa chỉ:", key="form_patient_hometown", placeholder="Hà Nội")
            
            # Khởi tạo biến cho phép lưu mặc định là True (đối với bệnh nhân mới)
            allow_to_save = True 
            
            if p_name.strip():
                is_old_patient = check_patient_exists(p_name)
                if is_old_patient:
                    st.warning(f"⚠️ PHÁT HIỆN: Bệnh nhân '{p_name.upper()}' đã có hồ sơ lịch sử trên đám mây.")
                    
                    # 🔴 THÊM CỔNG XÁC NHẬN: Hỏi ý kiến bác sĩ có muốn cập nhật thêm mốc khám mới không
                    confirm_update = st.radio(
                        "Bác sĩ có muốn cập nhật thêm mốc khám/ảnh mới cho bệnh nhân này không?",
                        options=["Chưa chọn", "Có, ghi nhận thêm mốc khám mới", "Không, đây là một bệnh nhân khác trùng tên"],
                        index=0
                    )
                    
                    if confirm_update == "Có, ghi nhận thêm mốc khám mới":
                        allow_to_save = True
                        st.caption("🟢 Hợp lệ: Nút lưu đã được mở khoá.")
                    elif confirm_update == "Không, đây là một bệnh nhân khác trùng tên":
                        allow_to_save = False
                        st.error("🛑 Khóa: Vui lòng đổi lại tên hoặc thêm Mã số định danh vào ô Họ tên để tạo hồ sơ riêng biệt.")
                    else:
                        allow_to_save = False
                        st.info("💡 Vui lòng tích chọn xác nhận ở trên để mở khóa nút Lưu bệnh án.")
                else:
                    st.success("✨ HỒ SƠ MỚI: Bệnh nhân mới (Sẽ tạo tài khoản hồ sơ mới)")
            
            if st.button("🗑️ Xóa nhanh thông tin (Reset Form)"):
                st.session_state["form_patient_name"] = ""
                st.session_state["form_patient_age"] = 25
                st.session_state["form_patient_hometown"] = ""
                st.rerun()

            st.markdown("---")
            st.header("⚙️ CẤU HÌNH HỆ THỐNG")
            min_conf = st.slider("Safety gate threshold (tau_c)", 0.3, 0.95, 0.60, 0.01)

        uploaded = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])
        
        if uploaded is not None:
            if st.session_state["last_uploaded_file_name"] != uploaded.name:
                st.session_state["last_uploaded_file_name"] = uploaded.name
                st.session_state["result"] = None
                st.session_state["messages"] = []
                st.session_state["analysis_time"] = None
                st.session_state["saved_local_img_path"] = None

            image = Image.open(uploaded).convert("RGB")
            img_rgb = np.array(image)

            col1, col2 = st.columns(2)
            with col1: st.image(image, caption="Input image", use_container_width=True)

            if st.button("Run analysis", type="primary"):
                st.session_state["analysis_time"] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                with st.spinner("Running inference..."):
                    tmp_dir = tempfile.mkdtemp()
                    tmp_path = os.path.join(tmp_dir, "input.png")
                    image.save(tmp_path)
                    st.session_state["saved_local_img_path"] = tmp_path
                    
                    result = get_pipeline(min_conf).run(tmp_path, return_mask=True)
                    st.session_state["result"] = result
                    st.session_state["messages"] = []

            result = st.session_state.get("result")
            if result:
                mask = result.get("segmentation_mask")
                mask_img = _mask_to_image(mask, img_rgb.shape[:2])
                with col2:
                    if mask_img is not None:
                        st.image(mask_img, caption="Segmentation mask", use_container_width=True, clamp=True, channels="L")

                metrics = result.get("metrics", {})
                cls = result.get("classification", {}) or {}
                
                current_pred = cls.get("prediction", "N/A")
                current_vi_name = get_vietnamese_diagnosis(current_pred)
                conf_percent = float(cls.get("confidence", 0.0)) * 100
                
                st.subheader("📊 Số liệu phân tích định lượng")
                c1, c2, c3, c4 = st.columns(4)
                
                c1.metric("Thời gian tiếp nhận", st.session_state["analysis_time"])
                c2.metric("Area ratio", f"{metrics.get('area_ratio', 0.0):.4f}")
                c3.metric("Border complexity", f"{metrics.get('border_complexity', 0.0):.4f}")
                
                with c4:
                    st.markdown(
                        f"""
                        <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 5px solid #ff4b4b; height: 105px;">
                            <p style="margin: 0; font-size: 0.85rem; color: #6c757d; font-weight: bold;">ĐỘ TIN CẬY MẮC BỆNH ({current_pred})</p>
                            <h3 style="margin: 2px 0; color: #212529; font-size: 1.5rem;">{conf_percent:.1f}%</h3>
                            <p style="margin: 0; font-size: 0.8rem; color: #1e88e5; font-weight: 500; line-height: 1.2;">📌 {current_vi_name}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

            st.divider()
            st.subheader("💬 VQA Chat Space")
            for msg in st.session_state["messages"]:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])

            prompt = st.chat_input("Ask a question about the lesion", disabled=not bool(result))
            if prompt:
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.markdown(prompt)
                answer = llm_postprocess(prompt, result, os.getenv("OPENAI_API_KEY"), st.session_state["messages"])
                st.session_state["messages"].append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"): st.markdown(answer)

            if result:
                st.divider()
                st.subheader("💾 Đồng bộ Bệnh án điện tử")
                
                # Biến allow_to_save sẽ chủ động kiểm soát trạng thái bấm của nút Lưu dưới này
                if st.button("Xác nhận & Lưu toàn bộ hồ sơ lên Google Cloud", type="secondary", disabled=not allow_to_save):
                    if not p_name:
                        st.error("❌ Không thể lưu! Vui lòng điền Họ tên bệnh nhân trước.")
                    elif not st.session_state["saved_local_img_path"]:
                        st.error("❌ Không tìm thấy file ảnh phân tích.")
                    else:
                        timestamp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        with st.spinner("Đang đẩy file ảnh lên mây trực tuyến ImgBB..."):
                            public_img_url = upload_image_to_imgbb(st.session_state["saved_local_img_path"])
                        
                        if not public_img_url:
                            st.error("❌ Lỗi tải ảnh lên mây ImgBB.")
                        else:
                            patient_info = {
                                "name": p_name.strip(),
                                "age": int(p_age),
                                "hometown": p_hometown.strip() if p_hometown else "N/A"
                            }
                            
                            metrics = result.get("metrics", {})
                            cls = result.get("classification", {}) or {}
                            visit_data = {
                                "timestamp_id": timestamp_id,
                                "created_at": st.session_state["analysis_time"],
                                "image_url": public_img_url,
                                "ai_extracted_metrics": {
                                    "status": result.get("status"),
                                    "area_ratio": float(metrics.get('area_ratio', 0.0)),
                                    "border_complexity": float(metrics.get('border_complexity', 0.0)),
                                    "prediction": cls.get("prediction", "N/A"),
                                    "confidence": float(cls.get("confidence", 0.0))
                                },
                                "vqa_conversations": list(st.session_state["messages"])
                            }
                            
                            with st.spinner("Đang đồng bộ vào Cloud Firestore (Multi-timeline)..."):
                                if save_medical_record_to_gcp(p_name, patient_info, visit_data):
                                    st.success(f"🎉 Đã đồng bộ thành công mốc ảnh bệnh án mới vào hồ sơ của bệnh nhân '{p_name.upper()}'!")
                                    if os.path.exists(st.session_state["saved_local_img_path"]):
                                        os.remove(st.session_state["saved_local_img_path"])
                                        st.session_state["saved_local_img_path"] = None

    # --- TAB 2: MÀN HÌNH XEM LẠI CỦA BÁC SĨ ---
    with tab_doctor:
        render_doctor_dashboard()


if __name__ == "__main__":
    main()