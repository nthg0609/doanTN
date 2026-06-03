"""Streamlit Multi-modal Medical EHR Dashboard — Hybrid VQA v2.0
Nâng cấp: Fusion Prompt, Safety Gate UI, LLM Logging, Plotly Charts, ABCD Metrics.
"""

import json
import os
import io
import tempfile
import base64
import requests
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import plotly.graph_objects as go

import google.cloud.firestore as gcp_firestore
from google.oauth2 import service_account

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()

# ── Hằng số hệ thống ──────────────────────────────────────────────────────────
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

TRIAGE_REASON_VI: Dict[str, str] = {
    "empty_or_low_confidence_mask":  "Không phát hiện được vùng tổn thương rõ ràng trong ảnh",
    "area_ratio_out_of_bounds":      "Tỉ lệ diện tích tổn thương nằm ngoài ngưỡng hợp lệ",
    "border_complexity_out_of_bounds": "Độ phức tạp bờ quá cao — có thể do nhiễu ảnh",
    "classification_unavailable":   "Mô hình phân loại không khả dụng",
    "low_classification_confidence": "Độ tin cậy phân loại thấp hơn ngưỡng an toàn (τ_c)",
    "image_load_failed":             "Không thể đọc file ảnh",
}

MEDICAL_DISCLAIMER = (
    "⚠️ **TUYÊN BỐ MIỄN TRỪ TRÁCH NHIỆM:** Hệ thống này chỉ cấu thành một "
    "công cụ hỗ trợ sàng lọc sơ bộ bằng công nghệ AI, hoàn toàn **không thay thế** "
    "cho các chẩn đoán y khoa chuyên môn của bác sĩ da liễu."
)

# ── Tiện ích ──────────────────────────────────────────────────────────────────
def get_vietnamese_diagnosis(pred_label: str) -> str:
    return DIAGNOSIS_DICTIONARY.get(pred_label, "Bệnh lý da liễu khác")


# ==============================================================================
# MODULE LOG HỆ THỐNG (Dev-only, ẩn hoàn toàn khỏi UI)
# ==============================================================================
def write_dev_log(data: Dict[str, Any], action_type: str) -> None:
    """Ghi gói JSON thô (bao gồm LLM Prompt & Response) vào file log cục bộ."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "action_time": timestamp,
        "action_type": action_type,
        "payload": data,
    }
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


# ==============================================================================
# MODULE CLOUD STORAGE (ImgBB + Firestore)
# ==============================================================================
def upload_image_to_imgbb(local_image_path: str) -> Optional[str]:
    """Tải ảnh lên ImgBB, trả về URL vĩnh viễn."""
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


def save_medical_record_to_gcp(
    patient_name: str,
    patient_info: Dict[str, Any],
    visit_data: Dict[str, Any],
) -> bool:
    """Lưu hồ sơ EHR lên Firestore: .update() cho bệnh nhân cũ, .set() cho mới."""
    try:
        cred_path = "gcp-credentials.json"
        if os.path.exists(cred_path):
            creds = service_account.Credentials.from_service_account_file(cred_path)
            db = gcp_firestore.Client(credentials=creds, project=creds.project_id, database="(default)")
            doc_id = "".join(patient_name.strip().upper().split())
            doc_ref = db.collection("medical_records").document(doc_id)
            doc_snap = doc_ref.get()
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if doc_snap.exists:
                existing_visits = doc_snap.to_dict().get("visits", [])
                existing_visits.append(visit_data)
                doc_ref.update({
                    "patient_info": patient_info,
                    "updated_at": now_str,
                    "visits": existing_visits,
                })
            else:
                doc_ref.set({
                    "patient_id": doc_id,
                    "patient_info": patient_info,
                    "created_at": now_str,
                    "updated_at": now_str,
                    "visits": [visit_data],
                })
            write_dev_log({"patient_id": doc_id, "visit": visit_data}, action_type="SAVE_OR_UPDATE_RECORD")
            return True
        else:
            st.error("❌ Không tìm thấy file gcp-credentials.json!")
            return False
    except Exception as e:
        st.error(f"❌ Lỗi ghi dữ liệu lên Cloud Firestore: {e}")
    return False


def fetch_all_medical_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    try:
        cred_path = "gcp-credentials.json"
        if os.path.exists(cred_path):
            creds = service_account.Credentials.from_service_account_file(cred_path)
            db = gcp_firestore.Client(credentials=creds, project=creds.project_id, database="(default)")
            docs = db.collection("medical_records").order_by(
                "updated_at", direction=gcp_firestore.Query.DESCENDING
            ).stream()
            for doc in docs:
                records.append(doc.to_dict())
    except Exception:
        pass
    return records


# ==============================================================================
# MODULE PIPELINE
# ==============================================================================
@st.cache_resource
def get_pipeline(min_conf: float):
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
# MODULE VQA ENGINE — Fusion Prompt Architecture
# ==============================================================================
def _build_fusion_system_prompt(cv_context: Dict[str, Any]) -> str:
    """
    Xây dựng System Prompt theo kiến trúc 3 vùng:
      [IDENTITY] → [CV_CONTEXT] → [GUARDRAIL_RULES]
    CV context được nhúng cứng (hard-coded) vào System, không phải User message.
    """
    pred    = cv_context.get("prediction", "N/A")
    vi_name = get_vietnamese_diagnosis(pred)
    conf    = float(cv_context.get("confidence", 0.0))
    metrics = cv_context.get("metrics", {})
    probs   = cv_context.get("probabilities", {})

    prob_lines = "\n".join(
        f"    • {k} ({get_vietnamese_diagnosis(k)}): {v:.4f}"
        for k, v in sorted(probs.items(), key=lambda x: -x[1])
    )

    system_prompt = f"""[IDENTITY]
Bạn là Trợ lý Da liễu AI — một hệ thống hỗ trợ sàng lọc y tế tích hợp mô hình Thị giác Máy tính (CV) và Mô hình Ngôn ngữ Lớn (LLM). Bạn tư vấn dựa HOÀN TOÀN trên dữ liệu CV được cung cấp bên dưới, không được bịa đặt số liệu.

[CV_CONTEXT — DỮ LIỆU CHẮC CHẮN TỪ MÔ HÌNH CV]
Kết quả phân loại mô hình chuyên biệt EfficientNet-B1 + CBAM Attention:
  • Nhãn dự đoán cao nhất : {pred} → {vi_name}
  • Độ tin cậy            : {conf:.4f} ({conf*100:.1f}%)

Phân phối xác suất đầy đủ 7 nhãn bệnh lý (ISIC):
{prob_lines}

Chỉ số hình học tổn thương (DeepLabV3+ Segmentation):
  • Tỉ lệ diện tích (Area ratio)      : {metrics.get('area_ratio', 0.0):.4f}
  • Độ phức tạp bờ (Border complexity): {metrics.get('border_complexity', 0.0):.4f}
  • Bất đối xứng (Asymmetry score)    : {metrics.get('asymmetry', 0.0):.4f}  [0=đối xứng, 1=bất đối xứng]
  • Độ tròn (Circularity)             : {metrics.get('circularity', 0.0):.4f}  [0=không tròn, 1=tròn đều]

[GUARDRAIL_RULES — QUY TẮC BẮT BUỘC TUYỆT ĐỐI]

ĐƯỢC PHÉP:
  ✅ Giải thích cơ chế bệnh sinh, mô tả triệu chứng lâm sàng phổ biến của nhãn bệnh trên.
  ✅ Hướng dẫn chăm sóc da không dùng thuốc (làm sạch, tránh nắng, dưỡng ẩm, bảo vệ).
  ✅ Phân nhóm thuốc tổng quát (ví dụ: "nhóm kháng nấm bôi tại chỗ", "nhóm corticosteroid bôi ngoài").
  ✅ Giải thích ý nghĩa các chỉ số hình học CV ở trên khi người dùng hỏi.
  ✅ Luôn khuyên người dùng đến gặp bác sĩ da liễu chuyên khoa để được chẩn đoán chính xác.

QUY TẮC GIẢI THÍCH TIẾN TRIỂN & HẬU QUẢ BỆNH (CLINICAL PATHOLOGY PROGRESSION RULES):
  Khi người dùng hỏi về tiến triển, biến chứng hoặc hậu quả của bệnh, hãy dựa vào NHÃN DỰ ĐOÁN CAO NHẤT ({pred}) từ mô hình CV để phản hồi chính xác:
  - Nếu nhãn dự đoán là LÀNH TÍNH (BKL, NV, VASC, DF):
    • Phải khẳng định rõ đây là tổn thương bản chất LÀNH TÍNH, không có khả năng tự biến đổi hoặc phát triển thành ung thư.
    • Làm rõ các ảnh hưởng chỉ dừng lại ở mặt thẩm mỹ, kích ứng tại chỗ (như cọ xát quần áo, ngứa nhẹ), hoặc tâm lý lo lắng.
    • Nhấn mạnh nguy cơ lớn nhất là "nhầm lẫn" (misdiagnosis) — tự chẩn đoán nhầm một tổn thương ác tính thực sự thành nốt lành tính, dẫn đến chủ quan không đi khám.
  - Nếu nhãn dự đoán là TIỀN ÁC TÍNH hoặc ÁC TÍNH (AKIEC, BCC, MEL):
    • Giải thích thận trọng, khách quan về nguy cơ tiến triển nếu không can thiệp (ví dụ: AKIEC có thể tiến triển thành ung thư biểu mô tế bào vảy xâm lấn; BCC xâm lấn phá hủy mô tại chỗ; MEL có thể di căn xa).
    • Tránh dùng từ ngữ gây hoảng loạn cực đoan cho bệnh nhân, nhưng phải nhấn mạnh tầm quan trọng của việc đi khám bác sĩ, sinh thiết và điều trị y khoa kịp thời để kiểm soát bệnh.

TUYỆT ĐỐI CẤM — MEDICATION GUARDRAIL:
  🚫 KHÔNG được nêu tên bất kỳ biệt dược cụ thể nào (Amoxicillin, Tretinoin, Mometasone, Hydrocortisone, Clotrimazole, Acyclovir, v.v.)
  🚫 KHÔNG được nêu liều lượng (mg, ml, %, IU, lần/ngày, tuần/lần).
  🚫 KHÔNG được nêu thời gian dùng thuốc (7 ngày, 2 tuần, 1 tháng).
  🚫 KHÔNG được đề xuất thuốc kể cả khi người dùng đặt câu hỏi dạng "ví dụ", "giả sử", "trường hợp giả định".
  🚫 KHÔNG được xác nhận hay phủ nhận một loại thuốc cụ thể người dùng tự đề xuất.
  → Nếu bị hỏi về tên thuốc cụ thể: Lịch sự từ chối, giải thích lý do y đức, và hướng dẫn gặp bác sĩ.

ĐỊNH DẠNG PHẢN HỒI:
  - Ngôn ngữ: Tiếng Việt, rõ ràng, chuyên nghiệp, dễ hiểu với bệnh nhân.
  - Độ dài: Tối đa 400 từ mỗi câu trả lời. Nếu câu hỏi phức tạp, chia thành đề mục ngắn.
  - Kết thúc mỗi câu trả lời bằng nhắc nhở đến khám bác sĩ da liễu.
"""
    return system_prompt


def _fallback_response(question: str, result: Dict[str, Any]) -> str:
    cls = result.get("classification") or {}
    return (
        f'Bạn hỏi: "{question}". '
        f'Mô hình dự đoán: {cls.get("prediction", "N/A")} '
        f'({float(cls.get("confidence", 0.0)):.3f}). '
        "Vui lòng tham khảo ý kiến bác sĩ da liễu."
    )


def generate_vqa_response(
    question: str,
    result: Dict[str, Any],
    api_key: Optional[str],
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Luồng Fusion VQA chính:
      1. Safety Gate Check — Chặn LLM nếu status == triage
      2. Build CV context dict
      3. Build System Prompt với CV context nhúng cứng
      4. Build message list: [system, *history_sans_last_user, user_question]
      5. Gọi LLM API
      6. Ghi log đầy đủ System Prompt + User message + Raw Response
      7. Return response
    """
    # ── BƯỚC 1: Safety Gate Check ─────────────────────────────────────────────
    if result.get("status") == "triage":
        triage_reason_raw = result.get("triage_reason", "unknown")
        triage_reason_vi  = TRIAGE_REASON_VI.get(triage_reason_raw, triage_reason_raw)
        return (
            f"🚨 **Safety Gate đã kích hoạt** — Hệ thống không thể đưa ra tư vấn vì:\n\n"
            f"> _{triage_reason_vi}_\n\n"
            "Vui lòng chụp lại ảnh tổn thương với ánh sáng tốt hơn, hoặc liên hệ trực tiếp "
            "với bác sĩ da liễu để được thăm khám chính xác."
        )

    # ── BƯỚC 2: Build CV context ───────────────────────────────────────────────
    cls_data = result.get("classification") or {}
    cv_context = {
        "prediction":    cls_data.get("prediction", "N/A"),
        "confidence":    float(cls_data.get("confidence", 0.0)),
        "probabilities": cls_data.get("probabilities", {}),
        "metrics":       result.get("metrics", {}),
    }

    # ── BƯỚC 3: Build System Prompt ───────────────────────────────────────────
    system_prompt = _build_fusion_system_prompt(cv_context)

    # ── BƯỚC 4: Build message list ────────────────────────────────────────────
    history = history or []
    # Loại bỏ tin nhắn user vừa append (sẽ được thêm riêng bên dưới để tránh duplicate)
    valid_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history
        if msg.get("role") in ("user", "assistant") and msg.get("content")
    ]
    # Bỏ tin nhắn user cuối cùng vì nó là câu hỏi hiện tại, sẽ truyền riêng
    if valid_history and valid_history[-1]["role"] == "user":
        valid_history = valid_history[:-1]

    messages = [
        {"role": "system", "content": system_prompt},
        *valid_history,
        {"role": "user",   "content": question},
    ]

    # ── BƯỚC 5: Gọi LLM ───────────────────────────────────────────────────────
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if OpenAI is None or not api_key:
        return _fallback_response(question, result)

    raw_response = ""
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            max_tokens=800,
            messages=messages,
        )
        raw_response = resp.choices[0].message.content or ""
    except Exception as exc:
        write_dev_log({"error": str(exc), "question": question}, action_type="LLM_ERROR")
        return _fallback_response(question, result)

    # ── BƯỚC 6: Ghi log đầy đủ ───────────────────────────────────────────────
    write_dev_log(
        {
            "system_prompt":  system_prompt,
            "user_message":   question,
            "chat_history_len": len(valid_history),
            "raw_response":   raw_response,
            "cv_context":     cv_context,
        },
        action_type="LLM_VQA_EXCHANGE",
    )

    return raw_response


# ==============================================================================
# MODULE VISUALIZATION
# ==============================================================================
def render_probability_chart(probabilities: Dict[str, float], patient_name: str = "", timestamp: str = "") -> None:
    """
    Vẽ Plotly Horizontal Bar Chart cho phân phối xác suất 7 nhãn.
    Đồng thời lưu ảnh PNG vào CHART_SAVE_DIR để dùng cho báo cáo.
    """
    if not probabilities:
        return

    labels_vi = [f"{k}<br><sub>{get_vietnamese_diagnosis(k)}</sub>" for k in probabilities.keys()]
    values    = list(probabilities.values())
    keys      = list(probabilities.keys())

    # Màu gradient: xanh lá → vàng → đỏ theo giá trị xác suất
    colors = []
    for v in values:
        r = int(255 * min(1.0, v * 2))
        g = int(255 * min(1.0, 2.0 - v * 2))
        colors.append(f"rgba({r},{g},60,0.85)")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=labels_vi,
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.3)", width=1)),
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(size=11, color="white"),
        hovertemplate="<b>%{y}</b><br>Xác suất: %{x:.4f}<extra></extra>",
    ))

    top_key = max(probabilities, key=probabilities.get)
    top_val = probabilities[top_key]

    fig.update_layout(
        title=dict(
            text=f"📊 Phân phối Xác suất Bệnh lý<br><sup>Dự đoán cao nhất: <b>{top_key}</b> — {top_val:.2%}</sup>",
            font=dict(size=14, color="white"),
            x=0.5,
        ),
        xaxis=dict(
            title="Xác suất",
            range=[0, 1.1],
            tickformat=".0%",
            gridcolor="rgba(255,255,255,0.1)",
            color="white",
        ),
        yaxis=dict(
            autorange="reversed",
            color="white",
            tickfont=dict(size=10),
        ),
        plot_bgcolor="rgba(17,25,40,0.9)",
        paper_bgcolor="rgba(17,25,40,0.9)",
        font=dict(color="white"),
        margin=dict(l=10, r=60, t=80, b=40),
        height=320,
    )

    st.plotly_chart(fig, width='stretch')

    # ── Lưu PNG báo cáo ───────────────────────────────────────────────────────
    try:
        ts_str = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize timestamp and patient name to be safe for filenames (no slashes, colons, or invalid chars)
        ts_clean = ts_str.replace("/", "").replace("\\", "").replace(":", "").replace(" ", "_")
        safe_name = "".join(c for c in patient_name.strip().upper() if c.isalnum() or c in ("-", "_")) or "UNKNOWN"
        png_filename = CHART_SAVE_DIR / f"prob_chart_{safe_name}_{ts_clean}.png"

        # Dùng kaleido (nếu có) hoặc fallback sang matplotlib
        try:
            img_bytes = fig.to_image(format="png", width=800, height=400, scale=2)
            with open(png_filename, "wb") as f:
                f.write(img_bytes)
        except Exception:
            # Fallback: matplotlib bar chart đơn giản
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig_mpl, ax = plt.subplots(figsize=(10, 5), facecolor="#111928")
            ax.set_facecolor("#111928")
            sorted_items = sorted(probabilities.items(), key=lambda x: -x[1])
            k_labels = [f"{k}\n{get_vietnamese_diagnosis(k)}" for k, _ in sorted_items]
            k_values = [v for _, v in sorted_items]
            bar_colors = [
                (min(1.0, v * 2), min(1.0, 2.0 - v * 2), 0.2)
                for v in k_values
            ]
            bars = ax.barh(k_labels, k_values, color=bar_colors)
            ax.set_xlim(0, 1.1)
            ax.set_xlabel("Xác suất", color="white")
            ax.set_title(f"Phân phối Xác suất — {safe_name} — {ts_str}", color="white")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("white")
            for bar, val in zip(bars, k_values):
                ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", color="white", fontsize=9)
            plt.tight_layout()
            plt.savefig(png_filename, dpi=150, bbox_inches="tight",
                        facecolor="#111928")
            plt.close(fig_mpl)

        st.caption(f"📁 Đã lưu biểu đồ báo cáo: `{png_filename}`")
    except Exception as save_err:
        st.caption(f"_(Không lưu được PNG: {save_err})_")


def render_radar_chart(metrics: Dict[str, Any]) -> None:
    """Vẽ Radar Chart 4 chỉ số hình học ABCD chuẩn hóa."""
    area        = float(metrics.get("area_ratio", 0.0))
    border_raw  = float(metrics.get("border_complexity", 0.0))
    asymmetry   = float(metrics.get("asymmetry", 0.0))
    circularity = float(metrics.get("circularity", 0.0))

    # Chuẩn hóa border_complexity về [0,1] (max ref = 8.0 theo safety gate)
    border_norm = min(border_raw / 8.0, 1.0)
    # Chuẩn hóa area_ratio về [0,1] (max ref = 0.75)
    area_norm   = min(area / 0.75, 1.0)

    categories  = ["Diện tích<br>(Area ratio)", "Độ phức tạp bờ<br>(Border)", "Bất đối xứng<br>(Asymmetry)", "Độ tròn<br>(Circularity)"]
    values_norm = [area_norm, border_norm, asymmetry, circularity]
    values_norm_closed = values_norm + [values_norm[0]]
    categories_closed  = categories  + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_norm_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(99,179,237,0.2)",
        line=dict(color="rgba(99,179,237,0.9)", width=2),
        name="Chỉ số ABCD",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(17,25,40,0.8)",
            radialaxis=dict(visible=True, range=[0, 1], color="white", gridcolor="rgba(255,255,255,0.15)"),
            angularaxis=dict(color="white", gridcolor="rgba(255,255,255,0.15)"),
        ),
        paper_bgcolor="rgba(17,25,40,0.9)",
        font=dict(color="white", size=10),
        showlegend=False,
        margin=dict(l=40, r=40, t=30, b=30),
        height=280,
    )
    st.plotly_chart(fig, width='stretch')


# ==============================================================================
# GIAO DIỆN DASHBOARD BÁC SĨ (EHR Timeline)
# ==============================================================================
def render_doctor_dashboard() -> None:
    st.header("📂 Hệ thống Tra cứu Hồ sơ Bệnh án Điện tử (Đa mốc thời gian)")
    st.info("Hỗ trợ hiển thị lịch sử tiến triển lâm sàng đa phương thức (nhiều ảnh ở các thời điểm khác nhau).")

    all_records = fetch_all_medical_records()
    if not all_records:
        st.warning("📭 Hiện tại kho lưu trữ đám mây chưa có dữ liệu bệnh án nào hợp lệ.")
        return

    patient_options: Dict[str, Dict[str, Any]] = {}
    for r in all_records:
        p_info  = r.get("patient_info", {})
        p_name  = p_info.get("name", "Ẩn danh")
        p_age   = p_info.get("age", "??")
        total   = len(r.get("visits", []))
        label   = f"👤 BN: {p_name} ({p_age} tuổi) — [{total} mốc ảnh bệnh án]"
        patient_options[label] = r

    selected_key = st.selectbox("🔍 Chọn bệnh nhân cần tra cứu:", list(patient_options.keys()))
    if not selected_key:
        return

    record  = patient_options[selected_key]
    p_info  = record.get("patient_info", {})
    visits  = sorted(record.get("visits", []), key=lambda x: x.get("timestamp_id", ""), reverse=True)

    st.markdown("---")
    st.subheader("👤 Thông tin hành chính bệnh nhân")
    cc1, cc2, cc3 = st.columns(3)
    cc1.markdown(f"**Họ và tên:** `{p_info.get('name', '').upper()}`")
    cc2.markdown(f"**Tuổi lâm sàng:** `{p_info.get('age', 'N/A')}`")
    cc3.markdown(f"**Địa chỉ thường trú:** `{p_info.get('hometown', 'N/A')}`")
    st.markdown("---")
    st.subheader("📅 Biên niên sử hình ảnh & Chẩn đoán qua các thời kỳ")

    for idx, visit in enumerate(visits):
        v_time      = visit.get("created_at", "N/A")
        ai_metrics  = visit.get("ai_extracted_metrics", {})
        image_url   = visit.get("image_url")
        conversations = visit.get("vqa_conversations", [])
        v_pred      = ai_metrics.get("prediction", "N/A")
        v_vi_name   = get_vietnamese_diagnosis(v_pred)

        with st.expander(f"📸 LẦN KHÁM THỨ {len(visits) - idx} — Ngày: {v_time}", expanded=(idx == 0)):
            col_img, col_data = st.columns([1, 1.2])
            with col_img:
                if image_url:
                    st.image(image_url, caption=f"Ảnh tổn thương: {v_time}", width='stretch')
                else:
                    st.warning("Không có tệp ảnh cho lần khám này.")
            with col_data:
                st.markdown("#### 🩺 Kết quả phân tích AI")
                conf_val = ai_metrics.get("confidence", 0.0)
                m1, m2 = st.columns(2)
                m1.metric("Nhãn dự đoán", v_pred)
                m2.metric("Độ tin cậy", f"{float(conf_val) * 100:.1f}%")
                st.markdown(
                    f"**Giải nghĩa:** <span style='color:#63b3ed;font-weight:bold;'>{v_vi_name}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("**Chỉ số hình học:**")
                st.write(f"- Area ratio: `{ai_metrics.get('area_ratio', 0.0):.4f}`")
                st.write(f"- Border complexity: `{ai_metrics.get('border_complexity', 0.0):.4f}`")
                st.write(f"- Asymmetry: `{ai_metrics.get('asymmetry', 0.0):.4f}`")
                st.write(f"- Circularity: `{ai_metrics.get('circularity', 0.0):.4f}`")

            st.markdown("##### 💬 Nhật ký tư vấn y khoa (VQA)")
            if not conversations:
                st.caption("Lần khám này không thực hiện hội thoại phụ.")
            else:
                chat_html = ""
                for msg in conversations:
                    role    = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        chat_html += (
                            f"<div style='background:rgba(99,179,237,0.15);border-left:3px solid #63b3ed;"
                            f"padding:6px 10px;margin:4px 0;border-radius:4px;overflow-wrap:break-word;'>"
                            f"<b>👤 Hỏi:</b> {content}</div>"
                        )
                    else:
                        chat_html += (
                            f"<div style='background:rgba(72,187,120,0.12);border-left:3px solid #48bb78;"
                            f"padding:6px 10px;margin:4px 0;border-radius:4px;overflow-wrap:break-word;"
                            f"max-height:200px;overflow-y:auto;'>"
                            f"<b>🤖 Đáp:</b> {content}</div>"
                        )
                st.markdown(chat_html, unsafe_allow_html=True)


# ==============================================================================
# MAIN APPLICATION INTERFACE
# ==============================================================================
def _inject_custom_css() -> None:
    """Inject CSS kiểm soát overflow & styling cho chat VQA."""
    st.markdown(
        """
        <style>
        /* ── Disclaimer banner ── */
        .medical-disclaimer {
            background: linear-gradient(135deg, rgba(245,101,101,0.15), rgba(236,153,75,0.15));
            border: 1px solid rgba(245,101,101,0.4);
            border-radius: 8px;
            padding: 10px 16px;
            margin-bottom: 12px;
            font-size: 0.82rem;
            color: #fbd38d;
        }
        /* ── Chat message overflow ── */
        [data-testid="stChatMessageContent"] p,
        [data-testid="stChatMessageContent"] li {
            word-break: break-word;
            overflow-wrap: break-word;
        }
        [data-testid="stChatMessageContent"] {
            max-height: 420px;
            overflow-y: auto;
        }
        /* ── Metric card ── */
        [data-testid="stMetricValue"] {
            font-size: 1.1rem !important;
        }
        /* ── Triage alert ── */
        .triage-banner {
            background: rgba(229,62,62,0.18);
            border: 2px solid #e53e3e;
            border-radius: 10px;
            padding: 14px 18px;
            text-align: center;
            font-size: 1rem;
            color: #feb2b2;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Dermatology VQA — Hybrid AI",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_custom_css()

    # ── Fixed Disclaimer ──────────────────────────────────────────────────────
    st.markdown(
        f"<div class='medical-disclaimer'>{MEDICAL_DISCLAIMER}</div>",
        unsafe_allow_html=True,
    )

    st.title("🔬 Hệ thống Trợ lý Da liễu Đa phương thức & EHR Dashboard")

    tab_diagnosis, tab_doctor = st.tabs(["🔬 Thực hiện Chẩn đoán VQA", "📂 Màn hình Xem lại của Bác sĩ"])

    # ── Session State Init ────────────────────────────────────────────────────
    for key, default in [
        ("messages", []),
        ("result", None),
        ("analysis_time", None),
        ("saved_local_img_path", None),
        ("last_uploaded_file_name", None),
        ("form_patient_name", ""),
        ("form_patient_age", 25),
        ("form_patient_hometown", ""),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── TAB 1: CHẨN ĐOÁN VQA ─────────────────────────────────────────────────
    with tab_diagnosis:
        with st.sidebar:
            st.header("📋 THÔNG TIN BỆNH NHÂN")
            p_name     = st.text_input("Họ và tên bệnh nhân:", key="form_patient_name", placeholder="Nguyễn Văn A")
            p_age      = st.number_input("Tuổi:", min_value=0, max_value=120, key="form_patient_age")
            p_hometown = st.text_input("Quê quán / Địa chỉ:", key="form_patient_hometown", placeholder="Hà Nội")

            allow_to_save = True
            if p_name.strip():
                is_old_patient = check_patient_exists(p_name)
                if is_old_patient:
                    st.warning(f"⚠️ PHÁT HIỆN: Bệnh nhân '{p_name.upper()}' đã có hồ sơ lịch sử.")
                    confirm_update = st.radio(
                        "Bác sĩ có muốn cập nhật thêm mốc khám mới không?",
                        options=["Chưa chọn", "Có, ghi nhận thêm mốc khám mới", "Không, đây là bệnh nhân khác trùng tên"],
                        index=0,
                    )
                    if confirm_update == "Có, ghi nhận thêm mốc khám mới":
                        allow_to_save = True
                        st.caption("🟢 Nút lưu đã được mở khoá.")
                    elif confirm_update == "Không, đây là bệnh nhân khác trùng tên":
                        allow_to_save = False
                        st.error("🛑 Vui lòng thêm Mã số định danh vào tên để tạo hồ sơ riêng.")
                    else:
                        allow_to_save = False
                        st.info("💡 Vui lòng tích chọn xác nhận để mở khoá nút Lưu.")
                else:
                    st.success("✨ HỒ SƠ MỚI: Sẽ tạo tài khoản hồ sơ mới.")

            if st.button("🗑️ Reset Form"):
                st.session_state["form_patient_name"]    = ""
                st.session_state["form_patient_age"]     = 25
                st.session_state["form_patient_hometown"] = ""
                st.rerun()

            st.markdown("---")
            st.header("⚙️ CẤU HÌNH HỆ THỐNG")
            min_conf = st.slider("Safety gate threshold (τ_c)", 0.30, 0.95, 0.60, 0.01)
            st.caption(f"LLM Model: `{os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}`")

        # ── Upload & Image Reset ──────────────────────────────────────────────
        uploaded = st.file_uploader("📤 Tải ảnh tổn thương da lên:", type=["jpg", "jpeg", "png"])

        if uploaded is not None:
            if st.session_state["last_uploaded_file_name"] != uploaded.name:
                # Smart Reset: ảnh mới → xóa sạch toàn bộ state cũ
                st.session_state["last_uploaded_file_name"] = uploaded.name
                st.session_state["result"]                  = None
                st.session_state["messages"]                = []
                st.session_state["analysis_time"]           = None
                st.session_state["saved_local_img_path"]    = None

            image   = Image.open(uploaded).convert("RGB")
            img_rgb = np.array(image)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="📷 Ảnh đầu vào", width='stretch')

            if st.button("🔍 Chạy Phân tích CV", type="primary"):
                st.session_state["analysis_time"] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                with st.spinner("Đang chạy Segmentation + Classification..."):
                    tmp_dir  = tempfile.mkdtemp()
                    tmp_path = os.path.join(tmp_dir, "input.png")
                    image.save(tmp_path)
                    st.session_state["saved_local_img_path"] = tmp_path
                    result = get_pipeline(min_conf).run(tmp_path, return_mask=True)
                    st.session_state["result"]   = result
                    st.session_state["messages"] = []

            result = st.session_state.get("result")

            if result:
                # ── Hiển thị mask ─────────────────────────────────────────────
                mask     = result.get("segmentation_mask")
                mask_img = _mask_to_image(mask, img_rgb.shape[:2])
                with col2:
                    if mask_img is not None:
                        st.image(mask_img, caption="🎭 Mặt nạ phân vùng", clamp=True, channels="L", width='stretch')

                metrics  = result.get("metrics", {})
                cls      = result.get("classification") or {}
                status   = result.get("status", "ok")
                pred     = cls.get("prediction", "N/A")
                vi_name  = get_vietnamese_diagnosis(pred)
                conf_pct = float(cls.get("confidence", 0.0)) * 100

                # ── Safety Gate Status Banner ─────────────────────────────────
                if status == "triage":
                    triage_reason_raw = result.get("triage_reason", "unknown")
                    triage_vi = TRIAGE_REASON_VI.get(triage_reason_raw, triage_reason_raw)
                    st.markdown(
                        f"<div class='triage-banner'>"
                        f"🚨 <b>SAFETY GATE KÍCH HOẠT</b><br>"
                        f"Lý do: <i>{triage_vi}</i><br>"
                        f"Hệ thống VQA bị khoá — Vui lòng chụp lại ảnh hoặc chuyển ca cho bác sĩ."
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.success(f"✅ Phân tích thành công — Độ tin cậy: {conf_pct:.1f}% ≥ τ_c")

                # ── 4 Metric Cards ────────────────────────────────────────────
                st.subheader("📊 Số liệu Phân tích Định lượng")
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("⏱ Thời gian", st.session_state["analysis_time"] or "—")
                c2.metric("📐 Area ratio", f"{metrics.get('area_ratio', 0.0):.4f}")
                c3.metric("〰️ Border", f"{metrics.get('border_complexity', 0.0):.4f}")
                c4.metric("🔀 Asymmetry", f"{metrics.get('asymmetry', 0.0):.4f}")
                c5.metric("⭕ Circularity", f"{metrics.get('circularity', 0.0):.4f}")

                with c6:
                    border_color = "#e53e3e" if status == "triage" else "#38a169"
                    st.markdown(
                        f"""
                        <div style="background:rgba(17,25,40,0.8);padding:10px;border-radius:6px;
                                    border-left:5px solid {border_color};height:105px;">
                            <p style="margin:0;font-size:0.78rem;color:#a0aec0;font-weight:bold;">
                                ĐỘ TIN CẬY ({pred})</p>
                            <h3 style="margin:2px 0;color:white;font-size:1.4rem;">{conf_pct:.1f}%</h3>
                            <p style="margin:0;font-size:0.75rem;color:#63b3ed;font-weight:500;
                                      line-height:1.2;overflow:hidden;text-overflow:ellipsis;
                                      white-space:nowrap;">📌 {vi_name}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # ── Probability Chart + Radar Chart ──────────────────────────
                probs = cls.get("probabilities", {})
                if probs:
                    chart_col, radar_col = st.columns([1.6, 1])
                    with chart_col:
                        st.markdown("##### 📈 Phân phối Xác suất 7 Nhãn Bệnh lý")
                        render_probability_chart(
                            probs,
                            patient_name=p_name,
                            timestamp=st.session_state.get("analysis_time", ""),
                        )
                    with radar_col:
                        st.markdown("##### 🕸️ Radar Chart ABCD")
                        render_radar_chart(metrics)

            # ── VQA Chat ──────────────────────────────────────────────────────
            st.divider()
            st.subheader("💬 VQA Chat Space")

            for msg in st.session_state["messages"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Chặn hoàn toàn chat khi triage hoặc chưa có result
            is_triage      = result is not None and result.get("status") == "triage"
            chat_disabled  = (not bool(result)) or is_triage
            placeholder_txt = (
                "⛔ Safety Gate đang kích hoạt — VQA bị khoá."
                if is_triage else
                "Đặt câu hỏi về tổn thương da (sau khi chạy phân tích)..."
            )

            prompt = st.chat_input(placeholder_txt, disabled=chat_disabled)
            if prompt:
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                answer = generate_vqa_response(
                    question=prompt,
                    result=result,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    history=st.session_state["messages"],
                )
                st.session_state["messages"].append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)

            # ── Lưu EHR ──────────────────────────────────────────────────────
            if result:
                st.divider()
                st.subheader("💾 Đồng bộ Bệnh án điện tử")

                if st.button(
                    "Xác nhận & Lưu toàn bộ hồ sơ lên Google Cloud",
                    type="secondary",
                    disabled=not allow_to_save,
                ):
                    if not p_name:
                        st.error("❌ Vui lòng điền Họ tên bệnh nhân trước.")
                    elif not st.session_state["saved_local_img_path"]:
                        st.error("❌ Không tìm thấy file ảnh phân tích.")
                    else:
                        timestamp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                        with st.spinner("Đang đẩy ảnh lên ImgBB..."):
                            public_img_url = upload_image_to_imgbb(st.session_state["saved_local_img_path"])
                        if not public_img_url:
                            st.error("❌ Lỗi tải ảnh lên ImgBB.")
                        else:
                            patient_info = {
                                "name":     p_name.strip(),
                                "age":      int(p_age),
                                "hometown": p_hometown.strip() if p_hometown else "N/A",
                            }
                            metrics  = result.get("metrics", {})
                            cls      = result.get("classification") or {}
                            visit_data = {
                                "timestamp_id":     timestamp_id,
                                "created_at":       st.session_state["analysis_time"],
                                "image_url":        public_img_url,
                                "ai_extracted_metrics": {
                                    "status":            result.get("status"),
                                    "area_ratio":        float(metrics.get("area_ratio", 0.0)),
                                    "border_complexity": float(metrics.get("border_complexity", 0.0)),
                                    "asymmetry":         float(metrics.get("asymmetry", 0.0)),
                                    "circularity":       float(metrics.get("circularity", 0.0)),
                                    "prediction":        cls.get("prediction", "N/A"),
                                    "confidence":        float(cls.get("confidence", 0.0)),
                                },
                                "vqa_conversations": list(st.session_state["messages"]),
                            }
                            with st.spinner("Đang đồng bộ vào Cloud Firestore..."):
                                if save_medical_record_to_gcp(p_name, patient_info, visit_data):
                                    st.success(
                                        f"🎉 Đồng bộ thành công hồ sơ bệnh nhân '{p_name.upper()}'!"
                                    )
                                    if (
                                        st.session_state["saved_local_img_path"]
                                        and os.path.exists(st.session_state["saved_local_img_path"])
                                    ):
                                        os.remove(st.session_state["saved_local_img_path"])
                                        st.session_state["saved_local_img_path"] = None

    # ── TAB 2: DASHBOARD BÁC SĨ ──────────────────────────────────────────────
    with tab_doctor:
        render_doctor_dashboard()


if __name__ == "__main__":
    main()