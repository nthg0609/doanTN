# Hệ thống Chẩn đoán Da liễu Tích hợp & Trợ lý VQA (Unified Dermatology Pipeline & VQA)

Dự án này cung cấp một đường ống xử lý (pipeline) chẩn đoán bệnh da liễu tích hợp, kết hợp giữa các mô hình thị giác máy tính chuyên biệt (Computer Vision - CV) cục bộ và Mô hình Ngôn ngữ Lớn (LLM) thông qua cơ chế **Fusion Prompt** để đảm bảo tính an toàn y tế và triệt tiêu hoàn toàn hiện tượng ảo giác (hallucination).

---

## 🚀 Tính năng nổi bật & Cải tiến mới

1. **Kiến trúc Hybrid VQA (Fusion Prompt)**:
   - Kết hợp mô hình CV cục bộ để chẩn đoán lâm sàng và phân vùng tổn thương trước.
   - Nhúng cứng (hard-coded) kết quả CV (xác suất chẩn đoán, chỉ số hình học ABCD) vào `system_prompt` của LLM, ép LLM phải phản hồi dựa trên dữ liệu thực tế đáng tin cậy.
   - Tích hợp **Medication Guardrails**: Tuyệt đối cấm LLM kê đơn thuốc hoặc biệt dược cụ thể, đảm bảo chuẩn mực y đức.

2. **Adaptive Safety Gate (Cổng an toàn động)**:
   - Tự động nhận diện loại ảnh chụp: ảnh nội soi chuyên dụng (**Dermoscopy**) hoặc ảnh chụp camera thông thường (**Phone**).
   - Áp dụng các ngưỡng động kiểm tra chất lượng ảnh và tính chính xác của tổn thương được quy định trong [safety_gate.json](file:///d:/DoAn_DaLieu/config/safety_gate.json), giúp nới lỏng các tiêu chí bờ và diện tích cho camera điện thoại để tránh từ chối nhầm ảnh.

3. **Test-Time Augmentation (TTA) cho Phân vùng (Segmentation)**:
   - Khi tiếp nhận ảnh chụp điện thoại (`phone`), hệ thống tự động kích hoạt cơ chế TTA đa tỷ lệ (scales: `1.0`, `0.75`, `0.5`) qua hàm `multiscale_segment_from_rgb()`.
   - Giúp ổn định mặt nạ phân vùng tổn thương khỏi các tác động của ánh sáng không đều, bóng mờ hoặc góc chụp không chuẩn.

4. **Cảnh báo Lâm sàng Lâm giới (Clinical Risk Warning)**:
   - Trên giao diện Streamlit, nếu nhãn chẩn đoán chính là lành tính nhưng các bệnh ác tính (Melanoma, BCC) có tổng xác suất hoặc xác suất riêng lẻ vượt quá **15%** (`MALIGNANT_ALERT_THRESHOLD = 0.15`), hệ thống sẽ lập tức hiển thị cảnh báo lâm sàng màu cam/đỏ và chuyển sang màu đỏ rực để nhắc nhở người dùng không được chủ quan.

---

## 🛠️ Kiến trúc Mô hình (Backbones)

* **Nhánh Phân loại (Classification)**:
  - **Mạng xương sống**: `EfficientNet-B1` tích hợp mô-đun chú ý kép **CBAM** (Convolutional Block Attention Module - chú ý không gian và kênh).
  - **Bộ dữ liệu**: Huấn luyện trên **20,030** mẫu da chuẩn ISIC. Đạt độ chính xác kiểm thử **95.01%** (Accuracy).
  - **Hỗ trợ 7 lớp bệnh da liễu chính**:
    * **MEL** (Melanoma - U hắc tố ác tính) `[Ác tính]`
    * **BCC** (Basal Cell Carcinoma - Ung thư biểu mô tế bào đáy) `[Ác tính]`
    * **AKIEC** (Actinic Keratosis / Bowen's disease - Tiền ung thư) `[Tiền ác tính]`
    * **BKL** (Benign Keratosis-like lesions - Tổn thương sừng hóa lành tính) `[Lành tính]`
    * **NV** (Melanocytic nevi - Nốt ruồi lành tính) `[Lành tính]`
    * **DF** (Dermatofibroma - U xơ da) `[Lành tính]`
    * **VASC** (Vascular lesions - Tổn thương mạch máu) `[Lành tính]`

* **Nhánh Phân vùng (Segmentation)**:
  - **Mạng xương sống**: `DeepLabV3+` với bộ mã hóa `ResNet50` trích xuất vùng tổn thương thông qua khối ASPP đa tỷ lệ.
  - **Trích xuất thuộc tính ABCD**:
    - **A** (Area ratio): Tỉ lệ diện tích tổn thương trên diện tích ảnh.
    - **B** (Border complexity): Độ phức tạp đường biên.
    - **C** (Circularity): Độ tròn đều của tổn thương.
    - **D** (Asymmetry score): Chỉ số bất đối xứng qua hai trục.
  - **Bộ dữ liệu**: Huấn luyện trên **2,594** ảnh phân vùng, đạt chỉ số **Dice = 0.9128** và **IoU = 0.8455**.

* **Mô hình VQA Cục bộ ngoại tuyến (Offline VQA - Thực nghiệm/Nghiên cứu)**:
  - Lớp `CPUMedicalVQAModel` định nghĩa trong [train_vqa_joint.py](file:///d:/DoAn_DaLieu/scripts/train_vqa_joint.py) sử dụng `EfficientNet-B1 + CBAM` làm Vision Encoder và `DistilGPT-2 + LoRA` làm Decoder văn bản. Không được sử dụng trong phiên bản ứng dụng Web thời gian thực (để đảm bảo chất lượng, ứng dụng Web gọi API OpenAI `gpt-4o-mini` hoặc thông qua các SLM).

---

## 📁 Cấu trúc Thư mục Quan trọng

```text
d:\DoAn_DaLieu\
├── config/                     # Cấu hình ngưỡng kiểm soát an toàn của Safety Gate
│   └── safety_gate.json
├── pipeline/                   # Nhân xử lý chẩn đoán da liễu tích hợp
│   ├── model_registry.py       # Tải và lưu trữ Singleton các mô hình cục bộ
│   ├── safety_gate.py          # Kiểm soát chất lượng ảnh và cảnh báo rủi ro
│   └── unified_pipeline.py     # Tích hợp luồng phân loại, phân vùng và đo chỉ số ABCD
├── docs/                       # Tài liệu hướng dẫn và phân tích hệ thống
│   ├── vqa_architecture_and_improvement_plan.md
│   └── vqa_response_evaluation.md
├── scripts/                    # Scripts bổ trợ huấn luyện, đánh giá, kiểm thử mô hình
│   ├── sanity_check_cls.py     # Kiểm tra mô hình classification độc lập
│   ├── run_inference.py        # Chạy inference qua giao diện CLI
│   └── train_vqa_joint.py      # Huấn luyện mô hình VQA cục bộ (Offline)
├── app_streamlit.py            # Giao diện chẩn đoán Web chính chạy bằng Streamlit
├── requirements.txt            # Danh sách các thư viện cần thiết
└── test_vqa_upgrade.py         # Unit test suite cho toàn bộ nâng cấp (Safety Gate, TTA, Warning)
```

---

## ⚙️ Hướng dẫn Cài đặt & Sử dụng

### 1) Cài đặt Môi trường
Cài đặt các gói phụ thuộc cần thiết cho dự án:
```powershell
pip install -r requirements.txt
```

### 2) Cấu hình API Key cho VQA Chat
Để kích hoạt tính năng hỏi đáp thông minh VQA trên giao diện Web, hãy cấu hình khóa OpenAI API key trong biến môi trường hệ thống hoặc tạo tệp `.env` tại thư mục gốc:
```env
OPENAI_API_KEY="sk-proj-..."
OPENAI_MODEL="gpt-4o-mini"
```

### 3) Khởi động Ứng dụng Web (Streamlit)
Để chạy giao diện Web tương tác đầy đủ, cho phép chụp/tải ảnh, chẩn đoán, xem biểu đồ phân phối xác suất, lưu trữ hồ sơ bệnh án Cloud Firestore và chat VQA:
```powershell
streamlit run app_streamlit.py
```

### 4) Kiểm tra Độc lập Mô hình Phân loại (Classification Sanity Check)
Nếu bạn muốn kiểm tra trực tiếp dự báo xác suất của mô hình classification EfficientNet-B1 mà không đi qua các bộ phân vùng hoặc Safety Gate:
```powershell
python scripts/sanity_check_cls.py --image path/to/image.jpg
```
Hoặc chỉ định đường dẫn trọng số riêng:
```powershell
python scripts/sanity_check_cls.py --image path/to/image.jpg --weights 4_Models/classification/efficientnet_attention_best.pth
```

### 5) Chạy CLI Inference
Thực hiện chẩn đoán trực tiếp trên ảnh và lưu trữ kết quả phân tích JSON:
```powershell
python scripts/run_inference.py --image path/to/image.jpg --output 5_Results/inference_output.json
```

### 6) Đánh giá tập dữ liệu Test (Dataset Evaluation)
Chạy script đánh giá độ chính xác chẩn đoán trên một thư mục ảnh kiểm thử:
```powershell
python scripts/evaluate_dataset.py --data_dir D:\DoAn_DaLieu\1_Data\Data_test
```

### 7) Chạy Unit Test kiểm thử nâng cấp
Để chạy các bộ kiểm tra tự động cho Safety Gate động, TTA, và tính năng Clinical Risk Warning:
```powershell
python test_vqa_upgrade.py
```
*(Báo cáo kết quả chi tiết sẽ được xuất ra màn hình và ghi lại trong thư mục `6_Test_Results`)*
