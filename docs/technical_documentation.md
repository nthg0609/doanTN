# BÁO CÁO KIẾN TRÚC HỆ THỐNG VÀ PHÂN TÍCH THUẬT TOÁN
## HỆ THỐNG TRỢ LÝ CHẨN ĐOÁN DA LIỄU ĐA PHƯƠNG THỨC TÍCH HỢP VQA & HỒ SƠ BỆNH ÁN ĐIỆN TỬ (EHR) ĐA MỐC THỜI GIAN

---

### TÓM TẮT HỆ THỐNG
Hệ thống là một giải pháp y tế số (Digital Health Solution) tích hợp đa phương thức (Multimodal AI), kết hợp các mô hình Thị giác Máy tính chuyên sâu (Computer Vision) dùng cho phân đoạn và phân loại tổn thương da liễu với Mô hình Ngôn ngữ Lớn (Large Language Model - LLM) để thực hiện hội thoại y khoa VQA (Visual Question Answering). Hệ thống được thiết kế với cơ chế tự động đánh giá an toàn dữ liệu (Safety Gate), quản lý bệnh án điện tử đa mốc thời gian (Multi-visit EHR) trên nền tảng đám mây Google Cloud Firestore, và giao diện trực quan hỗ trợ bác sĩ lâm sàng đưa ra quyết định sàng lọc sơ bộ tối ưu.

---

## 1. KIẾN TRÚC TỔNG QUAN VÀ SƠ ĐỒ LUỒNG DỮ LIỆU (System Architecture & Dataflow)

### 1.1 Khái quát mô hình tổng thể
Hệ thống hoạt động dựa trên cơ chế đồng hành song song (Parallel Pipeline Contract) giữa hai nhánh chính:
1. **Nhánh Trích xuất Đặc trưng Hình học & Phân đoạn (Segmentation Branch)**: Nhận ảnh gốc RGB, đi qua khối phân đoạn DeepLabV3+ (được tăng cường bằng kỹ thuật Test-Time Augmentation - TTA đối với ảnh chụp điện thoại) để sinh ra mặt nạ (binary mask) tổn thương da. Từ mặt nạ này, hệ thống tính toán các chỉ số hình học theo chuẩn ABCD lâm sàng.
2. **Nhánh Phân loại Bệnh lý (Classification Branch)**: Hoạt động hoàn toàn độc lập trên ảnh gốc RGB thông qua mạng EfficientNet-B1 kết hợp khối chú ý CBAM (Convolutional Block Attention Module) để xuất ra phân phối xác suất trên 7 nhóm bệnh lý da liễu thuộc chuẩn ISIC.

Kết quả từ hai nhánh được tổng hợp để đưa qua **Safety Gate**. Nếu vượt qua bộ lọc an toàn, dữ liệu hình học và phân loại sẽ được nhúng cứng (hard-coded) vào vùng ngữ cảnh hệ thống (System Prompt Context) của LLM tạo nên kiến trúc **Fusion Prompt**. LLM (GPT-4o-mini ở chế độ trực tuyến hoặc CPUMedicalVQAModel ở chế độ ngoại tuyến) sau đó sẽ sinh câu trả lời dạng dòng (Streaming Response) để trả lời các câu hỏi lâm sàng của người dùng (VQA Chat) dưới sự kiểm soát nghiêm ngặt của bộ quy tắc an toàn y đức (Medication Guardrail). Toàn bộ hồ sơ lâm sàng có thể được đồng bộ lên **Cloud Firestore** thông qua một cổng kiểm duyệt trùng lặp hồ sơ thông minh (**Confirmation Gate**).

### 1.2 Sơ đồ luồng dữ liệu (Dataflow ASCII Art)

```text
       [ Ảnh đầu vào (RGB Image) ]
                   │
                   ▼
     [ Image Type Detection Module ]
      - Phân tích khía cạnh hình học & độ phân giải ảnh
      - Phân loại luồng xử lý: { 'dermoscopy', 'phone' }
                   │
         ┌─────────┴────────────────────────┐
         ▼ (dermoscopy)                     ▼ (phone)
   [ Standard Segmentation ]         [ Multi-scale TTA Segment ]
   - Single pass ResNet50-DeepLab   - Scale factor: (1.0, 0.75, 0.5)
         │                                  - Averaged Probability Map
         └─────────┬────────────────────────┘
                   │
                   ▼
       [ Post-Processing Mask ]
       - Morphology Open & Close (Kernel 5x5)
       - Component Connection Analysis (Keep Largest)
       - [Classical Fallback Otsu] (nếu DeepLab không tạo được mask)
                   │
                   ├───► [ ABCD Geometric Feature Extraction ]
                   │      - Asymmetry (A), Border Complexity (B),
                   │        Circularity (C), Area Ratio (D)
                   │
                   │         [ Classification Branch ]
                   │         - Independent Forward pass on Raw RGB
                   │         - ImageNet normalization & Resize (224x224)
                   │         - EfficientNet-B1 + CBAM Network
                   │         - 7-class Probability Distribution
                   │                 │
                   └────────┬────────┘
                            ▼
                  [ Safety Gate Evaluation ]
                  - Đầu vào: ABCD metrics, cls_confidence, image_type
                  - Ngưỡng động: adaptive thresholds (dermoscopy vs phone)
                            │
         ┌──────────────────┴──────────────────┐
         ▼ (Chấp nhận - Accept)                ▼ (Từ chối - Reject: Triage Mode)
   [ Clinical Risk Warning check ]      [ Triage System Report Generated ]
   - Check if malignant prob >= 0.15    - Ghi nhận lỗi (area_ratio/border/confidence)
   - Render orange UI Warning banner    - Khoá hoàn toàn VQA Chat Input
         │                              - Render red Safety Gate Alert banner
         ▼                                     │
   [ Prompt Fusion Engine ]                    ▼
   - Embed CV context into System Prompt   [ Hidden Dev System Logging ]
   - Inject Medication Guardrails          - Serialize JSON raw data via Custom Encoder
         │                                 - Ghi xuống file logs cục bộ
         ▼                                   (5_Results/system_logs.log)
   [ LLM Chat Engine (OpenAI API) ]
   - gpt-4o-mini stream output
   - Live stream rendering to UI
         │
         ▼
   [ Cloud Confirmation Gate ]
   - Kiểm tra trùng lặp trên GCP Firestore
   - st.radio lựa chọn phân nhánh cho Bác sĩ:
     ├──► [ Đồng ý cập nhật ] ──► Firestore: Append to visits[] (Multi-visit EHR)
     └──► [ Báo trùng tên ]   ──► Block Save / Yêu cầu đổi ID hồ sơ bệnh nhân
```

---

## 2. CÁC CÔNG NGHỆ, MÔ HÌNH VÀ THƯ VIỆN LÕI (Core Technology Stack & Deep Learning Models)

### 2.1 Nhánh Phân vùng Tổn thương (DeepLabV3+ Segmentation)
* **Kiến trúc mô hình**: DeepLabV3+ sử dụng bộ trích xuất đặc trưng (backbone) **ResNet-50** đã loại bỏ các lớp phân loại cuối cùng. Khối Atrous Spatial Pyramid Pooling (ASPP) được áp dụng tại bottleneck để thu thập đặc trưng đa tỷ lệ thông qua các tốc độ giãn nở (dilation rates) khác nhau, giúp bảo toàn thông tin biên của tổn thương.
* **Thông số đầu vào/đầu ra**:
  * Kích thước ảnh đầu vào: $256 \times 256$ pixel, chuẩn hóa kênh màu qua cấu hình:
    $$\mu = [0.5, 0.5, 0.5], \quad \sigma = [0.25, 0.25, 0.25]$$
  * Đầu ra: Bản đồ xác suất đơn kênh (single-channel probability map) kích thước ban đầu $256 \times 256$. Bản đồ này được nội suy song tuyến (Bilinear Interpolation) về kích thước ảnh gốc $H \times W$, sau đó nhị phân hóa bằng ngưỡng $\text{seg\_threshold} = 0.3$.
* **Đánh giá chất lượng mô hình**: Phân đoạn tổn thương đạt chỉ số **Dice Coefficient = 0.9074** và **IoU (Intersection over Union) = 0.8394** trên tập dữ liệu kiểm thử chuẩn ISIC.

### 2.2 Nhánh Phân loại Bệnh lý (EfficientNet-B1 + CBAM)
* **Kiến trúc mạng**: Sử dụng mạng xương sống **EfficientNet-B1** làm bộ trích xuất đặc trưng (feature extractor) cơ sở, trích xuất đặc trưng tại lớp convolutional cuối cùng để thu được tensor đặc trưng kích thước $C \times H \times W$ (với $C = 1280$ kênh đặc trưng). Tensor này được đưa qua khối chú ý hỗn hợp **CBAM (Convolutional Block Attention Module)**.
* **Khối chú ý CBAM**: Tăng cường đặc trưng theo cả hai chiều:
  * **Channel Attention**: Sử dụng cả hai phép toán Adaptive Average Pooling và Adaptive Max Pooling song song trên tensor đặc trưng, đi qua một mạng MLP chia sẻ trọng số để tạo ra vector trọng số kênh, nhấn mạnh "cái gì" quan trọng.
    $$M_c(F) = \sigma(\text{MLP}(\text{AvgPool}(F)) + \text{MLP}(\text{MaxPool}(F)))$$
  * **Spatial Attention**: Thực hiện phép chiếu kênh trung bình và kênh cực đại dọc theo chiều sâu đặc trưng, nối chúng lại thành tensor $2 \times H \times W$, sau đó đi qua một lớp tích chập tích hợp bộ lọc kích thước $7 \times 7$ để tạo bản đồ chú ý không gian, định vị "ở đâu" quan trọng trên ảnh da.
    $$M_s(F) = \sigma(f^{7\times 7}([\text{AvgPool}(F); \text{MaxPool}(F)]))$$
  * Tensor đặc trưng sau cùng được nhân nhân bản với các bản đồ chú ý:
    $$F' = F \otimes M_c(F), \quad F'' = F' \otimes M_s(F')$$
  * Đặc trưng sau chú ý $F''$ được đưa qua lớp Global Average Pooling để tạo thành vector đặc trưng $1280$ chiều, qua lớp Dropout điều hòa tỷ lệ $0.3$ và cuối cùng kết nối với một lớp tuyến tính (Linear Layer) để phân loại sang 7 lớp bệnh lý.
* **Chuẩn hóa dữ liệu**: Ảnh được nội suy song tuyến về kích thước $224 \times 224$ và chuẩn hóa theo chuẩn ảnh ImageNet:
  $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
* **Tập nhãn bệnh lý (7 lớp ISIC)**:
  * `AKIEC` (Dày sừng quang hóa), `BCC` (Ung thư tế bào đáy), `BKL` (Tổn thương sừng hóa lành tính), `DF` (U xơ da), `MEL` (U hắc tố ác tính), `NV` (Nốt ruồi lành tính), `VASC` (Tổn thương mạch máu).

### 2.3 Hỏi đáp Y tế Đa phương thức (VQA Model)
Hệ thống hỗ trợ hai cơ chế VQA linh hoạt phụ thuộc vào điều kiện hạ tầng phần cứng:

#### A. Mô hình Trực tuyến (Online Production VQA)
Sử dụng mô hình ngôn ngữ lớn thương mại **gpt-4o-mini** của OpenAI. Để khắc phục hiện tượng "ảo tưởng" (hallucination) của LLM và đảm bảo tính chính xác lâm sàng, hệ thống áp dụng kiến trúc **Fusion Prompt**:
* **Không truyền ảnh trực tiếp vào LLM**: Thay vì truyền tệp ảnh thô vào API của GPT (gây tốn chi phí token và khó kiểm soát vùng tư vấn), hệ thống chuyển đổi ảnh sang dạng biểu diễn thông tin định lượng (Quantitative Representation).
* **System Prompt nhúng cứng ngữ cảnh CV**: Toàn bộ kết quả chẩn đoán của mô hình phân loại (nhãn dự đoán, xác suất chi tiết 7 lớp) và các chỉ số hình học trích xuất từ mô hình phân đoạn (Area ratio, Border complexity, Asymmetry, Circularity) được mã hóa thành văn bản và tiêm trực tiếp vào **System Prompt** ở khu vực dành riêng `[CV_CONTEXT]`.
* **Ưu điểm**: Giúp LLM chỉ hội thoại xung quanh dữ liệu thực tế do mô hình CV trích xuất, ngăn chặn việc suy diễn ra ngoài vùng tổn thương được chụp.

#### B. Mô hình Ngoại tuyến (Offline VQA - CPUMedicalVQAModel)
Đây là mô hình tự huấn luyện phục vụ chạy offline trên CPU mà không cần kết nối API ngoài. Cấu trúc mô hình bao gồm:
* **Vision Backbone**: Sử dụng `EfficientNet-B1` kết hợp khối `CBAM` trích xuất đặc trưng từ ảnh đầu vào $3 \times 224 \times 224$ thành vector đặc trưng kích thước $1 \times 1280$. Khi huấn luyện VQA, toàn bộ các tham số của mạng EfficientNet-B1 gốc bị đóng băng (frozen), chỉ mở khóa các tham số trong khối chú ý CBAM để tinh chỉnh thông tin không gian vùng da.
* **Projection Layer**: Khối MLP gồm 2 lớp tuyến tính có hàm kích hoạt GELU và Dropout ($0.3$) xen kẽ, đảm nhận vai trò ánh xạ vector đặc trưng ảnh từ không gian thị giác ($1280$ chiều) sang không gian biểu diễn ngôn ngữ ($768$ chiều) để tương thích với LLM:
  $$\text{Projection}(v) = W_2(\text{Dropout}(\text{GELU}(W_1(v) + b_1))) + b_2$$
  Vector thu được được định hình lại thành kích thước $1 \times 1 \times 768$, đóng vai trò như một "token hình ảnh" đặc biệt đại diện cho vùng tổn thương da.
* **Language Branch (DistilGPT-2 + LoRA)**:
  * Sử dụng kiến trúc transformer giải mã tự hồi quy (Causal Language Model) **DistilGPT-2** làm bộ sinh ngôn ngữ (đầu vào embedding kích thước $768$).
  * Nhúng bộ chuyển đổi tham số hiệu quả **LoRA (Low-Rank Adaptation)** trực tiếp vào các ma trận chiếu khóa-giá trị-truy vấn của lớp tự chú ý (các module `c_attn`).
  * Cấu hình tham số LoRA: Hạng $r = 8$, hệ số tỷ lệ $\alpha = 16$, tỷ lệ Dropout bằng $0.05$. Điều này giúp giảm thiểu 99% số lượng tham số cần cập nhật trong quá trình huấn luyện LLM.
* **Cơ chế ghép nối chuỗi mã hóa (Forward Pass)**:
  * Token ảnh chiếu ($1 \times 768$) được ghép nối trực tiếp vào phía trước các token embedding của chuỗi câu hỏi văn bản ($L \times 768$) tạo thành chuỗi embedding tổng hợp ($ (1 + L) \times 768 $):
    $$E_{\text{total}} = [\mathbf{v}_{\text{projected}} \; ; \; \mathbf{e}_{t_1} \; ; \; \mathbf{e}_{t_2} \; ; \; \dots \; ; \; \mathbf{e}_{t_L}]$$
  * Chuỗi embedding này được đưa vào mô hình DistilGPT-2 để sinh chuỗi câu trả lời tự hồi quy.

---

## 3. LOGIC XỬ LÝ CHI TIẾT TRÊN PIPELINE VÀ GIAO DIỆN (Detailed Workflow & State Management)

### 3.1 Khối Safety Gate & Triage Mode (Selective Prediction)
Nhằm bảo đảm an toàn tính mạng trong y tế số, hệ thống sử dụng thuật toán **Selective Prediction** thông qua khối kiểm soát **Safety Gate**. Khi dữ liệu đầu vào không đảm bảo chất lượng lâm sàng hoặc mô hình AI có độ tự tin quá thấp, hệ thống sẽ tự động chuyển sang **Triage Mode** (chế độ phân loại khẩn cấp/từ chối chẩn đoán tự động).

* **Đánh giá Ngưỡng Tin cậy Thích ứng (Adaptive Thresholds)**:
  Hệ thống tự động nhận dạng ảnh chụp bằng thiết bị chuyên dụng (`dermoscopy`) hay ảnh chụp tự do bằng điện thoại cá nhân (`phone`) dựa trên tỷ lệ khung hình và độ phân giải biên để kích hoạt các ngưỡng đánh giá phù hợp:

| Ngưỡng tham số | Chế độ `dermoscopy` | Chế độ `phone` (Adaptive) | Ý nghĩa lâm sàng |
| :--- | :---: | :---: | :--- |
| $\text{min\_mask\_area\_px}$ | $64\text{ px}$ | $64\text{ px}$ | Diện tích tổn thương tối thiểu để xử lý |
| $\text{min\_area\_ratio}$ | $0.001$ | $0.0005$ | Loại bỏ ảnh da bình thường không có nốt ruồi |
| $\text{max\_area\_ratio}$ | $0.75$ | $0.92$ | Loại bỏ ảnh chụp quá sát, mất biên tổn thương |
| $\text{max\_border\_complexity}$ | $8.0$ | $14.0$ | Nới lỏng độ phức tạp bờ do nhiễu hậu cảnh |
| $\text{min\_class\_confidence}$ ($\tau_c$) | $0.60$ | $0.60$ | Ngưỡng an toàn xác suất dự đoán nhãn |

* **Cơ chế hoạt động**:
  1. Nếu diện tích vùng tổn thương phân đoạn ($A_{\text{lesion}}$) nhỏ hơn $64$ pixel, hoặc chỉ số diện tích tỷ lệ nằm ngoài khoảng cho phép $\Rightarrow$ Từ chối chẩn đoán với lỗi `empty_or_low_confidence_mask` hoặc `area_ratio_out_of_bounds`.
  2. Nếu độ phức tạp bờ tổn thương vượt quá ngưỡng tối đa cho phép $\Rightarrow$ Từ chối chẩn đoán với lỗi `border_complexity_out_of_bounds` do ảnh bị nhiễu lông, vảy da sừng làm mất cấu trúc đường biên thực.
  3. Nếu xác suất dự đoán của nhãn phân loại lớn nhất thấp hơn ngưỡng an toàn $\tau_c$ (mặc định $0.60$) $\Rightarrow$ Từ chối chẩn đoán với lỗi `low_classification_confidence`.
  4. **Triage Mode Active**: Khi Safety Gate trả về kết quả từ chối (`accept = False`), giao diện Streamlit lập tức hiển thị Banner cảnh báo màu đỏ báo lỗi kỹ thuật chi tiết. Đồng thời, ô nhập chat VQA (`st.chat_input`) bị **khóa hoàn toàn (disabled)** nhằm ngăn chặn tuyệt đối việc LLM đưa ra lời khuyên y khoa dựa trên các số liệu đầu vào thiếu tin cậy.

### 3.2 Khối Cảnh báo Lâm sàng Nguy cơ Ác tính (Clinical Risk Warning)
Trên thực tế, mô hình phân loại có thể dự đoán nhãn có xác suất cao nhất là một bệnh lành tính (như nốt ruồi lành tính `NV`), nhưng xác suất dành cho lớp ác tính (như u hắc tố ác tính `MEL`) vẫn ở mức đáng lo ngại. Do đó, hệ thống triển khai cơ chế phát hiện sớm nguy cơ ác tính tiềm ẩn:
* **Logic toán học**: Định nghĩa tập hợp các lớp ác tính và tiền ác tính $\mathcal{M} = \{\text{MEL}, \text{BCC}, \text{AKIEC}\}$. Khi nhãn dự đoán chính của mô hình $y^* \notin \mathcal{M}$ (thuộc nhóm lành tính), hệ thống sẽ quét qua phân phối xác suất dự đoán $P$ để tìm giá trị lớn nhất trong nhóm nguy cơ cao:
  $$P_{\text{max\_malignant}} = \max_{m \in \mathcal{M}} P(m)$$
* **Ngưỡng cảnh báo**: Nếu $P_{\text{max\_malignant}} \ge 0.15$ (tức xác suất mắc bệnh ác tính tiềm ẩn đạt từ 15% trở lên), hệ thống sẽ tự động kích hoạt **Clinical Risk Warning Banner** trên giao diện với màu cam nổi bật:
  > ⚠️ **Cảnh báo Lâm sàng** — Dự đoán chính là **BKL** (lành tính), nhưng mô hình phát hiện xác suất **MEL** (U hắc tố ác tính) = **18.5%** ($\ge 15.0\%$). Đề nghị tham khảo bác sĩ da liễu để tiến hành làm sinh thiết loại trừ.
* Cơ chế này đóng vai trò như một màng lọc bảo vệ kép, tránh việc bỏ sót (False Negative) các ca bệnh nguy hiểm khi mô hình bị nhiễu phân loại.

### 3.3 Khối Cổng Kiểm Duyệt Trùng Lặp (Confirmation Gate)
Để bảo toàn tính toàn vẹn dữ liệu và tránh việc lưu đè dữ liệu vô tổ chức lên hệ thống bệnh án điện tử, hệ thống tích hợp khối logic kiểm soát trùng lặp:
1. **Kiểm tra sự tồn tại**: Khi bác sĩ nhập thông tin tên bệnh nhân vào sidebar, hệ thống chuẩn hóa tên bệnh nhân thành mã định danh viết hoa không dấu và không khoảng trắng (ví dụ: "Nguyễn Văn A" $\rightarrow$ ID document `NGUYENVANA`). Hàm `check_patient_exists()` sẽ thực hiện một truy vấn đọc nhanh lên Firestore.
2. **Kích hoạt cổng chặn**: Nếu hồ sơ bệnh nhân đã tồn tại, nút "Xác nhận & Lưu" bị khóa. Hệ thống hiển thị cảnh báo đỏ và kết xuất widget lựa chọn bắt buộc bác sĩ tương tác (`st.radio`):
   * *Lựa chọn 1: "Có, ghi nhận thêm mốc khám mới"* $\Rightarrow$ Đặt trạng thái biến cho phép ghi dữ liệu `allow_to_save = True`. Hệ thống sẽ chuẩn bị đẩy thông tin ảnh khám mới nhất vào mảng dòng thời gian của bệnh nhân cũ.
   * *Lựa chọn 2: "Không, đây là bệnh nhân khác trùng tên"* $\Rightarrow$ Đặt trạng thái `allow_to_save = False`. Bác sĩ bị chặn lưu hồ sơ và được yêu cầu thêm ký tự phân biệt (như mã căn cước công dân hoặc số thứ tự khám) vào tên để tạo lập một hồ sơ độc lập.
   * *Mặc định: "Chưa chọn"* $\Rightarrow$ Khóa tính năng đồng bộ và hướng dẫn bác sĩ chọn xác nhận để tiếp tục.

### 3.4 Kiến trúc Tiến triển Đa mốc thời gian (Multi-visit EHR Architecture)
Hệ thống sử dụng cơ sở dữ liệu tài liệu NoSQL Google Cloud Firestore làm kho lưu trữ EHR. Cấu trúc dữ liệu được thiết kế theo dạng **Nested Array of Objects** (Mảng đối tượng lồng nhau) trong một tài liệu duy nhất đại diện cho một bệnh nhân:

```json
{
  "patient_id": "NGUYENVANB",
  "patient_info": {
    "name": "Nguyễn Văn B",
    "age": 42,
    "hometown": "Đà Nẵng"
  },
  "created_at": "2026-06-01 09:30:15",
  "updated_at": "2026-06-03 14:20:00",
  "visits": [
    {
      "timestamp_id": "20260601_093015",
      "created_at": "2026-06-01 09:30:15",
      "image_url": "https://i.ibb.co/example1/image.png",
      "ai_extracted_metrics": {
        "status": "ok",
        "prediction": "NV",
        "confidence": 0.8845,
        "area_ratio": 0.0245,
        "border_complexity": 3.1205,
        "asymmetry": 0.1240,
        "circularity": 0.8920
      },
      "vqa_conversations": [
        {"role": "user", "content": "Nốt ruồi này có nguy hiểm không?"},
        {"role": "assistant", "content": "Dựa trên chỉ số phân tích, tổn thương của bạn được phân loại..."}
      ]
    },
    {
      "timestamp_id": "20260603_142000",
      "created_at": "2026-06-03 14:20:00",
      "image_url": "https://i.ibb.co/example2/image.png",
      "ai_extracted_metrics": {
        "status": "ok",
        "prediction": "NV",
        "confidence": 0.9120,
        "area_ratio": 0.0380,
        "border_complexity": 3.4560,
        "asymmetry": 0.1510,
        "circularity": 0.8540
      },
      "vqa_conversations": []
    }
  ]
}
```
* **Cơ chế cập nhật**: Khi lưu thêm mốc khám mới, hệ thống tải dữ liệu hiện tại xuống, thêm bản ghi khám mới vào mảng `visits` bằng phương thức `.update()` của Firestore Document Reference thay vì ghi đè lại toàn bộ tài liệu, giúp tiết kiệm chi phí băng thông đường truyền đám mây.
* **Giao diện Doctor Dashboard**: Hiển thị toàn bộ biên niên sử ảnh qua các thời kỳ dưới dạng cây thư mục thu gọn (`st.expander`). Bác sĩ có thể so sánh trực quan sự thay đổi kích thước (`area_ratio`), sự biến đổi đường viền, độ đối xứng và xem lại lịch sử hội thoại của từng lần khám trước đó để đánh giá tốc độ tiến triển của bệnh.

### 3.5 Khối Reset Trạng thái Thông minh (Smart Reset)
Streamlit chạy lại toàn bộ script từ đầu mỗi khi có bất kỳ tương tác UI nào. Để duy trì tính nhất quán của trạng thái phiên làm việc (Session State) mà không để xảy ra hiện tượng chồng chéo dữ liệu cũ-mới, hệ thống cài đặt cơ chế **Smart Reset**:
* Khi phát hiện tệp tin tải lên có tên khác với tên tệp tin lưu trong phiên khám hiện tại:
  $$\text{uploaded.name} \neq \text{st.session_state["last\_uploaded\_file\_name"]}$$
* Hệ thống lập tức thực thi chuỗi lệnh dọn dẹp bộ nhớ đệm:
  ```python
  st.session_state["last_uploaded_file_name"] = uploaded.name
  st.session_state["result"]                  = None
  st.session_state["messages"]                = []
  st.session_state["analysis_time"]           = None
  st.session_state["saved_local_img_path"]    = None
  ```
* Việc dọn dẹp này đảm bảo khi bác sĩ đổi sang phân tích ca bệnh mới, màn hình phân tích CV cũ, biểu đồ Plotly, biểu đồ Radar và nội dung trò chuyện VQA của bệnh nhân cũ sẽ biến mất hoàn toàn, tránh tình trạng bác sĩ đọc nhầm kết quả chẩn đoán của ca khám trước.

---

## 4. ĐẶC TẢ CÁC HÀM VÀ PHƯƠNG THỨC CHỦ CHỐT (Core Functions Specification)

### 4.1 Hàm khởi chạy pipeline phân tích tổng hợp
#### `UnifiedDermatologyPipeline.run(self, image_path: str, question: Optional[str] = None, return_mask: bool = False) -> Dict[str, Any]`
* **Mục đích**: Nhận ảnh da từ đường dẫn cục bộ, điều phối thực thi song song hai nhánh phân đoạn và phân loại, áp dụng Safety Gate để đưa ra kết quả chẩn đoán lâm sàng cuối cùng.
* **Đầu vào (Inputs)**:
  * `image_path` (`str`): Đường dẫn tuyệt đối đến tệp ảnh đầu vào.
  * `question` (`Optional[str]`): Câu hỏi VQA nếu có.
  * `return_mask` (`bool`): Cờ cho phép trả về ma trận numpy chứa mặt nạ nhị phân hay không.
* **Đầu ra (Outputs)**: Trả về một từ điển (`Dict[str, Any]`) có cấu trúc:
  * `status`: Trạng thái kết quả (`"ok"` hoặc `"triage"`).
  * `image_path`: Đường dẫn ảnh gốc đã giải quyết.
  * `triage_reason`: Lý do bị Safety Gate chặn (`None` nếu status là ok).
  * `preprocess`: Chứa thông tin tiền xử lý bệnh án (`image_type` là phone hay dermoscopy).
  * `segmentation`: Chi tiết phương pháp phân đoạn (`deeplab` hoặc `deeplab_tta`).
  * `metrics`: Từ điển chứa các chỉ số hình học ABCD.
  * `classification`: Từ điển chứa dự đoán nhãn cao nhất, độ tin cậy và phân phối xác suất 7 lớp.
  * `report`: Chuỗi văn bản báo cáo y khoa sơ bộ.
  * `segmentation_mask` (Tùy chọn): Ma trận nhị phân `np.ndarray` kích thước ảnh gốc.

### 4.2 Các hàm xử lý ảnh và trích xuất đặc trưng hình học
#### `UnifiedDermatologyPipeline._segment(self, img_rgb: np.ndarray, image_type: str = "dermoscopy") -> tuple[np.ndarray, Dict[str, Any]]`
* **Mục đích**: Thực hiện phân đoạn ảnh. Nếu `image_type == "phone"` và cờ `use_tta` được bật, hàm sẽ gọi bộ phân đoạn đa tỷ lệ TTA để triệt tiêu nhiễu môi trường, ngược lại sẽ phân đoạn đơn luồng chuẩn.
* **Đầu vào (Inputs)**:
  * `img_rgb` (`np.ndarray`): Ma trận ảnh đầu vào màu RGB dạng `uint8` có kích thước $H \times W \times 3$.
  * `image_type` (`str`): Kiểu ảnh chụp (`"dermoscopy"` hoặc `"phone"`).
* **Đầu ra (Outputs)**: Trả về `tuple` gồm:
  * `mask` (`np.ndarray`): Mặt nạ nhị phân nhãn $0/1$ kích thước $H \times W$.
  * `seg_info` (`Dict[str, Any]`): Từ điển lưu trữ phương pháp phân đoạn thực thi và các tham số kỹ thuật đi kèm.

#### `UnifiedDermatologyPipeline._get_lesion_metrics(self, mask: np.ndarray) -> Dict[str, Any]`
* **Mục đích**: Tính toán các chỉ số hình học ABCD từ mặt nạ phân đoạn nhị phân của tổn thương.
* **Đầu vào (Inputs)**:
  * `mask` (`np.ndarray`): Mặt nạ nhị phân vùng tổn thương.
* **Đầu ra (Outputs)**: `Dict[str, Any]` chứa:
  * `area_ratio` (`float`): Tỷ số diện tích tổn thương trên diện tích ảnh da.
  * `border_complexity` (`float`): Chỉ số đo lường độ phức tạp của đường bao.
  * `asymmetry` (`float`): Hệ số bất đối xứng của tổn thương $\in [0,1]$.
  * `circularity` (`float`): Chỉ số đo độ tròn của tổn thương $\in [0,1]$.
  * `lesion_area` (`int`): Diện tích tổn thương tính bằng số pixel.
  * `image_area` (`int`): Tổng diện tích ảnh tính bằng số pixel.
  * `low_confidence` (`bool`): Đánh giá nhanh xem diện tích có quá nhỏ để tính toán hay không.

#### `UnifiedDermatologyPipeline._postprocess_mask(self, mask: np.ndarray) -> np.ndarray`
* **Mục đích**: Loại bỏ nhiễu nhị phân bằng thuật toán hình thái học (Morphological Operations) và giữ lại thành phần liên thông có diện tích lớn nhất (Largest Connected Component).
* **Đầu vào (Inputs)**: `mask` (`np.ndarray`) nhị phân thô chứa nhiễu.
* **Đầu ra (Outputs)**: `np.ndarray` mặt nạ nhị phân đã được làm sạch biên và chỉ chứa một vùng tổn thương chính duy nhất.

#### `UnifiedDermatologyPipeline._classical_fallback_mask(self, img_rgb: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]`
* **Mục đích**: Thuật toán phân đoạn dự phòng cổ điển sử dụng ngưỡng Otsu kết hợp phân tích thành phần liên thông khi mô hình học sâu DeepLab không phát hiện được tổn thương trong ảnh.
* **Đầu vào (Inputs)**: `img_rgb` (`np.ndarray`) ảnh đầu vào màu gốc.
* **Đầu ra (Outputs)**: Trả về mặt nạ nhị phân dự phòng và thông tin kiểm duyệt hình học (tỷ lệ khung bao, độ nén, khoảng cách tới tâm ảnh) xem mặt nạ cổ điển này có đáng tin cậy để sử dụng tiếp hay không.

### 4.3 Các hàm giao tiếp dịch vụ đám mây (Cloud API Services)
#### `check_patient_exists(patient_name: str) -> bool`
* **Mục đích**: Kiểm tra sự tồn tại của hồ sơ bệnh án bệnh nhân dựa trên tên hành chính của họ trên Firestore.
* **Đầu vào (Inputs)**: Tên bệnh nhân đầy đủ (`patient_name` dạng `str`).
* **Đầu ra (Outputs)**: Trả về `True` nếu document ID tương ứng đã tồn tại trong collection `medical_records`, ngược lại trả về `False`.

#### `save_medical_record_to_gcp(patient_name: str, patient_info: Dict[str, Any], visit_data: Dict[str, Any]) -> bool`
* **Mục đích**: Đồng bộ dữ liệu khám bệnh lên Cloud Firestore.
* **Đầu vào (Inputs)**:
  * `patient_name` (`str`): Họ tên bệnh nhân.
  * `patient_info` (`Dict[str, Any]`): Thông tin hành chính bệnh nhân (tuổi, quê quán).
  * `visit_data` (`Dict[str, Any]`): Từ điển lưu trữ thông tin chi tiết của mốc khám (đường dẫn ảnh đám mây ImgBB, kết quả CV, nhật ký VQA).
* **Đầu ra (Outputs)**: Trả về `True` nếu thực hiện ghi/cập nhật thành công lên Firestore, ngược lại trả về `False`.

### 4.4 Hàm sinh hội thoại y khoa đa phương thức
#### `generate_vqa_response_stream(question: str, result: Dict[str, Any], api_key: Optional[str], history: Optional[List[Dict[str, str]]] = None)`
* **Mục đích**: Triệu gọi mô hình ngôn ngữ lớn ở chế độ sinh dòng (Streaming Generator), tích hợp ngữ cảnh CV nhúng cứng, sinh câu trả lời từng ký tự cho giao diện VQA.
* **Đầu vào (Inputs)**:
  * `question` (`str`): Câu hỏi hiện tại của người dùng.
  * `result` (`Dict[str, Any]`): Kết quả phân tích CV từ pipeline.
  * `api_key` (`Optional[str]`): Khóa API OpenAI.
  * `history` (`Optional[List[Dict[str, str]]]`): Danh sách lịch sử tin nhắn trong phiên trò chuyện.
* **Đầu ra (Outputs)**: Trình tạo (`Generator` yielding `str`) sinh ra từng từ/cụm từ của phản hồi hỗ trợ lâm sàng cho đến khi kết thúc chuỗi tin nhắn.

---

## 5. CÁC THÔNG SỐ CẤU HÌNH VÀ QUẢN LÝ HỆ THỐNG (Thresholds, Mappings & Monitoring)

### 5.1 Bảng ánh xạ bệnh lý lâm sàng (`DIAGNOSIS_DICTIONARY`)
Để phục vụ hiển thị trên giao diện người dùng tiếng Việt và hỗ trợ LLM nhận thức đúng thuật ngữ lâm sàng, hệ thống xây dựng bảng ánh xạ danh pháp y khoa chi tiết:

```python
DIAGNOSIS_DICTIONARY: Dict[str, str] = {
    "AKIEC": "Dày sừng quang hóa / Tiền ung thư",
    "BCC":   "Ung thư biểu mô tế bào đáy",
    "BKL":   "Tổn thương sừng hóa lành tính",
    "DF":    "U xơ da",
    "MEL":   "U hắc tố ác tính (Melanoma)",
    "NV":    "Nốt ruồi lành tính",
    "VASC":  "Tổn thương mạch máu",
}
```

### 5.2 Các hằng số cấu hình CSS định hình giao diện UI
Để tối ưu hóa giao diện Streamlit, tránh hiện tượng đè chữ, tràn dòng khi hiển thị trên các thiết bị màn hình khác nhau và loại bỏ hiệu ứng làm mờ giao diện (dimming effect) gây khó chịu cho bác sĩ khi mô hình đang chạy nền, hệ thống chèn trực tiếp mã CSS tùy chỉnh:

```css
/* Tránh tràn từ và đè chữ trong hộp thoại chat VQA */
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li {
    word-break: break-word;
    overflow-wrap: break-word;
}

/* Đặt giới hạn chiều cao cho khung hiển thị VQA Chat và thanh cuộn */
[data-testid="stChatMessageContent"] {
    max-height: 420px;
    overflow-y: auto;
}

/* Chuẩn hóa kích thước font chữ hiển thị của các thẻ chỉ số định lượng */
[data-testid="stMetricValue"] {
    font-size: 1.1rem !important;
}

/* Vô hiệu hóa hiệu ứng làm mờ giao diện khi Streamlit chạy lại code */
div[data-testid="stAppViewContainer"] {
    opacity: 1 !important;
    filter: none !important;
}
[data-st-mode="running"] {
    opacity: 1 !important;
}
div.element-container {
    opacity: 1 !important;
}
.stApp.running, [data-st-mode="running"] * {
    opacity: 1 !important;
}
```

### 5.3 Cơ chế Ghi nhật ký ẩn phục vụ giám sát kỹ thuật (Dev-only System Logging)
Hệ thống không hiển thị log kỹ thuật lên UI để tránh gây nhiễu cho bác sĩ. Thay vào đó, toàn bộ dữ liệu giao tiếp thô giữa người dùng, kết quả mô hình CV và chuỗi prompt gửi lên LLM được ghi tự động vào tệp tin nhật ký cục bộ `5_Results/system_logs.log` dưới dạng JSON thô.

* **Bộ mã hóa numpy an toàn (`_NumpySafeEncoder`)**:
  Do các thư viện OpenCV và PyTorch trả kết quả về dạng kiểu dữ liệu numpy (ví dụ: `np.float32`, `np.int64`, ma trận `np.ndarray`), các kiểu dữ liệu này không thể được giải tuần tự hóa (serialize) trực tiếp bằng thư viện `json` tiêu chuẩn của Python. Hệ thống thiết kế lớp chuyển đổi dữ liệu tùy biến:
  ```python
  class _NumpySafeEncoder(json.JSONEncoder):
      def default(self, obj):
          if isinstance(obj, (np.integer,)):
              return int(obj)
          if isinstance(obj, (np.floating,)):
              return float(obj)
          if isinstance(obj, np.ndarray):
              return obj.tolist()
          return super().default(obj)
  ```
* **Hàm ghi log `write_dev_log(data: Dict[str, Any], action_type: str)`**:
  Hàm tự động đóng gói dữ liệu cùng nhãn thời gian thực tế, chuyển hóa thành chuỗi văn bản JSON không đổi dấu và ghi tiếp nối (append) vào cuối tệp tin nhật ký:
  ```python
  log_entry = {
      "action_time": timestamp,
      "action_type": action_type,
      "payload": data,
  }
  ```

### 5.4 Phân tích toán học hàm mất mát (Loss Function) của VQA Offline
Trong quá trình Joint Fine-tuning mô hình offline `CPUMedicalVQAModel`, hệ thống sử dụng hàm mất mát hồi quy ngôn ngữ **Causal Language Modeling Loss**:
* Định nghĩa chuỗi tokens đầu vào có chiều dài $T$. Tại mỗi bước thời gian $t$, mô hình nhận chuỗi các token trước đó $x_{<t}$ và dự đoán phân phối xác suất cho token tiếp theo $x_t$.
* Hàm mất mát được tính bằng phép toán **Cross Entropy Loss** trên các token thuộc chuỗi câu trả lời (các token câu hỏi và token ảnh được gán nhãn $-100$ để bỏ qua trong quá trình tính gradient):
  $$\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t \in \mathcal{A}} \log P(x_t^{(i)} \mid x_{<t}^{(i)}; \theta)$$
  Trong đó $\mathcal{A}$ là tập chỉ số các token thuộc chuỗi câu trả lời (Answer tokens), $N$ là kích thước lô (batch size), và $\theta$ là tập hợp các tham số có thể huấn luyện (các trọng số LoRA của DistilGPT-2 và trọng số của các lớp Projection Layer cùng khối CBAM).

---

## 6. PHÂN TÍCH TOÁN HỌC TRÍCH XUẤT ĐẶC TRƯNG HÌNH HỌC (ABCD Metrics in Computer Vision)

Để mô phỏng quy trình chẩn đoán lâm sàng ABCD truyền thống của bác sĩ da liễu, hệ thống thực hiện trích xuất 4 tham số hình học quan trọng từ mặt nạ tổn thương nhị phân $M(y, x) \in \{0, 1\}$:

### 6.1 Asymmetry (Hệ số bất đối xứng - A)
* **Tính toán tâm khối (Centroid)**:
  Sử dụng các moment hình học bậc 0 và bậc 1 của đường bao tổn thương lớn nhất $\mathcal{C}$ để tìm tọa độ trọng tâm $(c_x, c_y)$ của vùng tổn thương:
  $$M_{00} = \sum_{y} \sum_{x} M(y, x)$$
  $$M_{10} = \sum_{y} \sum_{x} x \cdot M(y, x), \quad M_{01} = \sum_{y} \sum_{x} y \cdot M(y, x)$$
  $$c_x = \frac{M_{10}}{M_{00}}, \quad c_y = \frac{M_{01}}{M_{00}}$$
* **Phép lật đối xứng và tính sai lệch**:
  Hệ thống thực hiện cắt mặt nạ thành các phần dọc theo hai trục vuông góc đi qua trọng tâm $(c_x, c_y)$.
  * **Trục ngang (Horizontal split)**: Chia mặt nạ thành nửa trên $M_{\text{top}}$ và nửa dưới $M_{\text{bottom}}$. Thực hiện lật dọc nửa dưới $M_{\text{bottom\_flipped}} = \text{FlipY}(M_{\text{bottom}})$ và căn đệm kích thước tương đồng. Tính sai số tuyệt đối về mặt diện tích:
    $$\text{diff}_h = \sum_{y} \sum_{x} \left| M_{\text{top\_padded}}(y, x) - M_{\text{bottom\_flipped\_padded}}(y, x) \right|$$
  * **Trục dọc (Vertical split)**: Chia mặt nạ thành nửa trái $M_{\text{left}}$ và nửa phải $M_{\text{right}}$. Thực hiện lật ngang nửa phải $M_{\text{right\_flipped}} = \text{FlipX}(M_{\text{right}})$. Tính sai số tuyệt đối:
    $$\text{diff}_v = \sum_{y} \sum_{x} \left| M_{\text{left\_padded}}(y, x) - M_{\text{right\_flipped\_padded}}(y, x) \right|$$
* **Chuẩn hóa điểm bất đối xứng**:
  Điểm bất đối xứng tổng hợp được chuẩn hóa về khoảng $[0, 1]$ dựa trên diện tích vùng tổn thương thực tế:
  $$\text{Asymmetry Score} = \min\left(1.0, \frac{\text{diff}_h + \text{diff}_v}{2 \cdot M_{00}}\right)$$
  * *Ý nghĩa lâm sàng*: Điểm số bằng $0$ thể hiện tổn thương đối xứng hoàn hảo trên cả 2 trục (lành tính), điểm số tiến gần về $1.0$ thể hiện mức độ bất đối xứng cực kỳ cao (dấu hiệu melanoma).

### 6.2 Border Complexity (Độ phức tạp biên tổn thương - B)
* **Tính toán**:
  Đo lường mức độ gồ ghề, răng cưa hoặc tua rua của đường viền tổn thương da. Chỉ số được tính dựa trên tỷ lệ giữa chu vi tổn thương ($P$) và căn bậc hai diện tích của nó ($A = M_{00}$):
  $$\text{Border Complexity} = \frac{P}{\sqrt{A}}$$
  Trong đó, chu vi $P$ là độ dài đường bao lớn nhất $\mathcal{C}$ tính bằng thuật toán xấp xỉ chuỗi điểm biên của OpenCV (`cv2.arcLength`).
  * *Ý nghĩa lâm sàng*: Đối với một hình tròn hoàn hảo, tỷ lệ này đạt giá trị tối thiểu $\approx 2\sqrt{\pi} \approx 3.54$. Tổn thương lành tính thường có biên mịn màng nên chỉ số này dao động thấp từ $3.5 \rightarrow 5.0$. Các tổn thương ác tính phát triển mất kiểm soát ra xung quanh tạo các cạnh hình răng cưa phức tạp, khiến chỉ số này tăng vọt vượt lên $\ge 6.0$.

### 6.3 Circularity (Độ tròn tổn thương - C)
* **Tính toán**:
  Chỉ số độ tròn đánh giá mức độ tương đồng giữa hình dạng tổn thương với một hình tròn lý tưởng:
  $$\text{Circularity} = \frac{4\pi \cdot A}{P^2}$$
  Giá trị Circularity được giới hạn nghiêm ngặt trong đoạn $[0.0, 1.0]$.
  * *Ý nghĩa lâm sàng*: Nốt ruồi lành tính thông thường (`NV`) hầu như có dạng tròn hoặc oval rất đều, chỉ số Circularity sẽ tiệm cận sát $1.0$. Ngược lại, các mảng ung thư tế bào đáy hoặc u hắc tố có dạng méo mó, kéo dài dị hình làm chỉ số Circularity rơi sâu xuống sát $0.0$.

### 6.4 Area Ratio (Tỉ lệ diện tích tổn thương - D)
* **Tính toán**:
  Tỉ số diện tích được dùng để ước lượng gián tiếp kích thước của tổn thương trên vùng da thu nhận được qua ống kính camera:
  $$\text{Area Ratio} = \frac{M_{00}}{H \times W}$$
  Với $H \times W$ là tổng số pixel của toàn bộ ảnh đầu vào.
  * *Ý nghĩa lâm sàng*: Kết hợp cùng loại ảnh phát hiện để xác định xem tổn thương có kích thước quá lớn (dấu hiệu lan rộng nguy hiểm của các mảng sừng hóa BKL lớn hoặc Melanoma thời kỳ muộn) hoặc quá nhỏ dưới ngưỡng phân tích tin cậy của mô hình AI.
