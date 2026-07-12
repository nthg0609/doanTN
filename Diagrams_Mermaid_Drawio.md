# HƯỚNG DẪN IMPORT SƠ ĐỒ MERMAID VÀO DRAW.IO (TẠO BIỂU ĐỒ VECTOR CHỈNH SỬA ĐƯỢC)

Để chuyển các sơ đồ kỹ thuật dưới đây thành dạng sơ đồ vector sắc nét, có thể tự do chỉnh sửa màu sắc, cỡ chữ trên Draw.io, bạn hãy làm theo các bước sau:

1. Truy cập trang web **[draw.io](https://app.diagrams.net/)** (hoặc mở phần mềm Draw.io trên máy tính).
2. Tạo một bản vẽ mới hoặc mở bản vẽ hiện có.
3. Trên thanh công cụ phía trên, nhấn vào biểu tượng dấu cộng **`+` (Insert)** -> chọn **`Advanced`** -> chọn **`Mermaid`**.
4. Copy toàn bộ đoạn mã Mermaid của sơ đồ tương ứng dưới đây và paste vào khung soạn thảo của Draw.io.
5. Nhấn nút **`Insert`** (hoặc **`Chèn`**). Draw.io sẽ tự động dựng thành sơ đồ vector 100% chỉnh sửa được cho bạn!

---

## 1. Sơ đồ Kiến trúc Tổng quan Hệ thống (Slide 7)
Sơ đồ mô tả luồng dữ liệu song song độc lập từ ảnh đầu vào qua Safety Gate và phân nhánh xử lý trước khi ra quyết định lâm sàng.

```mermaid
graph TD
    %% Định nghĩa phong cách màu sắc HUST (Đỏ - Trắng - Xám)
    classDef hustRed fill:#990000,stroke:#660000,stroke-width:2px,color:#fff;
    classDef hustWhite fill:#fff,stroke:#990000,stroke-width:2px,color:#1e293b;
    classDef hustGray fill:#f1f5f9,stroke:#94a3b8,stroke-width:1.5px,color:#1e293b;

    Input["Ảnh Da Đầu Vào (RGB / DICOM)"]:::hustWhite
    
    subgraph SG ["Bộ lọc an toàn tiền xử lý (Safety Gate)"]
        BlurCheck["Kiểm tra độ mờ (Laplacian Var < Threshold)"]:::hustGray
        ExposureCheck["Kiểm tra phơi sáng (Fitzpatrick Scale)"]:::hustGray
        BlurCheck --> ExposureCheck
    end
    
    Input --> SG
    
    SG -->|Đạt chất lượng| Split{Tách Nhánh Song Song}:::hustRed
    SG -->|Ảnh lỗi| Reject["Yêu cầu chụp lại (Triage)"]:::hustRed

    %% Nhánh 1: Hình học
    subgraph SegBranch ["Nhánh 1: Phân đoạn & Chỉ số ABCD"]
        SegAlg["Phân đoạn tổn thương (SAM / DeepLabV3+ / OTSU)"]:::hustGray
        ABCDCalc["Đo đạc chỉ số hình học ABCD (Asymmetry, Border, Color, Diameter)"]:::hustGray
        SegAlg --> ABCDCalc
    end
    
    %% Nhánh 2: Phân loại
    subgraph ClsBranch ["Nhánh 2: Phân loại & Dịch tễ"]
        Classifier["Phân loại ảnh (EfficientNet-B1 + CBAM)"]:::hustGray
        LateFusion["Late Fusion Bayes (Hợp nhất xác suất với Tuổi, Giới tính, Vị trí)"]:::hustGray
        Classifier --> LateFusion
    end
    
    Split --> SegAlg
    Split --> Classifier
    
    %% Post-processing
    PostSG["Hậu kiểm lâm sàng (Safety Gate Post-check)"]:::hustWhite
    ABCDCalc --> PostSG
    LateFusion --> PostSG
    
    %% Output
    PostSG -->|Bệnh lý ác tính| UrgentTriage["Cảnh báo chuyển tuyến cấp thiết (Triage)"]:::hustRed
    PostSG -->|Lành tính / Tham vấn| VQA["Trợ lý đàm thoại VQA y văn RAG (DistilGPT-2 LoRA)"]:::hustWhite
    
    VQA --> OutputEHR["Lưu trữ CSDL Firestore EHR & Xuất báo cáo PDF"]:::hustWhite
```

---

## 2. Biểu đồ Use Case Chức năng Hệ thống (Slide 8)
Biểu đồ mô tả quyền hạn và các ca sử dụng của Bác sĩ lâm sàng trực tiếp trên giao diện CDSS (không có hệ thống phân quyền phức tạp).

```mermaid
left-to-right direction
graph TD
    %% Quyết định màu sắc
    classDef actor fill:#990000,stroke:#660000,stroke-width:2px,color:#fff;
    classDef usecase fill:#fff,stroke:#990000,stroke-width:2px,color:#1e293b;

    %% Actors
    Doc(("Bác sĩ lâm sàng")):::actor

    %% Use Cases
    ConfigSG["Cấu hình cổng an toàn (Safety Gate)"]:::usecase
    Upload["Tải ảnh / tệp DICOM"]:::usecase
    Segment["Phân đoạn tương tác (SAM / GrabCut)"]:::usecase
    ABCD["Xem kết quả ABCD & Biện giải lâm sàng"]:::usecase
    Fusion["Tùy chỉnh tham số Bayes (Slider Lambda)"]:::usecase
    ChatVQA["Hỏi đáp y văn ngoại tuyến (VQA Bot)"]:::usecase
    EHR["Đồng bộ & Quản lý bệnh án EHR"]:::usecase
    Timeline["Xem biểu đồ tiến triển timeline bệnh lý"]:::usecase
    PDF["Xuất báo cáo PDF bệnh nhân"]:::usecase

    %% Relationships
    Doc --> ConfigSG
    Doc --> Upload
    Doc --> Segment
    Doc --> ABCD
    Doc --> Fusion
    Doc --> ChatVQA
    Doc --> EHR
    EHR -.->|include| Timeline
    EHR -.->|include| PDF
```

---

## 3. Sơ đồ Mã hóa Bảo mật EHR & Tối ưu hóa Lưu trữ (Slide 14)
Sơ đồ chi tiết luồng mã hóa thông tin nhạy cảm của bệnh nhân tuân thủ tiêu chuẩn HIPAA và kỹ thuật song song hóa ThreadPool.

```mermaid
graph TD
    classDef hustRed fill:#990000,stroke:#660000,stroke-width:2px,color:#fff;
    classDef hustWhite fill:#fff,stroke:#990000,stroke-width:2px,color:#1e293b;
    classDef hustGray fill:#f1f5f9,stroke:#94a3b8,stroke-width:1.5px,color:#1e293b;

    RawData["Thông tin bệnh án nhạy cảm (Tên, CCCD, Bệnh lý)"]:::hustWhite
    
    %% Phân luồng bảo mật
    RawData --> SplitSec{Phân luồng bảo mật}:::hustRed
    
    %% Mã hóa ID
    SplitSec -->|Định danh| HashID["Băm SHA-256 (Tạo Document ID duy nhất)"]:::hustGray
    
    %% Mã hóa dữ liệu
    SplitSec -->|Dữ liệu y khoa| EncryptData["Mã hóa đối xứng XOR + Base64 (Ẩn danh thông tin)"]:::hustGray
    
    HashID --> SyncEHR["Đồng bộ lên Cloud Firestore (Tuân thủ HIPAA)"]:::hustWhite
    EncryptData --> SyncEHR
    
    %% Nhánh ảnh
    subgraph ThreadExec ["ThreadPoolExecutor (Tải lên song song)"]
        Thread1["Thread 1: Tải ảnh gốc RGB lên Firebase Storage"]:::hustGray
        Thread2["Thread 2: Tải ảnh mặt nạ tổn thương lên Firebase Storage"]:::hustGray
        Thread3["Thread 3: Tải ảnh nhiệt Grad-CAM lên Firebase Storage"]:::hustGray
    end
    
    RawData -->|Tài nguyên ảnh| ThreadExec
    ThreadExec -->|Đường dẫn URL| SyncEHR
    
    SyncEHR --> Finish["Hoàn thành lưu trữ (Thời gian UI lock giảm từ 5s xuống 1.5s)"]:::hustWhite
```

---

## 4. Biểu đồ lớp UML (Class Diagram) cấu trúc Phần mềm (Slide 15)
Biểu đồ lớp UML thể hiện thiết kế hướng đối tượng hoàn chỉnh của pipeline xử lý y tế.

```mermaid
classDiagram
    class UnifiedDermatologyPipeline {
        +SafetyGate safety_gate
        +InteractiveSegmenter segmenter
        +ModelRegistry registry
        +run(image_path, age, gender, lambda_val) InferenceResult
        -_safe_load_rgb(image_path) ndarray
        -_segment(image, type) ndarray
        -_classify(image, age, gender, lambda) dict
    }

    class SafetyGate {
        +SafetyGateConfig config
        +evaluate(metrics, cls_confidence, image_type) GateResult
        -_check_blur(image) bool
        -_check_exposure(image) bool
    }

    class InteractiveSegmenter {
        +DeepLabV3Model model
        +segment_by_point(image, pt_x, pt_y) ndarray
        -_apply_grabcut_refinement(image, mask) ndarray
        -_run_otsu_fallback(image) ndarray
    }

    class MultimodalBayesianFusion {
        +EfficientNetB1 classifier
        +dict demographics_prior
        +fuse(img_probs, patient_data, lambda_val) dict
        -_calculate_posterior(prior, likelihood) ndarray
    }

    class EHRManager {
        +FirestoreClient db
        +encrypt_patient_data(dict_data) dict
        +sync_record(patient_id, record_data) bool
        +generate_report_pdf(record_data) str
    }

    %% Mối quan hệ
    UnifiedDermatologyPipeline --> SafetyGate : sử dụng để tiền kiểm/hậu kiểm
    UnifiedDermatologyPipeline --> InteractiveSegmenter : gọi phân đoạn ảnh
    UnifiedDermatologyPipeline --> MultimodalBayesianFusion : gọi phân loại & fusion
    UnifiedDermatologyPipeline --> EHRManager : gọi để mã hóa & lưu CSDL
```

---

## 5. Quy trình đàm thoại VQA y văn & RAG Ngoại tuyến (Slide 12)
Sơ đồ thể hiện cách truy xuất thông tin hướng dẫn điều trị của Bộ Y tế từ ChromaDB và tích hợp màng lọc an toàn thuốc (Medication Guardrails) trước khi sinh câu trả lời.

```mermaid
graph TD
    classDef hustRed fill:#990000,stroke:#660000,stroke-width:2px,color:#fff;
    classDef hustWhite fill:#fff,stroke:#990000,stroke-width:2px,color:#1e293b;
    classDef hustGray fill:#f1f5f9,stroke:#94a3b8,stroke-width:1.5px,color:#1e293b;

    Q["Câu hỏi y khoa từ Bác sĩ (Giọng nói / Văn bản)"]:::hustWhite
    
    subgraph OfflineRAG ["Hệ thống RAG ngoại tuyến (Offline RAG)"]
        Embed["Mã hóa câu hỏi (All-MiniLM-L6-v2)"]:::hustGray
        Chroma["Truy vấn ChromaDB (Kho tài liệu Bộ Y tế)"]:::hustGray
        Context["Trích xuất ngữ cảnh liên quan nhất (Cosine Similarity)"]:::hustGray
        Embed --> Chroma
        Chroma --> Context
    end
    
    Q --> Embed
    
    subgraph LLM ["Mô hình ngôn ngữ lớn cục bộ (Local LLM)"]
        Prompt["Prompt Injection (Nhồi câu hỏi + Ngữ cảnh y văn chuẩn)"]:::hustGray
        Model["DistilGPT-2 LoRA / Ollama (Qwen 3B/7B)"]:::hustGray
        Prompt --> Model
    end
    
    Context --> Prompt
    
    subgraph Guard ["Màng lọc an toàn (Medication Guardrails)"]
        Rule["Kiểm tra từ khóa kê đơn thuốc nhạy cảm"]:::hustGray
        Block{"Phát hiện tự kê đơn?"}:::hustRed
        Refuse["Chặn & Fallback: 'Hệ thống không kê đơn thuốc. Hãy tham khảo ý kiến bác sĩ chuyên khoa.'"]:::hustRed
        Accept["Đạt yêu cầu an toàn"]:::hustWhite
        Rule --> Block
        Block -->|Có| Refuse
        Block -->|Không| Accept
    end
    
    Model --> Rule
    
    Accept --> Answer["Sinh câu trả lời tư vấn chuẩn y văn (Hiển thị văn bản)"]:::hustWhite
```

---

## 6. Sơ đồ luồng Hợp nhất xác suất Late Fusion Bayes (Slide 11)
Sơ đồ biểu diễn luồng dịch tễ học Toán - Tin giúp kiểm soát thiên lệch Hamlin10000 thông qua trọng số λ.

```mermaid
graph TD
    classDef hustRed fill:#990000,stroke:#660000,stroke-width:2px,color:#fff;
    classDef hustWhite fill:#fff,stroke:#990000,stroke-width:2px,color:#1e293b;
    classDef hustGray fill:#f1f5f9,stroke:#94a3b8,stroke-width:1.5px,color:#1e293b;

    Img["Ảnh tổn thương da (RGB)"]:::hustWhite
    Demo["Demographics bệnh nhân (Tuổi, Giới tính, Vị trí)"]:::hustWhite
    
    subgraph Vision ["Nhánh thị giác máy tính"]
        CNN["EfficientNet-B1 + CBAM"]:::hustGray
        P_Vision["Xác suất hình ảnh: P(C | Image)"]:::hustGray
        CNN --> P_Vision
    end
    
    subgraph Math ["Nhánh dịch tễ y học (Toán - Tin)"]
        Prior["Bảng phân phối tiền nghiệm dịch tễ: P(C | Demo)"]:::hustGray
    end
    
    Img --> CNN
    Demo --> Prior
    
    subgraph Fusion ["Bộ điều phối late fusion Bayes"]
        Lambda["Nhập trọng số Lambda (λ ∈ [0,1] kiểm soát thiên lệch HAM10000)"]:::hustGray
        Bayes["Tính xác suất hậu nghiệm (Posterior Probability)"]:::hustGray
        Lambda --> Bayes
    end
    
    P_Vision --> Bayes
    Prior --> Bayes
    
    Bayes --> Output["Xác suất hợp nhất cuối cùng & Grad-CAM giải thích"]:::hustWhite
```

---

## 7. Giải thuật Phân đoạn tương tác SAM + GrabCut & Dự phòng OTSU (Slide 9)
Sơ đồ giải thuật phân vùng tổn thương nâng cao giúp đo đạc chỉ số ABCD ổn định ngay cả khi mô hình Deep Learning gặp ảnh lỗi.

```mermaid
graph TD
    classDef hustRed fill:#990000,stroke:#660000,stroke-width:2px,color:#fff;
    classDef hustWhite fill:#fff,stroke:#990000,stroke-width:2px,color:#1e293b;
    classDef hustGray fill:#f1f5f9,stroke:#94a3b8,stroke-width:1.5px,color:#1e293b;

    Img["Ảnh chụp vùng da tổn thương"]:::hustWhite
    Click["Click chuột mồi của Bác sĩ (Interactive Point)"]:::hustWhite
    
    Click --> InitPoint{"Kiểm tra tọa độ nhấp"}:::hustRed
    
    subgraph SAMBranch ["Phân đoạn tương tác nâng cao"]
        SAM["Bộ phân đoạn SAM (Sinh mặt nạ mồi ban đầu)"]:::hustGray
        GrabCut["Thuật toán GrabCut (GC_INIT_WITH_MASK tối ưu biên bờ tinh tế)"]:::hustGray
        SAM --> GrabCut
    end
    
    InitPoint -->|Có điểm nhấp| SAM
    InitPoint -->|Không nhấp| DeepLab["Chạy tự động DeepLabV3+ (Multi-scale TTA)"]:::hustGray
    
    GrabCut --> CheckSize{"Kiểm tra kích thước mặt nạ (Sum < 100 pixels)?"}:::hustRed
    DeepLab --> CheckSize
    
    subgraph Fallback ["Giải pháp dự phòng y khoa"]
        OTSU["Thuật toán phân ngưỡng OTSU nghịch đảo (cv2.THRESH_BINARY_INV)"]:::hustGray
    end
    
    CheckSize -->|Có (Mặt nạ quá nhỏ/lỗi)| OTSU
    CheckSize -->|Không (Đạt chuẩn)| Output["Mặt nạ phân đoạn tối ưu (Lesion Mask)"]:::hustWhite
    
    OTSU --> Output
    
    Output --> ABCD["Tính toán chỉ số ABCD y văn & trích xuất DICOM PixelSpacing"]:::hustWhite
```

