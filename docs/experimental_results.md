# KẾT QUẢ THỰC NGHIỆM VÀ SỐ LIỆU ĐỊNH LƯỢNG HỆ THỐNG CHẨN ĐOÁN DA LIỄU

Tài liệu này tổng hợp toàn bộ các thông số huấn luyện, kết quả định lượng, phân bố lớp bệnh lý và thời gian suy luận benchmark của hệ thống chẩn đoán da liễu đa phương thức. Dữ liệu này được trích xuất trực tiếp từ logs, checkpoints và môi trường thực nghiệm của hệ thống để làm chất liệu viết chương Kết quả Thực nghiệm của Luận văn tốt nghiệp.

---

## 1. Bài toán phân đoạn tổn thương da (Lesion Segmentation)

Đánh giá trên tập kiểm thử ISIC 2018 (390 mẫu):

| Phương pháp / Mô hình | Dice Score | IoU | Số Epochs huấn luyện | Trạng thái dừng | Epoch tốt nhất |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **U-Net (ResNet34) đơn lẻ** | 0.894260 | 0.817698 | 29 / 50 | Early Stopping | 18 |
| **DeepLabV3+ (ResNet50) đơn lẻ** | 0.912763 | 0.845523 | 42 / 50 | Early Stopping | 31 |
| **Hybrid U-Net (Ablation)** | 0.889853 | 0.811094 | - | - | - |
| **Hybrid DeepLabV3+ (Ablation)** | 0.909266 | 0.843339 | - | - | - |
| **Hợp nhất trung bình (Average/Weighted)** | 0.901974 | 0.830662 | - | - | - |
| **Hybrid-Max Fusion (Đề xuất)** | **0.913169** | **0.847019** | - | - | - |

*Ghi chú:* 
- Hàm mất mát sử dụng: $\mathcal{L}_{seg} = 0.6 \cdot \mathcal{L}_{Dice} + 0.4 \cdot \mathcal{L}_{BCE}$.
- Thuật toán tối ưu: **Adam**, Learning Rate bắt đầu = $10^{-4}$, sử dụng scheduler `ReduceLROnPlateau` với factor = 0.5.

---

## 2. Bài toán phân loại bệnh lý da liễu (Lesion Classification)

Đánh giá trên tập kiểm thử HAM10000 ROI (3.005 mẫu):

### 2.1. Chỉ số tổng quan mô hình (EfficientNet-B0 + CBAM Attention)
- **Độ chính xác kiểm thử (Test Accuracy):**
  * *Mô hình tối ưu (Archived):* **96.51%** (Best Val Accuracy: **97.50%**, Test Loss: **0.2255**, 50 Epochs).
  * *Mô hình hiện tại:* **95.01%** (Best Val Accuracy: **96.90%**, Test Loss: **0.2764**, 38 Epochs).

### 2.2. Báo cáo hiệu năng chi tiết theo lớp bệnh lý (Lấy từ mô hình Baseline 88.65% Accuracy)
Được trích xuất từ test split HAM10000 (2.766 mẫu):

| Lớp bệnh lý | Precision | Recall | F1-Score | Số mẫu (Support) |
| :--- | :---: | :---: | :---: | :---: |
| **AKIEC** (Dày sừng ánh sáng / Tiền ung thư) | 0.7209 | 0.7750 | 0.7470 | 80 |
| **BCC** (Ung thư biểu mô tế bào đáy) | 0.8714 | 0.8714 | 0.8714 | 140 |
| **BKL** (Dày sừng lành tính) | 0.7467 | 0.7619 | 0.7542 | 294 |
| **DF** (U xơ da) | 0.9167 | 0.7857 | 0.8462 | 28 |
| **MEL** (U hắc tố ác tính) | 0.7568 | 0.7417 | 0.7492 | 302 |
| **NV** (Nốt ruồi lành tính) | 0.9383 | 0.9373 | 0.9378 | 1882 |
| **VASC** (Tổn thương mạch máu) | 0.8500 | 0.8500 | 0.8500 | 40 |
| **Toàn hệ thống (Accuracy)** | | | **0.8865** | **2766** |
| **Macro Average** | 0.8287 | 0.8176 | 0.8222 | 2766 |
| **Weighted Average** | 0.8869 | 0.8865 | 0.8866 | 2766 |

---

## 3. Trợ lý y tế hội thoại (CPUMedicalVQAModel - DistilGPT-2 + LoRA)

### 3.1. Phân tích tham số mạng VQA
- **Tổng tham số mô hình (All Parameters):** **90,352,514**
- **Tham số huấn luyện được (Trainable Parameters):** **1,926,754** (Adapter LoRA hạng $r=8$ nhúng tại các lớp `c_attn` + Projection layer)
- **Tỷ lệ phần trăm tham số huấn luyện:** **2.1325%**

### 3.2. Đường cong hội tụ Loss (Causal Language Modeling Loss)
- **Epoch 1 (Khởi đầu):** Train Loss = **2.9062** | Val Loss = **2.7685**
- **Epoch 12 (Tối ưu):** Train Loss = **2.2140** | Val Loss = **2.1209**
- **Epoch 15 (Dừng):** Train Loss = **2.1778** | Val Loss = **2.1466** (mô hình bắt đầu có xu hướng quá khớp - overfitting nhẹ trên tập xác thực).

### 3.3. Đánh giá Định lượng chất lượng câu trả lời VQA (Quantitative BLEU Evaluation)
Để đánh giá khả năng sinh văn bản của trợ lý hội thoại y khoa VQA, chúng tôi thực hiện đo lường chỉ số BLEU (Bilingual Evaluation Understudy) trên tập dữ liệu kiểm thử gồm 12 mẫu câu hỏi lâm sàng da liễu. Thực nghiệm so sánh giữa hai mô hình:
1. **Mô hình Ngoại tuyến (Offline VQA Model):** `CPUMedicalVQAModel` (DistilGPT-2 + Linear Projection layer + LoRA fine-tuning).
2. **Mô hình Trực tuyến (Online VQA Model):** `GPT-4o-mini` kết hợp cơ chế `RAG Prompt Fusion` (truy xuất tri thức Bộ Y tế và nhúng kết quả chẩn đoán của Computer Vision).

**Bảng so sánh hiệu năng VQA:**

| Mô hình VQA | Số lượng mẫu kiểm thử (Val Samples) | BLEU-1 trung bình | BLEU-2 trung bình | Ghi chú |
| :--- | :---: | :---: | :---: | :--- |
| **CPUMedicalVQAModel (Offline)** | 12 | **0.7269** | **0.6812** | Khớp mẫu câu trả lời huấn luyện tốt, câu trả lời ngắn gọn. |
| **Online VQA (GPT-4o-mini + RAG)** | 12 | **0.1091** | **0.0538** | Sinh văn bản tự nhiên, chi tiết, độ chính xác y khoa cao nhưng không khớp từ ngữ gốc. |

### 3.4. Phân tích chi tiết chất lượng câu trả lời trên 12 mẫu kiểm thử (Qualitative & Error Analysis)
Dựa trên kết quả thực nghiệm chi tiết từ tập kiểm thử 12 mẫu, chúng tôi thực hiện phân tích chất lượng câu trả lời của mô hình ngoại tuyến (`CPUMedicalVQAModel`) theo 3 nhóm trường hợp cụ thể:

#### Nhóm 1: Các trường hợp chính xác và trùng khớp cao với Ground Truth (BLEU-1 >= 0.80)
Các mẫu đạt điểm tuyệt đối hoặc tiệm cận tuyệt đối như:
- **Mẫu 2 (NV - "What type of mole is this?"):** Đạt BLEU-1 = **1.00**, BLEU-2 = **1.00**. Mô hình tái hiện chính xác câu trả lời chuẩn lâm sàng về nốt ruồi lành tính (Melanocytic nevus), khuyên tự theo dõi tại nhà.
- **Mẫu 7 (BKL - "What is this brown growth?"):** Đạt BLEU-1 = **0.9683**, BLEU-2 = **0.9683**. Mô hình mô tả chính xác seborrheic keratosis lành tính, tuy nhiên bị dừng (truncate) nhẹ ở từ cuối do giới hạn độ dài (`becomes` thay vì `becomes irritated`).
- **Mẫu 4 (BCC - "Is this serious?"):** Đạt BLEU-1 = **0.9100**, BLEU-2 = **0.8468**. Mô hình nhận diện đúng Basal Cell Carcinoma và khuyên điều trị ngoại khoa sớm. Tuy nhiên, ở cuối câu trả lời, mô hình gặp hiện tượng lặp cụm từ (repetition loop) do bộ suy luận cục bộ: `"Early detection and removal is recommended. Early"`.
- **Mẫu 8 (NV - "Is this mole dangerous?"):** Đạt BLEU-1 = **0.8752**, BLEU-2 = **0.8752**. Mô hình khẳng định nốt ruồi lành tính và hướng dẫn theo dõi định kỳ.

#### Nhóm 2: Các trường hợp trả lời đúng hướng nhưng bị cắt cụt (Truncated Responses - BLEU-1 từ 0.50 đến 0.80)
Đây là các trường hợp mô hình cục bộ nắm bắt đúng kiến trúc ngữ nghĩa và sinh câu trả lời bám sát nhãn chẩn đoán, nhưng bị dừng đột ngột trước khi hoàn thành câu do giới hạn chiều dài phát sinh (`max_new_tokens = 50` hoặc `max_length` của token generator):
- **Mẫu 1 (Melasma - "What causes these dark patches?"):** Đạt BLEU-1 = **0.6514**. Câu trả lời bị cắt cụt ở cụm từ `"Treatment includes"`.
- **Mẫu 3 & 5 (Melanoma - "Is this mole cancerous?"):** Đạt BLEU-1 = **0.6807**. Mô hình nhận diện chính xác các dấu hiệu cảnh báo ABCDE của u hắc tố ác tính, khuyên khám da liễu gấp nhưng bị dừng ở `"Immediate dermatological"`.
- **Mẫu 9 (SCC - "What is this scaly lesion?"):** Đạt BLEU-1 = **0.7510**. Nhận dạng đúng ung thư biểu mô tế bào vảy (SCC) nhưng bị dừng ở `"likely to metastas"`.
- **Mẫu 12 (Melanoma - "Is it dangerous?"):** Đạt BLEU-1 = **1.00**, BLEU-2 = **0.9832** (tính trên chuỗi sinh ra thực tế), mô hình trả lời đầy đủ ý và lặp lại nhẹ cụm từ cuối: `"Professional evaluation"`.

#### Nhóm 3: Các trường hợp lỗi y khoa và nhiễu thông tin (Errors & Hallucinations - BLEU-1 < 0.50)
Phân tích sâu các ca thất bại giúp làm rõ các giới hạn nội tại của mô hình ngoại tuyến quy mô nhỏ:
- **Mẫu 6 & 10 (Wart - "What is this hard bump?"):** Đạt BLEU-1 = **0.5444**, BLEU-2 = **0.3689**. Mặc dù nhận diện được từ khóa chính là `"wart"` (mụn cóc), mô hình sinh ra thông tin nhiễu y khoa (hallucination): `"contagious and treatable with topical or oral antifungal medications..."` (điều trị nấm thay vì điều trị virus HPV gây mụn cóc). Nguyên nhân của sự nhầm lẫn này có thể do kích thước tập dữ liệu huấn luyện VQA quá nhỏ (khoảng 74-80 mẫu), khiến biểu diễn nhúng ngữ nghĩa của các dạng tổn thương lành tính dạng bump (như nốt sừng hóa, mụn cóc, nấm da) bị chồng lấn trong không gian ẩn của bộ mã hóa DistilGPT-2.
- **Mẫu 11 (BCC - "Is this serious?"):** Đạt BLEU-1 = **0.1167**, BLEU-2 = **0.0000**. Đây là ca lỗi y khoa nghiêm trọng nhất (False Negative). Đối với ảnh tổn thương BCC (Ung thư biểu mô tế bào đáy), mô hình ngoại tuyến lại sinh câu trả lời mô tả một nốt ruồi lành tính: `"This appears to be a benign nevus (mole) with regular, symmetric borders..."`.
  - *Giải thích nguyên nhân:* Qua phân tích dữ liệu phân vùng và phân loại đầu vào của mẫu này, chúng tôi nhận thấy mô hình thị giác máy tính (`EfficientNet-B1 + CBAM`) dự đoán nhãn BCC nhưng với độ tin cậy thấp, hoặc các chỉ số hình học trích xuất từ mặt nạ phân đoạn có độ bất đối xứng rất nhỏ. Bộ chiếu tuyến tính (Projection layer) khi chuyển đổi vector ảnh này có xu hướng định hướng decoder nghiêng về phía lớp bệnh lý phổ biến nhất trong tập train là NV (Nốt ruồi lành tính chiếm tới 68% phân phối dữ liệu gốc HAM10000) trong một số trường hợp đặc thù. Hiện tượng này phản ánh sự mất cân bằng dữ liệu ở tập huấn luyện và khả năng khái quát hóa còn hạn chế của Projection layer tuyến tính khi đối mặt với các đặc trưng ảnh trung gian nằm gần biên quyết định (decision boundary).

### 3.5. Đối luận Học thuật về Hiện tượng có thể xem như Nghịch lý BLEU trong VQA Y khoa (Academic Discussion on BLEU)
Một quan sát đáng chú ý từ thực nghiệm VQA là **hiện tượng có thể được xem như nghịch lý đánh giá của chỉ số BLEU trong các bài toán sinh văn bản y văn**:
- **Tại sao mô hình Offline đạt BLEU rất cao (72.69%)?** Mô hình `CPUMedicalVQAModel` có kích thước nhỏ (82M tham số) được tinh chỉnh LoRA trên tập dữ liệu hẹp nên đã có xu hướng ghi nhớ (memorize) các câu trả lời mẫu. Khi gặp câu hỏi trùng khớp, mô hình tái hiện gần như nguyên văn từ ngữ của Ground Truth, dẫn đến điểm BLEU cực kỳ cao. Tuy nhiên, khả năng tùy biến ngôn ngữ và mở rộng tri thức ngoài tập huấn luyện của mô hình này rất kém.
- **Tại sao mô hình Online có BLEU rất thấp (10.91%)?** Mô hình trực tuyến sử dụng `GPT-4o-mini` sinh câu trả lời rất chi tiết, có cấu trúc đề mục rõ ràng, phân tích sâu sắc các yếu tố dịch tễ và đưa ra khuyến nghị lâm sàng cực kỳ chính xác. Tuy nhiên, vì mô hình sử dụng từ ngữ tự nhiên phong phú và không trùng khớp từng từ (n-gram matching) với Ground Truth ngắn của tập kiểm thử, chỉ số BLEU của nó bị kéo xuống mức rất thấp.
- **Kết luận khoa học:** Chỉ số BLEU chỉ phản ánh mức độ trùng khớp bề mặt từ vựng (lexical overlap), hoàn toàn không đại diện cho chất lượng lâm sàng hoặc độ chính xác y khoa (clinical accuracy) của hệ thống VQA. Đối với các hệ thống hỗ trợ quyết định lâm sàng thực tế, việc sử dụng các chỉ số như BLEU cần được kết hợp chặt chẽ với đánh giá định tính của các chuyên gia y tế hoặc các hệ thống chấm điểm ngữ nghĩa chuyên biệt (như MedBLEU hoặc BERTScore).

---

## 4. Benchmark thời gian xử lý toàn hệ thống (System Latency)

Đo lường trung bình trên 20 lần chạy thực tế (CPU/GPU):

* **Tổng thời gian xử lý trung bình:** **232.29 ms** (± 14.17 ms)

### Phân rã thời gian và tỷ lệ sử dụng tài nguyên:
1. **Tiền xử lý (Preprocessing & Load):** **1.12 ms** (Tỷ lệ: **0.48%**)
2. **Phân đoạn tổn thương (Hybrid-Max Segmentation):** **168.73 ms** (Tỷ lệ: **72.64%**)
3. **Trích xuất thuộc tính ABCD (Metrics & ROI):** **1.62 ms** (Tỷ lệ: **0.70%**)
4. **Phân loại bệnh lý (Classification):** **60.80 ms** (Tỷ lệ: **26.17%**)
