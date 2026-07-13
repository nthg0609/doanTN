# Tài liệu Phản biện Đồ án Tốt nghiệp: Hệ thống Chẩn đoán Da liễu CDSS

Tài liệu này tổng hợp toàn bộ câu hỏi và câu trả lời phản biện liên quan đến thuật toán Thị giác máy tính (CV), Hợp nhất dữ liệu (Bayesian Fusion) và Xử lý ngôn ngữ tự nhiên (NLP VQA) trong hệ thống.

---

## Phần I: Chất lượng ảnh & Bộ lọc tiền xử lý (Safety Gate)

### 1. Phương pháp Laplacian Variance đo độ mờ (Blurry) hoạt động thế nào? Ngưỡng 80.0 ý nghĩa là gì?
* **Phương pháp:** Toán tử Laplacian là một toán tử vi phân bậc hai lấy đạo hàm của ảnh để phát hiện các vùng có sự thay đổi đột ngột về cường độ sáng (đường biên/cạnh). Trong không gian ảnh xám 8-bit rời rạc, toán tử này được xấp xỉ bằng phép tích chập ảnh với nhân (kernel) Laplacian:
  $$\mathbf{K}_L = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$
* **Ý nghĩa giá trị 80.0:** 
  * Con số **80.0** ở đây **không phải là pixel** (không đo khoảng cách hay kích thước vật lý), mà là **Phương sai của ảnh Laplacian** (đơn vị là $(\text{gray level})^2$ - bình phương mức xám trong khoảng $[0, 255]$).
  * Ảnh rõ nét có các cạnh biên rõ ràng, độ chuyển đổi mức xám đột ngột làm phương sai đạo hàm lớn. Ảnh nhòe mờ sẽ mịn hóa các biên, triệt tiêu tần số cao dẫn đến phương sai Laplacian cực nhỏ. 
  * Ngưỡng **80.0** được chọn qua thực nghiệm y khoa: ảnh có phương sai dưới 80.0 bị coi là mờ/out-focus và bị hệ thống từ chối (`warning`).

### 2. Fitzpatrick Scale là gì và hoạt động thế nào trong chương trình?
* **Khái niệm:** Thang Fitzpatrick phân loại da người thành 6 nhóm từ tuýp I (da trắng sáng) đến tuýp VI (da sẫm/đen).
* **Hoạt động giảm định kiến (Algorithmic Bias Mitigation):**
  * Ảnh của người có tông da tối tự nhiên (tuýp V, VI) có độ sáng xám trung bình thấp và dễ bị thuật toán lọc chất lượng đánh giá nhầm là "thiếu sáng" (ngưỡng tối mặc định `DARK_THRESHOLD = 50.0`).
  * Để khắc phục, chương trình triển khai cơ chế **thích ứng động**: Nếu ảnh có độ nét rất cao (phương sai Laplacian $\geq 100.0$) nhưng tối, hệ thống tự động nhận diện đây là da sẫm màu lành tính và **hạ ngưỡng tối thiểu xuống `30.0`** để tránh phân biệt đối xử lâm sàng, cho phép tiếp tục chẩn đoán.

---

## Phần II: Phân đoạn (Segmentation) & Đo đạc ABCD

### 3. Mô hình DeepLabV3+ dùng mạng xương sống (Backbone) nào? Tại sao chọn mạng này?
* **Backbone:** Sử dụng mạng **`ResNet-50`**.
* **Lý do lựa chọn:**
  * **Cân bằng tối ưu:** ResNet-50 đủ sâu để trích xuất các đặc trưng không gian đa thang đo phức tạp qua các khối tích chập tích hợp Residual Connection, nhưng không quá nặng như ResNet-101/152, đảm bảo tốc độ suy luận nhanh trên CPU.
  * **Tránh quá khớp (Overfitting):** Tập dữ liệu phân đoạn da liễu có kích thước vừa phải (2.594 mẫu). Việc dùng các backbone tham số quá lớn rất dễ dẫn đến Overfitting trên tập dữ liệu này.

### 4. Thuật toán phân ngưỡng Otsu hoạt động thế nào? Dùng ngưỡng nào để phân tách?
* **Hoạt động:** Otsu là thuật toán phân ngưỡng tự động không tham số. Nó **không sử dụng một ngưỡng cố định** (như 127) mà duyệt qua toàn bộ dải mức xám $[0, 255]$ để tìm ra một ngưỡng tối ưu $T^*$ sao cho **phương sai giữa hai lớp (intra-class variance) đạt cực đại**.
* **Phân tách tối/sáng:** Sau khi tìm được $T^*$, chương trình áp dụng bộ lọc nhị phân nghịch đảo (`cv2.THRESH_BINARY_INV`). Các pixel có giá trị xám $< T^*$ (vùng tổn thương sắc tố tối màu) được gán thành **`255` (Trắng - Foreground)**, các pixel $> T^*$ (vùng da lành sáng màu) được gán thành **`0` (Đen - Background)**.

### 5. Chỉ số ABCD lâm sàng được tính toán bằng cách nào?
* **A — Asymmetry (Bất đối xứng):** Chia mặt nạ tổn thương thành hai nửa theo trục ngang và trục dọc đi qua trọng tâm (Centroid) hình học. Lật ngược các nửa để so sánh sự chồng chéo. Chỉ số $A \in [0, 1]$: giá trị 0 là đối xứng tuyệt đối, 1 là bất đối xứng hoàn toàn.
* **B — Border (Biên bờ):** Tính độ phức tạp bờ dựa trên chu vi ($P$) và diện tích ($A$): $\text{Border\_Complexity} = \frac{P}{\sqrt{A}}$. Viền càng lồi lõm, răng cưa thì chỉ số càng cao (ngưỡng báo động lâm sàng $>5.0$).
* **C — Color (Màu sắc):** Tính độ lệch chuẩn (Standard Deviation) của các kênh R, G, B trên các pixel thuộc vùng tổn thương, lấy trung bình cộng 3 kênh và chuẩn hóa về dải $[0, 1]$ bằng cách chia cho độ lệch chuẩn tối đa $127.5$.
* **D — Diameter (Đường kính):** Tính đường kính tương đương của tổn thương hình tròn có cùng diện tích: $D = 2 \times \sqrt{\frac{Area}{\pi}}$ (đơn vị pixel). Nếu là file ảnh DICOM có chứa siêu dữ liệu `PixelSpacing` ($mm/pixel$), chỉ số sẽ tự động được nhân quy đổi sang đơn vị **$mm$** thật.

---

## Phần III: Phân loại & Hợp nhất Bayes đa phương thức

### 6. Tại sao dùng backbone EfficientNet-B1 và cơ chế CBAM hoạt động ra sao?
* **EfficientNet-B1:** Sử dụng phương pháp Compound Scaling để cân bằng đồng thời độ sâu, độ rộng mạng và độ phân giải ảnh. Mô hình có tham số cực kỳ gọn nhẹ (~7.8M params) nhưng độ chính xác vượt trội DenseNet/ResNet, tối ưu cho suy luận thời gian thực trên Cloud.
* **Cơ chế CBAM (Convolutional Block Attention Module):** CBAM chèn sau backbone để tăng cường tập trung vào tổn thương:
  * **Channel Attention:** Đi qua luồng AvgPool và MaxPool song song, thu nhỏ số kênh rồi tái tạo lại nhằm tìm ra các kênh đặc trưng bệnh lý quan trọng (ví dụ: kênh biểu diễn mạng lưới sắc tố).
  * **Spatial Attention:** Tính toán trị trung bình và cực đại dọc theo trục kênh của tensor đặc trưng, gộp lại đưa qua lớp tích chập $7 \times 7$ để tạo bản đồ chú ý không gian, định vị chính xác vùng tổn thương da trên ảnh.

### 7. Thông tin nhân khẩu học tác động đến kết quả phân loại bằng cách nào qua Bayes?
* **Cách tác động:** Hợp nhất Bayes đa phương thức hiệu chỉnh xác suất đầu ra của bộ phân loại ảnh $P(C_i | \text{Ảnh})$ bằng xác suất tiên nghiệm của bệnh nhân (Tuổi, Giới tính, Vị trí tổn thương) thống kê từ HAM10000:
  $$P(C_i | \text{Ảnh}, \text{Tuổi}, \text{Giới tính}, \text{Vị trí}) \propto P(C_i | \text{Ảnh}) \times P(\text{Tuổi} | C_i) \times P(\text{Giới tính} | C_i) \times P(\text{Vị trí} | C_i)$$
* **Cách tính:**
  1. *Tuổi:* Ước lượng xác suất qua phân phối mật độ Gaussian $N(\mu, \sigma^2)$ của từng bệnh. (Ví dụ: U ác Melanoma hay xuất hiện ở người lớn tuổi $\mu=59.6$, nốt ruồi lành tính xuất hiện ở người trẻ $\mu=38.2$).
  2. *Giới tính & Vị trí:* Áp dụng bảng phân phối xác suất rời rạc $P(\text{Gender} | C_i)$ và $P(\text{Location} | C_i)$.
  3. *Late Fusion:* Kết hợp theo hệ số niềm tin $\lambda$ (mặc định = 0.85): 
     $$\text{Final\_P} = \lambda \times P(C_i | \text{Ảnh}) + (1 - \lambda) \times P(C_i | \text{Hợp nhất})$$

---

## Phần IV: Trợ lý đàm thoại ngôn ngữ lớn VQA (NLP)

### 8. Tại sao chọn DistilGPT-2 làm Decoder? Cấu hình tinh chỉnh LoRA ra sao và tại sao chọn như vậy?
* **Lý do chọn DistilGPT-2:** Là mô hình ngôn ngữ causal nhỏ gọn (~82 triệu tham số) được thu gọn từ GPT-2 gốc bằng chưng cất tri thức, tối ưu cho suy luận tạo sinh văn bản tư vấn trực tiếp trên CPU cục bộ ngoại tuyến.
* **Cấu hình LoRA (PEFT):**
  * Hạng ma trận **`r = 8`**, hệ số tỉ lệ **`lora_alpha = 16`**, dropout **`0.05`**.
  * Module nhắm mục tiêu: **`c_attn`** (gộp của ma trận $W_q, W_k, W_v$ trong khối Attention của GPT-2).
* **Lý do chọn cấu hình:** Cấu hình rank = 8 giúp đóng băng toàn bộ tham số gốc và chỉ cập nhật ma trận thích ứng LoRA siêu nhẹ (khoảng **`2.13%`** số lượng tham số huấn luyện). Thiết lập này ngăn chặn hiện tượng quá khớp (Overfitting) trên tập dữ liệu y học chuyên sâu hẹp và ngăn chặn hiện tượng "quên lãng thảm họa" (catastrophic forgetting) của mô hình ngôn ngữ nền tảng.

### 9. Kết quả đánh giá mô hình VQA (BLEU Score) cụ thể là bao nhiêu?
* **Kết quả mô hình ngoại tuyến (Offline Model):**
  * Đánh giá định lượng trên tập Validation (12 mẫu câu hỏi lâm sàng thực tế):
    * **Average BLEU-1:** **`0.7269`** (độ khớp từ vựng $72.69\%$)
    * **Average BLEU-2:** **`0.6812`** (độ khớp từ vựng $68.12\%$)
  * **Nhận xét học thuật:** Mô hình offline đạt điểm BLEU cao do học thuộc (memorize) tốt các cấu trúc câu trả lời y văn của chuyên gia trên tập dữ liệu hẹp. Tuy nhiên, khả năng tạo sinh linh hoạt ngôn ngữ bị giới hạn khi gặp câu hỏi có từ vựng nằm ngoài tập huấn luyện.
* **Mô hình trực tuyến (Online Model):**
  * Điểm BLEU-1 đạt khoảng **`10.91%`** do câu trả lời sinh ra tự nhiên, dài, mang nhiều chi tiết y học phong phú và cách dùng từ đa dạng hơn nhiều so với câu trả lời tham chiếu ngắn (dù độ chính xác y khoa và tính ứng dụng lâm sàng thực tế tốt hơn hẳn). Điều này cho thấy chỉ số BLEU có phần hạn chế khi đánh giá các hệ thống LLM VQA Y khoa do chỉ đo sự trùng khớp từ vựng bề mặt (lexical overlap).
