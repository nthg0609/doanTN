# Tài liệu Hướng dẫn Phản biện Đồ án Tốt nghiệp: Hệ thống CDSS Da liễu Đa phương thức

Tài liệu này tổng hợp chi tiết toàn bộ các khía cạnh toán học, giải thuật, cơ chế hoạt động thực tế trong mã nguồn của hệ thống hỗ trợ chẩn đoán lâm sàng da liễu CDSS. Tài liệu được thiết kế nhằm phục vụ hội đồng phản biện đồ án tốt nghiệp.

---

## Phần I: Chất lượng ảnh & Bộ lọc tiền xử lý (Safety Gate)

### 1. Giải thuật Laplacian Variance đo độ mờ (Blurry)
* **Khái niệm:** Toán tử Laplacian ($\nabla^2$) là toán tử vi phân bậc hai dùng để đo độ biến thiên không gian (không gian tần số cao) của hàm cường độ sáng ảnh $I(x, y)$, định nghĩa qua ma trận Hessian $\mathbf{H}(I)$:
  $$\nabla^2 I = \Delta I = \text{Tr}(\mathbf{H}(I)) = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$
* **Xấp xỉ rời rạc:** Trong không gian ảnh số rời rạc, toán tử này được xấp xỉ bằng phép tích chập ảnh xám $I_{\text{gray}}$ với nhân (kernel) Laplacian $\mathbf{K}_L$:
  $$\mathbf{K}_L = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$
* **Chỉ số phương sai (Variance):** Độ sắc nét (Sharpness) được biểu diễn thông qua phương sai $\sigma^2$ của các giá trị cường độ sáng sau tích chập:
  $$\text{blur-score} = \sigma^2(\nabla^2 I) = \frac{1}{H \cdot W} \sum_{x=1}^W \sum_{y=1}^H \left( (\nabla^2 I)(x, y) - \mu_{\nabla^2 I} \right)^2$$
  * **Giải thích chi tiết công thức:**
    * **Tên gọi:** Công thức tính phương sai của ảnh sau khi đi qua bộ lọc Laplacian.
    * **Thành phần:** 
      * $(\nabla^2 I)(x, y)$ là giá trị pixel tại tọa độ $(x, y)$ của ảnh sau khi được lọc vi phân bậc hai bằng nhân Laplacian.
      * $\mu_{\nabla^2 I}$ là giá trị trung bình mức xám của toàn bộ bức ảnh sau khi lọc Laplacian.
      * $H$ là chiều cao (Height - số dòng pixel) và $W$ là chiều rộng (Width - số cột pixel) của bức ảnh. Do đó, $H \cdot W$ chính là tổng số pixel của ảnh.
    * **Phép tính:** Lấy giá trị của từng pixel sau lọc trừ đi giá trị trung bình, bình phương lên (để đảm bảo khoảng cách luôn dương), cộng tổng tất cả các pixel lại rồi **chia cho tổng số pixel ảnh $H \cdot W$** để lấy trung bình. Phép tính này cho biết mức độ phân tán của độ tương phản đường biên.
* **Ngưỡng so sánh (Threshold):** Mặc định trong code là **`80.0`** (đơn vị là $(\text{gray level})^2$, tức bình phương mức xám trong khoảng $[0, 255]$).
  * Ý nghĩa: **80.0 không phải là pixel (độ dài/khoảng cách)**.
  * Nếu $\text{blur-score} < 80.0$: Các cạnh biên bị làm mịn mạnh (do nhòe ảnh, triệt tiêu các tần số cao), ảnh bị đánh giá là **mờ/out-focus** và bị từ chối chẩn đoán.

---

### 2. Thang Fitzpatrick & Cơ chế thích ứng giảm định kiến chủng tộc (Bias Mitigation)
* **Thang đo Fitzpatrick:** Phân loại da người thành 6 nhóm từ tuýp I (trắng sáng) đến tuýp VI (sẫm/đen).
* **Đo độ sáng trung bình:** 
  $$\text{brightness-score} = \mu_I = \frac{1}{H \cdot W} \sum_{x=1}^W \sum_{y=1}^H I_{\text{gray}}(x, y)$$
  * **Giải thích chi tiết công thức:**
    * **Tên gọi:** Công thức tính giá trị độ sáng trung bình của ảnh xám.
    * **Thành phần:** 
      * $I_{\text{gray}}(x, y)$ là giá trị độ sáng (mức xám từ 0 đến 255) của pixel tại tọa độ $(x, y)$.
      * $H \cdot W$ là tổng số pixel của ảnh.
    * **Phép tính:** Cộng tổng giá trị mức xám của tất cả các pixel trong ảnh lại với nhau, sau đó **chia cho tổng số pixel ảnh $H \cdot W$** để ra giá trị trung bình (kỳ vọng toán học).
* **Cơ chế thích ứng động:**
  * Ngưỡng tối an toàn mặc định là **`DARK_THRESHOLD = 50.0`**. Tuy nhiên, đối với bệnh nhân da sẫm màu tự nhiên (Fitzpatrick Type V, VI), giá trị $\mu_I$ thường tự nhiên rơi xuống dưới 50.0 dù điều kiện chụp đủ sáng.
  * Giải pháp chống định kiến: Nếu ảnh có độ sắc nét chi tiết cao ($\text{blur-score} \geq 100.0$), tức là ảnh lấy nét tốt và độ tối không phải do mờ hay rung ảnh mà có khả năng lớn do sắc tố da tự nhiên. Khi đó, hệ thống sẽ **hạ ngưỡng tối thiểu xuống `30.0`** để tránh từ chối sai lệch (false rejection) đối với người da màu.

---

## Phần II: Phân đoạn (Segmentation) & Đo đạc ABCD

### 3. Phân đoạn DeepLabV3+ dùng ResNet-50
* **Kiến trúc:** DeepLabV3+ sử dụng bộ mã hóa (Encoder) **ResNet-50** kết hợp với các khối tích chập giãn nở **Atrous Spatial Pyramid Pooling (ASPP)** ở nhiều tỉ lệ giãn nở (dilation rates = [6, 12, 18]) để khai thác ngữ cảnh đa thang đo.
* **Tại sao dùng ResNet-50:**
  * **Residual Connections:** Giúp giải quyết hiện tượng suy giảm đạo hàm (vanishing gradient) khi huấn luyện mạng sâu.
  * **Cân bằng hiệu năng:** Đủ sâu để học các đặc trưng biên dạng phức tạp của u hắc tố, nhưng số lượng tham số vừa phải giúp suy luận nhanh trên CPU so với ResNet-101/152.
  * **Phù hợp kích thước dữ liệu:** Tránh hiện tượng quá khớp (overfitting) do tập dữ liệu phân đoạn y tế (ISIC 2018) có quy mô trung bình (2.594 mẫu).

---

### 4. Thuật toán phân ngưỡng Otsu dự phòng y khoa
* **Nguyên lý hoạt động:** Otsu là thuật toán phân ngưỡng tự động tìm ngưỡng tối ưu $T^*$ dựa trên kỹ thuật tối đa hóa phương sai giữa hai lớp (intra-class variance $\sigma_B^2$):
  $$T^* = \arg\max_{0 \le T \le 255} \sigma_B^2(T)$$
  $$\sigma_B^2(T) = \omega_0(T) \omega_1(T) \left[ \mu_0(T) - \mu_1(T) \right]^2$$
  * **Giải thích chi tiết công thức:**
    * **Tên gọi:** Công thức tính phương sai giữa hai lớp (nền và đối tượng) ứng với một ngưỡng cắt $T$.
    * **Thành phần:**
      * $\omega_0(T)$ và $\omega_1(T)$ là tỉ lệ số lượng pixel (xác suất xuất hiện) của lớp nền (background) và lớp đối tượng (foreground) sau khi được phân chia bởi ngưỡng $T$.
      * $\mu_0(T)$ và $\mu_1(T)$ là giá trị cường độ sáng trung bình của lớp nền và lớp đối tượng tương ứng.
    * **Phép tính:** Lấy hiệu số giữa hai giá trị trung bình $\mu_0(T) - \mu_1(T)$, bình phương lên, sau đó **nhân với tỉ lệ pixel của cả hai nhóm $\omega_0(T) \cdot \omega_1(T)$**. Thuật toán sẽ quét mọi giá trị $T$ từ $0$ đến $255$ và chọn ra giá trị $T^*$ làm cho phương sai này lớn nhất. Phương sai giữa hai nhóm càng lớn nghĩa là sự phân tách giữa vùng tổn thương da và vùng da lành càng rõ ràng.
* **Ngưỡng gán:** Thuật toán duyệt qua mọi mức xám để tính $T^*$. Sau đó gán nhãn:
  $$\text{Pixel}(x, y) = \begin{cases} 255 & \text{nếu } I_{\text{gray}}(x, y) < T^* \text{ (Tổn thương - Tối)} \\ 0 & \text{nếu } I_{\text{gray}}(x, y) \ge T^* \text{ (Da lành - Sáng)} \end{cases}$$
  (Sử dụng flag `cv2.THRESH_BINARY_INV` để lật ngược vì tổn thương sắc tố thường có màu tối hơn vùng da lành xung quanh).

---

### 5. Công thức đo đạc 4 chỉ số ABCD lâm sàng
Từ mặt nạ nhị phân tổn thương $M \in \{0, 1\}^{H \times W}$, hệ thống tự động tính toán các chỉ số:
* **A — Asymmetry (Bất đối xứng):**
  1. Xác định tọa độ trọng tâm $(C_x, C_y)$ của vùng tổn thương qua các mô-men không gian bậc một:
     $$C_x = \frac{m_{10}}{m_{00}}, \quad C_y = \frac{m_{01}}{m_{00}}$$
     * **Giải thích phép chia:** Lấy mô-men bậc một theo hướng tương ứng ($m_{10}$ cho trục X, $m_{01}$ cho trục Y) **chia cho diện tích (mô-men bậc không $m_{00}$)** để tìm ra điểm cân bằng trọng lực (trọng tâm hình học) của hình dạng tổn thương.
     * Công thức mô-men: $m_{pq} = \sum_{x} \sum_{y} x^p y^q M(x, y)$
  2. Chia đôi mặt nạ theo trục ngang qua $C_y$, lật ngược nửa dưới chồng lên nửa trên để tính diện tích khác biệt ($Asym_H$):
     $$Asym_H = \sum_{x} \sum_{y} |M_{\text{top}}(x, y) - M_{\text{bottom-flipped}}(x, y)|$$
     * **Giải thích:** Trừ ma trận pixel của nửa trên cho ma trận pixel đã lật của nửa dưới. Lấy giá trị tuyệt đối để mọi sự lệch nhau đều mang giá trị dương, cộng tổng lại để ra số pixel không trùng khớp khi gấp ảnh theo trục ngang.
  3. Chia đôi mặt nạ theo trục dọc qua $C_x$, lật ngược nửa phải chồng lên nửa trái để tính diện tích khác biệt ($Asym_V$):
     $$Asym_V = \sum_{x} \sum_{y} |M_{\text{left}}(x, y) - M_{\text{right-flipped}}(x, y)|$$
     * **Giải thích:** Tương tự như trên, cộng tổng các điểm lệch nhau khi gấp ảnh theo trục dọc.
  4. Chỉ số bất đối xứng $A \in [0, 1]$:
     $$A = \frac{Asym_H + Asym_V}{2 \times \text{Area}(M)}$$
     * **Giải thích phép chia:** Cộng tổng hai lượng diện tích không khớp theo trục ngang và trục dọc ($Asym_H + Asym_V$), sau đó **chia cho 2 lần diện tích thực tế của tổn thương $\text{Area}(M)$** (diện tích thực tế chính là số pixel có giá trị 1 trên mặt nạ). Việc chia này giúp chuẩn hóa chỉ số về dải $[0, 1]$, độc lập với kích thước ảnh chụp gần hay xa.
* **B — Border (Biên bờ):** Tính độ phức tạp của biên bờ dựa trên tỷ lệ chu vi ($P$) và căn bậc hai diện tích ($A_{lesion}$):
     $$\text{Border-Complexity} = \frac{P}{\sqrt{A_{lesion}}}$$
     * **Giải thích phép chia:** Lấy chu vi tổn thương $P$ (độ dài đường biên bao quanh) **chia cho căn bậc hai diện tích tổn thương $\sqrt{A_{lesion}}$**. Phép chia này cho phép đánh giá mức độ gồ ghề của biên. Đối với hình tròn hoàn hảo, tỉ số này là nhỏ nhất ($2\sqrt{\pi} \approx 3.54$). Tổn thương càng ngoằn ngoèo, răng cưa thì chu vi $P$ càng kéo dài ra trong khi diện tích giữ nguyên, làm tỉ số này tăng cao (thông thường $>5.0$ là ranh giới bất thường).
* **C — Color (Màu sắc):** Đo độ biến động màu sắc bằng độ lệch chuẩn trung bình của 3 kênh màu RGB trên các pixel thuộc vùng tổn thương, chuẩn hóa về dải $[0, 1]$:
     $$C = \frac{1}{127.5} \times \left( \frac{\sigma_R + \sigma_G + \sigma_B}{3} \right)$$
     * **Giải thích phép chia:** 
       * Tính độ lệch chuẩn $\sigma_c$ của từng kênh màu $R, G, B$ trên các điểm ảnh thuộc vùng tổn thương để đo độ phân tán (loang lổ) màu sắc của kênh đó.
       * Cộng tổng độ lệch chuẩn của 3 kênh lại rồi **chia cho 3** để lấy giá trị trung bình.
       * Tiếp tục **chia cho 127.5** (là giá trị độ lệch chuẩn lớn nhất lý thuyết của một kênh ảnh 8-bit từ 0 đến 255) nhằm đưa chỉ số về dải chuẩn hóa $[0, 1]$. Số gần 0 nghĩa là tổn thương đơn sắc (đồng đều), gần 1 nghĩa là loang lổ nhiều màu (ác tính).
* **D — Diameter (Đường kính):** Tính đường kính tương đương của tổn thương hình tròn có cùng diện tích:
     $$D_{\text{px}} = 2 \times \sqrt{\frac{A_{lesion}}{\pi}}$$
     * **Giải thích công thức:** Diện tích hình tròn là $S = \pi r^2 \Rightarrow r = \sqrt{S / \pi}$. Đường kính bằng 2 lần bán kính nên ta lấy diện tích tổn thương $A_{lesion}$ **chia cho số $\pi$**, lấy căn bậc hai rồi **nhân với 2** để tính ra đường kính tương đương theo đơn vị pixel ($px$).
     * Quy đổi sang mm thực tế: $D_{\text{mm}} = D_{\text{px}} \times \text{PixelSpacing}$ (Nhân đường kính pixel với kích thước vật lý của một pixel trích xuất từ siêu dữ liệu ảnh DICOM để ra đường kính thật tính bằng milimet).

---

## Phần III: Phân loại & Hợp nhất Bayes đa phương thức

### 6. Backbone EfficientNet-B1 & Khối Attention CBAM
* **EfficientNet-B1:** Sử dụng phương pháp Compound Scaling để cân bằng đồng thời độ sâu ($d = \alpha^\phi$), độ rộng mạng ($w = \beta^\phi$) và độ phân giải ảnh ($r = \gamma^\phi$). Mô hình có kích thước gọn nhẹ (~7.8 triệu tham số), giảm độ trễ tối đa khi suy luận trên CPU.
* **CBAM (Convolutional Block Attention Module):** Gồm hai khối chú ý tuần tự chèn sau trích xuất đặc trưng:
  1. **Channel Attention (Chú ý kênh):** Tập trung vào việc mô hình hóa mối quan hệ giữa các kênh đặc trưng.
     $$M_c(F) = \sigma \left( \text{MLP}(\text{AvgPool}(F)) + \text{MLP}(\text{MaxPool}(F)) \right)$$
     * **Giải thích:** Ép đặc trưng không gian của ảnh qua Average Pooling và Max Pooling, đưa qua mạng Perceptron đa lớp MLP để học các trọng số kênh, cộng lại và đi qua hàm Sigmoid ($\sigma$) để ra vector phân phối chú ý kênh.
  2. **Spatial Attention (Chú ý không gian):** Tập trung vào vùng vị trí đặc trưng (định vị tổn thương).
     $$M_s(F) = \sigma \left( f^{7 \times 7} \left( [ \text{AvgPool}(F); \text{MaxPool}(F) ] \right) \right)$$
     * **Giải thích:** Tính giá trị trung bình và cực đại dọc theo chiều kênh của tensor đặc trưng, nối chúng lại với nhau (kí hiệu $[;]$), đi qua lớp tích chập với nhân lớn $7 \times 7$ rồi qua hàm Sigmoid ($\sigma$) để chỉ ra vùng tọa độ quan trọng trên ảnh.

---

### 7. Hợp nhất Bayes đa phương thức (Multimodal Bayesian Fusion)
* **Nguyên lý:** Kết hợp xác suất hình ảnh y khoa thu từ mạng học sâu CNN và xác suất dịch tễ của bệnh nhân (Tuổi, Giới tính, Vị trí tổn thương) dựa trên định lý Bayes:
  $$P(C_k | \text{Ảnh}, \text{Tuổi}, \text{Giới tính}, \text{Vị trí}) \propto P(C_k | \text{Ảnh}) \times P(\text{Tuổi} | C_k) \times P(\text{Giới tính} | C_k) \times P(\text{Vị trí} | C_k)$$
* **Cách tính toán:**
  1. *Kỳ vọng tuổi (Gaussian Likelihood):* Xác định qua hàm mật độ xác suất phân phối chuẩn $P(\text{Age} | C_k) = \mathcal{N}(\mu_{k}, \sigma_{k}^2)$:
     $$P(\text{Age} | C_k) = \frac{1}{\sigma_k \sqrt{2\pi}} \exp \left( -\frac{(\text{Age} - \mu_k)^2}{2\sigma_k^2} \right)$$
     * **Giải thích phép chia:** 
       * Tính khoảng cách lệch giữa tuổi bệnh nhân và tuổi trung bình mắc bệnh của nhóm bệnh lý $k$: $(\text{Age} - \mu_k)^2$, sau đó **chia cho $2\sigma_k^2$** (2 lần phương sai tuổi của nhóm bệnh $k$). Đi qua hàm mũ âm $\exp(-x)$ để tìm ra độ tương đồng.
       * Tiếp tục **chia cho hằng số chuẩn hóa $\sigma_k \sqrt{2\pi}$** để đảm bảo tổng diện tích dưới đường cong mật độ xác suất tích phân bằng 1.
  2. *Hợp nhất muộn (Late Fusion) hiệu chỉnh bằng tham số $\lambda$ (mặc định = 0.85):*
     $$\text{Final-P}(C_k) = \frac{(P(C_k | \text{Ảnh}))^\lambda \cdot (P(C_k | \text{Dịch tễ}))^{1-\lambda}}{\sum_j (P(C_j | \text{Ảnh}))^\lambda \cdot (P(C_j | \text{Dịch tễ}))^{1-\lambda}}$$
     * **Giải thích phép chia:** Tính toán tích xác suất hình ảnh mũ $\lambda$ nhân với xác suất dịch tễ mũ $1-\lambda$ cho từng lớp bệnh lý $C_k$, sau đó **chia cho tổng giá trị này của tất cả 7 lớp bệnh lý** để chuẩn hóa tổng xác suất đầu ra của hệ thống về đúng $1.0$ ($100\%$).

---

## Phần IV: Trợ lý đàm thoại ngôn ngữ lớn VQA (NLP)

### 8. Lựa chọn DistilGPT-2 làm Decoder & Tinh chỉnh LoRA
* **Lý do chọn DistilGPT-2:** Là phiên bản chưng cất tri thức (Knowledge Distillation) từ mô hình GPT-2 gốc, giảm số tầng Transformer xuống còn 6 tầng giúp giảm kích thước tham số (~82 triệu tham số) giúp suy luận tạo sinh văn bản tư vấn cực nhanh trên CPU mà không cần GPU.
* **Công thức tinh chỉnh LoRA (PEFT):**
  * Trong quá trình Fine-tuning, trọng số của lớp Attention gốc $W_0 \in \mathbb{R}^{d \times k}$ được đóng băng hoàn toàn. Hệ thống chỉ cập nhật ma trận biến thiên $\Delta W$ được phân tách thành tích của hai ma trận hạng thấp (low-rank) $A$ và $B$:
    $$W = W_0 + \Delta W = W_0 + B \cdot A$$
  * Giá trị đầu ra (forward pass) của lớp tích chập attention được tính bằng:
    $$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (B \cdot A) x$$
    * **Giải thích phép nhân và chia:** Lấy kết quả đầu ra của mạng đóng băng gốc $W_0 x$ cộng thêm tích của đầu vào $x$ với ma trận LoRA hạng thấp $(B \cdot A)x$, rồi **nhân với hằng số $\alpha$ và chia cho hạng rank $r$** ($\alpha / r$ là hệ số tỉ lệ giúp cân bằng độ lớn của các cập nhật trọng số LoRA mới thích ứng so với trọng số gốc có sẵn).

### 9. Kết quả đánh giá mô hình VQA (BLEU Score)
* **Kết quả mô hình ngoại tuyến (Offline Model):**
  * Đánh giá định lượng trên tập Validation (12 mẫu câu hỏi lâm sàng thực tế):
    * **Average BLEU-1:** **`0.7269`** (độ khớp từ vựng $72.69\%$)
    * **Average BLEU-2:** **`0.6812`** (độ khớp từ vựng $68.12\%$)
  * **Nhận xét học thuật:** Mô hình offline đạt điểm BLEU rất cao do học thuộc tốt các cấu trúc câu trả lời mẫu y văn của chuyên gia trên tập dữ liệu hẹp, nhưng khả năng linh hoạt ngôn ngữ bị hạn chế khi gặp câu hỏi ngoài tập huấn luyện.
* **Mô hình trực tuyến (Online Model):**
  * **BLEU-1 trung bình:** **`10.91%`** (mức độ trùng khớp từ vựng thấp do mô hình trực tuyến sinh câu trả lời tự nhiên, dài, đa dạng từ ngữ và mang nhiều chi tiết y học phong phú hơn hẳn câu trả lời tham chiếu ngắn, mặc dù có độ chính xác y khoa thực tế tốt hơn).
