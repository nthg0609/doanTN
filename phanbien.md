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
* **Tại sao phải sử dụng đa thang đo (Multi-scale):**
  1. **Sự biến thiên sinh học của tổn thương da:** Các nốt u, sẩn hoặc dát sắc tố trong thực tế da liễu có kích thước cực kỳ đa dạng (từ nốt ruồi chấm nhỏ vài milimet đến các mảng ung thư tế bào đáy/melanoma lan rộng vài centimet). Nếu mô hình chỉ sử dụng một tỷ lệ thu nhận (scale) cố định, nó sẽ:
     * Dễ bị mất dấu ngữ cảnh toàn cục (global context) khi gặp tổn thương quá lớn.
     * Bị bỏ sót các chi tiết biên tinh tế cục bộ (local context) khi gặp tổn thương quá nhỏ.
  2. **Giải pháp ASPP trong kiến trúc DeepLabV3+:**
     * ASPP áp dụng các phép tích chập giãn nở (Atrous/Dilated Convolution) song song với nhiều tỉ lệ giãn nở khác nhau (dilation rates = [6, 12, 18]).
     * Tỉ lệ nhỏ (rate=6) có trường thụ nhận hẹp giúp trích xuất các **đặc trưng cục bộ** sắc nét (rìa ngoài, đường viền bờ $B$ của ABCD).
     * Tỉ lệ lớn (rate=18) có trường thụ nhận rộng giúp trích xuất các **đặc trưng toàn cục** (hình thái tổng thể, ranh giới giữa u và da lành xung quanh).
     * Kết quả từ các nhánh song song này được concatenate lại để tạo ra một biểu diễn đặc trưng đa thang đo toàn diện mà không cần tăng số lượng tham số hay làm giảm độ phân giải của bản đồ đặc trưng.
  3. **Giải pháp Multi-scale TTA (Test-Time Augmentation) đối với ảnh điện thoại:**
     * Ảnh chụp từ điện thoại thực tế không có tiêu chuẩn khoảng cách cố định như kính soi da chuyên dụng (dermoscope).
     * Bằng cách co giãn ảnh đầu vào ở nhiều kích thước khác nhau (thang đo $\{1.0, \; 0.75, \; 0.5\}$) lúc suy luận, sau đó chạy phân đoạn độc lập và cộng trung bình các xác suất nhị phân, mô hình đạt được **tính bất biến về tỷ lệ (Scale Invariance)**. Điều này giúp hệ thống đạt độ chính xác ổn định cao bất kể khoảng cách chụp xa hay gần của bác sĩ.

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

### 5. Phương pháp phân đoạn tương tác SAM và GrabCut hoạt động như thế nào?
* **Mục đích:** Hỗ trợ bác sĩ điều chỉnh ranh giới phân đoạn u da một cách chủ động bằng cách nhấp chuột (click điểm) vào vùng tổn thương mong muốn, thay vì hoàn toàn phụ thuộc vào thuật toán phân đoạn tự động.
* **Cơ chế hoạt động:**
  1. **Nhánh 1: Segment Anything Model (SAM - MobileSAM):**
     * Khi có file checkpoint trọng số và thư viện được nạp, mô hình sử dụng kiến trúc mạng nơ-ron học sâu ViT-T (Vision Transformer) để xử lý ảnh đầu vào và nhận điểm nhấp chuột $(x, y)$ dưới dạng điểm mồi (Foreground point prompt với nhãn `label = 1`).
     * Mạng sinh ra các mặt nạ tương ứng kèm điểm số độ tin cậy (scores) và chọn mặt nạ có điểm số cao nhất để làm kết quả phân đoạn.
  2. **Nhánh 2: Bộ lọc GrabCut dự phòng (Fallback):**
     * Trong điều kiện tài nguyên CPU hạn chế trên Cloud, chương trình chạy thuật toán cắt đồ thị (Graph Cut) tối ưu hóa năng lượng dựa trên mô hình hỗn hợp Gaussian (GMM) để phân tách tiền cảnh/hậu cảnh.
     * **Cơ chế gán nhãn và Bounding Box động:**
       * **Giới hạn vùng tính toán (Bounding Box):** Thiết lập một khung bao quanh điểm click có kích thước bằng **60% chiều dài và rộng của ảnh** ($0.6W \times 0.6H$). Toàn bộ pixel nằm ngoài khung này (như thước đo ruler, viền ảnh đen) được gắn nhãn là **Hậu cảnh tuyệt đối (`cv2.GC_BGD = 0`)** để thuật toán bỏ qua hoàn toàn nhiễu từ bên ngoài.
       * **Khởi tạo nhãn trong khung:** Các pixel nằm trong khung được gán nhãn mặc định là **Hậu cảnh tiềm năng (`cv2.GC_PR_BGD = 2`)**.
       * **Gán nhãn điểm mồi:** Một vùng tròn bán kính nhỏ (bằng 8% kích thước ảnh) bao quanh điểm click chuột của bác sĩ được gán nhãn là **Tiền cảnh tiềm năng (`cv2.GC_PR_FGD = 3`)**, riêng điểm pixel click chính xác tại tâm được gán là **Tiền cảnh tuyệt đối (`cv2.GC_FGD = 1`)** để neo thuật toán.
     * **Lặp tối ưu hóa:** Thuật toán GrabCut chạy qua 5 vòng lặp để cập nhật GMM cho tiền cảnh/hậu cảnh, sau đó dựng đồ thị dòng cực đại - cắt cực tiểu (Max-flow Min-cut) để tách vùng.
     * **Hậu xử lý:** Sử dụng giải thuật Connected Components để lọc giữ lại thành phần liên thông lớn nhất có chứa trực tiếp tọa độ điểm click chuột ban đầu, đảm bảo mặt nạ ôm khít nốt tổn thương được chọn.
  3. **Cơ chế Fallback an toàn y khoa:**
     * Nếu mặt nạ tương tác sau khi tính toán quá nhỏ (diện tích $< 100$ pixel do click nhầm ra ngoài u hoặc thuật toán hội tụ lỗi), hệ thống sẽ tự động chuyển hướng sử dụng mặt nạ phân đoạn tự động của mạng **DeepLabV3+** để đảm bảo tính an toàn y khoa và luôn có dữ liệu chẩn đoán.

---

### 6. Công thức đo đạc 4 chỉ số ABCD lâm sàng
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

### 7. Backbone EfficientNet-B1 & Khối Attention CBAM
* **EfficientNet-B1:** Sử dụng phương pháp Compound Scaling để cân bằng đồng thời độ sâu ($d = \alpha^\phi$), độ rộng mạng ($w = \beta^\phi$) và độ phân giải ảnh ($r = \gamma^\phi$). Mô hình có kích thước gọn nhẹ (~7.8 triệu tham số), giảm độ trễ tối đa khi suy luận trên CPU.
* **CBAM (Convolutional Block Attention Module):** Gồm hai khối chú ý tuần tự chèn sau trích xuất đặc trưng:
  1. **Channel Attention (Chú ý kênh):** Tập trung vào việc mô hình hóa mối quan hệ giữa các kênh đặc trưng.
     $$M_c(F) = \sigma \left( \text{MLP}(\text{AvgPool}(F)) + \text{MLP}(\text{MaxPool}(F)) \right)$$
     * **Giải thích:** Ép đặc trưng không gian của ảnh qua Average Pooling và Max Pooling, đưa qua mạng Perceptron đa lớp MLP để học các trọng số kênh, cộng lại và đi qua hàm Sigmoid ($\sigma$) để ra vector phân phối chú ý kênh.
  2. **Spatial Attention (Chú ý không gian):** Tập trung vào vùng vị trí đặc trưng (định vị tổn thương).
     $$M_s(F) = \sigma \left( f^{7 \times 7} \left( [ \text{AvgPool}(F); \text{MaxPool}(F) ] \right) \right)$$
     * **Giải thích:** Tính giá trị trung bình và cực đại dọc theo chiều kênh của tensor đặc trưng, nối chúng lại với nhau (kí hiệu $[;]$), đi qua lớp tích chập với nhân lớn $7 \times 7$ rồi qua hàm Sigmoid ($\sigma$) để chỉ ra vùng tọa độ quan trọng trên ảnh.

---

### 8. Hợp nhất Bayes đa phương thức (Multimodal Bayesian Fusion)
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

### 9. Lựa chọn DistilGPT-2 làm Decoder & Tinh chỉnh LoRA
* **Lý do chọn DistilGPT-2:** Là phiên bản chưng cất tri thức (Knowledge Distillation) từ mô hình GPT-2 gốc, giảm số tầng Transformer xuống còn 6 tầng giúp giảm kích thước tham số (~82 triệu tham số) giúp suy luận tạo sinh văn bản tư vấn cực nhanh trên CPU mà không cần GPU.
* **Ý nghĩa của các ma trận $W_q, W_k, W_v$ với $q, k, v$ là gì?**
  * Trong cơ chế Tự chú ý (Self-Attention) của Transformers:
    * **$q$ (Query - Truy vấn/Câu hỏi):** Đại diện cho từ hoặc vị trí hiện tại đang xét, dùng để tìm kiếm các mối tương quan với các từ khác.
    * **$k$ (Key - Chìa khóa/Từ khóa):** Đại diện cho tất cả các từ trong câu, dùng để so khớp độ tương quan ngữ nghĩa với Query nhằm tính toán trọng số phân bố chú ý.
    * **$v$ (Value - Giá trị):** Chứa thông tin ngữ nghĩa thực sự của từng từ. Nó được nhân với trọng số chú ý thu được để tạo ra vector ngữ cảnh tổng hợp cuối cùng.
  * **Ví dụ trực quan y khoa (Clinical Example):**
    * Giả sử chuỗi đầu vào là cụm từ: `"nốt ruồi ác tính"`. Ta xét từ hiện tại là **`"ác tính"`**:
      * **Query ($q$) của từ `"ác tính"`** đại diện cho nhu cầu tìm kiếm: *"Tôi là tính chất 'ác tính', tôi cần tìm xem từ nào xung quanh liên quan mật thiết để bổ nghĩa cho tôi?"*
      * **Key ($k$) của các từ xung quanh:** Từ `"nốt"` có Key chỉ thực thể tròn; từ `"ruồi"` có Key chỉ nốt ruồi hắc tố; từ `"ác tính"` có Key tự chỉ chính nó.
      * **Tính toán Attention (So khớp):** Nhân vô hướng Query của `"ác tính"` với Key của các từ:
        * Tích vô hướng với Key của `"nốt"` $\rightarrow$ thấp.
        * Tích vô hướng với Key của `"ruồi"` $\rightarrow$ **rất cao** (vì tính chất ác tính liên quan mật thiết đến bệnh lý của nốt ruồi).
        * Tích vô hướng với Key của `"ác tính"` $\rightarrow$ cao (tự chú ý).
      * Phép tính này lọc ra trọng số chú ý (Attention Weight): ví dụ gán trọng số $0.7$ cho từ `"ruồi"`.
      * **Value ($v$):** Lấy trọng số $0.7$ nhân với thông tin ngữ nghĩa gốc (Value) của từ `"ruồi"` và cộng tổng lại. Kết quả giúp từ `"ác tính"` tích hợp trọn vẹn ngữ cảnh và hiểu rằng nó đang mô tả cho *bệnh lý da của một nốt ruồi*.
  * **Ví dụ giá trị số thực tế (Concrete Numerical Values):**
    * Các vector này thực chất là các mảng số thực 768 chiều trong DistilGPT-2. Ví dụ:
      * Vector Query của từ `"ác tính"` tại một Head Attention:
        $$\mathbf{q}_{\text{"ác tính"}} = [0.12, \; -0.45, \; 0.89, \; \dots, \; -0.71] \in \mathbb{R}^{768}$$
      * Vector Key của từ `"ruồi"` tại Head tương ứng:
        $$\mathbf{k}_{\text{"ruồi"}} = [0.09, \; -0.51, \; 0.72, \; \dots, \; -0.68] \in \mathbb{R}^{768}$$
      * Độ tương quan (Attention Score) chưa chuẩn hóa giữa hai từ được tính bằng tích vô hướng (Dot Product):
        $$\text{Score}(\mathbf{q}_{\text{"ác tính"}}, \; \mathbf{k}_{\text{"ruồi"}}) = \mathbf{q} \cdot \mathbf{k} = (0.12 \times 0.09) + (-0.45 \times -0.51) + \dots + (-0.71 \times -0.68) = 12.85$$
  * **Các ma trận trọng số chiếu $W_q, W_k, W_v$:**
    * Là các ma trận tuyến tính biến đổi vector nhúng đầu vào $x$ (kích thước $d=768$) thành các không gian vector Query, Key, Value tương ứng:
      $$Q = W_q x, \quad K = W_k x, \quad V = W_v x$$
    * Trong DistilGPT-2, 3 ma trận này được **ghép nối lại (concatenate)** theo chiều rộng thành một lớp tích duy nhất là **`c_attn`** có ma trận trọng số $W_{\text{orig}} \in \mathbb{R}^{d \times 3d}$ (với $d=768$ và $3d=2304$) để tăng tốc độ tính toán song song.
* **Công thức tinh chỉnh LoRA (PEFT):**
  * Trong quá trình Fine-tuning, trọng số của lớp Attention gốc $W_0 \in \mathbb{R}^{d \times k}$ được đóng băng hoàn toàn. Hệ thống chỉ cập nhật ma trận biến thiên $\Delta W$ được phân tách thành tích của hai ma trận hạng thấp (low-rank) $A$ và $B$:
    $$W = W_0 + \Delta W = W_0 + B \cdot A$$
  * Giá trị đầu ra (forward pass) của lớp tích chập attention được tính bằng:
    $$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (B \cdot A) x$$
    * **Giải thích phép nhân và chia:** Lấy kết quả đầu ra của mạng đóng băng gốc $W_0 x$ cộng thêm tích của đầu vào $x$ với ma trận LoRA hạng thấp $(B \cdot A)x$, rồi **nhân với hằng số $\alpha$ và chia cho hạng rank $r$** ($\alpha / r$ là hệ số tỉ lệ giúp cân bằng độ lớn của các cập nhật trọng số LoRA mới thích ứng so với trọng số gốc có sẵn).
* **Tại sao tỉ lệ tham số huấn luyện lại là 2.13% mà không phải con số khác?**
  1. **Chứng minh toán học theo kiến trúc của lớp `c_attn` (Math constraint):**
     * Trong DistilGPT-2, chiều ẩn (hidden size) là $d = 768$. Lớp tích chiếu attention `c_attn` tích hợp cả 3 ma trận $W_q, W_k, W_v$ nên có kích thước $W_{\text{orig}} \in \mathbb{R}^{768 \times 2304}$ (ở đó $2304 = 768 \times 3$). Số lượng tham số gốc của lớp này là: $768 \times 2304 = 1,769,472$ trọng số.
     * Khi áp dụng LoRA với hạng rank **`r = 8`**, ta phân tách thành hai ma trận $A \in \mathbb{R}^{768 \times 8}$ và $B \in \mathbb{R}^{8 \times 2304}$. Số lượng tham số cần huấn luyện mới của lớp này chỉ còn:
       $$\text{Params}_{\text{LoRA}} = (768 \times 8) + (8 \times 2304) = 6,144 + 18,432 = 24,576 \text{ tham số}$$
       (Tức là đã giảm tới **$72 \text{ lần}$** số lượng tham số cho riêng lớp chiếu attention này: $\frac{24,576}{1,769,472} \approx 1.39\%$).
     * Nhân với số tầng $L=6$ của DistilGPT-2, ta thu được tổng số tham số LoRA là: $24,576 \times 6 = 147,456$ tham số.
  2. **Công thức tính toán tổng thể 2.13% trong mô hình VQA liên kết (Joint Training):**
     * Trong mô hình VQA liên kết (`CPUMedicalVQAModel`), ngoài 147,456 tham số LoRA của DistilGPT-2, ta còn mở khóa huấn luyện hoàn toàn các khối thích ứng bổ trợ. Cụ thể:
       * **Lớp chiếu đặc trưng ảnh (Projection Layer):** Cầu nối chiếu đặc trưng ảnh kích thước $1280$ của EfficientNet về chiều ẩn $768$ của DistilGPT-2.
         * *Công thức tính:* Gồm 2 lớp tuyến tính (Linear): Lớp 1 chiếu $1280 \rightarrow 768$ (số tham số $= 1280 \times 768 + 768 \text{ bias} = 983,808$). Lớp 2 chiếu $768 \rightarrow 768$ (số tham số $= 768 \times 768 + 768 \text{ bias} = 590,592$).
         * *Tổng tham số:* $\text{Params-Projection} = 983,808 + 590,592 = 1,574,400$ tham số.
       * **Bộ tăng cường ngữ nghĩa (SemanticEnhancer):** Kiến trúc nút cổ chai (Bottleneck) $1280 \rightarrow 256 \rightarrow 1280$ không sử dụng bias chèn sau backbone để bù đắp điểm yếu thiên lệch kết cấu (texture bias) của mạng EfficientNet.
         * *Công thức tính:* Gồm lớp co hẹp $1280 \times 256 = 327,680$ và lớp giãn rộng $256 \times 1280 = 327,680$, cộng thêm 1 hệ số tỉ lệ học được (`scale`).
         * *Tổng tham số:* $\text{Params-Enhancer} = 327,680 + 327,680 + 1 = 655,361$ tham số.
       * **Cầu nối chú ý chéo sâu (DeepCrossAttentionBridge):** Xếp chồng 2 lớp Cross-Attention kết hợp với cơ chế DropKey và nhiệt độ học được để thực hiện lập luận đa phương thức sâu sắc giữa Vision & Language.
         * *Tổng tham số:* $\text{Params-Bridge} \approx 2,952,192$ tham số (ở chế độ tối giản hóa mạng FFN để tối ưu hóa CPU).
       * **Bộ tiêm cấu trúc lâm sàng (ClinicalStructureInjector):** Mạng MLP siêu nhẹ mã hóa 11 biến lâm sàng (4 chỉ số ABCD + 7 lớp phân phối xác suất bệnh) thành 1 vector đặc trưng lâm sàng $768$ chiều chèn trực tiếp vào luồng token.
         * *Công thức tính:* Chiếu tuyến tính đầu vào qua lớp ẩn $11 \rightarrow 64 \rightarrow 768$. Gồm MLP1 ($11 \times 64 + 64 \text{ bias} = 768$), MLP2 ($64 \times 768 + 768 \text{ bias} = 49,920$), và chuẩn hóa LayerNorm ($768 \times 2 = 1536$).
         * *Tổng tham số:* $\text{Params-Injector} = 768 + 49,920 + 1536 = 52,224$ tham số.
     * **Công thức tổng tham số huấn luyện:**
       $$\text{Params-Trainable} = \text{Params-LoRA} + \text{Params-Projection} + \text{Params-Enhancer} + \text{Params-Bridge} + \text{Params-Injector} + \text{Params-Prefix}$$
       $$\text{Params-Trainable} \approx 147,456 + 1,574,400 + 655,361 + 2,952,192 + 52,224 + 3,072 = 5,384,705 \text{ tham số}$$
     * **Công thức tỉ lệ phần trăm cuối cùng:**
       $$\text{Trainable-Percent} = \frac{\text{Params-Trainable}}{\text{Params-Total-VQA-Model}} \times 100\% = \frac{5.38 \text{ triệu}}{252.8 \text{ triệu}} \times 100\% \approx 2.13\%$$
       (Với $\text{Params-Total-VQA-Model} \approx 252.8 \text{ triệu}$ là tổng số tham số của toàn bộ mô hình VQA bao gồm EfficientNet-B1, CBAM, các khối Projection, và DistilGPT-2 base).

---

### 10. Kết quả đánh giá mô hình VQA (BLEU Score)
* **Kết quả mô hình ngoại tuyến (Offline Model):**
  * Đánh giá định lượng trên tập Validation (12 mẫu câu hỏi lâm sàng thực tế):
    * **Average BLEU-1:** **`0.7269`** (độ khớp từ vựng $72.69\%$)
    * **Average BLEU-2:** **`0.6812`** (độ khớp từ vựng $68.12\%$)
  * **Nhận xét học thuật:** Mô hình offline đạt điểm BLEU rất cao do học thuộc tốt các cấu trúc câu trả lời mẫu y văn của chuyên gia trên tập dữ liệu hẹp, nhưng khả năng linh hoạt ngôn ngữ bị hạn chế khi gặp câu hỏi ngoài tập huấn luyện.
* **Mô hình trực tuyến (Online Model):**
  * **BLEU-1 trung bình:** **`10.91%`** (mức độ trùng khớp từ vựng thấp do mô hình trực tuyến sinh câu trả lời tự nhiên, dài, đa dạng từ ngữ và mang nhiều chi tiết y học phong phú hơn hẳn câu trả lời tham chiếu ngắn, mặc dù có độ chính xác y khoa thực tế tốt hơn).

---

### 11. Cấu trúc hệ thống RAG (Retrieval-Augmented Generation) ngoại tuyến
* **Mục đích:** Khắc phục hiện tượng ảo giác (hallucination) của mô hình ngôn ngữ lớn (LLM) bằng cách trích xuất văn bản hướng dẫn chẩn đoán và điều trị chính thức từ Bộ Y tế để làm căn cứ/ngữ cảnh (Context) trước khi LLM sinh câu trả lời tư vấn cho bác sĩ.
* **Các thành phần cốt lõi trong hệ thống RAG của chương trình:**
  1. **Kho ngữ cảnh / Tài liệu cơ sở (Medical Corpus):**
     * Tệp tài liệu y văn chuẩn `9_VQA/medical_guidelines.txt` chứa toàn bộ hướng dẫn điều trị chi tiết bằng tiếng Việt cho 7 nhóm bệnh lý da liễu trong chương trình.
     * **Luật phân đoạn tài liệu (Chunking Rule):** Hệ thống **không** sử dụng phương pháp cắt theo số ký tự hay số từ cố định (sliding window) vì dễ làm đứt gãy ngữ cảnh y khoa giữa chừng. Thay vào đó, tài liệu được **phân đoạn logic dựa trên cấu trúc thẻ phân loại bằng biểu thức chính quy (Regex Splitter)**:
       $$\text{Pattern Split} = \text{r"\[BỆNH LÝ \backslash d+:"}$$
       Cứ mỗi khi gặp tiêu đề chương mục dạng `[BỆNH LÝ 1: ...]`, `[BỆNH LÝ 2: ...]`, hệ thống sẽ cắt ra thành một khối dữ liệu (chunk) riêng biệt chứa trọn vẹn toàn bộ hướng dẫn chẩn đoán và điều trị của một bệnh cụ thể, đảm bảo tính nguyên vẹn ngữ cảnh của y văn.
  2. **Mô hình mã hóa Vector (Embedding Model):**
     * Sử dụng mô hình mã hóa **`all-MiniLM-L6-v2`** từ thư viện `sentence-transformers`.
     * **Mã hóa đối tượng nào (What is embedded):** Mã hóa **cả hai phía (câu hỏi truy vấn và tài liệu y văn)** để đưa chúng về cùng một không gian vector so sánh:
       * *Mã hóa tài liệu (Document/Chunk Embedding):* Lúc khởi tạo cơ sở dữ liệu, toàn bộ tài liệu y văn được cắt nhỏ và mã hóa trước thành các vector, lưu sẵn vào ChromaDB.
       * *Mã hóa câu hỏi (Query Embedding):* Khi bác sĩ nhập câu hỏi tự nhiên $q$, hệ thống mã hóa nó thành vector truy vấn $\mathbf{v}_q$ tại thời điểm chạy (runtime).
     * **Tại sao là 384 chiều (Why 384 dimensions):**
       * Con số **384** là kích thước đầu ra đặc trưng cố định (embedding dimension / hidden size) được thiết kế và huấn luyện sẵn của mô hình mạng nơ-ron transformer `all-MiniLM-L6-v2`.
       * Trọng số của mô hình này được tối ưu để ánh xạ bất kỳ đoạn văn bản nào thành một vector số thực có đúng 384 chiều, đảm bảo sự cân bằng xuất sắc giữa:
         1. *Độ nhẹ và tốc độ:* Tiết kiệm bộ nhớ RAM/ổ đĩa tối đa và tính khoảng cách Cosine cực nhanh trên CPU mà không cần GPU.
         2. *Độ chính xác ngữ nghĩa:* Mô hình đạt điểm chất lượng cao trên các bảng xếp hạng tìm kiếm ngữ nghĩa học thuật (Sentence-Transformers Benchmarks).
  3. **Cơ sở dữ liệu Vector (Vector Database):**
     * Sử dụng **ChromaDB** chạy ở chế độ lưu trữ ngoại tuyến (`chromadb.PersistentClient`) lưu tại thư mục `5_Results/chroma_db`.
     * Nhiệm vụ: Lưu trữ các vector nhúng 384 chiều của tài liệu y văn Bộ Y tế cùng metadata nhãn bệnh để phục vụ truy vấn tốc độ cao mà không cần kết nối mạng Internet.
  4. **Thuật toán truy xuất (Retrieval Algorithm):**
     * Khi người dùng gửi câu hỏi y khoa $q$, mô hình `all-MiniLM-L6-v2` sẽ mã hóa câu hỏi đó thành vector $\mathbf{v}_q$.
     * Hệ thống tiến hành so sánh độ tương đồng giữa $\mathbf{v}_q$ với toàn bộ các vector tài liệu $\mathbf{v}_d$ được lưu trữ trong ChromaDB bằng công thức tính **Khoảng cách Cosine (Cosine Distance)**:
       $$D_C(\mathbf{v}_q, \mathbf{v}_d) = 1 - \text{Cosine-Similarity}(\mathbf{v}_q, \mathbf{v}_d) = 1 - \frac{\mathbf{v}_q \cdot \mathbf{v}_d}{\|\mathbf{v}_q\| \|\mathbf{v}_d\|}$$
       * *Giải thích phép chia:* Lấy tích vô hướng của hai vector câu hỏi và tài liệu ($\mathbf{v}_q \cdot \mathbf{v}_d$) **chia cho tích độ dài Euclid (L2-norm) của chúng ($\|\mathbf{v}_q\| \|\mathbf{v}_d\|$)** để tính cosin góc giữa hai vector. Lấy 1 trừ đi tỉ số này để quy đổi thành khoảng cách (khoảng cách càng gần $0$ thì tài liệu càng tương đồng lớn với câu hỏi).
     * Hệ thống tự động chọn ra phân đoạn tài liệu có khoảng cách ngắn nhất để làm ngữ cảnh tin cậy.
  5. **Nhồi ngữ cảnh vào Prompt (Prompt Injection):**
     * Phân đoạn y văn chuẩn tìm được sẽ được tiêm trực tiếp vào trường dữ liệu `[CV_CONTEXT]` hoặc hệ thống prompt đầu vào của LLM (DistilGPT-2 LoRA hoặc Qwen chạy trên Ollama) để trói buộc phạm vi tư vấn của LLM hoàn toàn dựa trên y văn, triệt tiêu khả năng tự bịa đặt thuốc của mô hình.

---

### 12. Đặc tả đầu vào (Input) và đầu ra (Output) của từng thành phần trong hệ thống
Để nắm rõ luồng truyền dữ liệu tuần tự trong Pipeline của hệ thống, dưới đây là chi tiết đầu vào và đầu ra của từng khối chức năng:

| Khối chức năng / Thành phần | Đầu vào (Input) | Đầu ra (Output) |
| :--- | :--- | :--- |
| **1. Cổng lọc tiền xử lý (Safety Gate QA)** | - Ảnh chụp màu thô dạng RGB (JPEG/PNG/DICOM). | - Điểm số độ nét Laplacian (`blur_score`) và độ sáng trung bình (`brightness_score`).<br>- Quyết định từ chối (`warning` yêu cầu chụp lại) hoặc chấp nhận (`ok`). |
| **2. Phân đoạn tự động (DeepLabV3+)** | - Ảnh RGB đã được chấp nhận, resize về $256 \times 256 \times 3$ và chuẩn hóa Z-score. | - Mặt nạ nhị phân tổn thương da nháp $M_{\text{raw}} \in \{0, 1\}^{256 \times 256}$. |
| **3. Phân đoạn tương tác (SAM / GrabCut)** | - Ảnh gốc RGB.<br>- Tọa độ điểm click chuột $(x, y)$ từ bác sĩ. | - Mặt nạ nhị phân tương tác ôm khít vùng tổn thương được chọn. |
| **4. Trích xuất chỉ số hình học ABCD** | - Mặt nạ phân đoạn nhị phân cuối cùng $M \in \{0, 1\}^{H \times W}$.<br>- Ảnh gốc RGB (dùng riêng cho tính toán kênh màu). | - Dictionary chứa 4 chỉ số thực: $A$ (Asymmetry), $B$ (Border), $C$ (Color), $D$ (Diameter - quy đổi $mm$ hoặc pixel). |
| **5. Phân loại ảnh (EfficientNet-B1 + CBAM)** | - Ảnh ROI (Cắt ảnh gốc theo Bounding Box của mặt nạ tổn thương, đệm 10px, resize về $224 \times 224 \times 3$, chuẩn hóa ImageNet). | - Vector phân phối xác suất dự đoán của 7 nhãn bệnh da liễu $P(C_i \| \text{Ảnh})$. |
| **6. Hợp nhất Bayes đa phương thức** | - Vector xác suất hình ảnh $P(C_i \| \text{Ảnh})$ từ khối phân loại.<br>- Tuổi bệnh nhân (số thực).<br>- Giới tính (Nam/Nữ).<br>- Vị trí giải phẫu u (Văn bản). | - Vector xác suất cuối cùng đã hiệu chỉnh dịch tễ $P(C_i \| \text{Ảnh, Nhân khẩu})$ để đưa ra chẩn đoán chính xác nhất. |
| **7. Truy xuất y văn (ChromaDB RAG)** | - Câu hỏi tự nhiên từ người dùng (Văn bản $q$) được mô hình `all-MiniLM-L6-v2` mã hóa thành vector 384 chiều. | - Đoạn văn bản hướng dẫn điều trị tương đồng nhất trích từ cSDL y văn của Bộ Y tế. |
| **8. Trợ lý tư vấn VQA (DistilGPT-2 LoRA)** | - Prompt văn bản tích hợp bao gồm: *Chẩn đoán phân loại*, *Độ tin cậy*, *4 chỉ số ABCD*, *Đoạn ngữ cảnh RAG Bộ Y tế*, và *Câu hỏi của người dùng*. | - Chuỗi văn bản tư vấn lâm sàng chi tiết tiếng Việt và âm thanh nói phát ra (TTS). |


