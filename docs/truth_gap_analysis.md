# BÁO CÁO PHÂN TÍCH KHOẢNG CÁCH CÔNG NGHỆ (TRUTH GAP ANALYSIS) & CẨM NANG BẢO VỆ LUẬN VĂN

Tài liệu này phân tích sâu sắc các hạn chế kỹ thuật thực tế (Tech Gaps) của hệ thống **Hybrid Medical AI System (Perception + Reasoning + VQA)** hiện tại so với các mô hình Vision-Language Model (VLM) hiện đại (sát thời điểm 2025–2026), và đề xuất chiến lược trả lời thuyết phục trước Hội đồng phản biện.

---

## I. BẢN CHẤT KỸ THUẬT HỆ THỐNG HIỆN TẠI (SYSTEM PATHWAY)

Hệ thống của bạn thuộc nhóm: **Pre-VLM Hybrid Medical AI System** (Hệ thống y tế lai trước kỷ nguyên VLM tích hợp đầu-cuối).
* **Bản chất luồng:** Image $\rightarrow$ CV Feature Extractor (CNN + Attention) $\rightarrow$ MLP Projection Layer $\rightarrow$ Decoder (Autoregressive LM).
* **Vị trí nâng cấp:** Đã thay thế nút thắt cổ chai 1-token (Average Pooling) bằng **Multi-Token Spatial Projection** (bản đồ đặc trưng $7 \times 7 = 49$ tokens).

---

## II. CHI TIẾT 4 KHOẢNG CÁCH CÔNG NGHỆ (THE 4 TECH GAPS)

Hội đồng hoặc phản biện có chuyên môn sâu về học máy (Machine Learning) có thể sẽ đưa ra câu hỏi xoay quanh 4 vấn đề cốt lõi sau:

### Gap #1: Trích xuất Đa Token nhưng Thiếu Cơ chế Cross-Attention Thực sự (Đã Khắc Phục)
* **Giải pháp Kỹ thuật đã Triển khai:** Hệ thống đã được tích hợp lớp **Query-Conditioned Cross-Attention Bridge (`QueryConditionedAttentionBridge`)**. Cầu nối này sử dụng text context trích xuất từ câu hỏi người dùng kết hợp với các query tokens học được để làm truy vấn ($Q$), thực hiện Attention chéo với các đặc trưng ảnh không gian ($7 \times 7 = 49$ tokens) đóng vai trò Keys ($K$) và Values ($V$). Cơ chế này giúp nén và căn chỉnh động đặc trưng ảnh theo ngữ nghĩa câu hỏi thành 4 tokens điều hướng trước khi đưa vào Decoder.
* **Tác dụng:** Giải quyết triệt để vấn đề "nối đặc trưng thô" (concatenation) của các kiến trúc tiền-VLM, cho phép mô hình Decoder "chú ý" (attend) vào các vùng ảnh khác nhau dựa trên nội dung câu hỏi cụ thể của người dùng (question-conditioned visual selection).

### Gap #2: Thiếu Inductive Bias về Không gian (Lack of Spatial Position Encodings - Đã Khắc Phục)
* **Giải pháp Kỹ thuật đã Triển khai:** Tích hợp bộ **Mã hóa Vị trí Không gian (Spatial Position Embeddings)** kích thước $1 \times 49 \times 768$ (nn.Parameter) cộng trực tiếp vào các đặc trưng ảnh chiếu qua MLP.
* **Tác dụng:** Bổ sung cấu trúc hình học lưới (2D grid layout bias), giúp mô hình Decoder nhận diện được vị trí tương quan vật lý giữa các tokens ảnh (ví dụ: góc biên, trung tâm tổn thương) thay vì chỉ xử lý chúng như một chuỗi 1D phẳng vô định dạng.

### Gap #3: Sự phụ thuộc vào CNN Backbone (EfficientNet-B1 + CBAM)
* **Bản chất kỹ thuật:** Dù CBAM đã cải thiện khả năng tập trung vùng tổn thương, EfficientNet vẫn mang các đặc tính cục bộ của tích chập (inductive bias of convolution) thay vì có tầm nhìn toàn cục không giới hạn như các bộ mã hóa dựa trên Transformer (như ViT, CLIP ViT).
* **Hậu quả:** Khả năng mô hình hóa mối quan hệ ngữ nghĩa vĩ mô giữa tổn thương với các vùng da xung quanh bị giới hạn. Độ phân giải không gian cuối cùng ($7 \times 7$) là tương đối thấp cho việc phân tích các chi tiết cấu trúc cực nhỏ (fine-grained features) của tổn thương ác tính.

### Gap #4: Rào cản Dữ liệu huấn luyện (Data Constraint Bottleneck)
* **Bản chất kỹ thuật:** Tập dữ liệu huấn luyện VQA cục bộ cực kỳ nhỏ (~74-80 cặp câu hỏi - trả lời).
* **Hậu quả:** Việc nâng kích thước nhúng ảnh từ 1 token lên 49 tokens làm tăng số lượng tham số tương tác (do chiều dài chuỗi đầu vào tăng), làm tăng nguy cơ **quá khớp (overfitting)** trên tập dữ liệu nhỏ. Mô hình chủ yếu ghi nhớ các mẫu câu (memorization) hơn là học được khả năng suy luận lâm sàng linh hoạt.

---

## III. CẨM NĂNG BẢO VỆ LUẬN VĂN: CÂU HỎI PHẢN BIỆN & GỢI Ý TRẢ LỜI

Khi bảo vệ luận văn, thay vì tìm cách che giấu các hạn chế kỹ thuật trên (rất dễ bị thầy cô có chuyên môn phát hiện và đánh giá thấp), **chiến lược tốt nhất là chủ động thừa nhận và giải thích rõ bản chất khoa học đằng sau các lựa chọn thiết kế**. Dưới đây là các câu hỏi rủi ro cao và gợi ý phản biện:

### Câu hỏi 1: "Kiến trúc VQA của em có thực sự là một mô hình Vision-Language Model (VLM) đồng nhất hay không? Việc nối đặc trưng ảnh vào LLM như vậy có hạn chế gì?"
* **Gợi ý trả lời:**
  > *"Kính thưa Hội đồng, hệ thống VQA ngoại tuyến cục bộ của em thuộc kiến trúc lai sơ khởi (Pre-VLM Hybrid System), chưa phải là một mô hình Vision-Language Foundation Model đồng nhất tích hợp Cross-Attention động như LLaVA hay Med-Flamingo. Hạn chế lớn nhất của phương pháp nối trực tiếp đặc trưng ảnh vào không gian nhúng văn bản (Linear Projection + Concat) là Decoder ngôn ngữ (DistilGPT-2) phải xử lý các đặc trưng ảnh này như các token từ ngữ thông thường qua cơ chế Self-Attention, làm thiếu đi sự tương tác chéo đa phương thức ở các tầng sâu (deep cross-modal alignment). Tuy nhiên, lựa chọn thiết kế này mang tính thực tiễn cao: nó giúp mô hình cực kỳ nhẹ (chỉ khoảng 82M tham số cho Decoder), có thể huấn luyện LoRA rất nhanh và chạy mượt mà trên môi trường CPU cục bộ không cần hạ tầng GPU đắt đỏ."*

### Câu hỏi 2: "Khi chuyển từ 1 token sang 49 tokens không gian bằng cách làm phẳng bản đồ đặc trưng, em có thêm mã hóa vị trí không gian (Spatial Position Encoding) không? Nếu không thì mô hình hiểu cấu trúc hình học thế nào?"
* **Gợi ý trả lời:**
  > *"Trong phiên bản hiện tại, khi làm phẳng bản đồ đặc trưng $7 \times 7$ thành 49 tokens ảnh, em chưa tích hợp thêm tọa độ không gian 2D (Spatial Position Encoding) vào chuỗi nhúng này. Đây quả thực là một hạn chế về mặt lý thuyết khiến mô hình Decoder-only coi 49 đặc trưng ảnh như một chuỗi tuần tự thông thường thay vì một lưới không gian 2 chiều. Tuy nhiên, tính chất hình học (bất đối xứng, bờ răng cưa) vẫn được bảo toàn một phần từ bộ mã hóa EfficientNet-B1 nhờ các lớp tích chập cục bộ và khối chú ý CBAM đã học trước đó. Trong định hướng phát triển tiếp theo, em sẽ tích hợp thêm mã hóa vị trí Grid Position Embeddings để tối ưu hóa khả năng hiểu anatomy của Decoder."*

### Câu hỏi 3: "Dữ liệu huấn luyện VQA của em rất nhỏ (khoảng 80 mẫu). Làm thế nào em chứng minh được mô hình không bị quá khớp (overfitting) khi tăng số lượng token ảnh từ 1 lên 49?"
* **Gợi ý trả lời:**
  > *"Dữ liệu VQA quy mô 74-80 mẫu quả thực là một giới hạn lớn về mặt thống kê và là thách thức chung đối với các hệ thống AI y tế giai đoạn thử nghiệm lâm sàng (clinical prototype). Để chống quá khớp khi nâng lên cơ chế 49-token, em đã thực hiện đồng thời: (1) Cố định toàn bộ trọng số của bộ mã hóa hình ảnh EfficientNet-B1, (2) Chỉ huấn luyện lớp Projection tuyến tính siêu nhẹ và áp dụng LoRA (r=8) lên các lớp Attention của DistilGPT-2, (3) Áp dụng Dropout cao (0.3) tại các lớp chiếu đặc trưng. Dữ liệu thực nghiệm cho thấy mô hình nâng cấp cải thiện đáng kể khả năng phân biệt bệnh lý và đa dạng hóa câu trả lời thay vì chỉ sụp đổ về nhãn NV phổ biến. Dù vậy, em hoàn toàn thừa nhận đây là kết quả thử nghiệm trên tập test nhỏ, và đóng góp chính của luận văn là đề xuất một giải pháp tích hợp (End-to-End CDSS Pipeline) chạy được thực tế hơn là tối ưu hóa độ chính xác tuyệt đối trên dữ liệu lớn."*

### Câu hỏi 4: "Tại sao hệ thống của em phải chia làm hai nhánh Online (Cloud GPT-4o-mini + RAG) và Offline (CPUMedicalVQAModel)? Tại sao không dùng RAG cho Offline?"
* **Gợi ý trả lời:**
  > *"Em thiết kế hệ thống theo cấu trúc 3 lớp (Perception, Reasoning, Language) chia hai nhánh dựa trên triết lý triển khai thực tế (Deployment-aware). Nhánh Online tận dụng sức mạnh lập luận vượt trội và khả năng truy xuất y văn thời gian thực (RAG) của GPT-4o-mini nhằm cung cấp chẩn đoán chất lượng cao nhất cho bác sĩ khi có Internet. Nhánh Offline hướng tới sự độc lập, bảo mật dữ liệu tuyệt đối (privacy-first) và khả năng hoạt động ở các vùng sâu vùng xa không có mạng internet. Do mô hình cục bộ DistilGPT-2 có dung lượng quá nhỏ (82M) và tài nguyên CPU hạn chế, việc chạy một hệ thống Vector DB và nhúng RAG cục bộ sẽ làm chậm đáng kể thời gian phản hồi. Vì vậy, em đã xây dựng một 'Cơ sở tri thức chuyên gia thu nhỏ' (Clinical Knowledge Base) tích hợp sẵn trong ứng dụng để hỗ trợ mô hình Offline đưa ra các khuyến cáo y tế an toàn mà không cần truy xuất RAG."*

---

## IV. TỔNG KẾT THẾ MẠNH CỦA ĐỒ ÁN (WHAT MAKES IT STRONG)

Dù có các khoảng cách công nghệ về mặt mô hình hóa VLM, đồ án của bạn vẫn có những điểm cực kỳ mạnh mẽ để ghi điểm tuyệt đối:
1. **Thiết kế Cổng An toàn (Safety Gate):** Thể hiện tư duy y đức chuyên nghiệp của Clinical AI. Không để LLM tự do suy diễn trên ảnh nhiễu, mờ hoặc phân loại có độ tin cậy thấp.
2. **Quy tắc Lâm sàng Cứng (Clinical Guardrails):** Thể hiện sự hiểu biết sâu sắc về y tế (tuyệt đối không kê đơn thuốc, cảnh báo có chừng mực giữa u lành và u ác).
3. **Cơ chế tương thích ngược (Backward Compatibility):** Chứng minh năng lực công nghệ phần mềm thực tế, biết cách quản lý phiên bản kiến trúc mô hình (production-grade engineering).
4. **Hệ thống hóa (System-level AI):** Đây không phải là một mô hình đơn lẻ mà là một hệ thống CDSS hoàn chỉnh từ Phân vùng $\rightarrow$ Phân loại $\rightarrow$ Chỉ số ABCD $\rightarrow$ VQA $\rightarrow$ Bệnh án điện tử Cloud Firestore.
