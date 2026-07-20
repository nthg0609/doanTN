# Chốt danh sách sửa slide — đối chiếu bản 28-slide hiện tại với code đã cập nhật

Đã đọc lại toàn bộ 28 trang bạn gửi. Tin tốt: phần lớn các lỗi ở vòng review trước đã được sửa đúng (Safety Gate text, EHR collection, số 88.65%/89.39%, TTA 0.6pp/217-390, Bayes fusion đổi tên, thêm slide "Điểm mới"...). Nhưng có **3 lỗi mới/còn sót ở mức nghiêm trọng** cần sửa trước, cộng thêm vài chỗ nhỏ.

---

## A. BẮT BUỘC SỬA NGAY (nghiêm trọng nhất trong toàn bộ đợt review)

### A1. Slide 26 "HẠN CHẾ" — thân slide không hề nói về hạn chế

Đây là lỗi nặng nhất hiện tại: tiêu đề slide 26 là **"HẠN CHẾ"**, nhưng nội dung bên trong lại là *"Hỗ trợ tự đào tạo bác sĩ..."* và *"Kết luận: Xây dựng thành công hệ thống..."* — **giống hệt** nội dung của slide 27 ngay sau nó. Đây rõ ràng là lỗi copy-paste nhầm, thân slide "Hạn chế" đã bị dán trùng nội dung "Kết luận" thay vì viết hạn chế thật. Nếu hội đồng thấy 2 slide liên tiếp trùng nội dung gần như 100%, đây là điểm trừ rất dễ thấy về sự cẩn thận.

**Nội dung đề xuất thay hẳn cho slide 26** (đã cập nhật theo đúng các fix code vừa làm — phần nào đã khắc phục thì ghi rõ đã khắc phục, không giấu):
> - Dữ liệu huấn luyện VQA còn nhỏ (~74-80 mẫu gốc, đánh giá trên 12 mẫu, cùng trích từ một tệp gốc) — rủi ro ghi nhớ mẫu hơn là suy luận lâm sàng tổng quát.
> - Model phân loại có dấu hiệu overfit nhẹ (train ~99% / val ~89%); đã áp dụng Dropout, Weighted Sampler, Early Stopping, augmentation để kiểm soát nhưng chưa loại bỏ hoàn toàn.
> - Cơ chế mã hóa EHR hiện ở mức nguyên mẫu (XOR + khóa cố định trong mã nguồn), chưa đạt chuẩn production như AES-256/KMS.
> - Lớp hậu kiểm guardrail thuốc (quét từ khóa/liều lượng trên câu trả lời) hiện là danh sách khởi tạo ban đầu, cần mở rộng và kiểm thử thêm.
> - λ tự động theo entropy là một heuristic dựa trên softmax chưa hiệu chỉnh (uncalibrated), chưa được tối ưu bằng dữ liệu thật.
> - Ablation "+7% khi cắt ROI" chưa xác định lại được nguồn số liệu gốc — cần kiểm chứng hoặc nêu định tính thay vì con số cụ thể (xem mục B3).

### A2. Slide 9 — sơ đồ pipeline vẫn còn Laplacian/Fitzpatrick, và sai cả vị trí Safety Gate

Slide 8 (text) đã sửa đúng mô tả Safety Gate, nhưng **sơ đồ hình ảnh ở slide 9 chưa được vẽ lại** — vẫn còn nguyên 2 hộp "Kiểm tra độ mờ (Laplacian Var < Threshold)" và "Kiểm tra phơi sáng (Fitzpatrick Scale)". Hai slide liền kề nhau giờ **tự mâu thuẫn nhau**: slide 8 nói một kiểu, sơ đồ slide 9 vẽ kiểu khác — còn dễ bị bắt lỗi hơn cả lúc trước vì giờ mâu thuẫn ngay trong nội bộ deck.

Thêm nữa, tôi xác nhận qua code (`unified_pipeline.py::run()`): **chỉ có 1 lần gọi Safety Gate duy nhất**, và nó chạy **sau khi cả 2 nhánh (phân đoạn + phân loại) đã chạy xong**, dùng chính kết quả của cả 2 nhánh làm input. Tức là hộp "Bộ lọc an toàn tiền xử lý (Safety Gate)" đặt **trước** khi tách nhánh trên sơ đồ hiện tại **không tồn tại** — chỉ có hộp "Hậu kiểm lâm sàng" là đúng vị trí thật.

**Cách sửa:** dùng sơ đồ mermaid tôi đã đưa ở lượt trả lời trước (bản `flowchart LR`, đã có class màu khớp bản gốc) — xóa hẳn hộp Safety Gate tiền xử lý ở đầu, chỉ giữ 1 Safety Gate duy nhất sau khi 2 nhánh hợp nhất, với đúng 2 tiêu chí: diện tích/độ phức tạp biên + độ tin cậy phân loại.

### A3. Slide 22 — sơ đồ vẫn còn "(Tuân thủ HIPAA)"

Text ở slide 21 đã bỏ HIPAA đúng như khuyến nghị, nhưng **hộp trong sơ đồ ở slide 22** vẫn ghi nguyên: *"Đồng bộ lên Cloud Firestore (Tuân thủ HIPAA)"*. Text và hình lại mâu thuẫn nhau lần nữa. Sửa hộp này thành *"Đồng bộ lên Cloud Firestore"* hoặc *"(Ẩn danh hóa)"* — bỏ hẳn chữ HIPAA khỏi mọi hình ảnh, không chỉ khỏi văn bản.

---

## B. NÊN SỬA (chính tả, từ tuyệt đối còn sót)

1. **Lỗi chính tả — dấu chấm đôi**, xuất hiện ở 3 nơi:
   - Slide 6: *"...nhánh VQA chạy offline hoàn toàn trên CPU.."* → bỏ 1 dấu chấm.
   - Slide 14: *"...khi ảnh có kèm metadata DICOM.."* → bỏ 1 dấu chấm.
   - Slide 26/27 (Kết luận): *"...(232ms/lượt).."* → bỏ 1 dấu chấm.
2. **Lỗi ngoặc thừa** ở slide 17: *"...Online GPT-4o-mini+RAG BLEU-1 = 0.12). BLEU không phản ánh..."* — có dấu `)` thừa không khớp cặp, sửa thành `...BLEU-1 = 0.12. BLEU không phản ánh...`.
3. **Từ tuyệt đối còn sót:**
   - Mục lục (6 slide: 2,3,7,11,19,23): *"...tối ưu đa luồng"* → đổi thành *"song song hóa đa luồng"* hoặc *"cải thiện hiệu năng đa luồng"*.
   - Slide 12: *"...đảm bảo đo đạc chỉ số ABCD luôn ổn định"* → đổi thành *"giúp giảm rủi ro đo đạc thất bại khi mô hình chính không phân đoạn được"* (đây là chỗ duy nhất từ review trước chưa được sửa).
   - Slide 27: *"cố vấn y khoa 24/7 trực quan"* → nên đổi thành *"công cụ hỗ trợ tra cứu y văn có sẵn liên tục"* để tránh gợi ý AI thay thế vai trò bác sĩ (rủi ro pháp lý/đạo đức nếu bị hỏi).
4. **Nội dung trùng lặp** ở slide 20: 2 gạch đầu dòng "CSDL Firestore" và "Hồ sơ EHR chi tiết" nói gần như cùng một ý (ảnh gốc, mask, Grad-CAM, ABCD, VQA) — gộp lại thành 1 bullet cho gọn.
5. **Câu lủng củng** ở slide 21: *"Băm SHA-256 CCCD/Tên làm Document ID, và Băm SHA-256 (16 ký tự đầu)..."* — lặp "Băm SHA-256" 2 lần trong 1 câu, đọc như lỗi ghép câu. Sửa gọn: *"Băm SHA-256 (16 ký tự đầu) của CCCD/Tên để tạo Document ID ẩn danh; thông tin nhạy cảm được che bằng XOR + Base64 ở mức nguyên mẫu..."*

---

## C. BỔ SUNG NỘI DUNG/HÌNH ẢNH ĐỂ KHỚP VỚI CODE MỚI (λ tự động, guardrail 2 lớp, ROI-crop)

1. **Slide 15 — chưa nhắc gì đến λ tự động (mới cài đặt trong code):** hiện chỉ nói *"trọng số λ do bác sĩ điều chỉnh"*. Cần bổ sung 1 câu: *"λ được tính tự động theo độ bất định (entropy) của phân phối xác suất ảnh — ảnh càng rõ ràng thì càng tin ảnh hơn; bác sĩ vẫn có thể chuyển sang chế độ đặt λ thủ công."* Đây là tính năng thật vừa code xong, nên đưa vào để tăng điểm, đừng bỏ lỡ.
2. **Slide 6 (Điểm mới) — thiếu đúng yêu cầu "nêu tên cụ thể":** hội đồng yêu cầu rõ *"nêu cụ thể tên chương trình thực tế cũ"*, nhưng slide hiện chỉ viết chung chung "các ứng dụng tự soi da phổ biến hiện nay". Tôi đã tra cứu thật (web search) 2 hệ thống có thể trích dẫn an toàn:
   - **SkinVision** (ứng dụng thương mại đang hoạt động): độ nhạy 80%, độ đặc hiệu 78% (n=267, theo tổng hợp hệ thống trên PubMed/PMC) — chỉ đánh giá rủi ro, không giải thích, cần mạng.
   - **Google DermAssist**: gợi ý 280+ bệnh da từ ảnh + triệu chứng, CE-mark Class I tại EU, không có ABCD/Grad-CAM, không hợp nhất dịch tễ có hệ thống.
   Nên thêm 1 bảng so sánh ngắn (3 cột: Tiêu chí | SkinVision/DermAssist | Đồ án này) vào slide 6 — đây là hình/bảng còn thiếu quan trọng nhất của toàn deck, và có thể gộp trực tiếp vào slide 6 hiện tại (không cần thêm slide mới, giữ đúng ngân sách 28 trang).
3. **Slide 18 — không cần sửa, đã đúng:** kiểm tra kỹ, sơ đồ này thật ra đã đặt bước "Kiểm tra từ khóa kê đơn thuốc nhạy cảm" SAU khi LLM sinh câu trả lời (đúng thiết kế output-side) — và code guardrail 2 lớp tôi vừa cài đặt khớp đúng sơ đồ này. Không cần vẽ lại, chỉ cần khi nói nên nhấn: *"đây là lớp hậu kiểm tất định, độc lập với việc LLM có tuân thủ system prompt hay không"*.
4. **Slide 24 (UML) — tên hàm private của SafetyGate vẫn là hư cấu:** class `SafetyGate` trong sơ đồ liệt kê `-_check_blur(image) bool` và `-_check_exposure(image) bool` — đây là 2 hàm **không tồn tại** trong code thật (`safety_gate.py` chỉ có `evaluate()`, không tách blur/exposure riêng). Cần xóa 2 dòng này khỏi UML hoặc đổi thành đại diện đúng, ví dụ chỉ giữ `+evaluate(metrics, cls_confidence, image_type) GateResult`.
5. **Slide 5 — bổ sung 1-2 dòng về bộ dữ liệu** (không cần slide riêng, tránh phình số trang): thêm ngắn gọn "HAM10000: 10.015 ảnh (train 14.021/val 3.004/test 3.005 sau chia ROI); phân đoạn: ISIC 2018, 390 ảnh test" vào cuối bullet "Đối tượng nghiên cứu" đã có sẵn.

---

## D. CẤU TRÚC & THỜI LƯỢNG

- Tổng hiện tại: **28 trang**. Nếu tính đúng luật "không tính mục lục" (6 trang mục lục: 2,3,7,11,19,23) → còn **22 trang nội dung thật**, vẫn nằm trong khung 20-25. Nhưng nếu hội đồng đếm tất cả 28 trang theo mặt, sẽ bị coi là vượt — nên vẫn giữ khuyến nghị cũ: gộp bớt 6 slide mục lục/divider thành 1-2 slide, đưa tổng về khoảng 22-24 trang cho an toàn tuyệt đối với mọi cách đếm.
- Thời lượng ước tính sau khi thêm slide "Điểm mới" (30s) và "Hạn chế" (35s, sau khi viết đúng nội dung): khoảng **14 phút 50 giây – 15 phút**, sát trần trên của khung 12-15 phút. Nên cắt bớt 4-5 slide mục lục divider (mỗi cái ~8-12s) để có buffer, tổng có thể lấy lại được khoảng 40-50 giây.

---

## Tóm tắt thứ tự làm

1. Viết lại đúng nội dung slide 26 "Hạn chế" (A1) — **quan trọng nhất**.
2. Vẽ lại sơ đồ slide 9 theo mermaid đã đưa, bỏ Laplacian/Fitzpatrick và sửa đúng vị trí Safety Gate (A2).
3. Sửa hộp "(Tuân thủ HIPAA)" trong sơ đồ slide 22 (A3).
4. Sửa 3 lỗi dấu chấm đôi + 1 lỗi ngoặc thừa + các từ tuyệt đối còn sót (B).
5. Thêm câu về λ tự động vào slide 15, thêm bảng so sánh SkinVision/DermAssist vào slide 6, sửa UML slide 24, thêm số liệu dataset vào slide 5 (C).
6. Cân nhắc gộp bớt slide mục lục để chắc chắn về số trang và thời lượng (D).
