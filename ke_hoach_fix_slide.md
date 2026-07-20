# Kế hoạch sửa slide — dựa trên đối chiếu checkpoint thật, log thật, code thật

Tài liệu này **không dựa vào `docs/experimental_results.md`** làm nguồn chính (đúng như bạn lưu ý — đó chỉ là file text tổng hợp thủ công, có thể lỗi thời). Mọi con số dưới đây được lấy trực tiếp từ: file checkpoint `.pth` (đọc bằng `torch.load`, kiểm tra cả kiến trúc thật qua cấu trúc `state_dict`), file JSON log (`3_Checkpoints/*.json`, `5_Results/*.json`), và source code đang chạy thật (`app_streamlit.py`, `pipeline/*.py`).

Quyết định bạn đã chọn: **dùng đúng số thật của model đang chạy (88.65% / 89.39%)**, và tôi chỉ đưa kế hoạch bằng văn bản để bạn tự sửa trong PowerPoint.

---

## 0. Phát hiện quan trọng nhất — phải sửa trước tiên

Tôi đã mở trực tiếp file `4_Models/classification/efficientnet_attention_best.pth` — đây chính xác là file mà `pipeline/model_registry.py` load vào app (`fallback=... "efficientnet_attention_best.pth"`), tức là **model đang chạy thật trong ứng dụng của bạn**.

**Bằng chứng đọc trực tiếp từ checkpoint (không suy đoán):**
```
epoch: 43 (best), tổng 50 epoch
val_acc:      89.39283101682517
best_val_acc: 89.39283101682517
classes: 7 lớp (AKIEC, BCC, BKL, DF, MEL, NV, VASC)
train_acc 5 epoch cuối: ~99% | val_acc 5 epoch cuối: ~85-89%  → dấu hiệu overfit rõ
```
Số này khớp chính xác từng chữ số với `3_Checkpoints/06_classification_complete.json` → `test_acc: 88.64786695589298`, `best_val_acc: 89.39283101682517`.

Tôi cũng kiểm tra **kiến trúc thật** bằng cách đếm số block trong từng stage của `state_dict` (không đọc tên biến, đọc cấu trúc số layer thật):
- File **`efficientnet_attention_best.pth`** (đang dùng): số block mỗi stage = `[2,3,3,4,4,5,2]` → đây chính xác là cấu trúc **EfficientNet-B1** (depth coefficient 1.1).
- File **backup** `efficientnet_attention_best_backup_20260401...pth` và `_20260415...pth`: số block mỗi stage = `[1,2,2,3,3,4,1]` → đây là cấu trúc **EfficientNet-B0** (depth coefficient 1.0), và có `val_acc = 96.90412782956058` — khớp với con số 96.90%/95.01% trong `FINAL_REPORT.txt`.

**Kết luận không thể chối cãi:** con số 95.01% (test) / 96.90% (val) trên slide 24 thuộc về **một checkpoint B0 đã bị thay thế**, hiện chỉ còn tồn tại dưới dạng file backup, **không phải model đang chạy**. Model thật đang chạy (B1) có test accuracy **88.65%**, best val **89.39%**, và có dấu hiệu overfit (train~99% vs val~88%).

`FINAL_REPORT.txt` (file mà `docs/experimental_results.md` copy lại) tự nó cũng ghi nhầm tên kiến trúc ở mục 4 ("EfficientNet-B0") trong khi phần VQA ở mục 5 lại nhắc "CBAM EfficientNet-B1" — tức là ngay cả file report cũng đã lẫn giữa 2 lần chạy khác nhau. Đây là lý do tại sao không nên tin thẳng file `.md`/`.txt` mà phải bắt tận checkpoint, đúng như bạn nghi ngờ.

---

## 1. Nguyên tắc ưu tiên khi sửa

- **P0 — bắt buộc, rủi ro cao nếu không sửa:** sai lệch sẽ bị hỏi trúng và không có gì để trả lời.
- **P1 — nên sửa:** làm yếu lý lẽ hoặc dễ bị vặn nếu hội đồng đào sâu.
- **P2 — nên bổ sung:** không sai, nhưng thiếu và dễ bị đánh giá là "chưa đầy đủ".
- Với mỗi mục: nêu **vị trí**, **nội dung cũ**, **nội dung mới đề xuất**, **vì sao**, **mức ưu tiên**.

---

## 2. Danh sách sửa theo từng slide

### Slide 2, 3, 6, 10, 18, 22 (mục lục / divider) — P0 + P1

- **Lỗi chính tả P0**, lặp lại ở cả 6 vị trí: *"Phần 5: **Kết Kết** Quả Thử Nghiệm & Kết Luận"* → sửa thành *"Phần 5: **Kết Quả** Thử Nghiệm & Kết Luận"*.
- **Cụm "tuân thủ HIPAA" P0**, lặp lại ở cả 6 vị trí (dòng mô tả Phần 4): *"Cấu trúc Firestore phân cấp, mã hóa XOR-Base64 tuân thủ HIPAA và tối ưu đa luồng."* → *"Cấu trúc Firestore phân cấp, ẩn danh hóa dữ liệu và tối ưu đa luồng."* (bỏ hẳn HIPAA và chữ "mã hóa" chuẩn, xem lý do ở mục Slide 20).
- **P1 (tùy chọn):** gộp bớt 5/6 slide divider này thành 1-2 slide để giảm cảm giác lặp — không bắt buộc nếu bạn muốn giữ nhịp chuyển phần rõ ràng khi nói.

### Slide 7 — Kiến trúc tổng quan (mô tả Safety Gate) — P0

- **Cũ:** *"Bộ lọc Safety Gate: Tự động đánh giá chất lượng ảnh (độ mờ Laplacian, phơi sáng thích ứng chủng tộc Fitzpatrick) trước khi chạy."*
- **Mới (khớp đúng `pipeline/safety_gate.py`):** *"Bộ lọc Safety Gate: Đánh giá diện tích và độ phức tạp biên của mặt nạ tổn thương cùng độ tin cậy phân loại, với ngưỡng thích ứng riêng cho ảnh dermoscopy và ảnh chụp điện thoại, trước khi cho phép phân tích lâm sàng."*
- **Vì sao:** code không có Laplacian/Fitzpatrick ở bất kỳ đâu (đã grep toàn repo, 0 kết quả ngoài slide). Đây là chi tiết rất cụ thể, dễ bị hỏi trúng và không trả lời được.

### Slide 8 — Sơ đồ pipeline (hình ảnh/diagram nhúng) — P0

Hộp "Bộ lọc an toàn tiền xử lý (Safety Gate)" đang ghi "Kiểm tra độ mờ (Laplacian Var < Threshold)" và "Kiểm tra phơi sáng (Fitzpatrick Scale)". Đây là hình vẽ (không phải text thường), cần **vẽ lại 2 hộp con** thành:
- Hộp 1: "Kiểm tra diện tích & độ phức tạp biên mặt nạ (area_ratio, border_complexity)"
- Hộp 2: "Kiểm tra độ tin cậy phân loại (cls_confidence ≥ 0.60)"

Nếu không có thời gian vẽ lại diagram, **tối thiểu phải nói đúng bằng lời khi thuyết trình** (xem script mục 5) — nhưng vẫn nên sửa vì hội đồng thường nhìn vào slide khi hỏi lại.

### Slide 11 — Kết quả phân đoạn — P2 (không bắt buộc)

Số liệu Dice/IoU trên slide (U-Net 89.43/81.77, DeepLabV3+ 91.28/84.55, Hybrid DeepLabV3+ 90.93/84.33, Hybrid-Max 91.32/84.70) **khớp chính xác** với `5_Results/ablation_fusion_results.json` và `FINAL_REPORT.txt` — giữ nguyên, đây là phần dữ liệu đáng tin nhất trong deck.

Tôi kiểm tra thêm claim "Multi-scale TTA tối ưu phân đoạn" bằng benchmark thật `tta_vs_standard_benchmark_isic.json` (**390 ảnh ISIC thật**, không phải benchmark rỗng 1-2 ảnh khác trong cùng thư mục): TTA thắng 217/390 ảnh (56%), IoU trung bình tăng từ 0.8138 → 0.8197 (+0.6 điểm phần trăm). Đây là cải thiện **có thật nhưng khiêm tốn**, không nên nói "tối ưu" (từ tuyệt đối) — đổi thành "TTA cải thiện IoU trung bình khoảng 0.6 điểm phần trăm trên tập ISIC 390 ảnh, thắng trên 56% số ảnh kiểm thử" nếu muốn có số liệu chính xác để bảo vệ.

### Slide 13 — Chỉ số ABCD — P1

- **Cũ:** *"...được trích xuất để quy đổi pixel sang kích thước thực tế, **đảm bảo** tính chính xác lâm sàng."*
- **Mới:** *"...được trích xuất để quy đổi pixel sang kích thước thực tế, **góp phần cải thiện** độ chính xác quy đổi khi ảnh có kèm metadata DICOM."*
- **Vì sao:** "đảm bảo" là từ tuyệt đối, và chỉ đúng khi có PixelSpacing trong DICOM — ảnh JPG/PNG thường không có.

### Slide 14, 15 — Phân loại & Bayes fusion — P0 (nội dung) + P0 (số liệu)

**(a) Số liệu classification — đây là điểm sửa quan trọng nhất toàn deck.**
- **Cũ (slide 24, nhưng liên quan trực tiếp tới model nói ở slide 14):** ngầm định model EfficientNet-B1+CBAM đạt 95.01%.
- **Mới:** Model EfficientNet-B1 + CBAM Attention (đang chạy thật trong ứng dụng) đạt **độ chính xác kiểm thử 88.65%**, độ chính xác validation tốt nhất **89.39%** (epoch 43/50), trên tập HAM10000-ROI 3.005 mẫu kiểm thử.
- Đưa số này vào slide 14 (nơi giới thiệu classifier) thay vì chỉ để ở slide 24, để tránh mâu thuẫn nếu hội đồng lật lại slide.

**(b) Cách gọi tên fusion:**
- **Cũ:** *"Late Fusion Bayes: Hợp nhất xác suất hình ảnh với demographics dịch tễ ... theo công thức Bayes."*
- **Mới:** *"Hợp nhất xác suất kiểu Bayes có trọng số: tính xác suất hậu nghiệm P(bệnh|tuổi, giới, vị trí) theo đúng công thức Bayes, sau đó trộn với xác suất từ ảnh bằng trọng số λ do bác sĩ điều chỉnh (log-linear weighted fusion)."*
- **Vì sao:** code (`multimodal_fusion.py`) chỉ đúng Bayes ở bước tính tiền nghiệm dịch tễ; bước trộn cuối `P(image)^λ · P(demo)^(1-λ)` là log-linear pooling, không phải hậu nghiệm Bayes chuẩn (không nhân trực tiếp likelihood × prior kiểu Bayes gốc). Nói đúng để không bị bắt bẻ nếu có ai hỏi kỹ về xác suất thống kê.

### Slide 16, 17 — VQA & RAG — P0 (guardrail) + P2 (bổ sung eval)

**(a) Guardrail thuốc:**
- **Cũ:** sơ đồ có nhánh "Kiểm tra từ khóa kê đơn thuốc nhạy cảm" → "Phát hiện tự kê đơn? Có/Không" → chặn.
- **Mới:** *"Ràng buộc an toàn ở tầng system prompt: LLM được chỉ dẫn rõ 'tuyệt đối cấm nêu tên biệt dược, liều lượng cụ thể', chỉ được giải thích cơ chế bệnh và khuyến nghị gặp bác sĩ."* Bỏ sơ đồ nhánh rẽ kiểm tra từ khóa (không tồn tại trong code), hoặc — nếu chọn hướng code fix ở mục 4 — giữ sơ đồ và làm nó thành sự thật.
- **Vì sao:** `app_streamlit.py::_build_system_prompt` chỉ có đoạn văn bản `[GUARDRAIL_RULES]` gửi cho LLM, không có bước hậu kiểm/regex nào trên câu trả lời sinh ra.

**(b) Bổ sung đánh giá chất lượng VQA (số liệu thật, đọc trực tiếp từ `5_Results/vqa_evaluation_report.json` và `vqa_online_evaluation_report.json`, không phải từ file `.md`):**
```
VQA Offline (CPUMedicalVQAModel): 12 mẫu kiểm thử, BLEU-1 trung bình = 0.7269, BLEU-2 = 0.6812
VQA Online (GPT-4o-mini + RAG):   12 mẫu kiểm thử, BLEU-1 trung bình = 0.1236, BLEU-2 = 0.0581
```
(Lưu ý: 2 con số online hơi khác `docs/experimental_results.md` — 0.1236 mới là số đọc trực tiếp từ file JSON gốc, dùng số này.)
Thêm nhận định: *"BLEU đo trùng khớp từ vựng bề mặt, không phản ánh chất lượng lâm sàng — model offline điểm cao vì học thuộc câu mẫu trên tập rất nhỏ (74-80 mẫu), model online điểm thấp dù trả lời chi tiết và đúng y khoa hơn."* Đây là một nhận định rất tốt, nên đưa lên slide thay vì chỉ để trong tài liệu ngầm — nó cho thấy bạn hiểu rõ giới hạn công cụ đánh giá của mình.

### Slide 19 — Thiết kế EHR — P0

- **Cũ:** *"Phân tách dữ liệu thành 3 collection chính: `users`, `patients` và `records`."*
- **Mới:** *"Lưu trữ hồ sơ trong một collection Firestore duy nhất `medical_records`, mỗi document tương ứng một bệnh nhân, chứa ảnh gốc, mask, Grad-CAM, chỉ số ABCD và lịch sử hỏi đáp VQA."*
- **Vì sao:** grep `app_streamlit.py` chỉ thấy `db.collection("medical_records")` ở 4 vị trí, không có collection `users`/`patients` riêng.

### Slide 20, 21 — Bảo mật EHR — P0

- **Cũ:** *"...mã hóa đối xứng XOR + Base64 toàn bộ thông tin nhạy cảm ... để tuân thủ HIPAA."*
- **Mới:** *"Băm SHA-256 (16 ký tự đầu) để tạo định danh hồ sơ ẩn danh; thông tin nhạy cảm được che bằng XOR + Base64 ở mức nguyên mẫu (prototype-level obfuscation). Đây là bước đầu hướng tới bảo vệ dữ liệu y tế; hướng phát triển tiếp theo là chuyển sang mã hóa chuẩn AES-256-GCM với khóa quản lý tách biệt khỏi mã nguồn."*
- **Vì sao (bằng chứng cụ thể, nói được khi bị hỏi):** hàm `encrypt_data(data, key: str = "DermaSecureKey2026")` trong `app_streamlit.py` dùng **khóa hardcode ngay trong source**, dùng chung cho mọi bản ghi. XOR với khóa lặp lại là mã hóa yếu theo chuẩn mật mã học hiện đại. Gọi đây là "tuân thủ HIPAA" là khẳng định không thể bảo vệ được trước bất kỳ ai hiểu về bảo mật. Chủ động nói đúng mức độ (nguyên mẫu) sẽ an toàn hơn nhiều so với bị bắt bẻ.

### Slide 24 — Kết quả thử nghiệm — P0 (2 điểm)

**(a) "Độ chính xác toàn luồng 95.01%"**
- **Mới:** *"Độ chính xác phân loại bệnh lý (EfficientNet-B1 + CBAM, model đang triển khai): 88.65% trên tập kiểm thử HAM10000-ROI (3.005 mẫu), best validation accuracy 89.39% tại epoch 43/50."* Bỏ hẳn chữ "toàn luồng" trừ khi bạn đo được một con số end-to-end thật (segmentation→fusion→nhãn cuối so với ground truth) trước ngày bảo vệ.
- Nếu muốn giữ tinh thần "hệ thống hoạt động tốt toàn luồng", tách rõ 2 ý: độ trễ 232ms là con số toàn luồng (đúng), còn độ chính xác 88.65% là của riêng bước phân loại (không phải toàn luồng) — đừng gộp 2 khái niệm này lại như slide hiện tại.
- Cân nhắc trung thực: train_acc cuối ~99% vs val_acc ~88% cho thấy overfit — có thể chủ động nhắc ở slide "Hạn chế" (mục 3 dưới).

**(b) "Cắt ROI tăng độ chính xác thêm 7%"**
- Tôi đã tìm khắp `3_Checkpoints/`, `5_Results/`, `docs/`, `2_Notebooks/`, `scratch/` — **không tìm thấy file/log nào tính "classification accuracy trên ảnh thô (không cắt ROI) so với ảnh đã cắt ROI"**. Không có cơ sở nào cho con số "+7%" trong repo hiện tại.
- **Đề xuất:** (i) nếu bạn nhớ đã chạy thực nghiệm này ở đâu đó (notebook cũ, Colab, máy khác) — tìm lại file kết quả và trích đúng số; (ii) nếu không tìm lại được, đổi thành câu định tính không có số cụ thể: *"Quan sát định tính cho thấy việc cắt ROI theo mặt nạ giúp mô hình tập trung vào vùng tổn thương, giảm nhiễu từ nền da xung quanh"* — tránh nêu con số không có bằng chứng truy vết được, vì đây là dạng câu hỏi hội đồng rất hay hỏi ("số liệu này lấy từ đâu, cho tôi xem log").

### Slide 25 — Kết luận — P1

- **Cũ:** *"...hệ thống hỗ trợ chẩn đoán CDSS da liễu đa phương thức hoàn chỉnh, hoạt động offline ổn định, bảo mật cao và tối ưu tốc độ phản hồi lâm sàng."*
- **Mới:** *"...hệ thống nguyên mẫu (prototype) CDSS da liễu đa phương thức, hoàn thiện các module chính, hoạt động offline, áp dụng các biện pháp bảo vệ dữ liệu cơ bản, và cải thiện đáng kể tốc độ phản hồi lâm sàng (232ms/lượt)."*
- Đổi *"cố vấn y khoa 24/7"* → *"công cụ hỗ trợ tra cứu y văn có sẵn liên tục cho bác sĩ tuyến cơ sở"* (tránh gợi ý vai trò thay thế bác sĩ).

---

## 3. Hai slide mới nên thêm (nội dung cụ thể, sẵn sàng để bạn gõ vào PowerPoint)

**Slide mới A — chèn sau slide 5 — "Điểm mới so với giải pháp hiện có"**
- Các ứng dụng tự soi da phổ biến hiện nay: phân loại ảnh đơn phương thức, hộp đen (không giải thích được), cần kết nối mạng liên tục.
- Đồ án khắc phục bằng: chỉ số ABCD + Grad-CAM minh bạch, hợp nhất xác suất ảnh với dịch tễ (tuổi/giới/vị trí), nhánh VQA chạy offline hoàn toàn trên CPU.
- *(Không nêu tên sản phẩm cụ thể nếu bạn không có nguồn trích dẫn chắc chắn — dùng mô tả loại hình chung như trên để tránh bịa tên.)*

**Slide mới B — chèn trước slide Kết luận — "Hạn chế của hệ thống"**
- Model phân loại đạt 88.65% test accuracy nhưng có dấu hiệu overfit (train ~99% vs val ~89%); dữ liệu VQA huấn luyện rất nhỏ (74-80 mẫu, đánh giá trên 12 mẫu).
- Cơ chế mã hóa EHR ở mức nguyên mẫu, khóa còn hardcode trong source — chưa đạt chuẩn production.
- Guardrail thuốc hiện là ràng buộc ở tầng prompt, chưa có tầng kiểm soát output độc lập.
- Hướng phát triển: mở rộng dữ liệu, AES-256/KMS, hậu kiểm output LLM, kiểm định thống kê cho các so sánh ablation.

---

## 4. Phần có thể sửa trong CODE (nếu muốn, không bắt buộc cho việc sửa slide)

Đây là các chỗ sửa code **rẻ, nhanh, an toàn**, giúp code khớp đúng với những gì bạn sẽ nói trên slide đã sửa — không bắt buộc, chỉ làm nếu bạn muốn phần trình bày "chắc tay" hơn khi bị hỏi sâu.

| # | File | Sửa gì | Công sức | Rủi ro |
|---|---|---|---|---|
| 1 | `pipeline/multimodal_fusion.py`, dòng docstring của `fuse()` | Sửa công thức trong docstring cho khớp code thật: đổi `Final_P = lambda * Image_P + (1-lambda) * Fused_P` thành đúng công thức đang chạy `Final ∝ P(image)^λ · P(demo)^(1-λ)` | 2 phút | Không — chỉ sửa comment, không đổi hành vi |
| 2 | (Tùy chọn, lớn hơn) `app_streamlit.py`, hàm sinh câu trả lời VQA online | Thêm một bước hậu kiểm: quét câu trả lời sinh ra bằng danh sách từ khóa tên thuốc/liều lượng phổ biến (regex đơn giản, vd `\d+\s*mg`, tên hoạt chất thường gặp); nếu khớp → thay bằng `_fallback_response`. Việc này biến sơ đồ "Guardrail" trên slide 17 từ mô tả sai thành **đúng sự thật** | ~30-60 phút (cần test vài câu hỏi thật) | Thấp, nhưng cần test kỹ để tránh false positive chặn nhầm câu trả lời hợp lệ — **chỉ làm nếu còn đủ thời gian trước ngày bảo vệ, không nên làm gấp đêm trước** |
| 3 | (Tùy chọn) file report generator (script tạo `FINAL_REPORT.txt`, có thể ở `2_Notebooks/`) | Chạy lại để file report tự sinh khớp đúng checkpoint hiện tại (B1/88.65%) thay vì giữ số liệu B0 cũ đã lỗi thời | Tùy độ phức tạp script | Không ảnh hưởng app, chỉ là tài liệu tham khảo |

**Không nên làm:** đổi checkpoint quay lại B0, retrain lại để cố đạt >95%, hoặc vá bảo mật EHR thành AES thật ngay trước ngày bảo vệ — đều tốn thời gian không cân xứng với lợi ích, và bạn đã chọn hướng "nói đúng sự thật" thay vì "sửa hệ thống để khớp số cũ".

---

## 5. Checklist thực hiện (theo thứ tự làm)

1. [ ] Sửa lỗi chính tả "Kết Kết Quả" — 6 vị trí (slide 2,3,6,10,18,22).
2. [ ] Xóa "tuân thủ HIPAA" khỏi 6 slide mục lục + slide 20/21.
3. [ ] Sửa slide 7 + (nếu có thời gian) vẽ lại 2 hộp Safety Gate trong slide 8.
4. [ ] Sửa slide 14/24: đổi 95.01%/96.90% → 88.65%/89.39%, ghi rõ đây là model B1 đang triển khai.
5. [ ] Sửa hoặc bỏ số "+7% ROI" ở slide 24 (tìm lại nguồn hoặc chuyển sang câu định tính).
6. [ ] Sửa slide 19 (1 collection `medical_records`).
7. [ ] Sửa slide 20/21 (bỏ HIPAA, nói đúng XOR là ẩn danh hóa nguyên mẫu).
8. [ ] Sửa slide 16/17 (guardrail là prompt-level; thêm bảng BLEU thật).
9. [ ] Sửa slide 13, 15, 25 theo bảng từ ngữ tuyệt đối.
10. [ ] Thêm slide mới A (điểm mới so với giải pháp cũ) và slide mới B (hạn chế).
11. [ ] (Tùy chọn) sửa docstring `multimodal_fusion.py`.
12. [ ] (Tùy chọn, chỉ nếu còn thời gian) thêm hậu kiểm từ khóa cho guardrail thuốc.
13. [ ] Đọc lại toàn bộ script thuyết trình ở `chuan_bi_bao_ve_toan_dien.md` mục 8, chỉnh các đoạn liên quan đến các slide vừa sửa (14, 16, 17, 19, 20, 24) cho khớp nội dung mới.

Sau khi bạn sửa xong, nói tôi biết — tôi có thể đọc lại slide đã cập nhật để rà soát vòng 2, hoặc bắt đầu phiên vấn đáp thử để bạn tập trả lời các câu hỏi ở mục 9 của `chuan_bi_bao_ve_toan_dien.md`.
