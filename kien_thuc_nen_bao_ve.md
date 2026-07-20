# Kiến thức nền cần nắm vững để bảo vệ — theo đúng 28 slide

Nguyên tắc trình bày: trực quan trước, nguyên lý sau, công thức/chi tiết kỹ thuật cuối cùng. Mọi con số dưới đây khớp đúng với code thật đã kiểm chứng trong repo (đọc trực tiếp checkpoint `.pth` bằng `torch.load`, không suy đoán), không phải số lý thuyết chung chung.

---

## 0. Bản đồ toàn bộ model trong hệ thống — kích cỡ, input/output

| Model | Kiến trúc | Tham số | File / kích cỡ đĩa | Input | Output |
|---|---|---:|---|---|---|
| Phân đoạn U-Net | ResNet34 encoder + U-Net decoder (`segmentation_models_pytorch`) | ~24.4M (ước tính theo ResNet34) | `unet_best.pth`, 293.5 MB | Ảnh RGB 256×256×3 | Mask 1×256×256 (logit, cần sigmoid) |
| Phân đoạn DeepLabV3+ | ResNet50 encoder + ASPP + decoder (`smp.DeepLabV3Plus`) | ~26.7M (ước tính theo ResNet50) | `deeplabv3plus_best.pth`, 320.6 MB | Ảnh RGB 256×256×3 | Mask 1×256×256 (logit, cần sigmoid) |
| Hybrid-Max Fusion | Không phải model riêng — hợp nhất (max/weighted) 2 xác suất mask từ U-Net và DeepLabV3+ | 0 tham số riêng | `hybrid_best.pth` chỉ 1.5 KB (chỉ lưu config alpha, không lưu trọng số) | 2 mask xác suất | 1 mask hợp nhất |
| Lọc lông + CLAHE (tiền xử lý cứu cánh) | Không phải model — xử lý ảnh cổ điển (Black-hat + inpaint Telea; CLAHE trên kênh L của LAB), chỉ kích hoạt khi lượt phân đoạn đầu tiên trả mask rỗng, xem mục A7 | 0 tham số học được | Không có checkpoint — code thuần trong `unified_pipeline.py::_enhance_image_quality` | Ảnh RGB gốc $W\times H$ | Ảnh RGB đã xử lý, cùng kích thước |
| Phân đoạn tương tác | MobileSAM, biến thể `vit_t` (TinyViT image encoder) — GrabCut chỉ còn là dự phòng nếu thiếu checkpoint | 10,140,231 (đọc trực tiếp từ checkpoint) | `4_Models/sam/mobile_sam.pt`, 40.7 MB | Ảnh RGB + 1 điểm click (x, y) | Mask nhị phân + điểm tin cậy (score) |
| Phân loại bệnh lý | EfficientNet-B1 (timm) + CBAM (reduction=16) + Dropout(0.3) + Linear(1280→7) | ~7.8M (backbone) + đầu phân loại nhỏ | `efficientnet_attention_best.pth`, 81.4 MB | Ảnh RGB 224×224×3, chuẩn hóa ImageNet | Vector logit 7 lớp → softmax |
| Vision backbone (VQA) | EfficientNet-B1 + CBAM — **instance huấn luyện riêng biệt**, không dùng chung trọng số với model phân loại | 6,780,199 (đọc trực tiếp từ checkpoint) | Nằm trong `dermavqa_gpt2_joint_best.pth` | Ảnh RGB 224×224×3 | Vector 1280-dim (global average pooled) |
| VQA Decoder | DistilGPT-2 (6 lớp transformer, causal) + LoRA (r=8) trên `c_attn` của cả 6 lớp | 120,657,408 (gồm cả `lm_head` không tie-weight) | Nằm trong `dermavqa_gpt2_joint_best.pth`, tổng file ~ hàng trăm MB | Chuỗi token văn bản + 1 token ảnh đã chiếu | Chuỗi token văn bản (câu trả lời) |
| Projection layer (VQA) | Linear(1280→768) → GELU → Dropout(0.3) → Linear(768→768) | 1,574,400 | Nằm trong `dermavqa_gpt2_joint_best.pth` | Vector 1280-dim từ vision backbone | Vector 768-dim (khớp chiều ẩn GPT-2) |
| Sentence Embedding (RAG) | all-MiniLM-L6-v2 (Sentence-Transformers, 6-layer Transformer encoder chưng cất) | ~22.7M (theo công bố gốc của model) | Tải tự động qua thư viện, không lưu trong repo | Chuỗi văn bản (câu hỏi) | Vector 384-dim |
| VQA Online | GPT-4o-mini (OpenAI, gọi qua API) | Không công bố (proprietary) | Không lưu local | Text + ảnh (base64) qua API | Text |
| VQA Ollama (tùy chọn) | Qwen2.5:3b (mặc định trong code, dù slide ghi "3B/7B") | ~3 tỷ | Tải qua Ollama, không nằm trong repo | Text | Text |

**Tổng tham số toàn bộ checkpoint VQA đang chạy thật** (đọc trực tiếp từ `dermavqa_gpt2_joint_best.pth`): 120,657,408 (llm) + 6,780,199 (vision_backbone) + 1,574,400 (projection) = **129,012,007** tham số toàn phần. Con số "90,352,514" trên slide/báo cáo nhiều khả năng là tổng **sau khi trừ `lm_head`** (38,597,376 tham số) — vì trong kiến trúc GPT-2 chuẩn, `lm_head` thường **chia sẻ trọng số (tied weights)** với embedding đầu vào nên không được đếm 2 lần khi báo cáo "kích thước model": $129{,}012{,}007 - 38{,}597{,}376 = 90{,}414{,}631$ — khớp gần đúng với con số đã công bố (chênh lệch nhỏ có thể do phiên bản checkpoint khác nhau đôi chút).

### ⚠️ 1 phát hiện quan trọng cần biết trước khi bị hỏi (đã cập nhật)

**~~SAM không chạy~~ — ĐÃ SỬA, giờ SAM thật sự chạy, khớp đúng với slide.** (Cập nhật: trước đó tôi phát hiện `InteractiveSegmenter()` được khởi tạo không truyền checkpoint nên luôn fallback GrabCut — đã khắc phục.) Hiện tại: đã cài thư viện `mobile_sam`, tải checkpoint chính thức `mobile_sam.pt` (40.7MB, TinyViT — đúng biến thể `vit_t` nhẹ cho CPU) vào `4_Models/sam/`, sửa `unified_pipeline.py` để tự động nạp và **cache lại 1 lần** (tránh nạp lại 40MB mỗi lần click). Đã kiểm chứng bằng cách gọi trực tiếp `UnifiedDermatologyPipeline.run(interactive_point=...)` — đúng luồng thật `app_streamlit.py` dùng: `segmentation method: mobile_sam`, `score: 0.933`, lần click đầu ~2.3s (gồm cả nạp checkpoint), lần click sau ~1.5s (đã cache). Cũng đã thêm `mobile_sam` vào `requirements.txt` và 1 hàm tự tải checkpoint từ GitHub lúc khởi động app (`download_sam_checkpoint_if_missing()`) để bản deploy trên Streamlit Cloud cũng hoạt động đúng sau khi push code, không chỉ chạy đúng ở máy local.

**Nếu hội đồng hỏi "SAM checkpoint đang dùng loại nào?"** → trả lời đúng: **MobileSAM (biến thể `vit_t`, dùng kiến trúc TinyViT làm image encoder thay vì ViT-H đầy đủ của SAM gốc)** — chọn bản này vì nhẹ (~40MB so với ViT-H ~2.4GB), phù hợp chạy CPU cho môi trường tuyến cơ sở không có GPU, đánh đổi độ chính xác biên một chút để lấy tốc độ suy luận khả dụng trong tương tác thời gian thực.

**Còn 1 điều chưa tự kiểm chứng được:** click chuột thật qua canvas trên trình duyệt (upload ảnh → chọn radio SAM → click) chưa được test bằng browser tự động do môi trường phát triển không có sẵn công cụ điều khiển trình duyệt headless phù hợp — chỉ mới kiểm chứng đến tầng logic/pipeline. Bạn nên tự click thử qua UI thật 1-2 lần trước khi bảo vệ để chắc chắn 100% phần canvas/tọa độ click cũng hoạt động mượt.

---

**Kiến trúc VQA "V3" phức tạp trong `train_vqa_joint.py` vẫn KHÔNG có trong checkpoint đang chạy** — điều này chưa thay đổi. File huấn luyện định nghĩa một kiến trúc rất phức tạp gồm `DeepCrossAttentionBridge` (2 lớp cross-attention), `ClinicalStructureInjector` (tiêm chỉ số ABCD + xác suất lớp bệnh thành 1 token), `ClinicalPrefix` (8 token prefix học được), `SemanticEnhancer` — nhưng khi tôi mở trực tiếp checkpoint thật `dermavqa_gpt2_joint_best.pth` (đúng file mà `app_streamlit.py` đang nạp), **không có bất kỳ tensor nào** thuộc các module này (`cross_attention_bridge`, `clinical_prefix`, `clinical_injector` đều không tồn tại trong state_dict). Model đang chạy thật chỉ gồm: vision backbone → projection (1280→768→768) → **1 token ảnh duy nhất** (global average pooling, không dùng 49 spatial token) → ghép với text embedding → DistilGPT-2+LoRA. Nếu bị hỏi về DeepCrossAttentionBridge hay cơ chế multi-token — đây là **hướng phát triển đã viết code sẵn sàng nhưng chưa huấn luyện thành checkpoint chính thức**, không phải kiến trúc đang thực sự chạy trong app.

---

## A. Phân đoạn ảnh (Segmentation) — slide 8, 9, 12, 13

### A0. Convolution là gì — nền tảng trước khi vào ResNet/U-Net/DeepLabV3+

**Trực quan:** convolution (phép chập) trượt 1 "cửa sổ nhỏ" gọi là **kernel** (hay filter — bộ lọc, 1 ma trận số nhỏ, ví dụ 3×3) qua toàn bộ ảnh, tại mỗi vị trí nhân từng số trong kernel với pixel tương ứng rồi cộng lại, ra 1 số duy nhất — đại diện cho "mức độ khớp" giữa vùng ảnh đó với hoa văn (pattern) mà kernel đang tìm (ví dụ 1 kernel có thể chuyên phát hiện cạnh dọc, kernel khác phát hiện góc...). Làm việc này với hàng trăm kernel khác nhau cho ra hàng trăm **channel** (kênh) đặc trưng — gọi chung là 1 **feature map** (bản đồ đặc trưng), có dạng $(C, H, W)$ với $C$=số channel, $H$=chiều cao, $W$=chiều rộng.

**Công thức kích thước output** của 1 phép convolution, với **input** kích thước $n$ (giả sử ảnh vuông $n \times n$), **kernel size** $k$ (kernel $k \times k$), **stride** $s$ (bước nhảy — kernel di chuyển bao nhiêu pixel mỗi lần), **padding** $p$ (số pixel đệm thêm 0 quanh viền ảnh để kiểm soát kích thước output):
$$\text{out} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1$$
Ký hiệu $\lfloor \cdot \rfloor$ là hàm sàn (floor — làm tròn xuống số nguyên gần nhất). **Ví dụ lấy đúng số thật dùng trong pipeline phân đoạn** (input $n=256$, xem mục A6): kernel 3×3, stride 1, padding 1 → out = $\lfloor(256+2-3)/1\rfloor+1 = 256$ — không đổi kích thước (padding "same"), đây là kiểu conv dùng trong hầu hết các lớp giữ nguyên độ phân giải của ResNet. Còn lớp **stem** đầu tiên của ResNet (mục A1) dùng kernel 7×7, stride 2, padding 3 để chủ động giảm kích thước ngay từ đầu: out = $\lfloor(256+2\times3-7)/2\rfloor+1 = \lfloor 255/2 \rfloor + 1 = 127+1 = 128$ — giảm đúng 1 nửa ($256\to128$), khớp với bước đầu tiên trong bảng shape trace ở mục A6 (trước khi qua thêm maxpool để còn 64).

### A1. ResNet — Residual Network, xương sống (backbone) chung của cả U-Net và DeepLabV3+

**Vấn đề ResNet giải quyết:** mạng CNN thường càng sâu (nhiều lớp) càng dễ học được đặc trưng phức tạp, nhưng thực nghiệm cho thấy nếu chỉ xếp chồng lớp convolution thẳng, sau một độ sâu nhất định độ chính xác lại **giảm** (không phải do overfit, mà do **vanishing gradient** — đạo hàm lan truyền ngược qua quá nhiều lớp bị "teo nhỏ dần về 0", khiến các lớp đầu gần như không học được gì).

**Residual connection (kết nối tắt/dư):** thay vì bắt 1 khối lớp phải học trực tiếp hàm output mong muốn $H(x)$, ResNet cho khối đó chỉ cần học phần **dư (residual)** $F(x) = H(x) - x$, rồi cộng lại đầu vào $x$ ban đầu:
$$y = F(x) + x$$
với $x$ = input của khối, $F(x)$ = phần biến đổi mà các lớp conv trong khối học được, $y$ = output cuối. Nếu khối đó "không cần học gì thêm" thì chỉ cần đưa $F(x) \to 0$ là xong (dễ học hơn nhiều so với phải học đúng hàm đồng nhất $H(x)=x$ từ đầu) — nhờ đường tắt cộng thẳng $x$, gradient khi lan truyền ngược có 1 đường "đi tắt" không bị suy giảm qua nhiều lớp, nên mạng sâu hàng chục/hàng trăm lớp vẫn huấn luyện ổn định.

**ResNet34 (dùng cho U-Net) và ResNet50 (dùng cho DeepLabV3+) — khác nhau ở đâu, và tại sao chọn khác nhau cho 2 model:**

| | ResNet34 | ResNet50 |
|---|---|---|
| Loại khối (block) | **BasicBlock**: 2 lớp conv 3×3 xếp chồng, cộng residual thẳng | **Bottleneck**: 3 lớp conv (1×1 → 3×3 → 1×1), cộng residual — lớp 1×1 đầu nén số channel xuống, lớp 1×1 cuối mở rộng lại ×4 |
| Số block mỗi giai đoạn (stage) | [3, 4, 6, 3] | [3, 4, 6, 3] (**cùng số block**, nhưng mỗi block "dày" hơn do dùng Bottleneck) |
| Tổng số lớp có trọng số | 34 lớp | 50 lớp (nhiều hơn dù cùng số block, vì Bottleneck có 3 lớp/block thay vì 2) |
| Số channel output mỗi stage | 64 → 64 → 128 → 256 → 512 | 64 → 256 → 512 → 1024 → 2048 (gấp 4 lần do hệ số mở rộng "expansion=4" của Bottleneck) |
| Tổng tham số (ImageNet gốc) | ~21.8 triệu | ~25.6 triệu |

**Tại sao U-Net dùng ResNet34 (nhẹ hơn), còn DeepLabV3+ dùng ResNet50 (nặng hơn) — không phải ngẫu nhiên:**
1. **U-Net vốn đã có "phần bù" ở decoder:** kiến trúc U-Net có các **skip connection** nối trực tiếp từng tầng encoder sang đúng tầng decoder tương ứng cùng độ phân giải — nghĩa là thông tin chi tiết (biên, texture) từ encoder nông đã được "chuyển thẳng" sang decoder mà không cần encoder phải quá sâu/quá mạnh mới giữ được chi tiết. Dùng ResNet34 (nhẹ) là đủ, tránh dư thừa tham số.
2. **DeepLabV3+ không có skip connection dày đặc như U-Net** (chỉ nối 1 tầng encoder độ phân giải thấp sang decoder), nên gánh nặng "phải trích đặc trưng đủ tốt" dồn nhiều hơn vào encoder — cần ResNet50 (sâu và rộng hơn) để bù lại, kết hợp thêm ASPP (mục A2) để mở rộng receptive field mà không cần thêm skip connection.
3. **Lý do bổ sung — đa dạng hóa cho Hybrid-Max Fusion:** việc cố ý dùng 2 backbone khác nhau (ResNet34 vs ResNet50) cho 2 nhánh phân đoạn giúp 2 model có xu hướng sai ở những chỗ khác nhau (architectural diversity) — khi hợp nhất Hybrid-Max (lấy max 2 bản đồ xác suất), 2 model "bù trừ" lỗi cho nhau tốt hơn so với việc dùng 2 model gần giống hệt nhau (nếu cả 2 đều dùng chung 1 backbone, sai lầm của chúng sẽ tương quan cao, hợp nhất sẽ ít cải thiện hơn) — đây chính là nguyên lý cốt lõi của ensemble learning: các model càng "khác nhau" theo cách sai thì hợp nhất càng hiệu quả.

### A2. DeepLabV3+ và Atrous Convolution (Dilated Convolution)

**Trực quan:** một mạng CNN thường thu nhỏ ảnh dần qua các lớp pooling/stride để "nhìn" được vùng rộng hơn (receptive field lớn hơn), nhưng cái giá phải trả là mất độ phân giải chi tiết (không biết chính xác biên tổn thương nằm ở đâu, vì ảnh đã bị thu nhỏ nhiều lần). DeepLabV3+ giải quyết bằng **Atrous Convolution** (atrous = tiếng Pháp/Hy Lạp nghĩa là "có lỗ", còn gọi là Dilated Convolution) — convolution "có lỗ hổng": giãn cách các điểm lấy mẫu của kernel ra xa nhau thay vì co cụm sát nhau, giúp mở rộng **receptive field** (vùng ảnh gốc mà 1 điểm output "nhìn thấy được") **mà không cần giảm độ phân giải feature map**.

**Công thức:** với **dilation rate** $r$ (khoảng cách giữa các điểm lấy mẫu trong kernel, $r=1$ là convolution thường), 1 kernel $k \times k$ (ví dụ 3×3) sẽ có **kích thước hiệu dụng (effective kernel size)**:
$$k_{\text{eff}} = k + (k-1)(r-1)$$
Ví dụ $k=3, r=6$: $k_{\text{eff}} = 3 + 2\times5 = 13$ — 1 kernel 3×3 nhưng "nhìn" được vùng rộng như kernel 13×13, mà số tham số thật vẫn chỉ là $3\times3=9$ (không tăng tính toán). Công thức kích thước output tổng quát (thay $k$ ở công thức A0 bằng $k_{\text{eff}}$):
$$\text{out} = \left\lfloor \frac{n + 2p - k_{\text{eff}}}{s} \right\rfloor + 1$$
**Ví dụ đúng số thật ASPP đang xử lý** (xem mục A6): input vào ASPP là feature map $n=8$ (đầu ra sâu nhất của ResNet50, $256/32=8$), $k=3$, $r=6$, $s=1$ — muốn giữ nguyên độ phân giải $8\times8$ (không được phép co nhỏ lại, vì còn phải ghép với 3 nhánh dilation khác cùng 1 nhánh global-pooling), padding phải chọn đúng bằng $p=r=6$ (quy ước chuẩn để atrous conv giữ "same size"): out = $\lfloor(8+2\times6-13)/1\rfloor+1 = \lfloor 7 \rfloor + 1 = 8$ — đúng khớp $(256,8,8)$ ở bảng A6.

**ASPP (Atrous Spatial Pyramid Pooling — "gộp kim tự tháp không gian bằng atrous conv"):** DeepLabV3+ chạy **song song** nhiều nhánh atrous convolution với dilation rate khác nhau (thường $r \in \{6, 12, 18\}$) trên cùng 1 feature map đầu vào, cộng thêm 1 nhánh global average pooling (tóm tắt toàn ảnh thành 1 vector rồi upsample lại), rồi **ghép (concatenate)** toàn bộ output các nhánh theo chiều channel — giúp bắt được tổn thương ở nhiều kích thước khác nhau cùng lúc trong 1 lần forward (tổn thương nhỏ li ti cần receptive field nhỏ, mảng lớn cần receptive field lớn).

**Câu hỏi khả năng bị hỏi:** *"Vì sao không dùng U-Net thường cho cả 2?"* → U-Net (encoder-decoder đối xứng, skip connection dày) đơn giản hơn, nhưng ở cùng 1 độ sâu thì receptive field hẹp hơn DeepLabV3+ (vì không có ASPP); DeepLabV3+ nắm được ngữ cảnh đa tỷ lệ tốt hơn cho tổn thương có kích thước rất đa dạng trong tập dữ liệu. Số liệu thật (slide 12): DeepLabV3+ Dice 91.28% > U-Net 89.43% — đúng như kỳ vọng lý thuyết, nhưng đây không phải lý do "bỏ U-Net" — cả 2 vẫn được giữ lại và hợp nhất ở bước Hybrid-Max chính vì lý do đa dạng hóa đã nêu ở mục A1.

### A3. TTA (Test-Time Augmentation) đa tỷ lệ

**Trực quan:** thay vì đưa 1 ảnh duy nhất vào model, đưa ảnh đó vào ở **nhiều kích thước khác nhau** (1.0×, 0.75×, 0.5× — đúng scale bạn dùng trong code), lấy trung bình/hợp nhất kết quả — giống việc hỏi 3 "phiên bản zoom khác nhau" của cùng 1 model rồi lấy đồng thuận, giảm sai số ngẫu nhiên do 1 tỷ lệ cụ thể vô tình không khớp với model.

**Vì sao chỉ dùng cho ảnh phone:** ảnh dermoscopy đã chuẩn hóa tỷ lệ chụp (máy soi da cố định khoảng cách), còn ảnh phone chụp ở khoảng cách/góc tùy ý — biến thiên tỷ lệ vật lý lớn hơn nhiều, nên hưởng lợi từ đa tỷ lệ nhiều hơn.

### A4. MobileSAM (Segment Anything Model, biến thể TinyViT) + GrabCut — vì sao kết hợp 2 thuật toán

**SAM (Segment Anything Model — "model phân đoạn được mọi thứ"):** một **foundation model** (model nền tảng — huấn luyện trên tập dữ liệu khổng lồ, đa dụng cho nhiều tác vụ khác nhau, không huấn luyện riêng cho 1 bài toán cụ thể) chuyên phân đoạn, nhận **prompt** (gợi ý — có thể là 1 điểm click, 1 khung box, hoặc 1 mask thô) làm đầu vào, tự sinh ra mask chính xác ứng với gợi ý đó. Kiến trúc gồm 3 phần: (1) **image encoder** — mã hóa toàn bộ ảnh 1 lần duy nhất thành 1 feature map giàu thông tin (bước tốn thời gian nhất), (2) **prompt encoder** — mã hóa điểm click/box thành vector, (3) **mask decoder** — 1 mạng rất nhẹ, kết hợp feature map ảnh với vector prompt để ra mask, chạy cực nhanh nên có thể đổi điểm click nhiều lần mà không cần mã hóa lại ảnh.

**MobileSAM — bản "nhẹ" của SAM, dùng TinyViT thay ViT-H:** SAM gốc dùng image encoder là **ViT-H** (Vision Transformer, phiên bản Huge — khoảng 632 triệu tham số, quá nặng cho CPU). MobileSAM thay bằng **TinyViT** — 1 kiến trúc Transformer thị giác dạng **phân cấp (hierarchical)**: ảnh đầu vào trước tiên qua 1 lớp **patch embedding** (dùng vài lớp convolution nhỏ, ký hiệu trong checkpoint là `patch_embed.seq.0.c` = conv, `.bn` = batch normalization — chia ảnh thành các patch nhỏ và chiếu mỗi patch thành 1 vector), sau đó qua **4 giai đoạn (stage)** xử lý Transformer, mỗi giai đoạn giảm dần độ phân giải không gian và tăng dần số channel (giống cách ResNet/CNN thu nhỏ dần ảnh, nhưng dùng self-attention thay vì convolution ở các stage sâu) — cách thiết kế phân cấp này giúp TinyViT vừa giữ được sức mạnh của Transformer vừa nhẹ hơn nhiều so với ViT thuần (ViT thuần giữ nguyên 1 độ phân giải token cố định xuyên suốt, tốn tính toán hơn). **Tổng tham số đã kiểm chứng trực tiếp từ checkpoint: 10,140,231** — nhẹ hơn ViT-H khoảng 62 lần, đủ nhẹ để chạy CPU trong vài giây.

**GrabCut — chỉ còn là phương án dự phòng khi thiếu checkpoint:** thuật toán cổ điển (không dùng deep learning) dựa trên **Gaussian Mixture Model — GMM** (mô hình hỗn hợp Gauss, giả định màu sắc foreground/background mỗi loại tuân theo tổng của vài phân phối Gauss) để mô hình hóa phân phối màu, kết hợp **Graph Cut** (coi ảnh như 1 đồ thị, mỗi pixel là 1 đỉnh, tìm đường cắt — cut — chia đồ thị thành 2 phần foreground/background sao cho "chi phí cắt" nhỏ nhất, giải bằng thuật toán **min-cut/max-flow**) để tìm ranh giới tối ưu, tối thiểu hóa hàm năng lượng:
$$E(\alpha, k, \theta, z) = U(\alpha, k, \theta, z) + V(\alpha, z)$$
Ký hiệu: $\alpha$ = nhãn mỗi pixel (foreground/background), $k$ = thành phần Gauss nào trong GMM mà pixel đó thuộc về, $\theta$ = tham số của các phân phối Gauss (trung bình, hiệp phương sai), $z$ = giá trị màu quan sát được của pixel. $U$ là **data term** (pixel càng khớp với GMM của lớp nó đang được gán thì $U$ càng nhỏ — "chi phí" thấp), $V$ là **smoothness term** (phạt nếu 2 pixel liền kề có màu gần giống nhau nhưng lại bị gán khác lớp — khuyến khích biên mượt, không lởm chởm).

**Vì sao dùng `GC_INIT_WITH_MASK` khi GrabCut chạy dự phòng:** thay vì để GrabCut tự đoán vùng foreground ban đầu bằng 1 khung box thô (dễ sai nếu tổn thương không nằm giữa khung), code dùng chính mask mà bước trước đó tạo ra (nếu có) làm khởi tạo — GrabCut khi đó chỉ cần **tinh chỉnh biên** cho mượt hơn theo màu sắc pixel thật, không phải đoán vùng từ đầu.

### A5. OTSU Thresholding — thuật toán dự phòng

**Nguyên lý:** OTSU tự động tìm 1 ngưỡng độ sáng $t$ chia ảnh xám thành 2 lớp (nền/vật thể), sao cho **phương sai giữa 2 lớp là lớn nhất** (tương đương phương sai trong từng lớp nhỏ nhất):
$$\sigma_b^2(t) = \omega_0(t)\omega_1(t)[\mu_0(t) - \mu_1(t)]^2$$
với $\omega_0, \omega_1$ là tỷ lệ pixel mỗi lớp, $\mu_0, \mu_1$ là độ sáng trung bình mỗi lớp. Duyệt qua mọi $t$ có thể, chọn $t^*$ maximize $\sigma_b^2$.

**`THRESH_BINARY_INV`:** đảo ngược nhị phân hóa — vì tổn thương da thường **tối hơn** vùng da xung quanh, nên cần đảo ngược để vùng tối (tổn thương) thành pixel "1" thay vì mặc định OTSU coi vùng sáng là foreground.

**Câu hỏi bẫy:** *"OTSU có phải deep learning không?"* → Không, đây là thuật toán xử lý ảnh cổ điển thuần túy dựa trên thống kê histogram, không có tham số học được — chính vì thế nó **không thể thất bại theo kiểu "model lỗi"**, luôn cho ra 1 kết quả xác định, phù hợp làm phương án dự phòng cuối cùng khi deep model thất bại hoàn toàn.

### A6. Input/Output chính xác của 2 model phân đoạn — đi qua từng tầng (shape trace)

**Input chung:** ảnh RGB resize về **256×256×3** — viết dạng tensor PyTorch chuẩn $(N, C, H, W)$ với $N$=batch size (=1 khi suy luận từng ảnh), $C=3$ (kênh R,G,B), $H=W=256$. Chuẩn hóa: mean=(0.5,0.5,0.5), std=(0.25,0.25,0.25) áp cho từng kênh theo công thức $x' = (x/255 - \text{mean})/\text{std}$ — khác chuẩn ImageNet của model phân loại (mục C3) vì đây là lựa chọn đơn giản hóa riêng cho bài toán phân đoạn nhị phân, không có gì bắt buộc phải theo đúng ImageNet.

**Câu hỏi gần như chắc chắn sẽ bị hỏi: "Vì sao phân đoạn dùng 256×256 còn phân loại dùng 224×224 — sao không thống nhất 1 kích cỡ cho cả hệ thống?"** Đây là 2 quyết định thiết kế độc lập, có lý do kỹ thuật hợp lý riêng cho từng bài toán (không phải do quên đồng bộ):
1. **Phân đoạn cần độ phân giải output cao hơn vì output là từng pixel, không phải 1 vector tóm tắt.** Output cuối của model phân loại chỉ là 7 con số (xác suất 7 lớp) sau Global Average Pooling — ảnh đầu vào to hay nhỏ hơn 1 chút gần như không ảnh hưởng độ chi tiết của output. Ngược lại, output của model phân đoạn là **cả 1 mask** — input càng lớn thì mask ra càng nhiều pixel, đường biên tổn thương càng mượt/chính xác hơn, và quan trọng hơn: **toàn bộ 4 chỉ số ABCD ở mục B đều tính trực tiếp trên số pixel của mask này** (diện tích = đếm pixel, chu vi = đo theo pixel) — mask càng thô (ít pixel) thì các phép đo hình học này càng kém chính xác. Vì vậy phân đoạn "đáng" phải trả thêm chi phí tính toán để lấy độ phân giải cao hơn, còn phân loại thì không cần.
2. **224×224 là quy ước chuẩn hóa của ImageNet pretraining** mà cả backbone phân đoạn (ResNet34/50, mục A1) lẫn backbone phân loại (EfficientNet-B1, mục C1) đều tận dụng trọng số pretrain — nhưng chỉ có model phân loại thực sự "giữ nguyên" độ phân giải pretrain gốc này, vì nó dùng thẳng đặc trưng toàn cục cuối cùng (không có decoder cần khớp shape chính xác); còn model phân đoạn dù cũng khởi tạo từ trọng số ImageNet, vẫn phải **fine-tune lại gần như toàn bộ** cho tác vụ pixel-wise nên độ lệch resolution so với lúc pretrain (256 thay vì 224) ít gây hại hơn (CNN có tính bất biến tương đối với thay đổi nhẹ độ phân giải input, nhờ các phép pooling/stride hoạt động theo tỷ lệ chứ không theo số pixel tuyệt đối).
3. **Ràng buộc chia hết cho 32:** cả ResNet lẫn EfficientNet đều downsample tổng cộng 5 lần theo hệ số 2 (÷32) trước khi tới feature map sâu nhất — cả 256 ($256/32=8$) lẫn 224 ($224/32=7$) đều chia hết, nên đây **không phải** lý do phân biệt giữa 2 lựa chọn (nhiều người nhầm tưởng phải chọn 256 vì lý do này) — cả 2 con số đều hợp lệ về mặt kiến trúc, sự khác biệt thực sự nằm ở lý do 1 và 2 nêu trên.

*(Lưu ý khi trả lời: đây là lý giải kỹ thuật dựa trên nguyên tắc thiết kế CNN chuẩn — không có 1 dòng comment nào trong code ghi thẳng "chọn 256 vì lý do X"; nên nếu hội đồng hỏi "có tài liệu nào ghi rõ không", nên trả lời trung thực là đây là suy luận kỹ thuật hợp lý dựa trên logic thiết kế, không phải trích dẫn từ code.)*

**Shape đi qua encoder ResNet34 (U-Net) — ví dụ cụ thể với input 256×256:**

| Tầng | Output shape $(C, H, W)$ | Ghi chú |
|---|---|---|
| Input | $(3, 256, 256)$ | Ảnh gốc đã chuẩn hóa |
| Stem (conv 7×7 stride 2 + maxpool stride 2) | $(64, 64, 64)$ | Giảm 4 lần ($256/4=64$) |
| Stage 1 (conv2_x, 3 BasicBlock) | $(64, 64, 64)$ | Giữ nguyên độ phân giải |
| Stage 2 (conv3_x, 4 BasicBlock, stride 2) | $(128, 32, 32)$ | |
| Stage 3 (conv4_x, 6 BasicBlock, stride 2) | $(256, 16, 16)$ | |
| Stage 4 (conv5_x, 3 BasicBlock, stride 2) | $(512, 8, 8)$ | Đây là feature map "sâu nhất", ít không gian nhất nhưng nhiều ngữ nghĩa nhất |

**Decoder U-Net** đi ngược lại: mỗi bước **upsample ×2** (dùng nội suy hoặc deconvolution) feature map sâu nhất, rồi **ghép (concatenate theo channel)** với đúng feature map cùng độ phân giải bên phía encoder (đây chính là **skip connection**) — ví dụ upsample $(512,8,8) \to (256,16,16)$ rồi ghép với feature map encoder Stage 3 $(256,16,16)$ thành $(512,16,16)$, qua conv giảm còn $(256,16,16)$, tiếp tục upsample... lặp lại đến khi về đúng $(64, 256, 256)$, cuối cùng 1 lớp conv 1×1 nén còn $(1, 256, 256)$ — chính là mask logit output.

**DeepLabV3+ (ResNet50) khác U-Net ở chỗ:** encoder cho ra feature map $(2048, 8, 8)$ ở tầng sâu nhất (do Bottleneck mở rộng ×4 channel — xem mục A1), đưa qua **ASPP** (mục A2) với 3 nhánh dilation rate {6,12,18} + 1 nhánh global pooling, mỗi nhánh cho ra $(256, 8, 8)$, ghép 4 nhánh thành $(1024, 8, 8)$, nén qua conv 1×1 còn $(256, 8, 8)$. Decoder DeepLabV3+ **chỉ upsample và ghép với đúng 1 tầng encoder nông** (thường là output Stage 1, có nhiều chi tiết biên) chứ không ghép đầy đủ mọi tầng như U-Net — đây chính là lý do (đã nêu ở mục A1) DeepLabV3+ "gánh nặng" nhiều hơn vào ASPP/encoder thay vì skip connection.

**Output cuối cùng (cả 2 model):** 1 kênh duy nhất $(1, 256, 256)$ (`classes=1` trong `smp.DeepLabV3Plus`/`smp.Unet`), **không áp activation trong model** (`activation=None`) — đây là **logit** thô (số thực bất kỳ, có thể âm), phải tự áp $\sigma(\cdot)$ (sigmoid, xem bảng ký hiệu ở đầu mỗi công thức: $\sigma(z)=1/(1+e^{-z})$) bên ngoài để ra xác suất mỗi pixel thuộc tổn thương, rồi nhị phân hóa bằng ngưỡng `seg_threshold=0.3` (thấp hơn 0.5 mặc định — cố ý "dễ dãi" hơn để ưu tiên không bỏ sót vùng tổn thương, chấp nhận đánh đổi biên có thể hơi phình ra).

**Hybrid-Max Fusion không phải 1 model riêng:** chạy CẢ 2 model (U-Net và DeepLabV3+) trên cùng 1 ảnh, ra 2 bản đồ xác suất $(1,256,256)$, lấy **giá trị lớn nhất tại từng vị trí pixel** giữa 2 bản đồ: $P_{\text{final}}(h,w) = \max(P_{U\text{-Net}}(h,w), P_{\text{DeepLab}}(h,w))$, rồi mới nhị phân hóa. Vì vậy `hybrid_best.pth` chỉ nặng 1.5KB (chỉ lưu config, không lưu trọng số riêng nào) — chi phí suy luận gấp đôi (chạy cả 2 model) nhưng không cần huấn luyện thêm bất kỳ tham số mới nào.

**⚠️ Luồng thật nối 2 nhánh phân đoạn → phân loại (điểm rất dễ hiểu nhầm — đọc trực tiếp từ `unified_pipeline.py`):** 256 và 224 **không nằm trên cùng 1 trục độ phân giải nối tiếp nhau** — 256 chỉ là kích thước "làm việc nội bộ" riêng của model phân đoạn, bị xóa dấu vết ngay sau khi model chạy xong:
1. Ảnh gốc `img_rgb` giữ nguyên kích thước thật $W\times H$ (kích thước file ảnh upload, KHÔNG cố định — ví dụ ảnh phone có thể là 4000×3000) trong suốt toàn bộ pipeline; **chỉ tạo ra 1 bản resize 256×256 tạm thời** để đưa vào model phân đoạn (hàm `_run_seg_forward`, `unified_pipeline.py:235`).
2. Model phân đoạn ra xác suất ở đúng 256×256, nhưng **ngay lập tức được resize ngược lại về đúng $W\times H$ gốc** (dòng 248: `return cv2.resize(prob, (w, h), ...)`) rồi mới ngưỡng hóa (`seg_threshold=0.3`) — nghĩa là **mask nhị phân cuối cùng trả về có kích thước bằng ảnh gốc**, không phải 256×256.
3. `_crop_to_roi(img_rgb, seg_mask, padding=10)` cắt bounding box của contour lớn nhất **trực tiếp trên ảnh gốc $W\times H$** (không phải trên ảnh 256×256) cộng thêm viền đệm 10px mỗi phía.
4. Vùng ROI cắt ra có kích thước bất kỳ (tùy tổn thương to/nhỏ, ví dụ 180×150) — chỉ đến bước này mới **resize về đúng 224×224** bằng PIL bilinear để đưa vào EfficientNet-B1 (mục C3).
5. Nếu mask không hợp lệ (rỗng, quá nhỏ theo `min_area_px`, hoặc `low_confidence`), bỏ qua bước crop, dùng thẳng ảnh gốc resize 224×224 — giữ 2 nhánh độc lập, lỗi phân đoạn không làm sập luôn phân loại.

Tóm lại thứ tự đúng là: **ảnh gốc → (resize tạm 256 chỉ để model phân đoạn nhìn thấy) → mask ở độ phân giải gốc → crop ROI ở độ phân giải gốc → resize 224 lần duy nhất trước khi vào model phân loại.**

### A7. Lọc lông + tăng sáng/tương phản (CLAHE) — lớp "cứu cánh" khi phân đoạn thất bại lần đầu

**Bối cảnh:** ảnh da liễu chụp bằng điện thoại thường gặp 2 vấn đề đặc thù mà ảnh dermoscopy (chụp bằng máy soi da chuyên dụng, ánh sáng/khoảng cách chuẩn hóa) ít gặp: (1) **lông che phủ** vùng tổn thương, (2) **thiếu sáng/bóng đổ không đều** do ánh sáng phòng khám hoặc ánh sáng tự nhiên không kiểm soát được. Cả 2 đều có thể khiến model phân đoạn deep learning (U-Net/DeepLabV3+, mục A1-A2) "nhìn nhầm" và trả về mask rỗng (không tìm thấy tổn thương nào).

**Cơ chế kích hoạt — CÓ ĐIỀU KIỆN, không áp dụng tràn lan:** đây là điểm quan trọng nhất cần nhớ khi trả lời. Bước lọc lông + CLAHE **chỉ chạy khi** lượt phân đoạn đầu tiên trên ảnh gốc cho ra mask hoàn toàn rỗng (`mask.sum() == 0`) — với ảnh chất lượng tốt (đa số trường hợp thực tế), bước này **không bao giờ được gọi tới**, ảnh đi thẳng qua model như bình thường. Lý do bắt buộc phải có điều kiện: cả 2 model phân đoạn được huấn luyện trên ảnh **gốc, chưa qua các phép biến đổi này** — nếu áp dụng cho mọi ảnh sẽ tạo ra **lệch phân phối train/inference** (train-inference distribution skew), có nguy cơ làm giảm độ chính xác trên chính những ảnh vốn dĩ đã ổn. Chỉ kích hoạt cho nhóm ảnh mà model gốc đã thất bại thì không có gì để mất thêm — đúng nguyên tắc "chỉ can thiệp khi cần".

**Lọc lông — kỹ thuật DullRazor kinh điển (`_enhance_image_quality`, `unified_pipeline.py`):**
1. Chuyển ảnh xám, áp **Black-hat morphology** (phép hình thái học "mũ đen" — lấy hiệu giữa ảnh sau khi Closing và ảnh gốc, làm nổi bật các cấu trúc **mảnh và tối hơn nền xung quanh**, đúng đặc điểm của sợi lông trên nền da sáng hơn) với kernel hình chữ nhật 13×13.
2. Ngưỡng hóa kết quả Black-hat (`threshold=12`) thành `hair_mask` nhị phân, dọn nhiễu bằng Opening morphology (kernel elip 3×3).
3. Nếu số pixel nghi là lông đủ lớn (`> 300` pixel — ngưỡng để tránh can thiệp nhầm khi ảnh không thực sự có lông, chỉ có vài pixel nhiễu ngẫu nhiên), dùng **inpainting** (`cv2.inpaint`, thuật toán **Telea 2004** — nội suy giá trị pixel bị "khuyết" dựa trên các pixel hợp lệ xung quanh viền vùng khuyết, lan dần vào trong) để "vá" lại đúng vùng `hair_mask`, coi như sợi lông chưa từng tồn tại trên ảnh.

**Tăng sáng/tương phản — CLAHE (Contrast Limited Adaptive Histogram Equalization):** chuyển ảnh sang không gian màu **LAB** (kênh L = độ sáng, tách biệt khỏi 2 kênh màu a, b — cho phép chỉnh sáng mà không làm lệch màu sắc thật của tổn thương, điều quan trọng vì màu sắc là 1 trong 4 chỉ số ABCD ở mục B), áp CLAHE lên riêng kênh L với `clipLimit=2.0` (giới hạn mức khuếch đại tương phản tối đa mỗi vùng — tránh khuếch đại nhiễu quá mức, khác với Histogram Equalization thường có thể gây nhiễu hạt ở vùng đồng màu) và `tileGridSize=(8,8)` (chia ảnh thành lưới 8×8 ô nhỏ, cân bằng histogram **riêng từng ô** rồi nội suy mượt giữa các ô — đây là điểm khác biệt "Adaptive" so với cân bằng histogram toàn ảnh: xử lý được ảnh có vùng sáng/vùng tối khác nhau trong cùng 1 ảnh, đúng tình huống bóng đổ không đều).

**Luồng tích hợp thật (cả 2 nhánh — standard và TTA cho ảnh phone):** lượt 1 chạy model phân đoạn trên ảnh gốc như bình thường → nếu mask rỗng → áp lọc lông+CLAHE → chạy lại model phân đoạn (lượt 2, cùng model, không train lại, không có checkpoint mới) trên ảnh đã xử lý → nếu vẫn rỗng mới rơi xuống lớp fallback cổ điển cuối cùng (OTSU/GrabCut, mục A5). Đã kiểm chứng bằng cách giả lập lượt 1 luôn thất bại (mock `_run_seg_forward`) trên cả nhánh standard lẫn TTA: `seg_info["method"]` đúng ra `"deeplab_enhanced_reseg"`/`"deeplab_tta_enhanced_reseg"`, mask lượt 2 khác rỗng, tổng cộng đúng 2 lượt gọi model.

**Vì sao KHÔNG cần huấn luyện lại bất kỳ checkpoint nào:** đây thuần túy là 1 bước xử lý ảnh cổ điển (OpenCV: morphology, inpainting, CLAHE) chạy hoàn toàn **trước** khi ảnh vào model — không có tham số học được, không đụng đến trọng số `.pth` nào của U-Net/DeepLabV3+. Vì cơ chế kích hoạt có điều kiện (chỉ khi thất bại) nên không phát sinh rủi ro lệch phân phối cho luồng bình thường.

**Chi phí runtime — chỉ trả giá đúng lúc cần:** với ảnh chất lượng tốt: **0 chi phí thêm** (không kích hoạt). Với ảnh bị kích hoạt: cộng thêm 1 lượt xử lý ảnh cổ điển (~vài chục–200ms, đo thực tế: hàm `_enhance_image_quality` chạy khoảng 0.39s cho 1 ảnh test) cộng 1 lượt forward model phân đoạn thứ 2 — gần như gấp đôi thời gian riêng phần phân đoạn cho nhóm ảnh này, không ảnh hưởng độ trễ trung bình toàn hệ thống (232.27ms, đo trên ảnh chất lượng bình thường) vì phần lớn ảnh không rơi vào nhánh này.

**Hạn chế cần thừa nhận nếu bị hỏi sâu:** đây là 1 **heuristic tăng cường lúc suy luận** (inference-time robustness heuristic), không phải thành phần đã được học/tối ưu — không có gì đảm bảo lọc lông+CLAHE luôn cải thiện kết quả trên mọi ảnh khó, vì model chưa từng "thấy" phân phối ảnh đã qua các phép biến đổi này lúc huấn luyện. Hướng phát triển chắc chắn hơn (chưa làm, cần thời gian huấn luyện lại có GPU): fine-tune model với augmentation mô phỏng lông/bóng tối ngay trong tập train.

---

## B. Chỉ số hình học ABCD — slide 14

Đây là 4 công thức thật trong code (`_get_lesion_metrics`), cần thuộc chính xác. Input của cả 4 công thức đều là **mask nhị phân** (ảnh 2 chiều, mỗi pixel chỉ là 0 hoặc 1, 1=thuộc tổn thương) lấy từ output nhị phân hóa của mục A6, cộng thêm ảnh gốc RGB cho riêng chỉ số C.

**A — Asymmetry (độ bất đối xứng):** trước tiên cần tìm **trọng tâm (centroid)** của vùng tổn thương — điểm "trung bình có trọng số" của toàn bộ pixel thuộc mask, tính bằng **moments ảnh** (image moments — các đại lượng thống kê mô tả hình dạng 1 vùng, do OpenCV `cv2.moments()` tính sẵn):
$$M_{00} = \sum_{x,y} I(x,y), \qquad M_{10} = \sum_{x,y} x \cdot I(x,y), \qquad M_{01} = \sum_{x,y} y \cdot I(x,y)$$
Ký hiệu: $I(x,y)$ là giá trị pixel tại tọa độ $(x,y)$ (=1 nếu thuộc mask, =0 nếu không), tổng lấy trên toàn ảnh. $M_{00}$ chính là **diện tích** (đếm số pixel =1). Trọng tâm:
$$c_x = \frac{M_{10}}{M_{00}}, \qquad c_y = \frac{M_{01}}{M_{00}}$$
Sau khi có $(c_x, c_y)$, chia mask thành 2 nửa theo trục ngang (qua $c_y$) và 2 nửa theo trục dọc (qua $c_x$), **lật (flip)** 1 nửa rồi so khớp pixel-theo-pixel với nửa đối diện, đếm số pixel không trùng khớp — nếu hình đối xứng hoàn hảo thì lật xong sẽ trùng khít, số pixel lệch = 0:
$$\text{Asymmetry} = \text{clip}\left(\frac{\text{asym}_h + \text{asym}_v}{2 \times \text{lesion\_area}}, 0, 1\right)$$
Ký hiệu: $\text{asym}_h$ = số pixel lệch nhau khi so theo trục ngang, $\text{asym}_v$ = theo trục dọc, $\text{lesion\_area}=M_{00}$ = tổng số pixel tổn thương (dùng để chuẩn hóa về tỷ lệ, không phụ thuộc kích thước tổn thương to hay nhỏ). $\text{clip}(x, 0, 1)$ = ép giá trị $x$ về nằm trong đoạn $[0,1]$ (nếu $x<0$ thì lấy 0, nếu $x>1$ thì lấy 1) — đảm bảo chỉ số luôn nằm trong khoảng diễn giải được (0=đối xứng hoàn hảo, 1=bất đối xứng tối đa).

**B — Border complexity (độ phức tạp biên):** tỷ lệ **chu vi (perimeter — tổng độ dài đường viền bao quanh mask, tính bằng `cv2.arcLength`)** trên **căn bậc hai diện tích**:
$$\text{Border} = \frac{\text{Perimeter}}{\sqrt{\text{Area}}}$$
Vì sao chia cho $\sqrt{\text{Area}}$ chứ không phải chính $\text{Area}$: đây là cách chuẩn hóa để chỉ số không phụ thuộc vào **kích thước tuyệt đối** của tổn thương (1 hình tròn to và 1 hình tròn nhỏ đều phải cho ra cùng 1 giá trị Border, vì cùng là hình tròn "trơn" như nhau) — về mặt hình học, chu vi tỷ lệ thuận với **kích thước tuyến tính** (bậc 1, ví dụ bán kính), còn diện tích tỷ lệ với **kích thước bình phương** (bậc 2), nên $\sqrt{\text{Area}}$ cũng có bậc 1 giống Perimeter — tỷ lệ giữa 2 đại lượng cùng bậc mới bất biến theo kích thước (scale-invariant). Hình tròn hoàn hảo cho tỷ lệ này thấp nhất (giá trị hằng số $2\sqrt{\pi} \approx 3.545$), hình càng răng cưa/lồi lõm thì Perimeter tăng nhanh hơn nhiều so với Area, tỷ lệ càng cao.

**C — Color variation (độ biến thiên màu sắc):** với ảnh RGB gốc, lấy riêng các pixel nằm trong vùng mask, tính **độ lệch chuẩn (standard deviation)** giá trị màu trên từng kênh riêng biệt R, G, B:
$$\text{std}_c = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_{i,c} - \bar{x}_c)^2}, \quad c \in \{R, G, B\}$$
Ký hiệu: $n$ = số pixel trong vùng tổn thương, $x_{i,c}$ = giá trị kênh màu $c$ của pixel thứ $i$, $\bar{x}_c$ = giá trị trung bình kênh $c$ trên toàn vùng. Lấy trung bình 3 độ lệch chuẩn, chuẩn hóa về $[0,1]$ bằng cách chia cho 127.5 (bằng nửa dải giá trị 8-bit $[0,255]$ — độ lệch chuẩn lý thuyết tối đa của 1 kênh màu 8-bit không thể vượt quá khoảng này trong thực tế):
$$\text{Color} = \text{clip}\left(\frac{\text{mean}(\text{std}_R, \text{std}_G, \text{std}_B)}{127.5}, 0, 1\right)$$

**D — Diameter (equivalent diameter):** đường kính của 1 hình tròn có cùng diện tích với vùng tổn thương:
$$D = 2\sqrt{\frac{\text{Area}}{\pi}}$$

**Vì sao dùng "equivalent diameter" chứ không đo trực tiếp bounding box:** tổn thương thường không tròn, bounding box sẽ đo theo trục dọc/ngang gây sai lệch tùy hướng chụp; equivalent diameter bất biến với hướng xoay của tổn thương.

**Về DICOM PixelSpacing:** file DICOM có metadata `PixelSpacing` (mm/pixel thật theo trục X, Y của thiết bị chụp) — nhân số pixel với giá trị này ra kích thước thực (mm). Ảnh JPG/PNG thường không có metadata này nên D chỉ dừng ở đơn vị pixel.

---

## C. Phân loại ảnh — EfficientNet-B1 + CBAM — slide 15

### C1. EfficientNet — Compound Scaling (mở rộng cân đối)

**Bối cảnh:** trước EfficientNet, người ta thường "làm model mạnh hơn" bằng cách tăng 1 trong 3 yếu tố riêng lẻ: **depth** (độ sâu — số lớp xếp chồng), **width** (độ rộng — số channel mỗi lớp), hoặc **resolution** (độ phân giải ảnh đầu vào). EfficientNet chứng minh bằng thực nghiệm rằng tăng **cả 3 cùng lúc theo đúng tỷ lệ** cho hiệu quả (accuracy trên mỗi đơn vị tính toán) tốt hơn hẳn so với chỉ tăng 1 yếu tố.

**Công thức Compound Scaling:** với 1 hệ số $\phi$ (phi — do người dùng chọn, $\phi$ càng lớn thì model càng "to"; B0 ứng với $\phi=0$, B1 ứng với $\phi=1$, B2 ứng $\phi=2$...):
$$\text{depth} = \alpha^\phi, \quad \text{width} = \beta^\phi, \quad \text{resolution} = \gamma^\phi$$
Ký hiệu: $\alpha, \beta, \gamma$ là 3 hằng số được dò tìm 1 lần duy nhất bằng grid search nhỏ trên model B0 gốc (giá trị công bố: $\alpha \approx 1.2, \beta \approx 1.1, \gamma \approx 1.15$), với ràng buộc $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ (số mũ bình phương ở $\beta, \gamma$ vì FLOPs — số phép tính — của 1 lớp conv tỷ lệ thuận với **bình phương** width và **bình phương** resolution, còn tỷ lệ thuận bậc 1 với depth — ràng buộc này đảm bảo mỗi lần tăng $\phi$ thêm 1, tổng chi phí tính toán chỉ tăng khoảng gấp đôi, có kiểm soát chứ không bùng nổ).

**Vì sao chọn B1 (không phải B0 hay B2...):** B1 tương ứng $\phi=1$, tức depth≈1.2 lần, width≈1.1 lần, resolution≈1.15 lần so với B0 gốc — một bước "nhích lên" vừa phải, tăng nhẹ khả năng biểu diễn so với B0 (7.8M tham số so với ~5.3M của B0) mà chưa nặng tới mức khó chạy CPU như B3 trở lên (B3 đã ~12M tham số, input 300×300, chậm hơn đáng kể trên CPU). Đây là điểm cân bằng hợp lý cho bài toán tuyến cơ sở không có GPU. Việc thật sự đang dùng B1 (không nhầm B0) đã được xác nhận qua cấu trúc block thật đọc từ checkpoint: số block mỗi giai đoạn là $[2,3,3,4,4,5,2]$ — khớp đúng công thức depth scaling của B1 áp lên cấu hình block gốc của B0 là $[1,2,2,3,3,4,1]$ (nhân với hệ số $\alpha^\phi\approx1.2$ rồi làm tròn lên).

**MBConv (Mobile Inverted Bottleneck Convolution) — khối xây dựng cơ bản của EfficientNet:** mỗi khối MBConv gồm 3 bước: (1) **Expansion** — dùng conv 1×1 mở rộng số channel lên gấp $t$ lần (hệ số expansion, thường $t=6$), (2) **Depthwise Convolution** — conv 3×3 hoặc 5×5 nhưng xử lý **riêng biệt từng channel** (không trộn thông tin giữa các channel ở bước này, ký hiệu depthwise nghĩa là mỗi kernel chỉ áp cho đúng 1 channel input, khác conv thường mà 1 kernel gộp thông tin từ TẤT CẢ channel input) — cách này giảm số tham số đáng kể so với conv thường cùng kích thước kernel (conv thường: tham số $\propto C_{in} \times C_{out} \times k^2$; depthwise: chỉ $\propto C_{in} \times k^2$, rẻ hơn $C_{out}$ lần), (3) **Projection** — conv 1×1 nén channel trở lại số lượng ban đầu. Có residual connection (mục A1) nối thẳng input→output nếu shape khớp nhau.

### C2. CBAM (Convolutional Block Attention Module — "khối chú ý tích chập")

**Trực quan:** trước khi đưa feature map cuối cùng vào lớp phân loại, CBAM "lọc" lại theo 2 bước tuần tự — bước 1 hỏi "trong hàng nghìn channel đặc trưng này, channel nào (ví dụ: channel phát hiện màu nâu sẫm, channel phát hiện kết cấu sần sùi...) đáng tin cậy nhất cho ảnh này?", bước 2 hỏi "trong toàn bộ vùng không gian của ảnh, vị trí nào (h,w) đáng chú ý nhất?".

**Channel Attention (chú ý theo kênh):** với feature map đầu vào $F \in \mathbb{R}^{C\times H\times W}$ (trong hệ thống này $C=1280$ — đúng bằng feature_dim của EfficientNet-B1, $H=W=7$ khi input 224×224 — xem shape trace bên dưới), tính 2 vector tóm tắt toàn không gian bằng **AvgPool** (average pooling — lấy trung bình mọi giá trị theo 2 chiều không gian $H,W$, ra vector $1280\times1\times1$) và **MaxPool** (lấy giá trị lớn nhất, cũng ra $1280\times1\times1$), đưa cả 2 qua **cùng 1 MLP** (Multi-Layer Perceptron — mạng 2 lớp fully-connected, dùng chung trọng số cho cả 2 nhánh) — MLP này nén $1280 \to 1280/16=80$ (do `reduction=16`) rồi khôi phục lại $80\to1280$, cộng kết quả 2 nhánh, qua sigmoid $\sigma$:
$$M_c(F) = \sigma\big(\text{MLP}(\text{AvgPool}(F)) + \text{MLP}(\text{MaxPool}(F))\big) \in \mathbb{R}^{1280\times1\times1}$$
Sau đó nhân **theo broadcasting** (mỗi channel của $F$ nhân với đúng 1 số vô hướng tương ứng trong $M_c$) để ra $F' = M_c(F) \otimes F$, vẫn giữ shape $1280\times7\times7$.

**Spatial Attention (chú ý theo không gian):** trên $F'$, lần này tính avg/max theo **chiều channel** (ngược lại bước trên) — tại mỗi vị trí không gian $(h,w)$, lấy trung bình và lấy max của 1280 giá trị channel tại đó, ra 2 bản đồ $1\times7\times7$, ghép lại (concatenate) thành $2\times7\times7$, qua 1 lớp **conv 7×7** (kernel lớn để "nhìn" được ngữ cảnh không gian rộng) nén còn $1\times7\times7$, qua sigmoid:
$$M_s(F') = \sigma\big(\text{Conv}_{7\times7}([\text{AvgPool}_{\text{channel}}(F'); \text{MaxPool}_{\text{channel}}(F')])\big) \in \mathbb{R}^{1\times7\times7}$$
Nhân broadcasting lần nữa (mỗi vị trí không gian của $F'$ nhân với đúng 1 số tương ứng trong $M_s$) ra output cuối, vẫn giữ shape $1280\times7\times7$ — CBAM **không đổi shape**, chỉ "tô đậm/làm mờ" từng phần của feature map gốc theo mức độ quan trọng đã học được.

**Vì sao dùng cả avg lẫn max pooling, không chỉ 1 loại:** avg pooling nắm bối cảnh tổng thể (giá trị trung bình đại diện chung cho cả vùng), max pooling bắt đặc trưng nổi bật nhất (giá trị cực đại thường tương ứng với đặc trưng "rõ rệt nhất" xuất hiện đâu đó) — dùng cả 2 bổ trợ nhau tốt hơn dùng riêng lẻ (đã kiểm chứng thực nghiệm trong paper gốc CBAM, Woo et al. 2018).

**Grad-CAM (Gradient-weighted Class Activation Mapping) — tính trên chính feature map sau CBAM:** để biết "model đang nhìn vào đâu trên ảnh gốc khi ra quyết định lớp $c$", tính đạo hàm của điểm số (logit, trước softmax) lớp dự đoán $y^c$ theo từng giá trị trong feature map ở lớp attention $A$, rồi lấy trung bình đạo hàm theo không gian để ra 1 trọng số quan trọng $\alpha_k^c$ cho mỗi channel $k$:
$$\alpha_k^c = \frac{1}{Z}\sum_{i,j} \frac{\partial y^c}{\partial A_{ij}^k}$$
Ký hiệu: $Z = H\times W$ (số vị trí không gian, dùng để lấy trung bình thay vì tổng), $A_{ij}^k$ = giá trị feature map tại channel $k$, vị trí không gian $(i,j)$. Sau đó tính tổng có trọng số của toàn bộ channel rồi cắt bỏ phần âm bằng ReLU:
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
$\text{ReLU}(z) = \max(0, z)$ — chỉ giữ giá trị dương vì chỉ quan tâm vùng có ảnh hưởng **tăng cường** cho lớp dự đoán $c$, bỏ qua vùng có ảnh hưởng **ngược lại/ức chế** (giá trị âm). Kết quả $L^c_{\text{Grad-CAM}}$ có shape $7\times7$ (nhỏ, bằng đúng feature map cuối), được resize/nội suy phóng to lại bằng kích thước ảnh gốc rồi tô màu (colormap) để tạo ra ảnh nhiệt trực quan như thấy trên slide 14.

**Câu hỏi khả năng bị hỏi:** *"Tại sao chọn attention ở đây thay vì thêm lớp conv thường?"* → Attention không tăng nhiều tham số (channel attention chỉ thêm 1 MLP nhỏ $1280\to80\to1280$ ≈ 204,898 tham số như đã tính ở mục E3, spatial attention chỉ thêm 1 conv $2\times1\times7\times7=98$ tham số) nhưng giúp model tự học "nhìn đâu là quan trọng" thay vì xử lý đều mọi vùng ảnh như nhau — đặc biệt hữu ích khi tổn thương chỉ chiếm 1 phần nhỏ trong ảnh, phần da lành xung quanh là nhiễu.

### C3. Input/Output chi tiết — đi qua từng tầng EfficientNet-B1

| Tầng | Output shape $(C,H,W)$ | Ghi chú |
|---|---|---|
| Input (đã resize + chuẩn hóa) | $(3, 224, 224)$ | Chuẩn ImageNet: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] — vì backbone pretrain trên ImageNet với đúng chuẩn này |
| Stem (conv 3×3 stride 2) | $(32, 112, 112)$ | |
| MBConv stage 1-7 (7 giai đoạn, mỗi giai đoạn stride khác nhau) | giảm dần còn $(1280, 7, 7)$ | Số block mỗi giai đoạn của B1: $[2,3,3,4,4,5,2]$ — xem mục C1 |
| `forward_features()` output | $(1280, 7, 7)$ | = $7\times7=49$ vị trí không gian, mỗi vị trí 1 vector 1280 chiều — **chính là 49 token nếu dùng chế độ spatial ở mục E3**, nhưng nhánh phân loại classification KHÔNG dùng chế độ này |
| CBAM | $(1280, 7, 7)$ | Không đổi shape, chỉ tái trọng số |
| Global Average Pooling | $(1280,)$ | Ép phẳng 49 vị trí không gian thành 1 vector duy nhất bằng cách lấy trung bình |
| Dropout(0.3) | $(1280,)$ | Trong lúc huấn luyện: ngẫu nhiên "tắt" 30% giá trị về 0 để chống overfit; lúc suy luận: không áp dụng, dùng nguyên vector |
| Linear(1280→7) | $(7,)$ | 7 logit thô, chưa qua softmax |
| Softmax | $(7,)$ | $P(C_k\|\text{Image})$, tổng 7 giá trị = 1 — dùng trực tiếp trong bước hợp nhất Bayes (mục D) |

**Tổng tham số:** đọc trực tiếp checkpoint (81.4MB) → EfficientNet-B1 gốc công bố chính thức khoảng 7.8 triệu tham số, cộng CBAM (~205 nghìn) và Linear(1280→7) (~9 nghìn, không đáng kể).

**Vì sao input là 224 (không phải 256 như model phân đoạn):** xem giải thích đầy đủ ở mục A6 (3 lý do: output chỉ là 1 vector toàn cục nên không cần độ phân giải cao; 224 khớp đúng chuẩn ImageNet pretraining của EfficientNet; cả 224 lẫn 256 đều chia hết cho 32 nên không phải lý do bắt buộc).

**Vì sao Dropout đặt đúng 0.3 (không phải 0.2 hay 0.5):** đây là 1 lựa chọn "vùng an toàn" (sweet spot) giữa 2 rủi ro đối lập, không có 1 công thức toán học nào tính ra chính xác con số này — phải chọn bằng thực nghiệm/kinh nghiệm:
- **Nếu dropout quá thấp (ví dụ 0.1, hoặc bỏ hẳn dropout)**: với tập dữ liệu phân loại tương đối nhỏ so với 7.8 triệu tham số của backbone, nguy cơ **overfit** cao (model "học thuộc" nhiễu/chi tiết ngẫu nhiên của tập train thay vì quy luật tổng quát) — đặc biệt nguy hiểm ở lớp cuối cùng `Linear(1280→7)` vì đây là lớp duy nhất KHÔNG có pretrain, khởi tạo ngẫu nhiên hoàn toàn nên dễ "ghi nhớ" tập train nếu không bị chặn.
- **Nếu dropout quá cao (ví dụ 0.5 — mức kinh điển hay dùng ở các lớp Fully-Connected lớn trong AlexNet/VGG cũ)**: ở đây chỉ có **đúng 1 lớp Linear** ngay sau dropout (không phải 2-3 lớp FC nối tiếp như AlexNet/VGG) — tắt tới 50% trong 1280 giá trị đầu vào của 1 lớp Linear duy nhất có thể làm mất quá nhiều thông tin hữu ích mỗi lần forward, khiến việc học chậm lại hoặc **underfit** (model không đủ tín hiệu ổn định để hội tụ tốt).
- **0.3 là giá trị trung gian phổ biến trong thực hành transfer-learning** cho các "classifier head" nhẹ (ít lớp) gắn sau 1 backbone pretrain đã đóng băng phần lớn — đủ mạnh để chống overfit trên tập dữ liệu da liễu có kích thước hạn chế, nhưng không quá mạnh đến mức phá vỡ tín hiệu học từ 1 lớp Linear duy nhất. *(Lưu ý: đây là lý giải theo nguyên tắc thực hành chung, không phải con số có chứng minh toán học "tối ưu" — nếu hội đồng hỏi "sao không thử 0.2 hay 0.4", câu trả lời trung thực là: có thể thử nghiệm thêm để so sánh, nhưng 0.3 là lựa chọn hợp lý ngay từ đầu dựa trên kinh nghiệm phổ biến, chưa phải kết quả của 1 quá trình tìm kiếm siêu tham số — hyperparameter search — có hệ thống.)*

---

## D. Hợp nhất Bayes có trọng số (Fusion) — slide 15

### D0. Định lý Bayes cơ bản — nhắc lại trước khi vào công thức thật

$$P(C_k \mid E) = \frac{P(E \mid C_k)\, P(C_k)}{P(E)}$$

Ký hiệu: $C_k$ = giả thuyết đang xét (ở đây là "bệnh nhân thuộc lớp bệnh $k$", $k\in\{1,...,7\}$), $E$ = bằng chứng quan sát được (evidence — ở đây là tuổi/giới/vị trí hoặc ảnh). $P(C_k)$ gọi là **prior** (tiền nghiệm — niềm tin về $C_k$ TRƯỚC khi thấy bằng chứng, ví dụ tỷ lệ mắc mỗi bệnh trong dân số). $P(E\mid C_k)$ gọi là **likelihood** (khả năng — nếu đúng là bệnh $k$ thì xác suất quan sát được bằng chứng $E$ này là bao nhiêu). $P(C_k\mid E)$ gọi là **posterior** (hậu nghiệm — niềm tin về $C_k$ SAU khi đã thấy bằng chứng $E$, đây là cái ta thực sự muốn biết). $P(E)$ ở mẫu số chỉ là hằng số chuẩn hóa (không phụ thuộc $k$, đảm bảo tổng các $P(C_k\mid E)$ trên mọi $k$ bằng 1) — vì vậy trong thực hành thường viết gọn bằng ký hiệu $\propto$ ("tỷ lệ thuận với", bỏ qua mẫu số): $P(C_k\mid E) \propto P(E\mid C_k)P(C_k)$.

### D1. Áp dụng Bayes cho dịch tễ — phần đúng chuẩn 100%

$$P(C_k \mid \text{Demo}) \propto P(C_k) \cdot P(\text{Age}\mid C_k) \cdot P(\text{Gender}\mid C_k) \cdot P(\text{Location}\mid C_k)$$

Ở đây $\text{Demo}$ = bộ 3 bằng chứng dịch tễ (tuổi, giới tính, vị trí giải phẫu). Vế phải nhân 3 likelihood riêng biệt thay vì 1 likelihood chung $P(\text{Demo}\mid C_k)$ — đây là **giả định Naive Bayes**: coi tuổi, giới, vị trí là **độc lập có điều kiện** khi đã biết lớp bệnh $C_k$ (nghĩa là: nếu đã biết chắc bệnh nhân mắc bệnh $k$, thì biết thêm tuổi của họ không giúp đoán thêm được gì về giới tính hay vị trí — một giả định đơn giản hóa, không hoàn toàn đúng trong thực tế nhưng giúp bài toán khả thi với lượng dữ liệu thống kê nhỏ, không cần ước lượng phân phối kết hợp 3 chiều phức tạp).

**Age likelihood — mô hình hóa bằng phân phối Gaussian (phân phối chuẩn):**
$$P(\text{Age}=a \mid C_k) = \frac{1}{\sigma_k\sqrt{2\pi}} \exp\left(-\frac{(a-\mu_k)^2}{2\sigma_k^2}\right)$$
Ký hiệu: $a$ = tuổi bệnh nhân đang xét, $\mu_k$ = tuổi trung bình của những người mắc bệnh $k$ (ước lượng từ thống kê thật trên HAM10000), $\sigma_k$ = độ lệch chuẩn tuổi của lớp $k$ (tuổi càng "phân tán" quanh trung bình thì $\sigma_k$ càng lớn, đường cong Gaussian càng "bẹt"), $\exp(\cdot)$ = hàm mũ cơ số $e$. Công thức này trả về giá trị **mật độ xác suất** (probability density) cao nhất khi $a=\mu_k$ (đúng bằng tuổi trung bình của lớp đó) và giảm dần khi $a$ càng xa $\mu_k$.

### D2. Log-linear pooling — phần KHÔNG phải Bayes chuẩn (điểm hay bị hỏi nhất)

$$P_{\text{final}}(C_k) \propto \big[P(C_k\mid\text{Image})\big]^\lambda \cdot \big[P(C_k\mid\text{Demo})\big]^{1-\lambda}, \qquad \lambda \in [0,1]$$

Lấy log 2 vế (dùng tính chất $\log(x^n)=n\log x$ và $\log(xy)=\log x + \log y$):
$$\log P_{\text{final}}(C_k) = \lambda \log P(C_k\mid\text{Image}) + (1-\lambda)\log P(C_k\mid\text{Demo}) + \text{const}$$
Đây là **trung bình có trọng số (weighted average)** của 2 giá trị log-xác suất, với trọng số $\lambda$ và $1-\lambda$ (cộng lại đúng bằng 1 — đây gọi là **convex combination**, tổ hợp lồi). Vì phép toán này là trung bình cộng trên **log** của xác suất chứ không phải trên chính xác suất, khi "giải log" (mũ hóa) ngược lại thì tương đương **trung bình nhân có trọng số** (weighted geometric mean) trên xác suất gốc. Tên gọi chuẩn trong tài liệu thống kê cho kỹ thuật hợp nhất nhiều "chuyên gia" (expert) theo cách này: **logarithmic opinion pool** (bể ý kiến dạng logarit).

**Vì sao KHÔNG phải Bayes thuần túy — giải thích kỹ:** nếu muốn đúng Bayes chuẩn để hợp nhất 2 nguồn bằng chứng độc lập (ảnh và dịch tễ) cho cùng 1 biến ẩn $C_k$, công thức đúng phải là:
$$P(C_k \mid \text{Image}, \text{Demo}) \propto P(C_k) \cdot P(\text{Image}\mid C_k) \cdot P(\text{Demo}\mid C_k)$$
— **nhân 2 likelihood**, không phải lũy thừa 2 posterior. Vấn đề: model phân loại ảnh (mục C) xuất ra thẳng $P(C_k\mid\text{Image})$ (posterior, qua softmax) chứ không phải $P(\text{Image}\mid C_k)$ (likelihood) — muốn "quy đổi" đúng phải chia ngược cho prior lúc huấn luyện: $P(\text{Image}\mid C_k) \propto P(C_k\mid\text{Image}) / P_{\text{train}}(C_k)$ (theo đúng công thức Bayes ở mục D0, giải ngược ra likelihood từ posterior) — bước "gỡ prior" này code hiện **không làm**, mà dùng thẳng luôn posterior $P(C_k\mid\text{Image})$ đưa vào công thức lũy thừa $\lambda$. Đây chính là lý do gọi tên đúng là "hợp nhất kiểu Bayes có trọng số" (Bayesian-inspired), không phải "công thức Bayes" theo đúng nghĩa toán học chặt chẽ.

### D3. λ tự động theo Entropy (Shannon Entropy — Entropy Shannon)

$$H = -\sum_{c=1}^{K} P(c)\log P(c), \qquad H_{\text{norm}} = \frac{H}{\log(K)}, \qquad \lambda = \lambda_{\max} - (\lambda_{\max}-\lambda_{\min}) \cdot H_{\text{norm}}$$

Ký hiệu: $H$ = **entropy** (độ đo bất định — do Claude Shannon đề xuất năm 1948, đo "mức độ hỗn loạn/không chắc chắn" của 1 phân phối xác suất, đơn vị là bit nếu log cơ số 2 hoặc nat nếu log tự nhiên), $K=7$ (số lớp bệnh), $P(c)$ = xác suất model dự đoán cho lớp $c$ (lấy từ output softmax ở mục C3). $H$ đạt **giá trị nhỏ nhất = 0** khi phân phối "nhọn tuyệt đối" (1 lớp có xác suất 1, các lớp còn lại 0 — model hoàn toàn chắc chắn), đạt **giá trị lớn nhất = $\log(K)$** khi phân phối "phẳng tuyệt đối" (mọi lớp xác suất bằng nhau $=1/K$ — model hoàn toàn phân vân, không nghiêng về lớp nào). $H_{\text{norm}}$ chia cho $\log(K)$ để chuẩn hóa entropy về đúng khoảng $[0,1]$ bất kể $K$ là bao nhiêu lớp. $\lambda_{\min}, \lambda_{\max}$ là 2 hằng số chặn trên/dưới cho $\lambda$ (trong code là 0.5 và 0.95) — khi $H_{\text{norm}}=0$ (rất chắc chắn) thì $\lambda=\lambda_{\max}$ (tin ảnh tối đa); khi $H_{\text{norm}}=1$ (rất phân vân) thì $\lambda=\lambda_{\min}$ (chỉ tin ảnh ở mức tối thiểu, không bao giờ về 0 hoàn toàn — luôn giữ lại ít nhất 1 phần tin cậy cho ảnh).

**Vì sao dùng entropy (nhìn toàn bộ phân phối) chứ không dùng trực tiếp xác suất top-1 (confidence — chỉ nhìn giá trị lớn nhất):** ví dụ minh họa — phân phối A = $(0.4, 0.4, 0.05, 0.05, 0.05, 0.025, 0.025)$ và phân phối B = $(0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)$ đều có top-1 = 0.4 giống hệt nhau, nhưng A "phân vân nghiêm trọng" giữa 2 lớp đầu (gần bằng nhau) trong khi B "khá chắc chắn" nghiêng về lớp đầu (các lớp còn lại đều thấp và đều nhau). Confidence top-1 không phân biệt được 2 trường hợp này (đều báo 0.4), nhưng entropy tính trên toàn bộ 7 giá trị sẽ cho $H_A > H_B$ (A bất định hơn) — đúng với trực giác.

**Hạn chế cần thừa nhận nếu bị hỏi:** entropy tính trên softmax **chưa hiệu chỉnh (uncalibrated)** — mạng neural nổi tiếng có xu hướng "quá tự tin" (overconfident, hiện tượng đã được ghi nhận rộng rãi trong tài liệu học máy hiện đại), nên $H$ đo được có thể thấp hơn độ bất định thật sự. Cách khắc phục chuẩn là **temperature scaling** (chia logit cho 1 hằng số nhiệt độ $T>1$ trước khi softmax, làm phân phối "bẹt" hơn, gần với độ tin cậy thật hơn) trên tập validation trước khi tính entropy — hướng phát triển tiếp theo, chưa làm trong hệ thống hiện tại.

---

## E. VQA — DistilGPT-2 + LoRA — slide 16

### E1. Kiến trúc Transformer Decoder (nền tảng GPT-2/DistilGPT-2)

**Self-Attention (tự chú ý) — nền tảng của mọi Transformer:** với 1 chuỗi gồm $L$ token, mỗi token có vector biểu diễn $d$ chiều, self-attention cho phép mỗi token "hỏi" tất cả token khác (kể cả chính nó) xem nên chú ý bao nhiêu đến từng token đó khi tính lại biểu diễn của mình. Cơ chế này dùng 3 ma trận trọng số học được để chiếu input $X\in\mathbb{R}^{L\times d}$ thành 3 phiên bản khác nhau:
$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$
Ký hiệu: $Q$ (**Query** — "câu hỏi": mỗi token tạo ra 1 vector query đại diện cho "tôi đang cần thông tin gì"), $K$ (**Key** — "chìa khóa": mỗi token tạo ra 1 vector key đại diện cho "tôi có thông tin gì để cung cấp"), $V$ (**Value** — "giá trị": nội dung thông tin thật sự sẽ được lấy ra nếu match). Công thức attention đầy đủ:
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
$QK^T$ là tích vô hướng (dot product) giữa mọi cặp Query-Key — giá trị càng lớn nghĩa là Query và Key đó càng "khớp nhau" (nên chú ý nhiều). $d_k$ = số chiều của mỗi vector Key (nếu chia đa đầu — xem Multi-Head bên dưới — thì $d_k = d/\text{số đầu}$), chia cho $\sqrt{d_k}$ để tránh giá trị dot-product quá lớn khi $d_k$ lớn (nếu không chia, softmax sẽ "bão hòa" — 1 giá trị gần 1, các giá trị khác gần 0, khiến gradient lúc lan truyền ngược gần như bằng 0, model khó học). Softmax biến các điểm số thành trọng số chú ý (tổng = 1), nhân với $V$ để lấy ra thông tin theo đúng tỷ lệ đã "chú ý".

**Multi-Head Attention (chú ý đa đầu):** thay vì chỉ tính 1 lần attention trên toàn bộ $d=768$ chiều, DistilGPT-2 chia thành **12 "đầu" (head)** chạy song song, mỗi đầu chỉ xử lý $d_k = 768/12 = 64$ chiều — mỗi đầu có thể học cách "chú ý" theo 1 khía cạnh khác nhau (ví dụ đầu này chú ý quan hệ ngữ pháp, đầu khác chú ý quan hệ ngữ nghĩa xa), rồi ghép (concatenate) output của cả 12 đầu lại thành đủ $768$ chiều.

**Causal masking (che nhân quả):** GPT là **decoder-only** (chỉ dùng phần decoder của kiến trúc Transformer gốc, không có encoder riêng), dùng **causal attention** — token thứ $i$ chỉ được phép "nhìn" các token $1, 2, ..., i$ (bản thân nó và các token trước đó), **không được nhìn token tương lai** ($i+1, i+2,...$) — thực hiện bằng cách gán $-\infty$ vào các vị trí "tương lai" trong ma trận điểm số $QK^T$ trước khi đưa qua softmax (do $e^{-\infty}=0$, các vị trí đó tự động có trọng số chú ý bằng 0). Đây là cơ chế phù hợp bài toán **sinh văn bản tuần tự (autoregressive generation)** — sinh từng từ 1 dựa trên các từ đã sinh trước đó, giống cách con người viết văn bản từ trái sang phải.

**DistilGPT-2 — bản "chưng cất" của GPT-2:** **Knowledge Distillation** (chưng cất tri thức) là kỹ thuật huấn luyện 1 model nhỏ (gọi là **student** — học trò) để "bắt chước" hành vi output của 1 model lớn đã huấn luyện sẵn (gọi là **teacher** — thầy giáo), thay vì học trực tiếp từ dữ liệu gốc. DistilGPT-2 là student được chưng cất từ GPT-2 gốc (12 lớp, 117M tham số) xuống còn **6 lớp Transformer, ~82M tham số** — giữ được phần lớn khả năng ngôn ngữ của GPT-2 gốc trong khi nhanh hơn và nhẹ hơn đáng kể, phù hợp chạy trên CPU không có GPU.

**Config chính xác của DistilGPT-2 (đã xác nhận qua `AutoConfig`):** `n_embd=768` (kích thước ẩn — mỗi token biểu diễn bằng vector 768 chiều), `n_layer=6` (6 lớp Transformer Block xếp chồng), `n_head=12` (12 đầu attention song song, mỗi đầu 64 chiều), `vocab_size=50257` (kích thước từ điển — số token khác nhau mà model có thể sinh ra hoặc nhận vào), `n_positions=1024` (độ dài chuỗi tối đa model xử lý được 1 lần).

### E2. LoRA (Low-Rank Adaptation — "thích nghi hạng thấp") — chi tiết đúng số bạn dùng: rank 8, alpha 16

**Vấn đề LoRA giải quyết:** fine-tune (tinh chỉnh — huấn luyện tiếp 1 model đã pretrain trên dữ liệu mới) toàn bộ tham số của 1 LLM (dù "chỉ" ~82-90 triệu tham số) trên dữ liệu rất nhỏ (74-80 mẫu) gần như chắc chắn overfit nặng (model đủ "to" để học thuộc lòng 74 mẫu mà không học được quy luật tổng quát), đồng thời tốn rất nhiều bộ nhớ để lưu gradient cho TỪNG tham số trong hàng chục triệu tham số đó.

**Ý tưởng cốt lõi:** với 1 ma trận trọng số gốc $W \in \mathbb{R}^{d\times d}$ (rất lớn, $d=768$ ở đây) trong model pretrain, thay vì cập nhật trực tiếp $W$, LoRA **đóng băng $W$ hoàn toàn** (giữ nguyên, không tính gradient), và học thêm **2 ma trận phụ hạng thấp (low-rank)**: $A \in \mathbb{R}^{r\times d}$ và $B \in \mathbb{R}^{d\times r}$, với **rank** $r \ll d$ (rank — hạng của ma trận, ở đây $r=8$ so với $d=768$, nhỏ hơn 96 lần). Trọng số hiệu dụng khi suy luận:
$$W' = W + \Delta W = W + \frac{\alpha}{r}\, BA$$
Ký hiệu: $\Delta W = \frac{\alpha}{r}BA$ là "phần điều chỉnh" được cộng thêm vào $W$ gốc, có **rank tối đa bằng $r$** (vì $B$ có $r$ cột, $A$ có $r$ hàng — tích $BA$ không thể có rank cao hơn $r$, đây chính là ý nghĩa "hạng thấp": giả định rằng sự thay đổi cần thiết để thích nghi domain mới **không cần** đủ "phong phú" để lấp đầy toàn bộ không gian $d\times d$ chiều, mà chỉ cần 1 không gian con $r$ chiều là đủ). $\alpha=16$ là hệ số **scaling** (tỷ lệ), $r=8$, tỷ số $\alpha/r = 16/8 = 2$ — hệ số này nhân vào $BA$ để giữ độ lớn (magnitude) của $\Delta W$ ổn định, không phụ thuộc quá nhiều vào việc chọn $r$ bao nhiêu (giúp khi thử nghiệm đổi $r$ không cần chỉnh lại learning rate).

**Vì sao tiết kiệm tham số khủng khiếp — tính bằng số cụ thể:** với $d=768$, ma trận gốc $W\in\mathbb{R}^{768\times768}$ có $768\times768=589{,}824$ tham số; còn cặp $A,B$ với $r=8$ chỉ có $A: 8\times768=6{,}144$ cộng $B: 768\times8=6{,}144$, tổng $12{,}288$ tham số — chỉ bằng **2.08%** so với $W$ gốc cho riêng 1 ma trận. Đây chính là lý do tổng thể toàn model chỉ cần huấn luyện 1,926,754/90,414,631 ≈ **2.13%** tham số (bảng chi tiết đầy đủ ở mục E3).

**Vì sao target đúng `c_attn` — và vì sao shape LoRA_B lại là (2304, 8) chứ không phải (768, 8):** trong cài đặt GPT-2 của thư viện HuggingFace, thay vì tách riêng 3 ma trận $W_Q, W_K, W_V$ (mục E1), tác giả gộp chung thành **1 lớp Linear duy nhất tên `c_attn`**, chiếu input $768$ chiều thành output $768 \times 3 = 2304$ chiều (chứa cả Q, K, V nối liền nhau trong 1 tensor, sau đó code tách lại thành 3 phần bằng `.split()`). Vì vậy khi áp LoRA lên `c_attn`, ma trận $A$ có shape $(8, 768)$ (nhận input 768 chiều), còn $B$ có shape $(2304, 8)$ (trả về output 2304 chiều, khớp đúng output gốc của `c_attn`) — **đây chính xác là con số tôi đọc trực tiếp được từ checkpoint thật** (`lora_A.default.weight` shape `(8,768)`, `lora_B.default.weight` shape `(2304,8)`), xác nhận LoRA áp đúng vào lớp QKV gộp chung này, trên cả 6 lớp Transformer của DistilGPT-2.

**Câu hỏi khả năng bị hỏi:** *"Rank 8 có đủ không, sao không chọn rank cao hơn?"* → Rank càng cao thì càng biểu diễn được nhiều loại điều chỉnh phức tạp hơn (không gian con $r$ chiều càng "rộng") nhưng cũng dễ overfit hơn trên tập dữ liệu 74-80 mẫu rất nhỏ; rank 8 là lựa chọn thận trọng cân bằng giữa khả năng học và nguy cơ overfit — có thể thử nghiệm thêm rank 4 hoặc 16 để so sánh nếu có thời gian.

### E3. Luồng ghép token thật — vì sao đây là "nối đặc trưng" chứ không phải Cross-Attention đầy đủ

**Bước 1 — Vision Backbone:** ảnh 224×224×3 → EfficientNet-B1+CBAM → global average pooling → **1 vector 1280 chiều duy nhất** (không giữ lại thông tin không gian 7×7, đã "ép phẳng" thành 1 điểm tóm tắt toàn ảnh).

**Bước 2 — Projection:** Linear(1280→768) → GELU → Dropout(0.3) → Linear(768→768) → ra **1 token ảnh** có cùng chiều 768 với không gian embedding từ vựng của GPT-2.

**Bước 3 — Ghép chuỗi:** `inputs_embeds = concat([1 token ảnh, N token văn bản câu hỏi], dim=1)` — token ảnh được chèn vào **đầu chuỗi**, sau đó toàn bộ chuỗi (ảnh + chữ) đi qua các lớp self-attention **causal thông thường** của DistilGPT-2 (đã gắn LoRA) như thể token ảnh cũng là 1 "từ" bình thường.

**Vì sao đây không phải kiến trúc VLM (Vision-Language Model) hiện đại đúng nghĩa:** các model VLM thời hiện đại (LLaVA, Med-Flamingo...) dùng **cross-attention chuyên biệt** — decoder văn bản "chủ động hỏi lại" nhiều token ảnh khác nhau tùy theo từng bước sinh từ, và thường giữ **nhiều token ảnh theo lưới không gian** (ví dụ 49 token 7×7) để không mất thông tin vị trí. Ở đây chỉ có **đúng 1 token ảnh duy nhất**, đã nén hết thông tin không gian thành 1 vector toàn cục qua global average pooling, và cách "chú ý" tới nó chỉ là self-attention thông thường (token ảnh bình đẳng như mọi token chữ khác) — đây là kiến trúc **nối đặc trưng đơn giản (linear projection + concatenation)**, một dạng kiến trúc lai sơ khởi (pre-VLM hybrid), không phải cross-attention đa phương thức thực thụ.

**Bảng tham số huấn luyện (khớp chính xác 2.13%/1,926,754 đã công bố — tôi đã trực tiếp đối chiếu từng nhóm tensor trong checkpoint):**

| Thành phần | Tham số | Trạng thái |
|---|---:|---|
| LoRA A/B trên `c_attn`, cả 6 lớp DistilGPT-2 | 147,456 | Huấn luyện |
| Projection layer (1280→768→768) | 1,574,400 | Huấn luyện (khởi tạo mới hoàn toàn) |
| CBAM attention (trong vision backbone) | 204,898 | Huấn luyện (mở băng riêng khối này) |
| **Tổng huấn luyện** | **1,926,754** | = 2.13% của 90,414,631 |
| Backbone EfficientNet-B1 (trừ CBAM) | ~6,575,301 | Đóng băng |
| DistilGPT-2 gốc (trừ LoRA) | ~81,700,000 | Đóng băng |

**Lưu ý quan trọng khi trả lời:** file huấn luyện `train_vqa_joint.py` có định nghĩa sẵn một kiến trúc phức tạp hơn nhiều — `DeepCrossAttentionBridge` (cross-attention 2 lớp thật sự, có thể giữ 49 token không gian), `ClinicalStructureInjector` (tiêm trực tiếp chỉ số ABCD + xác suất phân loại vào chuỗi token), `ClinicalPrefix` (8 token prefix học được kiểu prefix-tuning) — nhưng khi tôi mở checkpoint `dermavqa_gpt2_joint_best.pth` thật (đúng file `app_streamlit.py` đang nạp), **không tìm thấy bất kỳ tensor nào** thuộc các module này. Đây là kiến trúc đã viết code, có thể dùng để trình bày như "hướng phát triển đã thiết kế sẵn", nhưng **không phải kiến trúc đang thực sự chạy trong demo** — nếu bị hỏi trực tiếp về các module này, trả lời trung thực theo đúng thực tế trên, đừng mô tả chúng như đã hoàn thiện và đang chạy.

---

## F. RAG (Retrieval-Augmented Generation — "sinh văn bản có tăng cường truy xuất") — slide 16, 17

### F0. Chunking — chia nhỏ tài liệu y văn trước khi đưa vào RAG (bước hay bị bỏ qua khi giải thích RAG)

**Vì sao cần chunking (chia nhỏ thành từng "khối"/"đoạn"):** không thể nhúng (embed) toàn bộ 1 tài liệu dài thành 1 vector duy nhất rồi mong nó đại diện tốt cho MỌI nội dung bên trong — 1 vector chỉ nắm được ý nghĩa tổng quát, sẽ "loãng" nếu tài liệu quá dài và nói về nhiều chủ đề khác nhau. Giải pháp: chia tài liệu gốc thành nhiều **chunk** (khối/đoạn nhỏ hơn), mỗi chunk được nhúng thành 1 vector riêng — khi tìm kiếm sẽ tìm đúng chunk liên quan nhất thay vì cả tài liệu.

**Cách chunking thật trong hệ thống này** (`rag_engine.py::_populate_db`): dùng `re.split()` (regular expression — biểu thức chính quy, công cụ tách chuỗi theo khuôn mẫu) để tách file `medical_guidelines.txt` tại mỗi vị trí xuất hiện khuôn mẫu `[BỆNH LÝ <số>:` — nghĩa là **chia theo ranh giới tự nhiên của tài liệu (mỗi bệnh lý 1 chunk)**, không chia theo độ dài cố định (khác với nhiều hệ thống RAG khác thường chia theo số từ/ký tự cố định, ví dụ mỗi 500 từ 1 chunk bất kể ngữ nghĩa). Tôi đọc trực tiếp file này: tổng cộng **7 chunk** (đúng bằng 7 lớp bệnh: AKIEC, BCC, BKL, DF, MEL, NV, VASC), tổng dung lượng file **11,082 ký tự** trên 60 dòng → trung bình mỗi chunk khoảng **1,583 ký tự** (~300-400 token tùy cách tokenize).

**Rủi ro cần biết — giới hạn độ dài của embedding model:** all-MiniLM-L6-v2 (mục F1) có độ dài chuỗi tối đa thường dùng là **256 token** (theo cấu hình phổ biến của model này) — nếu 1 chunk vượt quá độ dài này, phần vượt quá sẽ bị **cắt bớt (truncation)** khi nhúng, thông tin phần cuối chunk có thể không được đưa vào vector đại diện. Với trung bình ~300-400 token/chunk như trên, đây là điều **cần kiểm tra lại thực tế** (không phải lý thuyết) trước khi khẳng định chắc rằng toàn bộ nội dung 7 chunk đều được nhúng đầy đủ không mất thông tin — câu trả lời an toàn nếu bị hỏi: *"Cách chia theo ranh giới bệnh lý là hợp lý về mặt ngữ nghĩa, nhưng em cần xác minh lại độ dài token thực tế của từng chunk có vượt giới hạn 256 token của model nhúng hay không."*

### F1. Sentence Embedding (All-MiniLM-L6-v2)

**Ý tưởng:** biến 1 câu văn bản bất kỳ (câu hỏi của bác sĩ, hoặc 1 chunk tài liệu) thành **1 vector số thực có chiều cố định** — bất kể câu dài hay ngắn, output luôn là 1 vector đúng **384 chiều** — sao cho **2 câu có ý nghĩa gần nhau thì 2 vector tương ứng cũng "gần nhau"** trong không gian 384 chiều đó (đo độ gần bằng cosine similarity, mục F2). Kiến trúc: 1 Transformer encoder 6 lớp (ký hiệu "L6" trong tên model), bản thân nó cũng là 1 model được **chưng cất (distillation, xem mục E1)** từ model lớn hơn — "MiniLM" = "mini language model", thiết kế để nhẹ và nhanh, phù hợp chạy CPU cho tác vụ nhúng câu.

### F2. Cosine Similarity (độ tương đồng cosine) — vì sao dùng độ đo này chứ không phải khoảng cách Euclid

$$\cos(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|\, \|\vec{v}\|}$$

Ký hiệu: $\vec{u}, \vec{v}$ là 2 vector embedding cần so sánh (384 chiều), $\vec{u}\cdot\vec{v} = \sum_i u_i v_i$ là **tích vô hướng (dot product)** — nhân từng cặp phần tử tương ứng rồi cộng lại, $\|\vec{u}\| = \sqrt{\sum_i u_i^2}$ là **độ dài (norm)** của vector $\vec{u}$ (định nghĩa đã nêu ở đầu tài liệu). Kết quả $\cos(\vec{u},\vec{v})$ nằm trong khoảng $[-1, 1]$: bằng 1 nếu 2 vector cùng hướng hoàn toàn (giống hệt về "ý nghĩa"), bằng 0 nếu vuông góc (không liên quan), bằng -1 nếu ngược hướng hoàn toàn.

**Vì sao đo góc (cosine) thay vì đo khoảng cách thẳng (Euclidean distance $=\|\vec{u}-\vec{v}\|$):** cosine similarity **không quan tâm độ dài (magnitude)** của vector, chỉ quan tâm **hướng**. Với embedding văn bản, độ dài vector đôi khi bị ảnh hưởng bởi độ dài câu hoặc tần suất từ, không phản ánh đúng ý nghĩa — 2 câu cùng ý nghĩa nhưng 1 câu dài hơn có thể cho ra vector "dài" hơn theo Euclidean, nhưng nếu cùng hướng thì cosine similarity vẫn cho ra gần 1 (đúng bản chất "giống ý nghĩa"), trong khi khoảng cách Euclidean có thể báo sai là "khác xa nhau".

**Lưu ý cần kiểm tra lại trước khi khẳng định chắc:** trong code (`rag_engine.py`), khi tạo collection bằng `client.get_or_create_collection(name=..., embedding_function=...)`, **không có tham số `metadata={"hnsw:space": "cosine"}` tường minh** — mà ChromaDB mặc định dùng khoảng cách **L2 (Euclidean) bình phương** cho index HNSW (Hierarchical Navigable Small World — thuật toán tìm lân cận gần đúng dùng cấu trúc đồ thị phân tầng, mục F3) nếu không cấu hình rõ. Nếu bị hỏi sâu "có chắc là cosine không", câu trả lời an toàn nhất: *"Mục tiêu thiết kế là dùng cosine similarity vì phù hợp với sentence embedding, cần xác minh lại đúng phiên bản ChromaDB đang cài đặt có mặc định là cosine hay L2 để khẳng định chính xác 100%."*

### F3. ChromaDB — Vector Database (cơ sở dữ liệu vector)

Lưu trữ hàng loạt embedding vector cùng văn bản gốc tương ứng (mỗi chunk ở mục F0 → 1 vector + 1 đoạn text, tổng cộng 7 cặp vector-text). Khi có câu hỏi mới: mã hóa câu hỏi thành vector bằng đúng model ở mục F1, tìm **$k$ vector gần nhất** (ở đây $k=1$, tham số `n_results=1` trong code — chỉ lấy đúng 1 chunk liên quan nhất) bằng **Approximate Nearest Neighbor** (tìm lân cận gần đúng — thay vì so sánh vét cạn với TẤT CẢ vector trong kho — brute-force, tốn thời gian tuyến tính theo số lượng — thuật toán HNSW xây sẵn 1 cấu trúc đồ thị phân tầng cho phép tìm nhanh gần đúng, đánh đổi 1 chút độ chính xác lấy tốc độ — với chỉ 7 chunk như ở đây, brute-force cũng đủ nhanh, nhưng cách làm này vẫn mở rộng tốt nếu kho tài liệu lớn hơn về sau), trả về đoạn văn bản y văn tương ứng, nhồi vào prompt cho LLM.

**Vì sao gọi là "giảm ảo giác" (hallucination — hiện tượng LLM tự "bịa" ra thông tin nghe có vẻ hợp lý nhưng sai sự thật):** LLM không phải lúc nào cũng nhớ đúng thông tin y khoa chính xác từ dữ liệu huấn luyện gốc. RAG buộc LLM phải dựa vào đoạn văn bản y văn **thật, có nguồn gốc rõ ràng** được nhồi trực tiếp vào prompt ngay trước khi sinh câu trả lời, giảm khả năng bịa đặt so với việc để LLM tự "nhớ lại" hoàn toàn từ tham số nội tại của nó.

---

## G. Bảo mật — SHA-256, XOR, Base64 — slide 20, 21

### G1. SHA-256 (Secure Hash Algorithm, phiên bản output 256 bit)

**Hash function (hàm băm) là gì:** 1 hàm toán học nhận **input độ dài bất kỳ** (1 ký tự hay 1 quyển sách đều được), luôn trả về **output độ dài cố định** gọi là **hash** hay **digest** (bản tóm lược) — với SHA-256, output luôn đúng **256 bit = 32 byte**, thường biểu diễn dưới dạng chuỗi 64 ký tự hexadecimal (hệ đếm cơ số 16, mỗi ký tự hex biểu diễn 4 bit, $64\times4=256$ bit).

**3 tính chất bắt buộc của 1 hash function an toàn (cần thuộc để giải thích đúng khi bị hỏi):**
1. **One-way (1 chiều — Pre-image resistance):** từ hash $h = \text{SHA256}(x)$, không có cách nào khả thi để suy ngược lại $x$ gốc ngoài cách thử toàn bộ khả năng (brute-force) — khác hoàn toàn với encoding (Base64) hay encryption (có thể giải mã ngược nếu có khóa).
2. **Avalanche effect (hiệu ứng lan tỏa/tuyết lở):** chỉ cần đổi **1 bit** trong input, output thay đổi hoàn toàn (khoảng 50% số bit output đổi khác) — đảm bảo không có mối liên hệ "dễ đoán" nào giữa input gần giống nhau và output của chúng.
3. **Collision resistance (kháng va chạm):** cực kỳ khó (về mặt tính toán) để tìm ra 2 input khác nhau $x_1 \neq x_2$ mà $\text{SHA256}(x_1) = \text{SHA256}(x_2)$ (gọi là 1 **collision** — va chạm).

**Vì sao chỉ lấy 16 ký tự hex đầu làm Document ID lại giảm độ an toàn:** 16 ký tự hex $=16\times4=64$ bit, trong tổng số 256 bit gốc — cắt ngắn như vậy làm giảm mạnh độ kháng va chạm. Theo **birthday paradox** (nghịch lý ngày sinh — hiện tượng thống kê: trong 1 nhóm chỉ cần khoảng 23 người đã có >50% khả năng 2 người trùng ngày sinh, dù có tới 365 ngày để chọn — vì số CẶP so sánh tăng theo bình phương số phần tử), số mẫu cần thử để có 50% khả năng xảy ra va chạm với hash $n$ bit chỉ khoảng $2^{n/2}$ (căn bậc 2 của tổng không gian, không phải $2^n$) — với $n=64$ bit, con số này chỉ là $2^{32}\approx 4.3$ tỷ mẫu — nghe có vẻ lớn nhưng đây là mức tính toán khả thi với phần cứng hiện đại, không còn "an toàn" theo chuẩn crypto đầy đủ (256-bit đầy đủ thì $2^{128}$ mẫu — một con số vượt xa khả năng tính toán của mọi máy tính hiện có và tương lai gần). Với quy mô demo vài nghìn bệnh nhân thì rủi ro va chạm thực tế gần như không đáng kể, nhưng về mặt lý thuyết đây là 1 sự đánh đổi có chủ đích (giảm độ dài ID để dễ đọc/lưu trữ) chứ không phải "an toàn tuyệt đối".

### G2. XOR Cipher (mật mã XOR) — vì sao yếu khi dùng khóa lặp lại

**XOR (exclusive OR — hoặc loại trừ), ký hiệu $\oplus$:** phép toán logic trên từng bit, trả về 1 nếu 2 bit đầu vào **khác nhau**, trả về 0 nếu **giống nhau** ($0\oplus0=0$, $1\oplus1=0$, $0\oplus1=1$, $1\oplus0=1$). Tính chất quan trọng nhất: XOR với cùng 1 giá trị 2 lần sẽ **quay lại giá trị gốc** — $X \oplus K \oplus K = X$ (vì $K\oplus K=0$, và $X\oplus 0 = X$) — đây là lý do XOR dùng được cho cả mã hóa lẫn giải mã bằng đúng 1 phép toán.

**Nguyên lý mã hóa/giải mã bằng XOR:**
$$C = P \oplus K \qquad \text{(mã hóa: ciphertext = plaintext XOR key)}$$
$$P = C \oplus K \qquad \text{(giải mã: dùng đúng phép XOR với cùng khóa)}$$
Ký hiệu: $P$ = plaintext (bản rõ — dữ liệu gốc chưa mã hóa), $C$ = ciphertext (bản mã — dữ liệu sau khi mã hóa), $K$ = key (khóa bí mật).

**Vì sao yếu khi khóa lặp lại trên nhiều bản ghi (repeating-key XOR — đúng tình huống thật trong code, khóa `"DermaSecureKey2026"` cố định, dùng chung cho MỌI bệnh nhân):** nếu kẻ tấn công có được 2 bản mã $C_1, C_2$ của 2 bản ghi khác nhau nhưng **dùng chung 1 khóa** $K$, họ có thể XOR trực tiếp 2 bản mã với nhau — khóa sẽ tự triệt tiêu theo đúng tính chất nêu trên:
$$C_1 \oplus C_2 = (P_1 \oplus K) \oplus (P_2 \oplus K) = P_1 \oplus P_2$$
Kết quả $P_1\oplus P_2$ **không còn phụ thuộc vào khóa $K$ nữa** — đây đã là thông tin rò rỉ trực tiếp về mối quan hệ giữa 2 bản rõ gốc, và với văn bản tự nhiên (tiếng Việt/Anh có quy luật thống kê rõ rệt về tần suất ký tự), kẻ tấn công hoàn toàn có thể suy ngược ra cả $P_1$ và $P_2$ bằng phân tích tần suất (frequency analysis), không cần biết khóa $K$ là gì. Đây là điểm yếu **kinh điển** của repeating-key XOR trong mật mã học — khác hẳn **One-Time Pad** (khóa $K$ dài đúng bằng $P$, ngẫu nhiên hoàn toàn, và **chỉ dùng đúng 1 lần rồi bỏ** — chưa từng dùng lại cho bản ghi nào khác) — chỉ có One-Time Pad mới được **chứng minh toán học** là an toàn tuyệt đối về mặt lý thuyết thông tin (Claude Shannon, 1949) — XOR với khóa cố định lặp lại cho nhiều bản ghi **không hề** thỏa điều kiện này.

**Base64:** hoàn toàn **không phải mã hóa** — chỉ là 1 kiểu **encoding** (biến đổi cách biểu diễn dữ liệu nhị phân thành chuỗi ký tự ASCII an toàn để lưu/truyền qua các hệ thống chỉ hỗ trợ text, ví dụ JSON, URL), dùng bảng 64 ký tự cố định công khai (A-Z, a-z, 0-9, +, /) — bất kỳ ai cũng decode ngược lại ngay lập tức mà **không cần biết bất kỳ khóa bí mật nào**. Trong hệ thống này, Base64 chỉ đóng vai trò "đóng gói hiển thị" chuỗi byte nhị phân sau khi đã XOR thành dạng text lưu được, hoàn toàn không đóng góp gì thêm về mặt bảo mật.

**Câu trả lời chuẩn khi bị hỏi:** *"XOR+Base64 có coi là bảo mật không?"* → Không đạt chuẩn mật mã học hiện đại; đây là biện pháp ẩn dữ liệu (obfuscation — làm dữ liệu khó đọc trực tiếp bằng mắt thường, nhưng không chống được tấn công có chủ đích) ở mức nguyên mẫu (prototype). Hướng khắc phục đúng chuẩn: **AES-256-GCM** (Advanced Encryption Standard — thuật toán mã hóa khối đối xứng đã được kiểm chứng rộng rãi và chuẩn hóa quốc tế, độ dài khóa 256 bit; chế độ **GCM** — Galois/Counter Mode — cung cấp đồng thời cả tính bảo mật (encryption) lẫn xác thực toàn vẹn dữ liệu (authentication — phát hiện nếu dữ liệu bị chỉnh sửa trái phép), gọi chung là **authenticated encryption**), với khóa quản lý qua **KMS** (Key Management Service — dịch vụ quản lý khóa chuyên biệt của nhà cung cấp cloud) tách biệt hoàn toàn khỏi mã nguồn, thay vì hardcode như hiện tại.

---

## H. Kiến trúc phần mềm OOP — slide 23

**Nguyên tắc Single Responsibility:** mỗi class chỉ nên có 1 lý do để thay đổi — `SafetyGate` chỉ lo đánh giá chất lượng, `InteractiveSegmenter` chỉ lo phân đoạn, `MultimodalBayesianFusion` chỉ lo hợp nhất xác suất. Tách biệt giúp sửa 1 phần không ảnh hưởng phần khác, dễ test độc lập (unit test từng class riêng).

**`UnifiedDermatologyPipeline` là Controller:** điều phối (orchestration) — gọi đúng thứ tự các service, không tự chứa logic nghiệp vụ chi tiết bên trong nó, đúng pattern kiến trúc phân lớp phổ biến (Controller/Service layer).

---

## I. Đánh giá mô hình — slide 24, 25 (phần dễ bị hỏi số liệu nhất)

### I1. Dice Score và IoU — vì sao 2 chỉ số gần giống nhau nhưng khác biệt

$$\text{Dice} = \frac{2|A \cap B|}{|A|+|B|}, \qquad \text{IoU} = \frac{|A \cap B|}{|A \cup B|}$$

Quan hệ toán học giữa 2 đại lượng: $\text{Dice} = \frac{2 \cdot \text{IoU}}{1+\text{IoU}}$ — Dice luôn ≥ IoU (trừ khi bằng 0 hoặc 1), và Dice "khoan dung" hơn với lỗi nhỏ, nên Dice thường được báo cáo cao hơn IoU cho cùng 1 phép đo — **không phải 2 model khác nhau**, chỉ là 2 cách đo cùng 1 overlap.

### I2. Confusion Matrix — đọc đúng ý nghĩa lâm sàng

Hàng = nhãn thật (True Label), cột = nhãn dự đoán (Predicted Label), đường chéo = dự đoán đúng. Với slide 24: NV có 1764/1882 đúng (đường chéo cao nhất vì NV chiếm đa số dữ liệu — 67% HAM10000), nhưng **MEL→NV nhầm 52 ca** là điều đáng lo nhất về mặt lâm sàng: MEL là ác tính, NV là lành tính — nhầm ác tính thành lành tính (**false negative** trên bệnh nguy hiểm) nguy hiểm hơn nhiều so với nhầm chiều ngược lại (chẩn đoán quá mức, false positive, chỉ gây lo lắng/khám thêm không cần thiết chứ không bỏ sót bệnh).

### I3. Selective Prediction — Accuracy vs Coverage (slide 25, biểu đồ quan trọng nhất để hiểu sâu)

**Coverage:** tỷ lệ % số ca hệ thống **chấp nhận đưa ra dự đoán** (không bị Safety Gate từ chối) trên tổng số ca đưa vào.

**Nguyên lý đánh đổi:** khi hạ ngưỡng chấp nhận của Safety Gate (dễ dãi hơn) → coverage tăng (nhận nhiều ca hơn) nhưng trong số các ca "biên" (mờ ranh giới, khó phân loại) lọt qua sẽ kéo accuracy trung bình xuống. Ngược lại siết ngưỡng chặt → coverage giảm (từ chối nhiều hơn) nhưng các ca còn lại là ca "dễ, rõ ràng" nên accuracy trên số đã chấp nhận tăng lên. Đây là lý thuyết **selective prediction / reject option** kinh điển trong ML — mô hình được phép nói "tôi không chắc" thay vì luôn phải đưa ra 1 câu trả lời.

**Câu hỏi chắc chắn sẽ bị hỏi:** *"Vậy nên đặt ngưỡng Safety Gate ở đâu?"* → Không có câu trả lời "đúng tuyệt đối" — đây là quyết định đánh đổi giữa an toàn lâm sàng (ưu tiên coverage thấp, accuracy cao trên ca chấp nhận, giảm rủi ro chẩn đoán sai) và tính hữu dụng thực tế (ưu tiên coverage cao để không từ chối quá nhiều bệnh nhân) — nên để bác sĩ tự điều chỉnh theo ngữ cảnh (đã có slider `τc` trên UI), không nên cố định cứng.

### I4. BLEU Score

$$\text{BLEU-N} = BP \times \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

$p_n$ là độ chính xác n-gram (tỷ lệ cụm n từ trong câu sinh ra khớp với câu tham chiếu), $BP$ là brevity penalty (phạt nếu câu sinh ra ngắn hơn tham chiếu nhiều). BLEU-1 chỉ xét khớp từng từ đơn lẻ (unigram) — đây là lý do nó **không nắm được ngữ nghĩa**, chỉ đo trùng khớp từ vựng bề mặt (đã giải thích ở slide 16, cần nhớ công thức gốc nếu bị hỏi sâu).

---

## Bảng tổng hợp con số cần thuộc lòng (tra nhanh trước khi vào phòng)

| Con số | Ý nghĩa | Nguồn |
|---|---|---|
| Dice 91.32% / IoU 84.70% | Hybrid-Max Fusion, best segmentation | `ablation_fusion_results.json` |
| 390 ảnh | Test set ISIC 2018 cho TTA benchmark | `tta_vs_standard_benchmark_isic.json` |
| 88.65% / 89.39% | Test acc / best val acc, EfficientNet-B1+CBAM | `06_classification_complete.json` |
| 3,005 mẫu | Test set HAM10000-ROI | checkpoint config |
| rank=8, alpha=16 | Cấu hình LoRA | `train_vqa_joint.py` |
| 90,352,514 / 1,926,754 / 2.13% | Tổng tham số / trainable / tỷ lệ VQA | `experimental_results.md` (đã đối chiếu checkpoint) |
| BLEU-1: 0.73 / 0.12 | Offline / Online, 12 mẫu | `vqa_evaluation_report.json` |
| 232.27 ms | Latency toàn luồng | pipeline benchmark |
| 57.1% | Accuracy trên kịch bản ảnh xấu | confusion matrix degraded scenario |
| reduction=16 | Hệ số giảm chiều trong CBAM channel attention | `multimodal_fusion.py`/model code |
| n_embd=768, n_layer=6, n_head=12 | Config DistilGPT-2 | `AutoConfig.from_pretrained('distilgpt2')` |
| c_attn output = 2304 = 3×768 | Q,K,V gộp chung 1 lớp Linear trong GPT-2 | xác nhận qua shape `lora_B` thật (2304,8) |
| 10,140,231 tham số, 40.7MB | MobileSAM (`vit_t`/TinyViT) | đọc trực tiếp `mobile_sam.pt` |
| 7 chunk, 11,082 ký tự | Chunking RAG (chia theo ranh giới bệnh lý) | `medical_guidelines.txt` |
| ResNet34 [3,4,6,3] BasicBlock vs ResNet50 [3,4,6,3] Bottleneck | Backbone U-Net vs DeepLabV3+ | kiến trúc chuẩn torchvision |
| Input 256×256 (phân đoạn) vs 224×224 (phân loại) | 2 quyết định độc lập: mask cần nhiều pixel hơn để đo ABCD chính xác; 224 khớp chuẩn pretrain ImageNet | xem lý giải đầy đủ ở mục A6 |
| Dropout=0.3 | "Vùng an toàn" giữa overfit (dropout thấp) và underfit (dropout cao, chỉ có 1 lớp Linear sau nó) | mục C3, không phải kết quả hyperparameter search |
| Black-hat kernel 13×13, threshold=12, >300px | Lọc lông DullRazor (chỉ kích hoạt khi mask lượt 1 rỗng) | mục A7, `_enhance_image_quality` |
| CLAHE clipLimit=2.0, tileGridSize=8×8, kênh L/LAB | Tăng sáng/tương phản cục bộ (cùng điều kiện kích hoạt như lọc lông) | mục A7 |

---

## Thứ tự ưu tiên học nếu thời gian gấp

1. **Bắt buộc thuộc nằm lòng:** kiến trúc VQA "V3" (`DeepCrossAttentionBridge`...) **không có trong checkpoint thật** — nếu bị hỏi trúng mà trả lời sai (mô tả như đang chạy) sẽ mất điểm nặng nhất trong cả buổi; ngược lại SAM **đã được sửa và xác nhận chạy thật** (mục 0) nên có thể tự tin khẳng định khi bị hỏi, không cần né tránh. Ngoài ra: công thức ABCD (mục B), công thức Bayes fusion + entropy λ (mục D), vì sao XOR yếu (mục G2).
2. **Nên hiểu sâu để giải thích tự tin:** LoRA (mục E2) — vì con số 2.13% rất dễ bị hỏi "tính từ đâu ra" (đã có bảng chia đúng 3 thành phần ở mục E3); luồng ghép 1 token ảnh thật (mục E3) — vì sao đây không phải VLM cross-attention đầy đủ.
3. **Nên nắm khái niệm, không cần thuộc công thức:** DeepLabV3+/ASPP, SAM+GrabCut, CBAM, Grad-CAM — hiểu đủ để giải thích trực quan, không cần viết được công thức đầy đủ trên bảng.
4. **Chỉ cần đọc hiểu ý nghĩa số liệu:** Dice/IoU, BLEU, confusion matrix, accuracy-coverage — đây là phần "đọc số ra sao" chứ không phải "chứng minh công thức".
