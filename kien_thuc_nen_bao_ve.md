# Kiến thức nền cần nắm vững để bảo vệ — theo đúng 28 slide

Nguyên tắc trình bày: trực quan trước, nguyên lý sau, công thức/chi tiết kỹ thuật cuối cùng. Mọi con số dưới đây khớp đúng với code thật đã kiểm chứng trong repo (đọc trực tiếp checkpoint `.pth` bằng `torch.load`, không suy đoán), không phải số lý thuyết chung chung.

---

## 0. Bản đồ toàn bộ model trong hệ thống — kích cỡ, input/output

| Model | Kiến trúc | Tham số | File / kích cỡ đĩa | Input | Output |
|---|---|---:|---|---|---|
| Phân đoạn U-Net | ResNet34 encoder + U-Net decoder (`segmentation_models_pytorch`) | ~24.4M (ước tính theo ResNet34) | `unet_best.pth`, 293.5 MB | Ảnh RGB 256×256×3 | Mask 1×256×256 (logit, cần sigmoid) |
| Phân đoạn DeepLabV3+ | ResNet50 encoder + ASPP + decoder (`smp.DeepLabV3Plus`) | ~26.7M (ước tính theo ResNet50) | `deeplabv3plus_best.pth`, 320.6 MB | Ảnh RGB 256×256×3 | Mask 1×256×256 (logit, cần sigmoid) |
| Hybrid-Max Fusion | Không phải model riêng — hợp nhất (max/weighted) 2 xác suất mask từ U-Net và DeepLabV3+ | 0 tham số riêng | `hybrid_best.pth` chỉ 1.5 KB (chỉ lưu config alpha, không lưu trọng số) | 2 mask xác suất | 1 mask hợp nhất |
| Phân đoạn tương tác | SAM/MobileSAM (ViT) **hoặc** GrabCut (`cv2.grabCut`) cổ điển | SAM: hàng chục-hàng trăm triệu (không dùng thực tế — xem cảnh báo bên dưới) | Không có checkpoint SAM nào trong repo | Ảnh RGB + 1 điểm click | Mask nhị phân |
| Phân loại bệnh lý | EfficientNet-B1 (timm) + CBAM (reduction=16) + Dropout(0.3) + Linear(1280→7) | ~7.8M (backbone) + đầu phân loại nhỏ | `efficientnet_attention_best.pth`, 81.4 MB | Ảnh RGB 224×224×3, chuẩn hóa ImageNet | Vector logit 7 lớp → softmax |
| Vision backbone (VQA) | EfficientNet-B1 + CBAM — **instance huấn luyện riêng biệt**, không dùng chung trọng số với model phân loại | 6,780,199 (đọc trực tiếp từ checkpoint) | Nằm trong `dermavqa_gpt2_joint_best.pth` | Ảnh RGB 224×224×3 | Vector 1280-dim (global average pooled) |
| VQA Decoder | DistilGPT-2 (6 lớp transformer, causal) + LoRA (r=8) trên `c_attn` của cả 6 lớp | 120,657,408 (gồm cả `lm_head` không tie-weight) | Nằm trong `dermavqa_gpt2_joint_best.pth`, tổng file ~ hàng trăm MB | Chuỗi token văn bản + 1 token ảnh đã chiếu | Chuỗi token văn bản (câu trả lời) |
| Projection layer (VQA) | Linear(1280→768) → GELU → Dropout(0.3) → Linear(768→768) | 1,574,400 | Nằm trong `dermavqa_gpt2_joint_best.pth` | Vector 1280-dim từ vision backbone | Vector 768-dim (khớp chiều ẩn GPT-2) |
| Sentence Embedding (RAG) | all-MiniLM-L6-v2 (Sentence-Transformers, 6-layer Transformer encoder chưng cất) | ~22.7M (theo công bố gốc của model) | Tải tự động qua thư viện, không lưu trong repo | Chuỗi văn bản (câu hỏi) | Vector 384-dim |
| VQA Online | GPT-4o-mini (OpenAI, gọi qua API) | Không công bố (proprietary) | Không lưu local | Text + ảnh (base64) qua API | Text |
| VQA Ollama (tùy chọn) | Qwen2.5:3b (mặc định trong code, dù slide ghi "3B/7B") | ~3 tỷ | Tải qua Ollama, không nằm trong repo | Text | Text |

**Tổng tham số toàn bộ checkpoint VQA đang chạy thật** (đọc trực tiếp từ `dermavqa_gpt2_joint_best.pth`): 120,657,408 (llm) + 6,780,199 (vision_backbone) + 1,574,400 (projection) = **129,012,007** tham số toàn phần. Con số "90,352,514" trên slide/báo cáo nhiều khả năng là tổng **sau khi trừ `lm_head`** (38,597,376 tham số) — vì trong kiến trúc GPT-2 chuẩn, `lm_head` thường **chia sẻ trọng số (tied weights)** với embedding đầu vào nên không được đếm 2 lần khi báo cáo "kích thước model": $129{,}012{,}007 - 38{,}597{,}376 = 90{,}414{,}631$ — khớp gần đúng với con số đã công bố (chênh lệch nhỏ có thể do phiên bản checkpoint khác nhau đôi chút).

### ⚠️ 2 phát hiện quan trọng cần biết trước khi bị hỏi

**1. Tính năng "SAM" trên slide hiện KHÔNG chạy trong pipeline thật.** Đọc trực tiếp `pipeline/interactive_sam.py` và `pipeline/unified_pipeline.py`: `InteractiveSegmenter()` được khởi tạo **không truyền `checkpoint_path`** (`segmenter = InteractiveSegmenter()`), mà điều kiện kích hoạt SAM là `checkpoint_path is not None and os.path.exists(checkpoint_path)` — luôn `False` trong pipeline thật. Ngoài ra không có file checkpoint SAM/MobileSAM nào tồn tại trong repo. **Kết quả: mọi lần "phân đoạn tương tác SAM" trong demo thực chất luôn chạy qua nhánh dự phòng `cv2.grabCut()` (GrabCut cổ điển, khởi tạo bằng box + circle quanh điểm click), không phải deep learning SAM.** Nếu hội đồng hỏi "cho xem SAM checkpoint đang dùng loại nào (ViT-B/L/H)?" — câu trả lời trung thực là: code đã viết sẵn tích hợp SAM/MobileSAM nhưng hiện chưa nạp checkpoint thật, pipeline đang chạy hoàn toàn bằng GrabCut. Nên hoặc (a) sửa lại text slide 8/9/12/13/15 để gọi đúng là "GrabCut tương tác (khởi tạo bằng box + điểm click)", hoặc (b) nếu còn thời gian, tải checkpoint MobileSAM thật (nhẹ, ~40MB, phù hợp CPU) và truyền đúng `checkpoint_path` để tính năng SAM thật sự hoạt động như mô tả.

**2. Kiến trúc VQA "V3" phức tạp trong `train_vqa_joint.py` KHÔNG có trong checkpoint đang chạy.** File huấn luyện định nghĩa một kiến trúc rất phức tạp gồm `DeepCrossAttentionBridge` (2 lớp cross-attention), `ClinicalStructureInjector` (tiêm chỉ số ABCD + xác suất lớp bệnh thành 1 token), `ClinicalPrefix` (8 token prefix học được), `SemanticEnhancer` — nhưng khi tôi mở trực tiếp checkpoint thật `dermavqa_gpt2_joint_best.pth` (đúng file mà `app_streamlit.py` đang nạp), **không có bất kỳ tensor nào** thuộc các module này (`cross_attention_bridge`, `clinical_prefix`, `clinical_injector` đều không tồn tại trong state_dict). Model đang chạy thật chỉ gồm: vision backbone → projection (1280→768→768) → **1 token ảnh duy nhất** (global average pooling, không dùng 49 spatial token) → ghép với text embedding → DistilGPT-2+LoRA. Nếu bị hỏi về DeepCrossAttentionBridge hay cơ chế multi-token — đây là **hướng phát triển đã viết code sẵn sàng nhưng chưa huấn luyện thành checkpoint chính thức**, không phải kiến trúc đang thực sự chạy trong app.

---

## A. Phân đoạn ảnh (Segmentation) — slide 8, 9, 12, 13

### A1. DeepLabV3+ và Atrous Convolution — vì sao chọn nó, không phải U-Net thường

**Trực quan:** một mạng CNN thường thu nhỏ ảnh dần qua các lớp pooling để "nhìn" được vùng rộng, nhưng cái giá phải trả là mất độ phân giải chi tiết (không biết chính xác biên tổn thương nằm ở đâu). DeepLabV3+ giải quyết bằng **Atrous Convolution (Dilated Convolution)** — convolution "có lỗ hổng", giãn cách các điểm lấy mẫu của kernel ra xa nhau thay vì co cụm, giúp mở rộng receptive field (vùng nhìn thấy) **mà không cần giảm độ phân giải ảnh**.

**Công thức:** với dilation rate $r$, kernel 3×3 thường sẽ "nhìn" một vùng $3+2(r-1)$ thay vì chỉ 3 pixel liên tiếp. Output size với input $n$, kernel $k$, dilation $r$, stride $s$, padding $p$:
$$\text{out} = \left\lfloor \frac{n + 2p - r(k-1) - 1}{s} + 1 \right\rfloor$$

**ASPP (Atrous Spatial Pyramid Pooling):** DeepLabV3+ chạy song song nhiều nhánh atrous convolution với dilation rate khác nhau (thường 6, 12, 18) trên cùng feature map, rồi ghép lại — giúp bắt được tổn thương ở nhiều kích thước khác nhau cùng lúc (tổn thương nhỏ li ti lẫn mảng lớn).

**Backbone ResNet50:** trích đặc trưng qua 50 lớp với residual connection $y = F(x) + x$ — giải quyết vấn đề vanishing gradient khi mạng sâu, cho phép huấn luyện ổn định.

**Câu hỏi khả năng bị hỏi:** *"Vì sao không dùng U-Net?"* → U-Net (encoder-decoder đối xứng, skip connection) đơn giản hơn, ít tham số hơn, nhưng receptive field hẹp hơn ở cùng độ sâu; DeepLabV3+ với ASPP nắm được ngữ cảnh đa tỷ lệ tốt hơn cho tổn thương có kích thước rất đa dạng. Số liệu bạn có (slide 12): DeepLabV3+ Dice 91.28% > U-Net 89.43% — đúng như kỳ vọng lý thuyết.

### A2. TTA (Test-Time Augmentation) đa tỷ lệ

**Trực quan:** thay vì đưa 1 ảnh duy nhất vào model, đưa ảnh đó vào ở **nhiều kích thước khác nhau** (1.0×, 0.75×, 0.5× — đúng scale bạn dùng trong code), lấy trung bình/hợp nhất kết quả — giống việc hỏi 3 "phiên bản zoom khác nhau" của cùng 1 model rồi lấy đồng thuận, giảm sai số ngẫu nhiên do 1 tỷ lệ cụ thể vô tình không khớp với model.

**Vì sao chỉ dùng cho ảnh phone:** ảnh dermoscopy đã chuẩn hóa tỷ lệ chụp (máy soi da cố định khoảng cách), còn ảnh phone chụp ở khoảng cách/góc tùy ý — biến thiên tỷ lệ vật lý lớn hơn nhiều, nên hưởng lợi từ đa tỷ lệ nhiều hơn.

### A3. SAM (Segment Anything Model) + GrabCut — vì sao kết hợp 2 thuật toán

**SAM:** một foundation model phân đoạn, nhận **prompt** (điểm click, box, hoặc mask) làm gợi ý, tự sinh ra mask ứng với điểm đó — kiến trúc gồm image encoder (ViT) mã hóa toàn ảnh 1 lần, rồi prompt encoder + mask decoder nhẹ chạy nhanh cho từng điểm click.

**GrabCut:** thuật toán cổ điển dựa trên **Gaussian Mixture Model (GMM)** để mô hình hóa phân phối màu foreground/background, kết hợp **Graph Cut (min-cut/max-flow)** để tìm đường phân chia tối ưu giữa 2 vùng, tối thiểu hóa năng lượng:
$$E(\alpha, k, \theta, z) = U(\alpha, k, \theta, z) + V(\alpha, z)$$
trong đó $U$ là data term (màu pixel khớp với GMM foreground/background đến đâu), $V$ là smoothness term (phạt biên gồ ghề không tự nhiên giữa 2 pixel liền kề màu khác biệt).

**Vì sao dùng `GC_INIT_WITH_MASK`:** thay vì GrabCut tự đoán vùng foreground ban đầu (dễ sai), dùng chính mask SAM sinh ra làm khởi tạo — GrabCut chỉ cần **tinh chỉnh biên** cho mượt và chính xác hơn, không phải đoán từ đầu. Đây là lý do 2 thuật toán bổ trợ nhau: SAM giỏi định vị vùng lớn theo ngữ nghĩa, GrabCut giỏi tinh chỉnh biên theo màu sắc pixel-level.

### A4. OTSU Thresholding — thuật toán dự phòng

**Nguyên lý:** OTSU tự động tìm 1 ngưỡng độ sáng $t$ chia ảnh xám thành 2 lớp (nền/vật thể), sao cho **phương sai giữa 2 lớp là lớn nhất** (tương đương phương sai trong từng lớp nhỏ nhất):
$$\sigma_b^2(t) = \omega_0(t)\omega_1(t)[\mu_0(t) - \mu_1(t)]^2$$
với $\omega_0, \omega_1$ là tỷ lệ pixel mỗi lớp, $\mu_0, \mu_1$ là độ sáng trung bình mỗi lớp. Duyệt qua mọi $t$ có thể, chọn $t^*$ maximize $\sigma_b^2$.

**`THRESH_BINARY_INV`:** đảo ngược nhị phân hóa — vì tổn thương da thường **tối hơn** vùng da xung quanh, nên cần đảo ngược để vùng tối (tổn thương) thành pixel "1" thay vì mặc định OTSU coi vùng sáng là foreground.

**Câu hỏi bẫy:** *"OTSU có phải deep learning không?"* → Không, đây là thuật toán xử lý ảnh cổ điển thuần túy dựa trên thống kê histogram, không có tham số học được — chính vì thế nó **không thể thất bại theo kiểu "model lỗi"**, luôn cho ra 1 kết quả xác định, phù hợp làm phương án dự phòng cuối cùng khi deep model thất bại hoàn toàn.

### A5. Input/Output chính xác của 2 model phân đoạn

- **Kích thước input:** cả U-Net và DeepLabV3+ đều resize ảnh về **256×256×3** (`seg_input_size=256` trong `ModelConfig`) trước khi đưa vào model — khác với model phân loại (224×224).
- **Chuẩn hóa (normalization):** mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25) — **không phải chuẩn ImageNet** như model phân loại. Đây là lựa chọn đơn giản hóa: đưa pixel về khoảng gần $[-2, 2]$ đều nhau trên cả 3 kênh, không dùng thống kê riêng theo kênh như ImageNet (vốn được tính từ hàng triệu ảnh tự nhiên, ảnh da liễu có phân phối màu khác biệt nên không nhất thiết phải theo đúng chuẩn đó).
- **Output:** 1 kênh duy nhất (`classes=1`), không áp activation trong model (`activation=None`) — logits thô, phải tự áp `torch.sigmoid()` bên ngoài để ra xác suất, rồi nhị phân hóa bằng ngưỡng `seg_threshold=0.3`.
- **Vì sao ngưỡng nhị phân hóa là 0.3, không phải 0.5 mặc định:** ngưỡng thấp hơn 0.5 làm model "dễ dãi" hơn khi quyết định 1 pixel có thuộc tổn thương hay không — ưu tiên **không bỏ sót** vùng tổn thương (giảm false negative ở cấp độ pixel) hơn là ưu tiên độ chính xác tuyệt đối của biên, chấp nhận đánh đổi có thể hơi "phình" biên ra một chút.
- **U-Net (ResNet34, encoder ~24.4M tham số theo kiến trúc ResNet34 chuẩn) và DeepLabV3+ (ResNet50, encoder ~26.7M tham số):** đều dùng backbone **pretrained trên ImageNet** rồi fine-tune lại cho bài toán phân đoạn nhị phân — tận dụng đặc trưng thị giác tổng quát (cạnh, texture, hình khối) đã học từ hàng triệu ảnh tự nhiên, giúp hội tụ nhanh hơn nhiều so với huấn luyện từ đầu (from scratch) trên tập dữ liệu da liễu tương đối nhỏ.
- **Hybrid-Max Fusion không phải 1 model riêng:** đây là bước hậu xử lý (post-processing) đơn giản — chạy CẢ 2 model (U-Net và DeepLabV3+) trên cùng 1 ảnh, lấy **giá trị lớn nhất (max)** tại mỗi pixel giữa 2 bản đồ xác suất, rồi mới nhị phân hóa. Vì vậy `hybrid_best.pth` chỉ nặng 1.5KB (chỉ lưu config, không lưu trọng số riêng) — chi phí suy luận gấp đôi (phải chạy cả 2 model) nhưng không cần huấn luyện thêm bất kỳ tham số mới nào.

---

## B. Chỉ số hình học ABCD — slide 14

Đây là 4 công thức thật trong code (`_get_lesion_metrics`), cần thuộc chính xác:

**A — Asymmetry:** chia mask theo 2 trục đi qua trọng tâm (centroid, tính bằng moments ảnh $c_x = M_{10}/M_{00}$, $c_y = M_{01}/M_{00}$), lật nửa dưới/phải rồi so khớp pixel-by-pixel với nửa trên/trái, đếm số pixel lệch nhau:
$$\text{Asymmetry} = \text{clip}\left(\frac{\text{asym}_h + \text{asym}_v}{2 \times \text{lesion\_area}}, 0, 1\right)$$

**B — Border complexity:** tỷ lệ chu vi trên căn bậc hai diện tích — hình tròn hoàn hảo có tỷ lệ này thấp nhất, hình càng răng cưa/lồi lõm thì tỷ lệ càng cao:
$$\text{Border} = \frac{\text{Perimeter}}{\sqrt{\text{Area}}}$$

**C — Color variation:** độ lệch chuẩn của giá trị RGB trong vùng tổn thương, trung bình 3 kênh, chuẩn hóa về [0,1] bằng cách chia cho 127.5 (nửa dải giá trị 8-bit):
$$\text{Color} = \text{clip}\left(\frac{\text{mean}(\text{std}_R, \text{std}_G, \text{std}_B)}{127.5}, 0, 1\right)$$

**D — Diameter (equivalent diameter):** đường kính của 1 hình tròn có cùng diện tích với vùng tổn thương:
$$D = 2\sqrt{\frac{\text{Area}}{\pi}}$$

**Vì sao dùng "equivalent diameter" chứ không đo trực tiếp bounding box:** tổn thương thường không tròn, bounding box sẽ đo theo trục dọc/ngang gây sai lệch tùy hướng chụp; equivalent diameter bất biến với hướng xoay của tổn thương.

**Về DICOM PixelSpacing:** file DICOM có metadata `PixelSpacing` (mm/pixel thật theo trục X, Y của thiết bị chụp) — nhân số pixel với giá trị này ra kích thước thực (mm). Ảnh JPG/PNG thường không có metadata này nên D chỉ dừng ở đơn vị pixel.

---

## C. Phân loại ảnh — EfficientNet-B1 + CBAM — slide 15

### C1. EfficientNet — compound scaling

**Ý tưởng cốt lõi:** thay vì tăng độ sâu (depth), độ rộng (width), hay độ phân giải ảnh (resolution) riêng lẻ như các kiến trúc cũ, EfficientNet tăng **cả 3 đồng thời theo 1 hệ số phối hợp** $\phi$:
$$\text{depth} = \alpha^\phi, \quad \text{width} = \beta^\phi, \quad \text{resolution} = \gamma^\phi$$
với ràng buộc $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ để giữ chi phí tính toán tăng có kiểm soát. **B1** là phiên bản scale nhẹ hơn B0 gốc (depth coefficient ≈1.1) — đã xác nhận qua cấu trúc block thật trong checkpoint (số block mỗi stage tăng so với B0: [2,3,3,4,4,5,2]).

**Khối cơ bản MBConv (Mobile Inverted Bottleneck):** mở rộng số kênh bằng conv 1×1, depthwise conv 3×3/5×5 xử lý không gian riêng từng kênh (tiết kiệm tham số so với conv thường), rồi nén lại bằng conv 1×1, có residual connection nếu input/output cùng shape.

### C2. CBAM (Convolutional Block Attention Module)

**Trực quan:** trước khi đưa feature map vào lớp phân loại, "lọc" lại 2 lần — lần 1 hỏi "kênh đặc trưng nào (màu, texture...) quan trọng?", lần 2 hỏi "vị trí không gian nào trên ảnh quan trọng?".

**Channel Attention:** với feature map $F$, tính cả **average pooling** và **max pooling** toàn cục theo không gian, đưa qua cùng 1 MLP chia sẻ trọng số (giảm chiều bằng `reduction=16` rồi khôi phục), cộng lại, qua sigmoid:
$$M_c(F) = \sigma\big(\text{MLP}(\text{AvgPool}(F)) + \text{MLP}(\text{MaxPool}(F))\big)$$
$F' = M_c(F) \otimes F$ (nhân theo kênh)

**Spatial Attention:** trên $F'$, tính avg và max theo chiều kênh (còn lại 2 kênh: trung bình và max tại mỗi vị trí không gian), ghép lại, qua conv 7×7 rồi sigmoid:
$$M_s(F') = \sigma\big(\text{Conv}_{7\times7}([\text{AvgPool}_c(F'); \text{MaxPool}_c(F')])\big)$$

**Vì sao dùng cả avg lẫn max pooling, không chỉ 1 loại:** avg pooling nắm bối cảnh tổng thể, max pooling bắt đặc trưng nổi bật nhất — dùng cả 2 bổ trợ nhau tốt hơn dùng riêng lẻ (đã được chứng minh thực nghiệm trong paper gốc CBAM).

**Grad-CAM (đưa vào Grad-CAM trên chính khối attention):** tính gradient của điểm số lớp dự đoán $y^c$ theo feature map ở lớp attention, average pooling gradient theo không gian để ra trọng số quan trọng $\alpha_k^c$ cho từng kênh $k$:
$$\alpha_k^c = \frac{1}{Z}\sum_{i,j} \frac{\partial y^c}{\partial A_{ij}^k}, \qquad L_{Grad-CAM}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
ReLU ở cuối vì chỉ quan tâm vùng có ảnh hưởng **dương** tới lớp dự đoán, bỏ qua vùng ảnh hưởng âm.

**Câu hỏi khả năng bị hỏi:** *"Tại sao chọn attention ở đây thay vì thêm lớp conv thường?"* → Attention không tăng nhiều tham số (chỉ thêm 1 MLP nhỏ + 1 conv 7×7) nhưng giúp model tự học "nhìn đâu là quan trọng" thay vì xử lý đều mọi vùng ảnh như nhau — đặc biệt hữu ích khi tổn thương chỉ chiếm 1 phần nhỏ trong ảnh, phần da lành xung quanh là nhiễu.

### C3. Input/Output chính xác

- **Input:** ảnh RGB **224×224×3**, chuẩn hóa theo **đúng thống kê ImageNet** (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) — khác với 2 model phân đoạn (dùng mean/std đơn giản 0.5/0.25). Lý do dùng chuẩn ImageNet ở đây: backbone EfficientNet-B1 được pretrained trên ImageNet với đúng chuẩn hóa này, giữ nguyên khi fine-tune giúp phân phối input khớp với những gì backbone "quen" trong lúc pretrain.
- **feature_dim = 1280:** số kênh đặc trưng đầu ra của EfficientNet-B1 sau `forward_features` (trước global pooling) — con số này giữ nguyên bất kể input resolution, chỉ phụ thuộc kiến trúc.
- **Output:** vector 7 logit (7 lớp bệnh) → softmax → phân phối xác suất, dùng trực tiếp làm $P(C_k|\text{Image})$ trong bước hợp nhất Bayes (mục D).
- **Tổng tham số:** đọc trực tiếp từ checkpoint (đếm toàn bộ tensor trong state_dict) — cỡ 81.4MB file → tương ứng khoảng 7-8 triệu tham số cho phần backbone+CBAM+classifier head (EfficientNet-B1 gốc khoảng 7.8M tham số theo công bố chính thức, cộng thêm phần CBAM nhỏ và Linear(1280→7) không đáng kể).

---

## D. Hợp nhất Bayes có trọng số (Fusion) — slide 15

### D1. Định lý Bayes — phần đúng chuẩn

$$P(C_k \mid \text{Demo}) \propto P(C_k) \cdot P(\text{Age}|C_k) \cdot P(\text{Gender}|C_k) \cdot P(\text{Location}|C_k)$$

Đây là posterior ∝ prior × likelihood, giả định các yếu tố dịch tễ **độc lập có điều kiện** cho trước lớp bệnh (giả định Naive Bayes) — đơn giản hóa nhưng thực dụng, vì mô hình hóa chính xác sự phụ thuộc giữa tuổi-giới-vị trí sẽ cần dữ liệu lớn hơn nhiều.

**Age likelihood (Gaussian):**
$$P(\text{Age}=a \mid C_k) = \frac{1}{\sigma_k\sqrt{2\pi}} \exp\left(-\frac{(a-\mu_k)^2}{2\sigma_k^2}\right)$$
với $\mu_k, \sigma_k$ ước lượng từ phân phối tuổi thật của từng lớp bệnh trong HAM10000.

### D2. Log-linear pooling — phần KHÔNG phải Bayes chuẩn (điểm hay bị hỏi nhất)

$$P_{\text{final}}(C_k) \propto \big[P(C_k|\text{Image})\big]^\lambda \cdot \big[P(C_k|\text{Demo})\big]^{1-\lambda}$$

Lấy log: $\log P_{\text{final}} = \lambda \log P(\text{Image}) + (1-\lambda)\log P(\text{Demo}) + \text{const}$ — đây là **trung bình có trọng số của log-xác suất**, tương đương **trung bình hình học có trọng số**. Tên gọi chuẩn trong tài liệu thống kê: **logarithmic opinion pool**.

**Vì sao KHÔNG phải Bayes thuần túy:** Bayes chuẩn để hợp nhất 2 nguồn bằng chứng độc lập sẽ là $P(C_k) \cdot P(\text{Image}|C_k) \cdot P(\text{Demo}|C_k)$ — nhân **likelihood**, không phải lũy thừa **posterior**. Model phân loại ảnh xuất ra $P(C_k|\text{Image})$ (posterior, qua softmax), muốn đúng Bayes phải "gỡ" prior huấn luyện ra trước ($P(\text{Image}|C_k) \propto P(C_k|\text{Image})/P_{\text{train}}(C_k)$) — bước này code không làm.

### D3. λ tự động theo Entropy

$$H = -\sum_c P(c)\log P(c), \qquad H_{\text{norm}} = \frac{H}{\log(K)}, \qquad \lambda = \lambda_{\max} - (\lambda_{\max}-\lambda_{\min}) \cdot H_{\text{norm}}$$

**Vì sao dùng entropy chứ không dùng trực tiếp xác suất top-1 (confidence):** entropy nhìn vào **toàn bộ phân phối** 7 lớp, không chỉ lớp cao nhất — 1 model có thể có top-1 = 0.4 nhưng vẫn "khá chắc chắn" nếu 6 lớp còn lại chia đều phần còn thấp; ngược lại top-1=0.4 nhưng lớp thứ 2 cũng gần 0.4 thì thực sự phân vân. Entropy phân biệt được 2 trường hợp này còn confidence top-1 thì không. $\log(K)$ với $K=7$ là entropy tối đa (phân phối đều), dùng để chuẩn hóa về [0,1].

**Hạn chế cần thừa nhận nếu bị hỏi:** entropy tính trên softmax **chưa hiệu chỉnh (uncalibrated)** — mạng neural nổi tiếng có xu hướng "quá tự tin" (overconfident), nên entropy đo được có thể thấp hơn độ bất định thật sự. Cách khắc phục chuẩn là temperature scaling trên tập validation trước khi tính entropy — hướng phát triển tiếp theo.

---

## E. VQA — DistilGPT-2 + LoRA — slide 16

### E1. Kiến trúc Transformer Decoder (nền tảng GPT-2/DistilGPT-2)

**Self-Attention:** mỗi token "hỏi" các token khác (kể cả chính nó) xem nên chú ý bao nhiêu, qua 3 ma trận học được Query/Key/Value:
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
Chia $\sqrt{d_k}$ để tránh giá trị dot-product quá lớn làm softmax bão hòa (gradient gần 0).

**Causal masking:** GPT là decoder-only, dùng **causal attention** — token thứ $i$ chỉ được nhìn token $1..i$, không được nhìn tương lai (che bằng ma trận tam giác $-\infty$ trước softmax) — phù hợp bài toán sinh văn bản tuần tự.

**DistilGPT-2:** bản chưng cất (distillation) từ GPT-2 gốc — huấn luyện 1 model nhỏ hơn để "bắt chước" output của model lớn (teacher), giữ được phần lớn khả năng nhưng giảm ~40% tham số, nhanh hơn, phù hợp chạy CPU.

### E2. LoRA (Low-Rank Adaptation) — chi tiết đúng số bạn dùng: rank 8, alpha 16

**Vấn đề LoRA giải quyết:** fine-tune toàn bộ tham số của 1 LLM (dù chỉ 82-90M tham số) trên dữ liệu nhỏ (74-80 mẫu) sẽ overfit nặng và tốn bộ nhớ lưu gradient cho mọi tham số.

**Ý tưởng:** thay vì cập nhật trực tiếp ma trận trọng số $W \in \mathbb{R}^{d\times d}$ (rất lớn), đóng băng $W$ hoàn toàn, chỉ học **2 ma trận hạng thấp** $A \in \mathbb{R}^{d\times r}$, $B \in \mathbb{R}^{r\times d}$ với $r \ll d$ (ở đây $r=8$):
$$W' = W + \Delta W = W + \frac{\alpha}{r} BA$$
$\alpha=16$ là hệ số scaling (giữ magnitude cập nhật ổn định khi đổi $r$), $\alpha/r = 16/8 = 2$.

**Vì sao tiết kiệm tham số khủng khiếp:** nếu $d=768$ (kích thước ẩn của GPT-2), ma trận gốc $W$ có $768 \times 768 \approx 590{,}000$ tham số; còn $A, B$ với $r=8$ chỉ có $768\times8 + 8\times768 \approx 12{,}300$ tham số — giảm ~98% cho riêng lớp đó. Đây chính là lý do tổng thể chỉ 1,926,754/90,352,514 ≈ **2.13%** tham số cần huấn luyện (đúng số bạn có trên slide).

**Target `c_attn`:** LoRA chỉ áp vào ma trận attention (query/key/value projection), không áp toàn bộ mạng — vì đây là nơi mô hình học cách "chú ý" theo ngữ cảnh mới, hiệu quả nhất để thích nghi domain y khoa với chi phí tối thiểu.

**Câu hỏi khả năng bị hỏi:** *"Rank 8 có đủ không, sao không chọn rank cao hơn?"* → Rank càng cao thì càng biểu diễn được nhiều loại điều chỉnh phức tạp hơn nhưng cũng dễ overfit hơn trên tập dữ liệu 74-80 mẫu rất nhỏ; rank 8 là lựa chọn thận trọng cân bằng giữa khả năng học và nguy cơ overfit — có thể thử nghiệm thêm với rank 4 hoặc 16 để so sánh nếu có thời gian.

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

## F. RAG — Retrieval-Augmented Generation — slide 16, 17

### F1. Sentence Embedding (All-MiniLM-L6-v2)

**Ý tưởng:** biến câu hỏi văn bản thành 1 vector số thực cố định chiều (model này ra vector 384 chiều) sao cho **câu có nghĩa gần nhau thì vector gần nhau** trong không gian đó — dùng kiến trúc Transformer encoder 6 lớp (L6), chưng cất từ model lớn hơn (MiniLM = "mini language model", nhẹ, nhanh).

### F2. Cosine Similarity — vì sao dùng độ đo này chứ không phải khoảng cách Euclid

$$\text{cos}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$$

Đo **góc** giữa 2 vector, không quan tâm độ dài (magnitude) — phù hợp với embedding văn bản vì độ dài vector đôi khi phản ánh độ dài câu chứ không phải ý nghĩa; 2 câu cùng ý nghĩa nhưng độ dài khác nhau vẫn có cosine similarity cao dù khoảng cách Euclid có thể lớn.

**Lưu ý cần kiểm tra lại trước khi khẳng định chắc:** trong code (`rag_engine.py`), khi tạo collection bằng `client.get_or_create_collection(name=..., embedding_function=...)`, **không có tham số `metadata={"hnsw:space": "cosine"}` tường minh** — mà ChromaDB mặc định dùng khoảng cách **L2 (Euclidean) bình phương** cho index HNSW nếu không cấu hình rõ. Nếu bị hỏi sâu "có chắc là cosine không", câu trả lời an toàn nhất: *"Mục tiêu thiết kế là dùng cosine similarity vì phù hợp với sentence embedding, cần xác minh lại đúng phiên bản ChromaDB đang cài đặt có mặc định là cosine hay L2 để khẳng định chính xác 100%."* — tránh khẳng định tuyệt đối một điều chưa tự tay kiểm chứng đến cùng.

### F3. ChromaDB — Vector Database

Lưu trữ hàng loạt embedding vector cùng văn bản gốc, khi có câu hỏi mới: mã hóa thành vector, tìm **k vector gần nhất** (ở đây $k=1$ theo code, `n_results=1`) bằng thuật toán tìm kiếm lân cận gần đúng (Approximate Nearest Neighbor — nhanh hơn brute-force khi kho tài liệu lớn), trả về đoạn văn bản y văn tương ứng, nhồi vào prompt cho LLM.

**Vì sao gọi là "giảm ảo giác" (hallucination):** LLM không phải lúc nào cũng nhớ đúng thông tin y khoa chính xác từ dữ liệu huấn luyện gốc (có thể "bịa" thông tin nghe hợp lý nhưng sai). RAG buộc LLM phải dựa vào đoạn văn bản y văn **thật** được nhồi vào ngay trong prompt, giảm khả năng bịa đặt.

---

## G. Bảo mật — SHA-256, XOR, Base64 — slide 20, 21

### G1. SHA-256

**Tính chất bắt buộc phải biết:** hàm băm 1 chiều (không thể đảo ngược để lấy lại input gốc), luôn ra output cố định **256 bit = 32 byte** bất kể input dài ngắn thế nào, có tính "hiệu ứng lan tỏa" (avalanche effect — đổi 1 bit input làm output đổi hoàn toàn khác), và có tính kháng va chạm (collision-resistant — cực khó tìm 2 input khác nhau ra cùng hash).

**Vì sao chỉ lấy 16 ký tự hex đầu:** 16 ký tự hex = 64 bit (mỗi ký tự hex = 4 bit) trong tổng số 256 bit — **cắt ngắn làm giảm mạnh độ kháng va chạm** (birthday bound cho 64-bit ≈ $2^{32}$ mẫu là có 50% khả năng đụng độ) — chấp nhận được cho quy mô vài nghìn bệnh nhân nhưng về lý thuyết không còn "an toàn" theo chuẩn crypto đầy đủ.

### G2. XOR Cipher — vì sao yếu

**Nguyên lý:** $C = P \oplus K$ (ciphertext = plaintext XOR key), giải mã: $P = C \oplus K$ (vì $X \oplus K \oplus K = X$).

**Vì sao yếu khi khóa lặp lại (repeating-key XOR):** nếu dùng cùng 1 khóa cho nhiều bản ghi (đúng tình huống code hiện tại — khóa `"DermaSecureKey2026"` cố định trong source), kẻ tấn công có 2 bản mã dùng chung khóa có thể XOR 2 ciphertext với nhau, khóa sẽ tự triệt tiêu: $C_1 \oplus C_2 = P_1 \oplus P_2$ — lộ thông tin thống kê về plaintext (tấn công dựa trên tần suất ký tự tiếng Anh/Việt hoàn toàn khả thi). Đây gọi là điểm yếu kinh điển của repeating-key XOR, khác hẳn One-Time Pad (khóa dùng đúng 1 lần, dài bằng plaintext, ngẫu nhiên hoàn toàn — mới thực sự an toàn tuyệt đối về lý thuyết thông tin, theo chứng minh của Shannon).

**Base64:** chỉ là **encoding** (biến đổi biểu diễn để có thể lưu/truyền dưới dạng text ASCII an toàn), hoàn toàn **không phải mã hóa** — ai cũng decode được ngay lập tức không cần khóa gì cả, chỉ dùng để "đóng gói hiển thị" chuỗi byte nhị phân sau XOR thành text.

**Câu trả lời chuẩn khi bị hỏi:** *"XOR+Base64 có coi là bảo mật không?"* → Không đạt chuẩn mật mã học hiện đại; đây là biện pháp ẩn dữ liệu (obfuscation) ở mức nguyên mẫu. Hướng khắc phục đúng chuẩn: **AES-256-GCM** (mã hóa khối đối xứng đã được kiểm chứng rộng rãi, có chế độ GCM cung cấp cả tính bảo mật lẫn xác thực toàn vẹn dữ liệu — authenticated encryption), với khóa quản lý qua KMS (Key Management Service) tách biệt hoàn toàn khỏi mã nguồn.

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

---

## Thứ tự ưu tiên học nếu thời gian gấp

1. **Bắt buộc thuộc nằm lòng:** 2 cảnh báo ở mục "0" (SAM không thực sự chạy, kiến trúc VQA V3 không có trong checkpoint thật) — đây là 2 chỗ nếu bị hỏi trúng mà trả lời sai sự thật sẽ mất điểm nặng nhất trong cả buổi; công thức ABCD (mục B), công thức Bayes fusion + entropy λ (mục D), vì sao XOR yếu (mục G2).
2. **Nên hiểu sâu để giải thích tự tin:** LoRA (mục E2) — vì con số 2.13% rất dễ bị hỏi "tính từ đâu ra" (đã có bảng chia đúng 3 thành phần ở mục E3); luồng ghép 1 token ảnh thật (mục E3) — vì sao đây không phải VLM cross-attention đầy đủ.
3. **Nên nắm khái niệm, không cần thuộc công thức:** DeepLabV3+/ASPP, SAM+GrabCut, CBAM, Grad-CAM — hiểu đủ để giải thích trực quan, không cần viết được công thức đầy đủ trên bảng.
4. **Chỉ cần đọc hiểu ý nghĩa số liệu:** Dice/IoU, BLEU, confusion matrix, accuracy-coverage — đây là phần "đọc số ra sao" chứ không phải "chứng minh công thức".
