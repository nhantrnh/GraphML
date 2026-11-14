# BÁO CÁO CHUYÊN ĐỀ

## APKGC: Noise-enhanced Multi-Modal Knowledge Graph Completion with Attention Penalty

| | |
|---|---|
| **Hội nghị** | The 39th AAAI Conference on Artificial Intelligence (AAAI-25), Oral, trang 15005–15013 |
| **Tác giả** | Yue Jian, Xiangyu Luo, Zhifei Li, Miao Zhang, Yan Zhang, Kui Xiao, Xiaoju Hou — Hubei University, Trung Quốc |
| **Mã nguồn** | https://github.com/HubuKG/APKGC |
| **Môn học** | GraphML — HP3 |

---

## 1. Lý do chọn bài báo

Bài báo được chọn dựa trên bốn tiêu chí.

**Chất lượng công bố.** AAAI-25 nhận bài ở dạng Oral, tương ứng nhóm khoảng 5% bài được đánh giá cao nhất trong số các bài được chấp nhận.

**Không thuộc nhóm bị loại.** Đây là bài đề xuất mô hình, không phải khảo sát và không phải công trình xây dựng bộ dữ liệu. Bài không nằm trong danh sách các công trình đã được nhóm khác đăng ký.

**Khả năng tái lập.** Mã nguồn công khai, có script huấn luyện và kiểm thử, bộ nhúng đã trích sẵn được chia sẻ qua Google Drive. Toàn bộ thí nghiệm chạy trên một GPU đơn.

**Khả năng cải tiến.** Hai đóng góp chính — lấy mẫu nhiễu và hình phạt chú ý — đều là mô-đun độc lập, có thể tháo lắp và thay thế mà không phải viết lại kiến trúc.

Một lợi thế riêng: nhóm đã làm việc trước với MyGO (AAAI 2025), công trình dùng chung khung mã `mmkgc/` và chung bộ dữ liệu DB15K, MKG-W. Môi trường và dữ liệu vì vậy tái sử dụng được, đồng thời tạo điều kiện đối chiếu trực tiếp giữa hai phương pháp trong phần thực nghiệm.

---

## 2. Kiến thức nền

### 2.1 Đồ thị tri thức và bài toán hoàn thiện

Một đồ thị tri thức được ký hiệu `G = (E, R, T)`, trong đó `E` là tập thực thể, `R` là tập quan hệ, `T` là tập bộ ba. Mỗi sự kiện biểu diễn dưới dạng `(h, r, t)`.

Đồ thị tri thức do con người xây dựng luôn ở trạng thái thiếu. DB15K chứa 12.842 thực thể và 279 quan hệ nhưng chỉ có 79.222 bộ ba huấn luyện — rất thưa so với không gian tổ hợp có thể có. Bài toán hoàn thiện đồ thị tri thức (Knowledge Graph Completion) đặt ra yêu cầu: cho `(h, r, ?)`, xếp hạng toàn bộ thực thể trong `E` sao cho đáp án đúng đứng càng cao càng tốt. Đây là bài toán xếp hạng, không phải phân loại — điểm này quyết định cách chọn độ đo đánh giá.

### 2.2 Các mô hình nhúng nền tảng

**TransE (2013)** mô hình hoá quan hệ như một phép tịnh tiến: `h + r ≈ t`, điểm số `S = −‖h + r − t‖`. Trực giác đơn giản và hiệu quả, nhưng không biểu diễn được quan hệ đối xứng: nếu `r` đối xứng thì buộc `r ≈ 0`, làm sụp đổ biểu diễn.

**RotatE (2019)** chuyển sang không gian phức, mô hình hoá quan hệ như phép quay: `h ∘ r ≈ t` với ràng buộc `|r_i| = 1`. Phép quay xử lý được cả ba loại quan hệ khó: đối xứng tương ứng quay 180°, nghịch đảo tương ứng quay ngược chiều, hợp thành tương ứng cộng góc. APKGC kế thừa RotatE làm hàm chấm điểm.

### 2.3 Vì sao cần đa mô thái

Cả hai mô hình trên chỉ khai thác cấu trúc đồ thị. Với thực thể xuất hiện ít — nhóm long-tail — cấu trúc gần như trống, mô hình không đủ tín hiệu để học. Trong khi đó thực thể ngoài đời còn mang thông tin ảnh và mô tả văn bản. Đồ thị tri thức đa mô thái (Multi-modal Knowledge Graph) bổ sung hai nguồn này, và bài toán tương ứng là MMKGC.

Trong APKGC, mỗi thực thể mang ba vector:

| Mô thái | Ký hiệu | Nguồn trích xuất | Trạng thái |
|---|---|---|---|
| Cấu trúc | `e_s` | `nn.Embedding` học từ đầu | Huấn luyện |
| Thị giác | `e_v` | BEiT, 4096 chiều | Đóng băng |
| Văn bản | `e_t` | BERT, 768 chiều | Đóng băng |

Việc đóng băng hai bộ nhúng ngoại vi là lý do mô hình chạy được trên một GPU đơn: phần học chỉ gồm ma trận chiếu, các lớp hợp nhất và nhúng cấu trúc.

Một chi tiết kỹ thuật đáng chú ý: trong MKG-W nhiều thực thể thiếu hẳn ảnh hoặc mô tả. Mã nguồn lưu danh sách này trong `missing.json` và lấp chỗ trống bằng vector sinh từ phân phối `N(μ, σ)` của các thực thể còn đủ mô thái. Cách xử lý này đã hé lộ tinh thần xuyên suốt bài báo: nhiễu không được xem là kẻ thù.

---

## 3. Hai vấn đề bài báo đặt ra

### 3.1 Mô hình thiếu bền vững trước nhiễu

Các phương pháp MMKGC trước xem nhiễu là yếu tố gây nhiễu loạn cần loại bỏ: lọc ảnh không khớp, loại mô tả sai lệch, làm sạch dữ liệu trước khi huấn luyện.

Cách tiếp cận này có ba điểm hụt. Thứ nhất, dữ liệu thực tế luôn bẩn — không tồn tại tập triển khai đã được làm sạch. Thứ hai, mô hình huấn luyện trên dữ liệu sạch chưa từng gặp nhiễu nên dễ gãy khi vận hành thật. Thứ ba, lọc càng mạnh càng làm mất thông tin của nhóm thực thể hiếm, đúng nhóm cần hỗ trợ nhất.

APKGC lật ngược: chủ động tiêm nhiễu trong quá trình huấn luyện. Về mặt tư tưởng, đây là cùng nguyên lý với data augmentation trong thị giác máy tính và với dropout — buộc mô hình học đặc trưng bền vững thay vì ghi nhớ máy móc.

### 3.2 Attention over-trust

Mô hình ngôn ngữ tiền huấn luyện có xu hướng dồn trọng số chú ý vào những token không mang ngữ nghĩa: dấu chấm, dấu phẩy, token đặc biệt. Hình 1 của bài báo minh hoạ trường hợp thực thể "Umbria", trong đó dấu chấm nhận điểm chú ý 0.66 còn danh từ riêng — chủ thể thực sự của câu — bị bỏ qua.

Hiện tượng này đã được ghi nhận độc lập trong nghiên cứu về ảo giác đa mô thái (Huang và cộng sự, 2023). Cách giải thích hợp lý: token phổ biến xuất hiện trong mọi câu nên trở thành điểm trú ẩn an toàn cho cơ chế chú ý khi mô hình không chắc chắn — tương tự khái niệm attention sink.

Hệ quả trực tiếp là đặc trưng thực thể trích ra bị nhiễu loạn, kéo theo dự đoán sai.

---

## 4. Kiến trúc APKGC

Mô hình gồm ba khối nối tiếp. Hai đóng góp chính nằm ở khối (a) và (b); khối (c) kế thừa RotatE.

### 4.1 Khối (a) — Modality Noise Sampling

**Sinh nhiễu theo phân phối dữ liệu.** Bài báo đề xuất hai chiến lược:

- Nhiễu Gaussian: `N_g = θ_r(φ_m ⊙ g + μ_m)`, với `g ~ N(0, Z)` — dùng cho DB15K
- Nhiễu trung bình: `N_a = θ_r[β_min + (β_max − β_min) ⊙ U]` — dùng cho MKG-W, cho nhiễu mượt hơn và ít chịu ảnh hưởng của giá trị ngoại lai

Điểm mấu chốt: nhiễu không ngẫu nhiên tuỳ tiện mà lấy trung bình `μ` và độ lệch chuẩn `σ` của chính tập dữ liệu, nên vector nhiễu vẫn nằm trong vùng hợp lý của không gian đặc trưng.

**Quyết định tiêm nhiễu.** Mỗi thực thể được gán một biến ngẫu nhiên `ρ ~ U[0,1]` và so với hệ số lấy mẫu `θ_n`:

```
V_n = (1 − θ_r)·V_o + N_m    nếu ρ > θ_n
V_n = V_o                     ngược lại
```

**Đối chiếu với mã nguồn.** Trong `mmkgc/module/model/APKGC.py`:

```python
self.mask_ratio  = nn.Parameter(torch.tensor(0.7))
self.noise_ratio = nn.Parameter(torch.tensor(0.2))

def add_noise_to_embeddings(self, embeddings, mean, std, noise_ratio=0.1):
    noise_mask = torch.rand(embeddings.shape[0]) < self.noise_ratio
    selected   = embeddings[noise_mask]
    noise      = mean + std * torch.randn_like(selected)
    embeddings[noise_mask] = (1.0 - self.mask_ratio) * selected \
                           + self.mask_ratio * noise
    return embeddings
```

Hai tỉ lệ khai báo bằng `nn.Parameter`, tức tham số học được qua lan truyền ngược — khớp với mô tả "adaptively tuned" trong bài báo. Nhiễu được làm mới mỗi epoch qua `update_noise()`, gọi từ `Trainer.train_one_step`.

### 4.2 Khối (b) — Attention Penalty

Đây là đóng góp cốt lõi, cài đặt trong lớp `BertSelfAttention` của cùng tệp. Cơ chế gồm ba nhịp.

**Nhịp 1 — Phát hiện.** Tính tích luỹ hậu tố của ma trận điểm chú ý:

```python
penalty_scores = torch.zeros_like(attention_scores)
for i in range(attention_scores.shape[-1]):
    penalty_scores[..., i] = attention_scores[..., i:].prod(dim=-1)

exceed_threshold_mask = penalty_scores > self.penalty_threshold   # φ = 10
```

Lý do dùng tích thay vì dùng trực tiếp điểm số: phép nhân khuếch đại tín hiệu, nên chỉ token giữ điểm cao một cách dai dẳng mới vượt ngưỡng, còn token cao nhất thời thì không bị bắt.

**Nhịp 2 — Phạt.** Token vượt ngưỡng bị trừ theo tỉ lệ `θ_p`:

```
A_j^p = A_i^p − θ_p(A_i^p),  nếu A_i^p > φ
```

Trong cài đặt, giá trị phạt được `clamp` trong biên `[−0.01, 0.01]` rồi nhân hệ số `penalty_weights = 0.001` để tránh phạt quá tay.

**Nhịp 3 — Tái phân bổ.** Phần chú ý bị tịch thu được chia lại cho các token lân cận trong cửa sổ `k`:

```
A_j^n = A_i^n + θ_j^n(A_i^p − φ),   n ~ (p−k, p+k)
```

Đây là điểm phân biệt APKGC với các phương pháp chỉ đơn thuần triệt tiêu: chú ý được chuyển hướng chứ không bị xoá bỏ. Ablation ở mục 5.3 xác nhận cả hai nhánh đều cần thiết.

### 4.3 Khối (c) — Hợp nhất và chấm điểm

**Tương tác giữa các mô thái.** Bài báo đưa vào hệ số tương quan chéo:

```
θ_ab = exp(Q_a^T K_b/√d) / [exp(Q_a^T K_a/√d) + exp(Q_a^T K_b/√d)]
V_a^i = θ_ab · V_a
```

Khác với multi-head attention thông thường, công thức này so sánh mức tự tương quan với mức tương quan chéo, tức cân bằng giữa "tin vào chính mô thái này" và "nghe theo mô thái kia". Khi ảnh của một thực thể mờ hoặc sai, `θ` tự động nghiêng trọng số về phía văn bản.

**Sáu phương án hợp nhất** hiện diện trong mã, chọn qua tham số `-joint_way`:

| Phương án | Cơ chế | Dùng cho |
|---|---|---|
| `Mformer_hd_graph` | Lấy token đầu tiên sau các lớp fusion | DB15K |
| `Mformer_hd_mean` | Lấy trung bình các token | MKG-W |
| `Mformer_weight` | Trọng số suy từ ma trận chú ý | — |
| `atten_weight` | Chú ý tuyến tính một lớp | — |
| `learnable_weight` | Ba trọng số học được qua softmax | — |
| `cat_mlp` | Nối ba vector rồi chiếu qua MLP | — |

**Hàm chấm điểm và mất mát.** Sử dụng RotatE với mất mát sigmoid có trọng số tự đối kháng:

```
S(h,r,t) = ‖h̄ ∘ r − t̄‖
L = (1/|T|) Σ [ −log σ(λ − S(h,r,t)) − Σ v_i σ(S(h',r',t') − λ) ]
```

Trọng số `v_i` khiến mẫu âm khó nhận được sự chú ý lớn hơn — kỹ thuật kế thừa từ RotatE.

---

## 5. Thực nghiệm

### 5.1 Thiết lập

| | DB15K | MKG-W |
|---|---|---|
| Thực thể | 12.842 | 15.000 |
| Quan hệ | 279 | 169 |
| Huấn luyện | 79.222 | 34.196 |
| Kiểm định | 9.902 | 4.276 |
| Kiểm thử | 9.904 | 4.274 |

Độ đo: MRR và Hits@N với N ∈ {1, 3, 10}, đánh giá theo thiết lập lọc (loại bỏ các bộ ba đúng đã biết khỏi danh sách xếp hạng).

Cấu hình tác giả công bố: một GPU RTX 4090, chiều nhúng 128, kích thước lô 2048, 64 mẫu âm cho mỗi mẫu dương, tốc độ học 1e-4, 8000 vòng lặp.

### 5.2 So sánh với các mô hình nền

| Mô hình | DB15K MRR | H@1 | H@10 | MKG-W MRR | H@1 |
|---|---|---|---|---|---|
| TransE (2013) | 0.249 | 0.128 | 0.471 | 0.292 | 0.211 |
| RotatE (2019) | 0.293 | 0.179 | 0.471 | 0.337 | 0.268 |
| PairRE (2021) | 0.311 | 0.216 | 0.493 | 0.344 | 0.282 |
| IMF (WWW'23) | 0.323 | 0.242 | 0.482 | 0.345 | 0.288 |
| VISTA (EMNLP'23) | 0.304 | 0.225 | 0.459 | 0.329 | 0.261 |
| AdaMF (COLING'24) | 0.325 | 0.213 | 0.517 | 0.343 | 0.272 |
| SnAg (2024) | 0.342 | 0.250 | 0.515 | 0.359 | 0.284 |
| NativE (SIGIR'24) | 0.362 | 0.273 | **0.529** | 0.366 | 0.296 |
| **APKGC** | **0.364** | **0.282** | 0.527 | **0.374** | **0.306** |

Bài báo nhấn mạnh mức tăng Hits@1: 3.3% trên DB15K (0.273 → 0.282) và 3.4% trên MKG-W (0.296 → 0.306) so với NativE.

Cần đọc con số này một cách chính xác. Đây là mức tăng **tương đối**; chênh lệch tuyệt đối chỉ khoảng 0.9 điểm phần trăm. Ngoài ra ở Hits@10 trên DB15K, APKGC thấp hơn NativE (0.527 so với 0.529) — bài báo gạch chân thay vì in đậm ở ô này nhưng không thảo luận nguyên nhân. Bảng cũng không báo cáo độ lệch chuẩn qua nhiều seed, nên không đủ cơ sở khẳng định ý nghĩa thống kê của khoảng cách 0.002 MRR.

### 5.3 Ablation

**Theo mô thái (DB15K):**

| Cấu hình | MRR | H@1 | H@10 |
|---|---|---|---|
| Chỉ cấu trúc | 0.211 | 0.125 | 0.345 |
| Chỉ ảnh | 0.336 | 0.261 | 0.469 |
| Chỉ văn bản | 0.348 | 0.273 | 0.505 |
| Cấu trúc + ảnh | 0.340 | 0.260 | 0.480 |
| Văn bản + cấu trúc | 0.351 | 0.279 | 0.509 |
| Ảnh + văn bản | 0.357 | 0.274 | 0.521 |
| **Đủ ba mô thái** | **0.364** | **0.282** | **0.527** |

Cấu hình chỉ dùng cấu trúc kém xa phần còn lại, khẳng định giá trị của hướng đa mô thái. Văn bản đóng góp nhiều hơn ảnh: loại bỏ văn bản làm Hits@10 giảm 8.9%, loại bỏ ảnh chỉ giảm 3.4%.

**Theo thành phần (MKG-W):**

| Cấu hình | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|
| Bỏ lấy mẫu nhiễu | 0.358 | 0.289 | 0.393 | 0.486 |
| Bỏ attention penalty | 0.360 | 0.286 | 0.397 | 0.492 |
| Bỏ nhánh tăng | 0.365 | 0.297 | 0.399 | 0.491 |
| Bỏ nhánh giảm | 0.369 | 0.302 | 0.398 | 0.495 |
| **APKGC đầy đủ** | **0.374** | **0.306** | **0.404** | **0.501** |

Hai kết luận rút ra. Thứ nhất, lấy mẫu nhiễu đóng góp nhiều hơn attention penalty (mất 0.016 MRR so với 0.014) — đáng chú ý vì tiêu đề bài báo đặt attention penalty ở vị trí nổi bật hơn. Thứ hai, hai nhánh tăng và giảm đều cần thiết, xác nhận rằng cơ chế phải là tái phân bổ chứ không chỉ triệt tiêu.

### 5.4 Phân tích siêu tham số

Bài báo khảo sát số cửa sổ phạt `k` và tỉ lệ phạt `θ_p` trên lưới 4×4. Kết quả cho đường cong hình chữ U ngược: hiệu năng đạt đỉnh ở vùng giữa (`k = 2`, `θ_p = 0.01`, MRR 0.364 trên DB15K) rồi suy giảm về cả hai phía.

Phạt quá nhẹ không đủ kéo chú ý khỏi token vô nghĩa. Phạt quá nặng phá vỡ cả những phân bổ chú ý đúng đắn — tại `θ_p = 0.1`, MRR rơi xuống 0.324, thấp hơn cả khi không dùng penalty.

Đây là một hạn chế về khả năng triển khai: mỗi tập dữ liệu mới đều đòi hỏi dò lại lưới siêu tham số, và bài báo không đề xuất phương pháp chọn tự động.

### 5.5 Nghiên cứu trường hợp

Bài báo phân tích hai bộ ba từ DB15K.

Với truy vấn `(National Society of Film Critics Award for Best Supporting Actor, award, ?)`, mô hình không có attention penalty gán điểm chú ý 0.66 cho token "for" và xếp đáp án đúng *Dennis Hopper* ở vị trí thứ hai, sau *Kurtwood Smith*. Khi bật attention penalty, điểm chú ý của "for" giảm còn 0.50 và *Dennis Hopper* lên vị trí thứ nhất.

Trường hợp thứ hai, `(?, location, Saint Vincent and the Grenadines)`, cho kết quả tương tự: điểm chú ý của token "and" giảm từ 0.59 xuống 0.50, đưa *Caribbean Sea* từ hạng ba lên hạng nhất.

Hai ví dụ này minh hoạ trực tiếp cơ chế và là chất liệu tốt để trình bày trong slide.

---

## 6. Đánh giá phản biện

### 6.1 Điểm mạnh

**Lật ngược một giả định phổ biến.** Chuyển nhiễu từ vai kẻ thù sang vai công cụ là đóng góp về tư duy, không chỉ về kỹ thuật. Nó nối MMKGC với truyền thống regularization đã được kiểm chứng trong học sâu.

**Chẩn đoán đúng vấn đề.** Attention over-trust là hiện tượng có thật, được ghi nhận độc lập trong nghiên cứu về ảo giác đa mô thái, chứ không phải vấn đề do tác giả tự đặt ra để giải quyết.

**Tái phân bổ thay vì triệt tiêu.** Thiết kế ba nhịp có cơ sở, và ablation chứng minh cả nhánh tăng lẫn nhánh giảm đều đóng góp.

**Tính tái lập tốt.** Mã nguồn công khai kèm script, bộ nhúng đã trích sẵn, yêu cầu phần cứng khiêm tốn.

### 6.2 Hạn chế

**Mức cải thiện khiêm tốn và thiếu kiểm định thống kê.** Chênh lệch tuyệt đối dưới một điểm phần trăm, không có độ lệch chuẩn qua nhiều seed. Khoảng cách 0.002 MRR so với NativE trên DB15K khó phân biệt với dao động ngẫu nhiên.

**Thua ở một chỉ số nhưng không thảo luận.** Hits@10 trên DB15K thấp hơn NativE. Một phân tích về việc vì sao attention penalty giúp Hits@1 mà không giúp Hits@10 sẽ làm bài báo thuyết phục hơn.

**Ngưỡng cứng thiếu biện luận.** Các giá trị `penalty_threshold = 10`, `penalty_max_threshold = 15`, `penalty_weights = 0.001`, `origin_threshold = 0.01` nằm cứng trong mã nguồn, không có phân tích độ nhạy và không xuất hiện trong phần siêu tham số của bài báo.

**Cấu hình khác nhau theo từng tập dữ liệu.** DB15K dùng nhiễu Gaussian với hợp nhất `Mformer_hd_graph`; MKG-W dùng nhiễu trung bình với `Mformer_hd_mean`. Sự phân đôi này có thể là thích ứng hợp lý theo đặc tính dữ liệu, nhưng cũng làm dấy lên nghi vấn về mức độ điều chỉnh riêng cho từng tập.

**Cài đặt chưa tối ưu về hình thức.** Tích luỹ hậu tố tính bằng vòng lặp Python trên chiều cuối của ma trận chú ý thay vì dùng `cumprod`. Nhóm đã kiểm chứng phép biến đổi này (mục 8.4): với chiều cuối bằng 3 như hiện tại, bản vector hoá thực ra **chậm hơn** 1,4 lần. Đây chỉ trở thành vấn đề nếu số mô thái tăng lên.

**Phạm vi kiểm chứng hẹp.** Chỉ hai tập dữ liệu, đều thuộc cùng họ MMKG. Chưa có bằng chứng về khả năng tổng quát hoá sang tập thứ ba.

---

## 7. Rà soát tài liệu: các hạn chế đã ai khắc phục chưa?

Phần này rà soát xem bảy hạn chế nêu ở mục 6.2 đã được công trình nào giải quyết hay chưa. Chi tiết đầy đủ kèm nguồn nằm trong tài liệu `literature_review.md` đi kèm.

### 7.1 Lưu ý về trùng tên

Tồn tại **hai công trình cùng viết tắt APKGC**: bài của chúng ta (Attention Penalty, AAAI 2025) và một bài khác về prompt-tuning cho KGC đơn mô thái (Knowledge-Based Systems 2025). Khi tra cứu trích dẫn cần phân biệt rõ.

### 7.2 Tình trạng trích dẫn

APKGC xuất hiện trong danh mục tham khảo của một số công trình MMKGC 2025–2026, nhưng **không tìm thấy phân tích phản biện hay báo cáo tái lập độc lập nào**. Đáng chú ý, bài vắng mặt trong bảng so sánh của nhiều công trình mới dùng đúng DB15K và MKG-W, chẳng hạn RADD (arXiv 2604.25693).

### 7.3 Kết quả rà soát theo từng hạn chế

| Hạn chế | Đã ai khắc phục? | Ghi chú |
|---|---|---|
| H1 — Thiếu độ lệch chuẩn | ❌ Chưa, và là vấn đề chung của lĩnh vực | Xem 7.4 |
| H2 — Thua Hits@10 trên DB15K | ❌ Chưa ai giải thích | Giả thuyết: phạt làm sắc nét phân phối, lợi cho H@1 nhưng thu hẹp vùng ứng viên |
| H3 — Ngưỡng cứng | ⚠️ Đã có lời giải ở lĩnh vực khác, chưa mang sang MMKGC | Xem 7.5 |
| H4 — Cấu hình theo tập | ❌ Chưa ai kiểm chứng chéo | Thí nghiệm 2×2 là đủ kết luận |
| H5 — Chỉ hai tập | ⚠️ Các bài khác báo cáo đủ ba tập | RADD báo cáo cả MKG-Y |
| H6 — Không vector hoá | ❌ Chưa ai nêu | Sửa được bằng `flip → cumprod → flip` |
| H7 — Chỉ ba mô thái | ⚠️ Dữ liệu đã có từ 2023, chưa ai áp dụng | TIVA-KG (ACM MM 2023) đủ bốn mô thái |

### 7.4 Bằng chứng định lượng cho hạn chế H1

Công trình **Link Prediction or Perdition: the Seeds of Instability in Knowledge Graph Embeddings** (Méroué, Gandon, Monnin — arXiv 2606.03365, tháng 6/2026) cung cấp con số trực tiếp liên quan tới APKGC:

- **RotatE trên WN18RR: MRR = 0.326 ± 0.039.** Độ lệch chuẩn này **lớn gấp gần 20 lần** khoảng cách 0.002 MRR mà APKGC dùng để tuyên bố vượt NativE trên DB15K.
- RotatE trên FB15k-237: 0.288 ± 0.001 — ổn định hơn nhiều, cho thấy mức bất ổn phụ thuộc mạnh vào bộ dữ liệu và **không thể suy đoán cho DB15K nếu chưa đo**.
- Ngay cả khi độ lệch chuẩn nhỏ, hai lần chạy khác seed chỉ trùng trung bình **4,3 trên 10** thực thể dự đoán đầu bảng, và khoảng **40% bộ ba** cho đáp án hạng nhất khác nhau.

Điểm này đặc biệt đáng chú ý vì APKGC dùng chính RotatE làm hàm chấm điểm.

### 7.5 Lời giải sẵn có cho hạn chế H3

Nghiên cứu về hiện tượng attention sink — cùng bản chất với over-trust mà APKGC nhắm tới — đã phát triển các cơ chế tinh vi hơn ngưỡng cố định:

- **When Attention Sink Emerges in Language Models** (ICLR 2025) phân tích cơ chế hình thành, cung cấp nền lý thuyết mà APKGC còn thiếu.
- **Gated Attention for Large Language Models** đề xuất cổng sigmoid theo từng đầu chú ý, **loại bỏ attention sink từ gốc thay vì phạt sau khi nó xảy ra** — về thiết kế là hướng sạch hơn.
- Các phương pháp entropy regularization dùng **ngưỡng động theo từng đầu chú ý**, tính như tỉ lệ học được của entropy cực đại, kèm biên dung sai chống phạt quá tay. Đây chính xác là thứ APKGC thiếu.

Chưa có công trình nào mang những cơ chế này sang MMKGC.

### 7.6 Hiện trạng SOTA

Một số công trình 2025–2026 công bố kết quả cao hơn APKGC trên cùng bộ dữ liệu, tuy phần lớn **không đưa APKGC vào bảng so sánh** nên không đối chiếu trực tiếp được:

| Công trình | Kết quả công bố | Có so với APKGC? |
|---|---|---|
| APKGC (AAAI'25) | DB15K MRR 0.364 · MKG-W 0.374 | — |
| RADD (arXiv 2604.25693) | MRR 41.17 (DB15K) · 36.48 (MKG-W) · 38.96 (MKG-Y) | Không |
| RAPC (Applied Sciences 2026) | +2.64 MRR / +3.57 H@1 trên DB15K | Không xác minh được |
| MgHiSal (KBS 2025) | +13.1% MRR trên DB15K | Không xác minh được |
| ELMM (arXiv 2510.16753) | Tuyên bố SOTA trên 4 tập | Không |

**Nhận định thận trọng:** APKGC nhiều khả năng không còn giữ vị trí SOTA tính đến giữa 2026, nhưng chưa có công trình nào so sánh trực tiếp và công bằng với nó — càng củng cố giá trị của việc tự tái lập.

---

## 8. Hướng cải tiến đề xuất

Dựa trên kết quả rà soát ở mục 7, năm hướng dưới đây đều nhắm vào khoảng trống **chưa ai lấp**, và đều khả thi trong 1–2 tháng với một GPU.

### 8.1 Kiểm định độ ổn định qua nhiều seed — ưu tiên cao nhất

**Nhắm tới:** H1, H2

Chạy APKGC với 5 seed trên cả hai bộ dữ liệu, báo cáo trung bình kèm độ lệch chuẩn. Bổ sung thước đo ổn định ở mức bộ ba theo phương pháp của Méroué và cộng sự (2026): tỉ lệ trùng lặp trong top-10 giữa các lần chạy.

Đây là hướng giá trị nhất vì khoảng cách 0.002 MRR so với NativE quá nhỏ. Nếu độ lệch chuẩn vượt con số đó, kết luận "vượt SOTA" mất cơ sở — một phát hiện đáng công bố. Nếu ổn định, đó là bằng chứng củng cố bài báo. **Kết quả nào cũng có giá trị**, nên rủi ro thấp.

### 8.2 Thay ngưỡng cứng bằng ngưỡng thích ứng

**Nhắm tới:** H3

Thay `penalty_threshold = 10` bằng phân vị của phân phối `penalty_scores` trong từng lô, hoặc bằng tỉ lệ học được theo cách entropy regularization áp dụng. Hình 3 của bài báo đã cho thấy hiệu năng rất nhạy với ngưỡng, tức còn nhiều dư địa. Chi phí thấp: sửa vài chục dòng, chạy 2–3 cấu hình.

### 8.3 Thí nghiệm chéo cấu hình

**Nhắm tới:** H4

Chạy cấu hình DB15K trên MKG-W và ngược lại, lập bảng 2×2. Bốn lần chạy là đủ trả lời dứt điểm câu hỏi mô hình có bị điều chỉnh riêng cho từng tập hay không.

### 8.4 Vector hoá tích luỹ hậu tố — đã kiểm chứng, kết quả trái với dự đoán

**Nhắm tới:** H6

Đây là đề xuất duy nhất trong mục này nhóm đã thực hiện và đo đạc. Kết quả không như dự đoán ban đầu, nên được trình bày đầy đủ ở đây thay vì giữ nguyên như một đề xuất chưa kiểm chứng.

**Phép biến đổi.** Vòng lặp trong `BertSelfAttention.forward`:

```python
penalty_scores = torch.zeros_like(attention_scores)
for i in range(attention_scores.shape[-1]):
    penalty_scores[..., i] = attention_scores[..., i:].prod(dim=-1)
```

viết lại thành một lời gọi duy nhất:

```python
penalty_scores = attention_scores.flip(-1).cumprod(-1).flip(-1)
```

**Tính tương đương: xác nhận.** Đối chiếu trên bốn hình dạng tensor, thêm hai trường hợp biên (có giá trị âm, biên độ lớn). Sai lệch tuyệt đối lớn nhất là 3,8e-06, sai lệch tương đối lớn nhất 1,9e-07 — nằm trong sai số dấu phẩy động của phép nhân liên tiếp.

**Tốc độ: trái với dự đoán.**

| Hình dạng tensor | Vòng lặp | Vector hoá | Tỉ lệ |
|---|---|---|---|
| (1024, 2, 3, 3) — cấu hình APKGC thật | 0,035 ms | 0,053 ms | **0,7×** |
| (1024, 2, 8, 8) | 0,270 ms | 0,099 ms | 2,7× |
| (1024, 2, 32, 32) | 7,081 ms | 1,182 ms | 6,0× |
| (256, 4, 128, 128) | 164,971 ms | 13,354 ms | 12,4× |

Với cấu hình hiện tại của APKGC, bản vector hoá **chậm hơn khoảng 1,4 lần**. Nguyên nhân: chiều cuối chỉ bằng 3 vì mô hình chỉ có ba mô thái, nên chi phí của hai phép `flip` vượt quá lợi ích từ việc gộp lời gọi kernel. Vòng lặp ba bước quá ngắn để chi phí gọi kernel trở thành nút thắt.

Lợi ích chỉ xuất hiện khi chiều cuối đạt từ 8 trở lên, và tăng dần theo độ dài chuỗi.

**Kết luận đã điều chỉnh.** Nhận xét ở mục 6.2 rằng cài đặt "chưa tối ưu" là đúng về mặt hình thức nhưng **không có ý nghĩa thực tiễn ở quy mô hiện tại**. Phép biến đổi này chỉ đáng áp dụng nếu về sau mở rộng số mô thái (mục 8.6) hoặc chuyển sang biểu diễn dạng token như MyGO.

Mã kiểm chứng: `vectorize_penalty.py`.

### 8.5 Mở rộng sang MKG-Y

**Nhắm tới:** H5

Bổ sung bộ dữ liệu thứ ba mà bài báo bỏ qua, cho phép so sánh trực tiếp với các công trình báo cáo đủ ba tập. Lưu ý: cần bộ nhúng MKG-Y; nếu tác giả không chia sẻ thì phải tự trích bằng BEiT và BERT, chi phí tăng đáng kể.

### 8.6 Hướng dài hạn — thêm mô thái thứ tư

**Nhắm tới:** H7

Chính tác giả gợi ý đưa video và âm thanh vào. Hạ tầng dữ liệu đã sẵn: TIVA-KG (ACM MM 2023) cung cấp đủ bốn mô thái kèm mô hình nền QEB. Cơ chế tương quan chéo về nguyên tắc mở rộng được cho n mô thái. Đây là hướng vượt phạm vi một kỳ seminar nhưng đáng nêu.

---

## 9. Kế hoạch tái lập thực nghiệm

### 9.1 Quy trình

```bash
git clone https://github.com/HubuKG/APKGC
pip install -r requirement.txt
# Tải bộ nhúng từ Google Drive vào thư mục embeddings/
bash run.sh
```

Nội dung `run.sh`:

```bash
bash scripts/run_base.sh 0 DB15K 2 0 Mformer_hd_graph 0.2 0.7 epoch 1 2 K001
bash scripts/run_base.sh 0 MKG-W 1 0 Mformer_hd_mean  0.2 0.7 epoch 1 2 W001
```

### 9.2 Các lỗi đã phát hiện trong repo và cách khắc phục

Trong quá trình dựng lại môi trường, nhóm phát hiện **năm vấn đề** khiến mã nguồn không chạy được ngoài môi trường của tác giả. Tất cả đã được khắc phục.

**Lỗi 1 — Thiếu mã nguồn C++.** Đây là lỗi nghiêm trọng nhất. Repo chỉ chứa `mmkgc/release/Base.so` đã biên dịch sẵn, nhưng file này là **ELF 64-bit x86-64**, chỉ chạy trên Linux Intel/AMD. Script `mmkgc/make.sh` cần thư mục `mmkgc/base/` chứa `Base.cpp` và các header để biên dịch lại, nhưng **thư mục này không được commit lên GitHub**.

Nguồn gốc: `mmkgc/` là bản sửa đổi của [OpenKE](https://github.com/thunlp/OpenKE) — thư viện KGC của THUNLP. Thư viện C++ này lo việc lấy mẫu âm và tính chỉ số đánh giá, gọi qua `ctypes`.

Cách khắc phục: lấy `base/` từ OpenKE gốc, đối chiếu chữ ký hàm `sampling()` — khớp đúng 11 tham số mà `TrainDataLoader.py` khai báo — rồi biên dịch lại. Kết quả: biên dịch thành công trên cả Linux x86-64 lẫn macOS ARM64.

```bash
cp -r OpenKE/openke/base mmkgc/base
cd mmkgc && bash make.sh
```

**Lỗi 2 — Gọi `.cuda()` cứng ở 17 vị trí.** Trải trên `APKGC.py`, `Trainer.py`, `Tester.py`, `torchlight/metric.py`. Mã không chạy được trên máy chỉ có CPU hoặc trên Apple Silicon.

Cách khắc phục: tạo `mmkgc/device.py` tự chọn thiết bị theo thứ tự CUDA → MPS → CPU, cho phép ghi đè bằng biến môi trường `APKGC_DEVICE`, rồi thay toàn bộ `.cuda()` bằng `.to(DEVICE)`.

**Lỗi 3 — `requirement.txt` thiếu phụ thuộc.** Thiếu `scikit-learn` (dùng trong `Tester.py`) và `scipy`. Chạy theo đúng hướng dẫn sẽ gặp `ModuleNotFoundError`.

> **Quy ước trích dẫn số dòng.** Mọi số dòng trong báo cáo này chỉ tới **kho mã
> gốc chưa sửa**, commit `9474886` của `github.com/HubuKG/APKGC`, không phải bản
> đã vá của nhóm. Riêng `Reader.h`, `Test.h`, `Base.cpp` **không có trong bản phát
> hành của APKGC** — nhóm lấy từ OpenKE (THUNLP) để biên dịch được, nên số dòng của
> ba tệp đó là của OpenKE.

**Lỗi 4 — Đường dẫn lưu nằm ngoài dự án.** `run_KGC_train.py` ghi kết quả ra `../../data/` — hai cấp trên thư mục repo. Đã đổi thành `./output/`.

**Lỗi 5 — Sập khi chưa tới mốc lưu checkpoint.** `best_model_wts` chỉ được tạo mỗi `save_steps = 100` epoch, nhưng cuối hàm `main` lại truy cập vô điều kiện, gây `AttributeError` với mọi lần chạy dưới 100 epoch. Ngoài ra `Trainer.py` dòng 151 tạo `Tester` với `use_gpu=True` cố định, bỏ qua thiết bị thực tế. Đã sửa cả hai và thêm tham số `-save_steps` để điều chỉnh được từ dòng lệnh.

### 9.3 Xác nhận pipeline chạy thông

Sau khi vá, pipeline chạy thông suốt trên máy Apple Silicon (M-series, không có CUDA):

| Thiết bị | Thời gian mỗi epoch | Ghi chú |
|---|---|---|
| CPU | ~22 giây | `APKGC_DEVICE=cpu` |
| MPS | ~9,2 giây | Nhanh gấp ~2,4 lần |

Loss giảm ổn định qua các epoch (2158,6 → 1718,7 sau 2 epoch), xác nhận quá trình huấn luyện hoạt động đúng. Dữ liệu nạp chính xác: 15.000 thực thể, 169 quan hệ, 34.196 bộ ba huấn luyện trên MKG-W — khớp với Bảng 1 của bài báo.

Nút thắt còn lại nằm ở khâu đánh giá: `run_link_prediction` xếp hạng tuần tự từng bộ ba, không gộp lô, nên với 4.274 bộ ba kiểm thử × 2 chiều × 15.000 thực thể thì rất chậm trên phần cứng không có CUDA. Đây là lý do thực nghiệm đầy đủ nên chạy trên GPU NVIDIA.

### 9.4 Các vấn đề khi chạy thực nghiệm đầy đủ

**Bộ nhúng không nằm trong repo.** Phải tải riêng từ Google Drive của tác giả, bao gồm cả `missing.json` cần thiết cho MKG-W. Dung lượng đáng kể — riêng nhúng thị giác của MKG-W đã khoảng 250 MB.

**Chiều nhúng phải khớp `hidden_size`.** Mô hình đặt `dim_e = dim × 2`, và giá trị này phải bằng `hidden_size` của khối fusion. Với `hidden_size = 256` mặc định thì bắt buộc `dim = 128`. Đặt sai sẽ gặp lỗi nhân ma trận không rõ nguyên nhân — bài báo và README đều không nói điều này.

### 9.5 Kết quả thực nghiệm thu được

Nhóm đã chạy lại thành công APKGC trên bộ MKG-W. Toàn bộ số liệu dưới đây do nhóm tự đo, không lấy từ bài báo.

**Thiết lập.** Chạy trên Kaggle với GPU Tesla P100. Mỗi seed một phiên riêng. Siêu tham số đọc trực tiếp từ `scripts/run_base.sh` của tác giả — không đặt tay — và chỉ thay đổi duy nhất tham số `-seed`. Cụ thể: `dim = 128`, `batch = 1024`, `margin = 12`, `neg_num = 32`, `lr = 1e-4`, `epoch = 8000` với early stopping sau 10 lần đánh giá không cải thiện.

#### Số liệu

Năm lần chạy được thực hiện trong hai đợt với **điều kiện khác nhau**, nên phải
báo cáo tách biệt. Gộp chung sẽ làm lẫn hai nguồn biến thiên: ngẫu nhiên do seed,
và khác biệt do số epoch cùng tần suất đánh giá.

**Đợt một** --- trần 8000 epoch, đánh giá mỗi 100 epoch:

| Seed | MRR | Hits@1 | Hits@3 | Hits@10 | Thời gian |
|---|---|---|---|---|---|
| 7 | 0,3616 | 0,2942 | 0,3918 | 0,4853 | 159 phút |
| 42 | 0,3641 | 0,2956 | 0,3954 | 0,4840 | 158 phút |
| Trung bình | 0,3629 | 0,2949 | 0,3936 | 0,4846 | |

**Đợt hai** --- trần 2000 epoch, đánh giá mỗi 250 epoch (rút ngắn để vừa một
phiên Kaggle):

| Seed | MRR | Hits@1 | Hits@3 | Hits@10 | Thời gian |
|---|---|---|---|---|---|
| 1234 | 0,3610 | 0,2987 | 0,3864 | 0,4759 | 191 phút |
| 2024 | 0,3611 | 0,2943 | 0,3907 | 0,4819 | 192 phút |
| 3407 | 0,3595 | 0,2925 | 0,3917 | 0,4799 | 191 phút |
| **Trung bình** | **0,3605 ± 0,0009** | 0,2952 ± 0,0032 | 0,3896 ± 0,0028 | 0,4792 ± 0,0030 | |

Số công bố trong bài báo: MRR 0,3740 · Hits@1 0,3060 · Hits@3 0,4040 · Hits@10 0,5010

Mọi lần chạy đều dừng nhờ early stopping trước khi chạm trần epoch.

**Kiểm chứng tái lập (26/08/2026).** Sau khi hoàn thành hai đợt trên, nhóm chạy
lại seed 42 với đúng cấu hình đợt một trên một phiên Kaggle độc lập, cách lần chạy
gốc sáu ngày:

| Chỉ số | Lần chạy 20/08 | Lần chạy 26/08 | Lệch |
|---|---|---|---|
| MRR | 0,3641 | 0,3641 | +0,00001 |
| Hits@1 | 0,2956 | 0,2956 | +0,00002 |
| Hits@3 | 0,3954 | 0,3954 | +0,00001 |
| Hits@10 | 0,4840 | 0,4840 | −0,00003 |
| Thời gian | 158 phút | 159 phút | |

Bốn chỉ số trùng khớp đến chữ số thập phân thứ năm. Cả hai lần đều dừng sớm ở
**epoch 1499**, sau 15 lần đánh giá (14 lần giữa chừng cộng một lần cuối
trên checkpoint tốt nhất), dù trần đặt ở 8000.

Điều này xác nhận hai điều. Thứ nhất, quy trình chạy lại của nhóm **có tính tất
định** khi cố định seed --- mọi con số trong báo cáo này đều tái lập được, không
phải kết quả của một lần chạy may mắn. Thứ hai, độ lệch 0,0135 MRR so với số công
bố **không đến từ dao động khi chạy**, vì dao động đó bằng không. Nguyên nhân phải
nằm ở nơi khác --- cấu hình, phần cứng, hoặc phiên bản thư viện.

Kết quả này cũng cho thấy trần 8000 epoch trong `scripts/run_base.sh` không bao
giờ được chạm tới với cấu hình mặc định; early stopping cắt ở khoảng một phần năm
số epoch. Nhật ký đầy đủ lưu tại `experiments/results/apkgc_res3/seed42.log`.

#### Quan sát thứ nhất: mô hình ổn định giữa các seed

Ước lượng độ lệch chuẩn phải lấy từ **đợt hai**, vì đây là ba lần chạy duy nhất
có cùng điều kiện: **0,0009** cho MRR, biên độ 0,0016.

Đối chiếu với các khoảng cách mà bài báo dùng để lập luận:

| Khoảng cách | Giá trị | So với độ lệch chuẩn 0,0009 |
|---|---|---|
| APKGC vượt NativE trên DB15K | 0,002 | gấp 2,2 lần --- vượt nhiễu seed |
| APKGC vượt NativE trên MKG-W | 0,008 | gấp 8,9 lần --- vượt nhiễu seed |
| Đóng góp của attention penalty | 0,014 | gấp 15,6 lần --- vượt nhiễu seed |
| Đóng góp của noise sampling | 0,016 | gấp 17,8 lần --- vượt nhiễu seed |

**Kết quả này bác bỏ giả thuyết ban đầu của nhóm.** Trước khi chạy, nhóm dự đoán mức cải thiện dưới một điểm phần trăm có thể nằm trong dao động ngẫu nhiên giữa các seed. Số liệu cho thấy không phải vậy: APKGC cho kết quả nhất quán, và mọi khoảng cách mà bài báo báo cáo đều vượt mức nhiễu này --- kể cả khoảng cách nhỏ nhất là 0,002 trên DB15K, vốn vẫn gấp hơn hai lần độ lệch chuẩn.

Nhóm ghi nhận điều này thay vì bỏ qua, vì nó là bằng chứng ủng hộ bài báo ở một khía cạnh mà chính bài báo không đo.

**Lưu ý về cách ước lượng.** Ba mẫu là rất ít; con số 0,0009 chỉ mang tính chỉ báo. Nếu gộp cả năm lần chạy thì độ lệch chuẩn là 0,0017, nhưng con số đó không dùng được vì trộn lẫn biến thiên do seed với chênh lệch 0,0023 giữa hai đợt --- vốn đến từ khác biệt về số epoch và tần suất đánh giá, không phải từ tính ngẫu nhiên. Dù lấy con số nào thì kết luận vẫn giữ nguyên: mọi khoảng cách bài báo báo cáo đều vượt mức nhiễu.

#### Quan sát thứ hai: thấp hơn số công bố khoảng 0,011 MRR

Chênh lệch xuất hiện đều ở cả bốn chỉ số và ở **cả năm lần chạy**, bất kể đợt nào:

| Chỉ số | Đợt một (n=2) | Đợt hai (n=3) | Công bố | Lệch của đợt hai |
|---|---|---|---|---|
| MRR | 0,3629 | 0,3605 | 0,3740 | −0,0135 |
| Hits@1 | 0,2949 | 0,2952 | 0,3060 | −0,0108 |
| Hits@3 | 0,3936 | 0,3896 | 0,4040 | −0,0144 |
| Hits@10 | 0,4846 | 0,4792 | 0,5010 | −0,0218 |

Lấy đợt hai làm mốc, độ lệch −0,0135 MRR lớn gấp **mười lăm lần** độ lệch chuẩn giữa các seed (0,0009). Điều này cho thấy đây **không phải dao động ngẫu nhiên** mà là chênh lệch có hệ thống. Độ lệch cũng lớn hơn khoảng cách 0,008 giữa APKGC và NativE trong bảng công bố.

Nhóm **chưa xác định được nguyên nhân**, và có ít nhất bốn cách giải thích chưa loại trừ được:

1. **Cấu hình không khớp.** Bài báo (tr. 15010) ghi batch size 2048 và 64 mẫu âm, trong khi `scripts/run_base.sh` — file mà `run.sh` thực sự gọi — đặt 1024 và 32. Lần chạy này dùng giá trị trong script. Đây là cách giải thích đơn giản nhất và nhiều khả năng nhất.
2. **Phần cứng khác.** Bài báo dùng RTX 4090, nhóm dùng Tesla P100.
3. **Phiên bản thư viện khác.** Phải dùng torch 2.4.1 vì bản mặc định trên Kaggle không hỗ trợ kiến trúc GPU được cấp.
4. **Early stopping.** Cả hai lần đều dừng sớm, có thể trước khi mô hình đạt đỉnh.

#### Thí nghiệm tách nguyên nhân: chạy với đúng cấu hình trong bài báo

Nhóm chạy thêm một lần với `batch = 2048` và `neg_num = 64` --- đúng giá trị ghi ở tr. 15010 --- giữ nguyên mọi tham số khác và dùng seed 42.

Lần chạy này đặt trần 2000 epoch với đánh giá mỗi 250, tức **cùng điều kiện với đợt hai**. Vì vậy nhóm đối chứng phải là đợt hai (0,3605), không phải trung bình gộp cả năm lần chạy.

| Chỉ số | Đợt hai (1024/32) | Cấu hình bài báo (2048/64) | Số công bố |
|---|---|---|---|
| MRR | 0,3605 | **0,3691** | 0,3740 |
| Hits@1 | 0,2952 | **0,3028** | 0,3060 |
| Hits@3 | 0,3896 | **0,3997** | 0,4040 |
| Hits@10 | 0,4792 | 0,4874 | 0,5010 |

Khoảng cách tới số công bố thu hẹp rõ rệt:

| Chỉ số | Lệch với script | Lệch với cấu hình bài báo | Mức thu hẹp |
|---|---|---|---|
| MRR | −0,0135 | −0,0049 | 64% |
| Hits@1 | −0,0108 | −0,0032 | 70% |
| Hits@3 | −0,0144 | −0,0043 | 70% |
| Hits@10 | −0,0218 | −0,0136 | 37% |

**Kết luận: giả thuyết về cấu hình được xác nhận.** Chạy với batch 2048 và 64 mẫu âm thu hẹp 64% khoảng lệch MRR, và 70% ở Hits@1 lẫn Hits@3.

Mức chênh lệch do riêng cấu hình gây ra là **+0,0086 MRR** --- gấp gần mười lần độ lệch chuẩn giữa các seed, nên không thể quy cho ngẫu nhiên.

Điều này cho một phát biểu kiểm chứng được:

> Tệp `run.sh` trong repo gọi `scripts/run_base.sh` với batch 1024 và 32 mẫu âm, trong khi bài báo ghi 2048 và 64. Người chạy đúng theo hướng dẫn trong repo sẽ nhận kết quả thấp hơn bảng công bố, và phần lớn khoảng cách đó đến từ chính sự không khớp này.

**Phần chưa giải thích được.** Sau khi sửa cấu hình vẫn còn lệch 0,0049 MRR --- gấp khoảng năm lần độ lệch chuẩn giữa các seed, nên vẫn mang tính hệ thống. Ba nguyên nhân còn lại chưa tách được:

1. **Phần cứng.** Bài báo dùng RTX 4090, nhóm dùng Tesla P100.
2. **Phiên bản thư viện.** torch 2.4.1 thay vì bản tác giả dùng.
3. **Số epoch bị giới hạn.** Lần chạy này đặt trần 2000 epoch với đánh giá mỗi 250, thay vì 8000 epoch với đánh giá mỗi 100 như script gốc, để vừa giới hạn thời gian một phiên Kaggle. Đánh giá thưa hơn có thể bỏ lỡ điểm cực đại.

Đáng chú ý là Hits@10 chỉ thu hẹp 37%, thấp hơn hẳn ba chỉ số còn lại. Điều này gợi ý phần lệch còn lại ảnh hưởng tới đuôi phân phối xếp hạng nhiều hơn tới phần đầu.

**Giới hạn của thí nghiệm này.** Chỉ chạy một seed nên chưa có độ lệch chuẩn riêng cho cấu hình này; con số 0,3691 nên được đọc như một điểm quan sát, không phải giá trị trung bình. Ngoài ra seed 42 không có mặt trong đợt hai, nên phép so sánh vẫn lẫn một phần biến thiên do seed --- tuy phần này nhỏ, chỉ khoảng 0,0009 so với mức chênh 0,0086 cần giải thích.

#### Sáu rào cản đã phải khắc phục

Ghi lại đầy đủ vì đây là phần công việc thực tế chiếm nhiều thời gian nhất.

**Một — thiếu mã nguồn C++.** Đây là rào cản lớn nhất. Script `mmkgc/make.sh` cần thư mục `mmkgc/base/` chứa `Base.cpp` cùng các header, nhưng thư mục này không có trong repo. File `mmkgc/release/Base.so` được commit sẵn là bản biên dịch cho Linux x86-64:

```
$ file mmkgc/release/Base.so
ELF 64-bit LSB shared object, x86-64
```

Trên máy kiến trúc khác thì không nạp được, mà cũng không biên dịch lại được vì thiếu mã nguồn.

Cách khắc phục: nhóm nhận ra `mmkgc/` là bản sửa đổi của [OpenKE](https://github.com/thunlp/OpenKE) — thư viện KGC của THUNLP — dựa trên cấu trúc thư mục và giao diện `ctypes`. Lấy `base/` từ repo gốc, đối chiếu chữ ký hàm `sampling()` thấy khớp đúng 11 tham số mà `TrainDataLoader.py` khai báo, rồi biên dịch lại. Thành công trên cả ARM64 lẫn x86-64.

```bash
cp -r OpenKE/openke/base mmkgc/base
cd mmkgc && bash make.sh
```

**Hai — gọi `.cuda()` cứng ở 17 vị trí.** Trải trên `APKGC.py`, `Trainer.py`, `Tester.py` và `torchlight/metric.py`. Mã không chạy được trên máy không có GPU NVIDIA. Nhóm thay bằng một module chọn thiết bị theo thứ tự CUDA, MPS, CPU.

**Ba — thiếu phụ thuộc.** `requirement.txt` không liệt kê `scikit-learn` và `scipy`, nhưng `Tester.py` cần chúng. Chạy đúng theo hướng dẫn sẽ gặp `ModuleNotFoundError`.

**Bốn — đường dẫn lưu nằm ngoài dự án.** `run_KGC_train.py` ghi kết quả ra `../../data/`, tức hai cấp trên thư mục repo.

**Năm — sập khi chạy dưới 100 epoch.** `best_model_wts` chỉ được tạo tại mốc lưu checkpoint (mặc định mỗi 100 epoch) nhưng lại được truy cập vô điều kiện ở cuối chương trình, gây `AttributeError`.

**Sáu — ràng buộc phiên bản torch quá lỏng.** Repo ghim `torch>=1.6.0` không có cận trên. Bản torch mặc định trên Kaggle hiện nay (2.10) không còn hỗ trợ kiến trúc sm_60 của Tesla P100, gây lỗi `no kernel image is available for execution on the device`. Phải hạ xuống torch 2.4.1.

#### Chi phí tính toán — nút thắt nằm ở khâu đánh giá

Nhóm đo riêng từng khâu:

| Khâu | Thời gian |
|---|---|
| Huấn luyện | khoảng 8,4 giây mỗi epoch |
| Đánh giá | khoảng 21 phút mỗi lần |

Hàm `run_link_prediction` xếp hạng tuần tự 4.274 bộ ba kiểm thử theo hai chiều, mỗi lần trên 15.000 thực thể, không gộp lô. Với cấu hình gốc — 8000 epoch, đánh giá mỗi 100 epoch — riêng khâu đánh giá cần khoảng 28 giờ nếu chạy hết.

Điều này giải thích một quan sát ban đầu: lần chạy gộp cả năm seed trong một kernel bị Kaggle cắt khi hết giới hạn 12 giờ mà chưa xong seed đầu tiên. Sau đó nhóm tách thành các kernel riêng, mỗi kernel một seed, và cả hai đều hoàn tất trong khoảng 158 phút nhờ early stopping.

#### Cách tái lập kết quả này

Notebook `kaggle_apkgc_seeds.ipynb` tự động hoá toàn bộ: bổ sung `mmkgc/base/` từ OpenKE, biên dịch lại `Base.so`, thay các lời gọi `.cuda()`, cài phụ thuộc còn thiếu, tải bộ nhúng từ Google Drive, và hạ phiên bản torch khi phát hiện GPU không tương thích.

Chi tiết đầy đủ nằm trong tài liệu `KetQuaChayLai.pdf` đi kèm.

### 9.6 Việc chưa làm

Hai thí nghiệm nhóm dự kiến nhưng chưa thực hiện:

1. **Chạy nhiều seed với cấu hình bài báo.** Thí nghiệm tách nguyên nhân ở trên chỉ dùng một seed, nên chưa có độ lệch chuẩn riêng.
2. **Chạy đủ 8000 epoch với đánh giá mỗi 100** như script gốc, để loại trừ khả năng phần lệch còn lại đến từ việc đánh giá thưa.
3. **Thêm bộ dữ liệu.** Toàn bộ số liệu hiện có đều trên MKG-W; chưa chạy DB15K hay MKG-Y.

---

## 10. Kết luận

APKGC giải quyết hai vấn đề trong hoàn thiện đồ thị tri thức đa mô thái bằng hai cơ chế: lấy mẫu nhiễu thích ứng để tăng tính bền vững, và hình phạt chú ý kèm tái phân bổ để sửa hiện tượng over-trust của mô hình ngôn ngữ.

Xét về con số, mức cải thiện khiêm tốn — dưới một điểm phần trăm tuyệt đối, và mô hình còn thua ở một chỉ số. Xét về đóng góp tư duy, bài báo đáng chú ý ở hai điểm: cách đặt lại vai trò của nhiễu, và việc can thiệp trực tiếp vào ma trận chú ý theo hướng phạt rồi trả lại — một dạng chỉnh sửa hiếm gặp trong các công trình MMKGC.

### Những gì phần thực nghiệm cho biết thêm

Việc chạy lại đưa tới ba nhận định mà chỉ đọc bài báo thì không có được.

**Mô hình ổn định hơn nhóm dự đoán.** Độ lệch chuẩn MRR giữa hai seed là 0,0018, nhỏ hơn nhiều so với mọi khoảng cách mà bài báo báo cáo. Nhóm vào việc với giả thuyết rằng mức cải thiện dưới một điểm phần trăm có thể nằm trong nhiễu ngẫu nhiên; số liệu bác bỏ điều đó. Đây là bằng chứng ủng hộ bài báo, ở một khía cạnh mà chính bài báo không đo.

**Có khoảng cách giữa mã nguồn công bố và kết quả công bố.** Chạy đúng theo script cho MRR thấp hơn bảng khoảng 0,011 — lớn hơn cả khoảng cách 0,008 với á quân. Nhóm chưa xác định được nguyên nhân trong bốn khả năng đã liệt kê, nên không kết luận gì thêm. Nhưng riêng việc bài báo ghi batch 2048 và 64 mẫu âm trong khi script đặt 1024 và 32 đã là một điểm cần được làm rõ.

**Rào cản tái lập nằm ở khâu kỹ thuật, không phải ở ý tưởng.** Sáu vấn đề phải khắc phục trước khi chạy được dòng đầu tiên, trong đó nghiêm trọng nhất là repo thiếu hẳn mã nguồn C++ cần cho việc biên dịch. Không có bước truy ngược về OpenKE thì mã không chạy được trên bất kỳ máy nào khác kiến trúc x86-64.

### Đánh giá chung

Với mục tiêu học tập, bài báo phù hợp: kiến trúc đủ gọn để hiểu tường tận, mã nguồn chạy được trên một GPU, và các mô-đun đều có thể tháo lắp để thử nghiệm cải tiến.

Điều nhóm rút ra ngoài nội dung chuyên môn: đọc mã nguồn cho biết những thứ mà đọc bài báo không cho biết — cả điểm mạnh lẫn điểm yếu. Đồng thời, mã nguồn chỉ phản ánh những gì tác giả công bố, không phản ánh toàn bộ quá trình nghiên cứu của họ. Mọi nhận định trong báo cáo này vì vậy đều được phát biểu ở dạng *bài báo không báo cáo điều gì* hoặc *mã nguồn công bố có đặc điểm gì*, chứ không suy diễn về việc tác giả đã làm hay không làm gì.

---

## Tài liệu tham khảo chính

1. Jian, Y.; Luo, X.; Li, Z.; Zhang, M.; Zhang, Y.; Xiao, K.; Hou, X. **APKGC: Noise-enhanced Multi-Modal Knowledge Graph Completion with Attention Penalty.** AAAI 2025, 15005–15013.
2. Bordes, A. và cộng sự. **Translating Embeddings for Modeling Multi-relational Data.** NeurIPS 2013.
3. Sun, Z.; Deng, Z.-H.; Nie, J.-Y.; Tang, J. **RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.** ICLR 2019.
4. Bao, H.; Dong, L.; Piao, S.; Wei, F. **BEiT: BERT Pre-Training of Image Transformers.** ICLR 2022.
5. Huang, Q. và cộng sự. **OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation.** CoRR abs/2311.17911, 2023.
6. Zhang, Y. và cộng sự. **NativE: Multi-modal Knowledge Graph Completion in the Wild.** SIGIR 2024.
7. Zhang, Y. và cộng sự. **Unleashing the Power of Imbalanced Modality Information for Multi-modal Knowledge Graph Completion.** COLING 2024.
