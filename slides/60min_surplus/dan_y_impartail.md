# Dàn ý trình bày — ImparTail

**Bài báo:** Towards Unbiased and Robust Spatio-Temporal Scene Graph Generation
and Anticipation — Peddi, Saurabh, Shrivastava, Singla, Gogate. CVPR 2025
(Highlight).

**Bộ slide:** `slides/ImparTail_Slides.pdf`, 26 trang. **Thời lượng:** 60 phút.

---

## Phân bổ thời gian

| Phần | Slide | Phút | Cộng dồn |
|---|---|---|---|
| 1. Mở đầu | 1–2 | 2 | 2 |
| 2. Kiến thức nền | 3–5 | 9 | 11 |
| 3. Vấn đề đuôi dài | 6–7 | 7 | 18 |
| 4. Phương pháp | 8–12 | 17 | 35 |
| 5. Đóng góp thứ ba (độ bền) | 13 | 6 | 41 |
| 6. Thực nghiệm | 14–16 | 8 | 49 |
| 7. Đọc phản biện | 17–19 | 8 | 57 |
| 8. Kết & hỏi đáp | 20 | 3 | 60 |

Nếu cháy giờ: cắt slide 7 (các hướng đã có) và rút gọn slide 16 xuống 2 phút.
**Không cắt** slide 12 (bộ ba đi xuyên mô hình) và 17 (phát hiện về curriculum) — đó là
hai chỗ người nghe nhớ nhất.

---

## Phần 1 — Mở đầu (2 phút)

Mở bằng một câu hỏi, không phải bằng tên bài báo:

> "Nhìn một đoạn video người ta pha cà phê. Máy tính cần trả lời được: ai đang
> làm gì với cái gì, và **sắp** làm gì tiếp. Đó là bài toán hôm nay."

Nói rõ đây là **CVPR 2025 Highlight** — top khoảng 3% số bài nộp. Rồi vào luôn.

---

## Phần 2 — Kiến thức nền (9 phút)

### Slide 3 — Scene graph: ảnh tĩnh sang video (3 phút)

Bắt đầu từ cái quen thuộc: scene graph tĩnh là đồ thị, đỉnh là vật thể, cạnh là
quan hệ. Một câu về Visual Genome.

Rồi chỉ vào hình ba khung: **cùng một cặp (người, cốc)**, quan hệ đổi từ *nhìn* →
*chạm* → *uống*.

> "Đây là thứ scene graph tĩnh không diễn tả được. Ba tấm ảnh rời cho ba đồ thị
> rời. Cái ta cần là **một chuỗi đồ thị có liên hệ với nhau theo thời gian**."

### Slide 4 — Ký hiệu (3 phút)

Đi chậm, vì cả bài dùng lại. Viết lên bảng nếu có:

- Đoạn video $V_1^T$ gồm $T$ khung
- Trong khung $t$: vật thể $o_k^t$, mỗi cái có hộp bao và nhãn lớp
- Vị ngữ $p_{ijk}^t$ nối cặp $(o_i^t, o_j^t)$
- **Bộ ba** $r_{ijk}^t = (o_i^t, p_{ijk}^t, o_j^t)$ — đây là đơn vị cơ bản

Rồi hai tác vụ, nhấn vào chỗ khác nhau:

> "VidSGG nhìn khung nào thì mô tả khung đó. SGA phải **đoán khung chưa nhìn
> thấy**. Cùng bộ máy, nhưng SGA khó hơn hẳn."

### Slide 5 — Ví dụ đầu vào (3 phút)

Đây là slide đầu tiên người nghe thực sự *thấy* bài toán. Đi từ trái sang phải:

1. Hai khung đã quan sát → VidSGG sinh (người, **looking_at**, sổ tay)
2. Mũi tên SGA
3. Hai khung chưa quan sát → dự đoán (người, **touching**, cốc) rồi
   (người, **drinking_from**, cốc)

> "Chú ý chuỗi *chạm* rồi *uống*. Mô hình phải học rằng chạm vào cốc thường dẫn
> tới uống. Đó là tri thức về **diễn tiến**, không phải về một khung hình."

---

## Phần 3 — Vấn đề đuôi dài (7 phút)

### Slide 6 — Phân phối đuôi dài (4 phút)

Chỉ vào biểu đồ cột. Bốn cột xanh cao vống, phần đuôi đỏ dẹt lét.

Câu chốt của cả phần này:

> "Vấn đề không phải là đuôi dài. Vấn đề là **lớp ở đuôi mới là lớp đáng giá**.
> Biết 'người ở phía trước cái cốc' gần như vô dụng. Biết 'người đang uống từ
> cái cốc' mới là hiểu cảnh. Mà mô hình lại học ngược lại."

Giải thích vì sao **mean Recall** chứ không Recall:

> "Recall@K cộng gộp mọi dự đoán, nên lớp head chiếm đa số và che mất điểm yếu
> ở đuôi. mean Recall tính recall **từng lớp rồi lấy trung bình** — mỗi lớp một
> phiếu. Đổi chỉ số là đổi cả bài toán."

### Slide 7 — Các hướng đã có (3 phút) — *cắt được nếu thiếu giờ*

Lướt nhanh bảng bốn hướng. Dừng ở dòng cuối:

> "Ba hướng trên đều **thêm tham số hoặc thêm mô-đun**. ImparTail không thêm gì
> cả — nó chỉ đổi hàm mất mát. Đó là lý do nó gắn được lên sáu mô hình nền khác
> nhau."

---

## Phần 4 — Phương pháp (17 phút, trọng tâm)

### Slide 8 — Toàn cảnh pipeline (3 phút)

Chỉ vào dấu ngoặc lớn: **ba khối đầu lấy nguyên từ mô hình nền**. Chỉ khối cuối
là của ImparTail.

> "Nói cách khác: forward pass **không đổi một dòng nào**. Chỉ backward đổi."

### Slide 9 — Masked Loss (5 phút, chậm nhất phần này)

Viết công thức lên bảng:

$$\widetilde{\mathcal{L}}_{p} = (1 - m) \ast \mathcal{L}_{p}$$

Rồi giải thích bằng lời trước khi giải thích bằng ký hiệu:

> "$m$ là một cái công tắc nhị phân cho **từng bộ ba một**. Bật lên thì mất mát
> của bộ ba đó bằng không — nó biến mất khỏi gradient, như thể không tồn tại
> trong batch này."

Nhấn chỗ khác biệt với cost-sensitive learning:

> "Đánh trọng số lại là *giảm tiếng nói* của lớp head. Che là **tắt hẳn**. Nghe
> thô, nhưng chính vì thô mà không phải dò tham số trọng số."

Chỉ vào hình bên phải: hai thanh xám gạch đứt là head bị che, hai thanh đỏ là
tail còn nguyên.

### Slide 10 — Hàm mục tiêu (3 phút)

Đừng đọc từng số hạng. Chỉ nói hai ý:

1. VidSGG có hai số hạng: vật thể và vị ngữ. **Chỉ vị ngữ bị che.**
2. SGA có năm số hạng. **Hai chỗ bị che**: vị ngữ ở khung đã quan sát và vị ngữ
   ở khung dự đoán. Hộp bao và tái tạo giữ nguyên.

> "Vì sao chỉ che nhánh vị ngữ? Vì chỉ nhánh đó mới bị đuôi dài. Phân bố lớp vật
> thể cân bằng hơn nhiều."

### Slide 11 — Curriculum (3 phút)

Ba thành phần: xếp hạng theo lượng dữ liệu, đo thành thục bằng recall, trộn dần
theo epoch. Tỉ lệ che $\mathcal{R}_m = e \times \mathcal{R}_s$ tăng dần.

Chỉ vào ba biểu đồ bên phải: phân phối nhãn còn lại dịch từ đuôi dài sang đều.

**Gieo mầm cho slide 17** — đừng nói hết ngay:

> "Ghi nhớ slide này. Lát nữa ta sẽ quay lại xem nó có thực sự được dùng không."

### Slide 12 — Một bộ ba đi xuyên mô hình (3 phút)

Đi từng bước với một bộ ba cụ thể:

1. Khung $I^{12}$ có người và cốc
2. ORPU cho đặc trưng, hộp bao, phân phối lớp
3. STPU cho biểu diễn quan hệ $z_{ij}^{12}$
4. Decoder cho phân phối vị ngữ
5. Nhãn thật là **looking_at** → thuộc head → $m = 1$ → **mất mát bằng 0**

> "Bộ ba này vừa đi hết mô hình rồi **không đóng góp gì cho việc học**. Nếu nhãn
> là *drinking_from* thì ngược lại — gradient chảy qua bình thường."

---

## Phần 5 — Đóng góp thứ ba (6 phút)

### Slide 13 — Robust VidSGG và Robust SGA (6 phút)

Giải thích thiết kế: **huấn luyện trên dữ liệu sạch, đánh giá trên dữ liệu
nhiễu**. Không phải huấn luyện tăng cường.

> "Đây mô phỏng đúng tình huống triển khai thật: mô hình được huấn luyện trong
> phòng, rồi đem ra chỗ có sương mù, chói nắng, camera nhiễu."

16 kiểu nhiễu, ba mức. Nhóm lại cho dễ nhớ: nhiễu cảm biến, thời tiết, ánh sáng.

Nhấn: **đây là hai tác vụ mới bài báo đặt ra**, không phải chỉ là thí nghiệm phụ.

---

## Phần 6 — Thực nghiệm (8 phút)

### Slide 14 — Thiết lập (2 phút)

Nói cách đọc bảng trước khi chiếu bảng:

> "Mỗi dòng là một mô hình nền, dòng ngay dưới là **chính mô hình đó** sau khi
> gắn ImparTail. So theo cặp, đừng so chéo giữa các mô hình."

Ba chế độ VidSGG khác nhau ở lượng thông tin cho sẵn: SGDET chỉ có khung hình,
SGCLS thêm hộp bao, PREDCLS thêm cả nhãn vật thể.

### Slide 15 — Kết quả VidSGG (3 phút)

Đừng đọc số. Chỉ vào các ô xanh:

> "Cải thiện ở **mọi ô**, không có ngoại lệ. Mức tăng mR@10 khoảng 12% cho
> SGDET, 22% cho SGCLS, trên 20% cho PREDCLS."

Nếu có người hỏi vì sao PREDCLS tăng nhiều nhất: vì cho sẵn nhãn vật thể nên phần
khó còn lại đúng là phân loại vị ngữ — chính chỗ ImparTail tác động.

### Slide 16 — SGA và độ bền (3 phút)

Bảng SGA: khoảng 12% cho STTran++/DSGDetr++, 20% cho SceneSayer.

Biểu đồ độ bền là chỗ đáng dừng:

> "Cả hai đều tụt khi gặp nhiễu — đó là bình thường. Điều đáng nói là mô hình
> ImparTail **tụt ít hơn**: 8,0 điểm so với 14,3. Và điểm sau khi nhiễu của nó
> vẫn **cao hơn điểm sạch** của mô hình nền."

Giải thích cơ chế theo bài báo: cân bằng phân phối làm mô hình không quá khớp
vào mẫu hình phổ biến, nên nhiễu ít phá hơn — giống tác dụng của dropout.

---

## Phần 7 — Đọc phản biện (8 phút)

### Slide 17 — Curriculum không được dùng để lấy số (4 phút)

Đây là điểm nhấn của cả bài nói. Dựng lên từ từ.

Nhắc lại slide 11, rồi chiếu nguyên văn Remark trang 7. Đọc to đoạn
*"we opted for a fixed Masking Ratio of 0.9"*.

Rồi đưa ba tầng bằng chứng:

| Nguồn | Bằng chứng |
|---|---|
| Bài báo | Remark trang 7 nói rõ dùng tỉ lệ cố định |
| Mã nguồn | `train_sgg_methods.py:402` → `is_curriculum = False` là mặc định |
| Script chạy | `METHODS=("sttran" "dsgdetr")` — không script nào gọi curriculum |

**Phát biểu cẩn thận** — đây là chỗ dễ nói quá:

> "Tôi chỉ đang mô tả **bài báo và mã đã công bố**. Không kết luận tác giả có
> chạy curriculum hay không trong quá trình nghiên cứu — rất có thể họ có chạy
> mà chỉ không đưa vào bản phát hành."

Câu hỏi để lại:

> "Nếu tỉ lệ cố định cho kết quả tương đương curriculum, thì đóng góp thật nằm ở
> curriculum hay chỉ ở việc che mất mát? Bài báo chưa tách được hai thứ đó."

### Slide 18 — Quan sát khác khi đọc mã (2 phút)

Lướt nhanh: đường dẫn cứng của bốn máy, không cố định seed, rào cản chạy lại.

Nói rõ vì sao nhóm không chạy lại được — và **không đổ cho mã hỏng**:

> "Không phải mã sai. Là chi phí dữ liệu: hàng chục GB video Charades, bước tách
> khung hình, và `torch 1.10.1+cu111` — bản năm 2021 mà GPU hiện đại không còn
> hỗ trợ."

Ghi nhận điểm cộng: giấy phép Apache 2.0 rõ ràng, có phát hành checkpoint — khá
hơn nhiều repo cùng lĩnh vực.

### Slide 19 — Mạnh yếu (2 phút)

Đọc lướt hai cột. Dừng ở dòng "tỉ lệ che 0,9":

> "Che 90% nhãn mà vẫn học được — con số này tự nó đã đáng ngạc nhiên. Bài báo
> không khảo sát vì sao lại là 0,9."

---

## Phần 8 — Kết (3 phút)

### Slide 20 — Ba điều mang về

Nói chậm, mỗi ý một câu:

1. **Đổi mục tiêu, đừng đổi kiến trúc.** Một phép nhân với mặt nạ nhị phân, không
   thêm tham số nào, cải thiện 12–22%.
2. **Chỉ số quyết định điều mình nhìn thấy.** Recall giấu điểm yếu ở đuôi, mean
   Recall phơi ra.
3. **Đọc bài báo cùng với mã.** Phát hiện về curriculum chỉ hiện ra khi đọc cả
   hai; đọc riêng cái nào cũng không thấy.

---

## Câu hỏi có thể bị hỏi

**1. Che 90% nhãn thì còn gì để học?**
Che theo **bộ ba**, không theo mẫu. Một khung hình có nhiều cặp vật thể; che hết
các cặp head vẫn còn các cặp tail. Và mất mát vật thể không bị che.

**2. Khác gì focal loss?**
Focal loss giảm trọng số mẫu dễ theo **độ tin cậy dự đoán**, liên tục và tự động.
ImparTail che theo **lớp**, nhị phân, và theo lịch trình định trước. Bài báo không
so trực tiếp với focal loss — đó là một thiếu sót.

**3. Vì sao không huấn luyện tăng cường với nhiễu cho robust?**
Vì tác vụ họ đặt ra cố tình mô phỏng triển khai thật: không biết trước sẽ gặp
nhiễu gì. Huấn luyện tăng cường sẽ trả lời một câu hỏi khác.

**4. Cải thiện có nằm trong nhiễu ngẫu nhiên không?**
**Không kiểm chứng được** — bài báo không báo cáo độ lệch chuẩn, không nói chạy
mấy seed. Với mức tăng 12–22% thì khả năng là thật, nhưng đúng ra phải có nhiều
seed.

**5. Nhóm có chạy lại không?**
Không, và lý do là chi phí dữ liệu chứ không phải mã hỏng — xem slide 18. Nhóm
có chạy lại đầy đủ bài APKGC trong phần báo cáo.

**6. ImparTail có gắn được lên mô hình mới hơn không?**
Về nguyên tắc có, miễn mô hình đó dùng pipeline biểu diễn hướng-đối-tượng và có
một số hạng mất mát phân loại vị ngữ tách rời. Bài báo đã chứng minh trên sáu mô
hình nền.

**7. Vì sao chọn Action Genome?**
Đây là bộ chuẩn de-facto cho VidSGG và SGA, chú giải trên video Charades. Nhưng
**chỉ một bộ dữ liệu** cũng là điểm yếu — không biết kết quả có chuyển sang bộ
khác không.
