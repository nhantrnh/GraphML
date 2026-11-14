# Dàn ý trình bày — APKGC

Kịch bản cho 60 phút, bám theo 27 trang slide. Cột thời gian là mốc tích luỹ.

---

## Phân bổ thời gian

| Phần | Slide | Thời lượng | Mốc |
|---|---|---|---|
| Mở đầu | 1–2 | 2 phút | 0:00 |
| Kiến thức nền | 4–6 | 8 phút | 0:02 |
| Hai vấn đề | 8–9 | 7 phút | 0:10 |
| Kiến trúc | 11–14 | 16 phút | 0:17 |
| Ví dụ chạy | 16 | 8 phút | 0:33 |
| Thực nghiệm bài báo | 18–20 | 8 phút | 0:41 |
| **Thực nghiệm của nhóm** | 22–23 | 9 phút | 0:49 |
| Đánh giá, cải tiến, kết | 25–27 | 6 phút | 0:58 |

Chừa 3–5 phút cuối cho câu hỏi. Nếu bị trễ, phần cắt được là mục 5.4 (siêu tham số) và mục 8 (hướng cải tiến) — nói gọn trong một câu mỗi mục.

---

## Phần 1 — Mở đầu (2 phút)

Slide 1–2.

Không đọc lại tiêu đề. Mở bằng một câu định vị vấn đề:

> "Bài hôm nay giải quyết hai chuyện tưởng như không liên quan: làm sao mô hình chịu được dữ liệu bẩn, và làm sao ngăn cơ chế attention nhìn nhầm chỗ. Điểm thú vị là cả hai đều được giải bằng cách can thiệp vào thứ mà người ta thường để yên."

Rồi lướt nhanh mục lục, nhấn: *"Phần cuối là số liệu nhóm tự chạy, không lấy từ bài báo."*

---

## Phần 2 — Kiến thức nền (8 phút)

### Slide 4 — Đồ thị tri thức

Chỉ vào hình. Nói theo mạch:
- Mỗi mũi tên là một bộ ba
- Mũi tên đỏ đứt nét là thứ cần dự đoán
- **Nhấn:** đây là bài toán *xếp hạng*, không phải phân loại — quyết định cách chọn độ đo, và cũng là lý do khâu đánh giá tốn thời gian (sẽ nói lại ở phần thực nghiệm)

### Slide 5 — TransE và RotatE

Đây là chỗ **dễ mất thời gian nhất**, phải kiểm soát.

Cách nói gọn:
- TransE: quan hệ là phép cộng vector. Hỏng với quan hệ đối xứng — nếu `r` đối xứng thì buộc `r ≈ 0`
- RotatE: quan hệ là phép quay. Đối xứng thành quay 180 độ, giải quyết được

**Không** đi sâu vào toán phức. Chỉ cần người nghe hiểu: APKGC dùng RotatE làm hàm chấm điểm, không phát minh lại phần này.

### Slide 6 — Ba mô thái

Nhấn hai điểm:
1. Ảnh và văn bản **đóng băng** — APKGC chỉ học cách chiếu và hợp nhất. Đây là lý do chạy được trên một GPU
2. Chi tiết `missing.json`: khi thiếu mô thái, họ lấp bằng vector sinh từ phân phối của các thực thể còn lành

Điểm 2 dẫn tự nhiên sang phần sau: *"cách xử lý này đã hé lộ tinh thần của bài — nhiễu không bị coi là kẻ thù."*

---

## Phần 3 — Hai vấn đề (7 phút)

### Slide 8 — Nhiễu

Cách kể hiệu quả: đặt câu hỏi ngược.

> "Nếu dữ liệu bẩn, bạn sẽ làm gì? Phản xạ đầu tiên là lọc. Bài này làm ngược lại."

Rồi giải thích ba điểm hụt của việc lọc, và liên hệ dropout: *"ai từng dùng dropout thì sẽ thấy quen — tắt ngẫu nhiên nơ-ron để mạng không phụ thuộc một đường dẫn. APKGC làm thế ở mức mô thái."*

### Slide 9 — Attention over-trust

Chỉ vào biểu đồ. Nói:
- Cột đỏ là dấu chấm, nhận điểm chú ý 0.66
- Danh từ riêng — chủ thể thật của câu — gần như bị bỏ qua

**Nhấn:** đây không phải vấn đề tác giả tự đặt ra để giải. Nó được ghi nhận độc lập trong nghiên cứu về ảo giác đa mô thái, và liên quan tới khái niệm *attention sink*.

---

## Phần 4 — Kiến trúc (16 phút) — phần trọng tâm

### Slide 11 — Toàn cảnh

Đi từ trái sang phải theo sơ đồ, mỗi khối một câu. Nói rõ: **hai đóng góp nằm ở khối (a) và (b); khối (c) kế thừa RotatE.**

Đừng dừng lâu ở slide này — nó chỉ để định hướng.

### Slide 12 — Adaptive Noise Sampling

Hai ý:
1. Nhiễu **không tuỳ tiện** — lấy trung bình và độ lệch của chính tập dữ liệu, nên vector nhiễu vẫn nằm trong vùng hợp lý
2. Mỗi thực thể tung đồng xu riêng, ngưỡng `θ_n` được mô tả là học được

Ở ý 2, gieo sẵn: *"chỗ này nhóm có kiểm chứng lại, sẽ nói ở phần thực nghiệm."* Đừng tiết lộ trước.

### Slide 13 — Attention Penalty (dừng lâu nhất, ~7 phút)

Đây là phần cốt lõi. Ba nhịp, mỗi nhịp giải thích *tại sao* chứ không chỉ *làm gì*.

**Nhịp 1 — phát hiện.** Câu hỏi then chốt cần trả lời trước khi ai đó hỏi:

> "Vì sao dùng tích luỹ chứ không dùng thẳng điểm attention?"

Trả lời: phép nhân khuếch đại. Token cao **dai dẳng** mới đẩy tích vượt ngưỡng; token cao **nhất thời** thì không. Nếu chỉ so điểm số đơn lẻ thì bắt nhầm cả hai.

**Nhịp 2 — phạt.** Trừ theo tỉ lệ, có `clamp` giới hạn biên độ.

**Nhịp 3 — tái phân bổ.** Nhấn mạnh đây là chỗ phân biệt APKGC với các cách chỉ triệt tiêu: phần chú ý bị tịch thu được **chia lại**, không biến mất.

Chốt bằng bằng chứng ablation: bỏ nhánh tăng còn 0.365, bỏ nhánh giảm còn 0.369, đủ cả hai được 0.374. **Chỉ phạt thôi chưa đủ.**

### Slide 14 — Hợp nhất

Nói ngắn. Điểm đáng nhấn: hệ số `θ_ab` cân giữa "tin vào mình" và "nghe mô thái kia" — nếu ảnh mờ thì tự nghiêng về văn bản.

Nếu thiếu thời gian, cắt bảng sáu phương án hợp nhất, chỉ nói: *"mã có sáu cách hợp nhất, hai tập dữ liệu dùng hai cách khác nhau — điểm này sẽ quay lại ở phần đánh giá."*

---

## Phần 5 — Ví dụ chạy (8 phút)

Slide 16. Đây là phần **người nghe dễ theo nhất**, nên đi chậm.

Kể như một câu chuyện:
1. Đầu vào: hỏi ai thắng giải diễn viên phụ xuất sắc
2. Tra ba vector, tung đồng xu — lần này ρ = 0.31 > 0.2 nên tiêm nhiễu vào vector ảnh
3. Chiếu về cùng chiều, tính attention — token "for" nhận 0.66, đây chính là over-trust
4. Tích luỹ hậu tố cho 12.4, vượt ngưỡng 10 → bị phạt xuống 0.50
5. Kết quả: không có penalty thì đáp án đúng xếp nhì; có penalty thì lên nhất

Chốt: *"đây là ví dụ trong bài báo, không phải nhóm dựng ra."*

---

## Phần 6 — Thực nghiệm trong bài báo (8 phút)

### Slide 18 — So sánh SOTA

Đọc bảng nhanh, rồi **nói thẳng ba điều bài báo không nhấn**:
1. Mức tăng 3.3% là **tương đối**; tuyệt đối chỉ khoảng 0.9 điểm phần trăm
2. Ở Hits@10 trên DB15K, APKGC **thua** NativE — bài báo gạch chân chứ không in đậm, và không thảo luận
3. Bảng **không có độ lệch chuẩn**

Điểm 3 là cầu nối sang phần thực nghiệm của nhóm.

### Slide 19 — Ablation

Hai quan sát:
- Chỉ dùng cấu trúc là tệ nhất rất xa → bằng chứng cho hướng đa mô thái
- **Lấy mẫu nhiễu đóng góp nhiều hơn attention penalty** (0.016 so với 0.014) — thú vị vì tiêu đề bài báo đặt attention penalty ở vị trí nổi bật hơn

### Slide 20 — Siêu tham số

Nói gọn: đường cong hình chữ U ngược, phạt quá nhẹ hoặc quá nặng đều hỏng. Tại `θ_p = 0.1` còn thấp hơn cả khi không phạt.

**Đây là mục cắt được nếu trễ giờ.**

---

## Phần 7 — Thực nghiệm của nhóm (9 phút) — phần ăn điểm

### Slide 22 — Kết quả 5 seed

Mở bằng cách nói rõ đây là số tự đo:

> "Bài báo không báo cáo độ lệch chuẩn, nên nhóm tự chạy năm seed để xem con số đó là bao nhiêu."

Đọc bảng, rồi trình bày **hai kết luận**:

**Một — mô hình ổn định.** Độ lệch chuẩn 0.0017. Mọi khoảng cách bài báo báo cáo đều vượt nhiễu seed, kể cả khoảng cách nhỏ nhất 0.002.

Nói thẳng: *"Nhóm vào việc với giả thuyết ngược lại — rằng mức cải thiện dưới một điểm phần trăm có thể nằm trong nhiễu. Số liệu bác bỏ điều đó. Đây là bằng chứng ủng hộ bài báo, ở khía cạnh chính bài báo không đo."*

Chỗ này quan trọng: **thừa nhận giả thuyết của mình sai là điểm cộng**, không phải điểm trừ.

**Hai — lệch hệ thống 0.0125 MRR.** Nhất quán ở cả năm seed và bốn chỉ số, gấp tám lần độ lệch chuẩn nên không phải ngẫu nhiên.

Nêu bốn khả năng, nhấn khả năng đầu: bài báo ghi batch 2048 / neg 64, script chạy 1024 / 32.

**Phải nói rõ:** *"Nhóm chưa xác định được nguyên nhân, nên không kết luận gì về việc bài báo có tái lập được hay không."*

### Slide 23 — Sáu rào cản

Phần này kể được, người nghe thích. Nhấn rào cản lớn nhất:

> "Repo thiếu hẳn thư mục chứa mã C++. File biên dịch sẵn chỉ chạy trên Linux x86-64. Nhóm phải truy ngược ra rằng `mmkgc/` là bản sửa của OpenKE, lấy mã gốc từ đó, đối chiếu chữ ký hàm — khớp đúng 11 tham số — rồi biên dịch lại."

Rồi bảng chi phí: huấn luyện 8.4 giây mỗi epoch, nhưng **đánh giá 21 phút mỗi lần**. Cấu hình gốc cần khoảng 28 giờ chỉ riêng khâu đánh giá.

---

## Phần 8 — Đánh giá và kết (6 phút)

### Slide 25 — Phản biện

Đọc lướt hai cột. Nhấn: mục "ổn định giữa các seed" ở cột điểm mạnh là **do nhóm kiểm chứng**, không phải bài báo báo cáo.

### Slide 26 — Hướng cải tiến

Nói gọn. Chỉ dừng ở dòng "Đã thử xong":

> "Nhóm đề xuất vector hoá phép tích luỹ hậu tố, rồi tự kiểm chứng. Kết quả trái với dự đoán: ở cấu hình hiện tại nó **chậm hơn 1.4 lần**, vì chiều cuối chỉ bằng 3. Chi phí hai phép lật vượt lợi ích gộp kernel. Chỉ có lợi khi chuỗi dài từ 8 trở lên."

Đây là ví dụ tốt cho việc **kiểm chứng trước khi khẳng định**.

### Slide 27 — Ba điều mang về

Đọc chậm, đây là câu cuối người nghe nhớ. Điều thứ ba là thông điệp của nhóm: *đọc mã nguồn cho biết những gì đọc bài báo không cho biết.*

---

## Câu hỏi thầy có thể hỏi

**"Vì sao tích luỹ hậu tố mà không dùng điểm attention trực tiếp?"**
→ Phép nhân khuếch đại. Token cao dai dẳng mới bị bắt; cao nhất thời thì không. Nếu so điểm đơn lẻ thì bắt nhầm cả hai.

**"Ngưỡng φ = 10 lấy từ đâu?"**
→ Nằm cứng trong mã, bài báo không giải thích và không có phân tích độ nhạy cho tham số này. Đây là một trong những hạn chế nhóm nêu. Hình 3 chỉ khảo sát `k` và `θ_p`, không có `φ`.

**"Vì sao kết quả nhóm thấp hơn bài báo?"**
→ Chưa xác định được. Bốn khả năng: cấu hình lệch (khả dĩ nhất), phần cứng, phiên bản thư viện, early stopping. Nhóm đang chạy thí nghiệm với đúng cấu hình bài báo để tách nguyên nhân. **Không khẳng định bài báo sai.**

**"Sao chỉ chạy MKG-W mà không chạy DB15K?"**
→ Giới hạn quota GPU miễn phí. MKG-W nhỏ hơn nên chọn làm tập khảo sát chính. DB15K nằm trong phần việc chưa làm, có ghi trong báo cáo.

**"noise_ratio không nhận gradient nghĩa là bài báo sai à?"**
→ Nghĩa là **mã nguồn công bố** có tham số đó không nhận gradient. Không suy ra được tác giả đã làm gì — họ có thể có bản khác. Phát biểu kiểm chứng được chỉ dừng ở: trong mã công bố, tham số này đứng yên ở giá trị khởi tạo.

**"Nhóm có đề xuất cải tiến nào không?"**
→ Có sáu hướng, một hướng đã thử và cho kết quả âm tính (vector hoá). Hướng đáng làm nhất là ngưỡng thích ứng thay ngưỡng cứng, vì Hình 3 cho thấy hiệu năng rất nhạy với ngưỡng.

**"Vì sao chọn bài này?"**
→ AAAI Oral, không phải khảo sát hay dataset, có mã nguồn công khai, dataset nhỏ chạy được trên một GPU, và hai đóng góp đều là mô-đun tháo lắp được nên dễ thử nghiệm cải tiến.

---

## Lưu ý khi nói

**Ba chỗ tuyệt đối không nói quá:**
1. Không nói "bài báo không tái lập được" — chỉ nói "kết quả nhóm đo được thấp hơn số công bố, chưa rõ nguyên nhân"
2. Không nói "tác giả không chạy nhiều seed" — chỉ nói "bài báo không báo cáo độ lệch chuẩn"
3. Không nói "phát hiện lỗi trong bài báo" — nói "mã nguồn công bố có những đặc điểm khiến người ngoài khó chạy lại"

Ranh giới này là giữa *quan sát kiểm chứng được* và *quy kết không có căn cứ*. Người nghe có kinh nghiệm sẽ để ý.

**Chỗ nên chậm lại:** slide 13 (attention penalty), slide 16 (ví dụ chạy), slide 22 (kết quả nhóm).

**Chỗ có thể lướt:** slide 5 (TransE/RotatE — chỉ cần ý chính), slide 20 (siêu tham số), slide 26 (hướng cải tiến).

**Nếu còn dư thời gian:** quay lại slide 23 nói kỹ hơn về chi phí đánh giá 21 phút mỗi lần và vì sao điều đó khiến cấu hình gốc không chạy trọn trong một phiên Kaggle.
