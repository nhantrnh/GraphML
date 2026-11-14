# Dàn ý trình bày — ImparTail (15 phút)

**Bài báo:** Towards Unbiased and Robust Spatio-Temporal Scene Graph Generation
and Anticipation — Peddi, Saurabh, Shrivastava, Singla, Gogate. CVPR 2025
(Highlight).

**Slide:** `slides/ImparTail_Slides.pdf`, 13 trang.

---

## Phân bổ thời gian

| Slide | Nội dung | Phút | Mốc |
|---|---|---|---|
| 1 | Bìa | 0,5 | 0:30 |
| 2 | Scene graph và video | 1,5 | 2:00 |
| 3 | Ví dụ đầu vào — đầu ra | 1,5 | 3:30 |
| 4 | Vấn đề đuôi dài | 1,5 | 5:00 |
| 5 | **Ý tưởng: masked loss** | 2,0 | 7:00 |
| 6 | Curriculum | 1,5 | 8:30 |
| 7 | **Bộ ba chạy qua mô hình** | 2,0 | 10:30 |
| 8 | Kết quả | 1,5 | 12:00 |
| 9 | Độ bền | 1,0 | 13:00 |
| 10 | **Đọc phản biện** | 1,5 | 14:30 |
| 11 | Ba điều mang về | 0,5 | 15:00 |
| 12 | Tài liệu tham khảo | — | (không nói) |

**Ba slide in đậm là trọng tâm** — chiếm 5,5/15 phút. Nếu cháy giờ thì rút slide
2 và 9, tuyệt đối không rút ba slide đó.

---

## Nguyên tắc xuyên suốt

Đề bài nói rõ: **hạn chế đọc lại slide**, phải dẫn dắt và phân tích sâu hơn. Mỗi
slide dưới đây ghi **điều KHÔNG có trên slide** mà bạn cần nói thêm.

---

## Slide 2 — Scene graph và video (1,5 phút)

**Trên slide:** ba khung hình, cùng cặp (người, cốc), vị ngữ đổi *nhìn → chạm →
uống*.

**Nói thêm:**
> "Lớp mình đã học đồ thị và GNN rồi, nên tôi không nhắc lại. Cái mới ở đây là
> đồ thị **sinh ra từ ảnh**: đỉnh là vật thể phát hiện được, cạnh là quan hệ giữa
> chúng."

Rồi chỉ vào ba khung:
> "Ba đồ thị này không độc lập. Cùng một cặp vật thể mà quan hệ chuyển dần. Đó
> là thứ scene graph tĩnh không diễn tả được, và là lý do bài toán này khó."

**Đừng** giải thích bộ ba $(o_i, p, o_j)$ bằng lời — công thức trên slide đã có
chú thích dưới từng thành phần, chỉ cần chỉ tay.

---

## Slide 3 — Ví dụ đầu vào, đầu ra (1,5 phút)

**Trên slide:** khung đã quan sát bên trái, khung tương lai bên phải, hai dòng
đầu ra.

**Nói thêm** — đây là chỗ người nghe nắm được bài toán, đi chậm:
> "VidSGG nhìn khung nào thì mô tả khung đó — bài toán nhận dạng. SGA phải sinh
> đồ thị cho khung **chưa nhìn thấy** — bài toán dự đoán. Cùng bộ máy, nhưng SGA
> khó hơn hẳn."

Chỉ vào chuỗi *chạm* rồi *uống*:
> "Mô hình phải học được rằng chạm vào cốc thường dẫn tới uống. Đó là tri thức
> về diễn tiến, không phải về một khung hình."

---

## Slide 4 — Vấn đề đuôi dài (1,5 phút)

**Trên slide:** biểu đồ cột, head xanh cao, tail đỏ dẹt.

**Nói thêm** — câu quan trọng nhất phần này:
> "Vấn đề không phải là phân phối lệch. Phân phối lệch thì ở đâu cũng có. Vấn đề
> là **lớp hiếm mới là lớp đáng giá**. Biết 'người ở phía trước cái cốc' gần như
> vô dụng. Biết 'người đang uống từ cái cốc' mới là hiểu cảnh. Mà mô hình lại
> học ngược lại."

Về chỉ số, nói ngắn:
> "Recall@K cộng gộp mọi dự đoán nên head lấn át. mean Recall tính từng lớp rồi
> trung bình — mỗi lớp một phiếu."

---

## Slide 5 — Masked loss (2 phút) — trọng tâm 1

**Trên slide:** công thức với ba mũi tên chú thích thành phần.

**Nói thêm** — giải thích bằng lời trước, ký hiệu sau:
> "$\mathbf{m}$ là một cái công tắc nhị phân, cho **từng bộ ba một**. Bật lên thì
> mất mát của bộ ba đó bằng không — nó biến mất khỏi gradient, như thể không tồn
> tại trong batch này."

Rồi nhấn chỗ khác biệt — đây là điều slide không nói:
> "Các phương pháp trước đánh trọng số lại, tức là **giảm tiếng nói** của lớp
> head. ImparTail **tắt hẳn**. Nghe thô, nhưng chính vì thô mà không phải dò
> tham số trọng số — mà dò trọng số mới là chỗ hay bị chê là tinh chỉnh quá tay."

---

## Slide 6 — Curriculum (1,5 phút)

**Trên slide:** ba biểu đồ, phân phối dịch từ đuôi dài sang đều.

**Nói thêm:**
> "Tỉ lệ che tăng theo epoch. Đầu buổi mô hình học trên phân phối gốc, càng về
> sau càng che nhiều head, đến cuối thì phân phối nhãn còn lại gần như đều."

**Gieo mầm cho slide 10** — nói rồi chuyển ngay, đừng giải thích:
> "Ghi nhớ slide này. Cuối bài ta quay lại xem nó có thực sự được dùng không."

---

## Slide 7 — Bộ ba chạy qua mô hình (2 phút) — trọng tâm 2

**Trên slide:** bốn khối, dấu ngoặc ghi "forward pass KHÔNG đổi".

**Nói thêm** — đi từng bước với ví dụ cụ thể:
> "Khung thứ 12 có người và cái cốc. ORPU và STPU cho ra biểu diễn quan hệ.
> Decoder cho phân phối trên các vị ngữ. Đến đây **mọi thứ giống hệt mô hình
> nền** — ImparTail chưa đụng gì cả."

Rồi chỉ xuống mũi tên đỏ:
> "Nhãn thật là *looking_at*, thuộc lớp head. Mặt nạ bật lên, mất mát bằng không.
> Bộ ba này vừa đi hết mô hình rồi **không đóng góp gì cho việc học**."

Câu chốt:
> "Vì chỉ backward đổi nên gắn được lên bất kỳ mô hình nào. Bài báo gắn lên
> **sáu** mô hình nền khác nhau — đó mới là điểm mạnh thật, không phải con số
> trong bảng."

---

## Slide 8 — Kết quả (1,5 phút)

**Trên slide:** bảng mean Recall, ô xanh là ImparTail.

**Đừng đọc số.** Nói cách đọc bảng:
> "Mỗi cặp dòng là cùng một mô hình, trước và sau khi gắn ImparTail. So theo cặp,
> đừng so chéo."

Rồi:
> "Cải thiện ở **mọi ô**, không ngoại lệ. Từ 12 đến 22 phần trăm."

Nếu bị hỏi vì sao PREDCLS tăng nhiều nhất: vì chế độ đó cho sẵn nhãn vật thể nên
phần khó còn lại đúng là phân loại vị ngữ — chính chỗ ImparTail tác động.

---

## Slide 9 — Độ bền (1 phút)

**Trên slide:** hai cặp cột, đường gạch nối thể hiện mức tụt.

**Nói thêm:**
> "Huấn luyện trên dữ liệu sạch, đánh giá trên khung hình đã làm nhiễu — mô phỏng
> đúng cảnh triển khai thật. Cả hai đều tụt, đó là bình thường. Điều đáng nói là
> ImparTail tụt 8 điểm còn mô hình nền tụt 14. Và điểm **sau khi nhiễu** của nó
> vẫn cao hơn điểm **sạch** của mô hình nền."

Cơ chế, một câu:
> "Cân bằng phân phối làm mô hình không quá khớp vào mẫu hình phổ biến — giống
> tác dụng của dropout."

---

## Slide 10 — Đọc phản biện (1,5 phút) — trọng tâm 3

**Trên slide:** trích nguyên văn Remark, ba khối bằng chứng.

Nhắc lại slide 6 trước:
> "Nhớ curriculum chứ. Giờ ta kiểm."

Đọc to đoạn trích, rồi chỉ ba khối:
> "Bài báo nói dùng tỉ lệ cố định. Mã nguồn đặt `is_curriculum = False` làm mặc
> định. Và không script chạy nào trong kho gọi biến thể curriculum. Ba nguồn độc
> lập, cùng một kết luận."

**Phát biểu cẩn thận** — đây là chỗ dễ nói quá:
> "Tôi chỉ đang mô tả bài báo và mã đã công bố. Không kết luận tác giả có chạy
> curriculum hay không — rất có thể họ có chạy mà chỉ không đưa vào bản phát
> hành."

Câu để lại:
> "Nhưng nếu tỉ lệ cố định cho kết quả tương đương, thì đóng góp thật nằm ở
> curriculum hay chỉ ở việc che? Bài báo chưa tách được hai thứ đó."

---

## Slide 11 — Nhóm chạy lại được (0,5 phút)

Nói ngắn, để bảng tự nói:

> "Nhóm chạy lại toàn bộ 1.750 video tập test, chế độ PredCls, dùng chính
> checkpoint tác giả phát hành. Chín con số mean Recall lệch trung bình
> **0,74 điểm** so với Bảng 1."

Nếu bị hỏi *"sao không chạy bản có phương pháp?"* — trả lời thẳng: chưa chạy.
Cái đã xác nhận là **pipeline đánh giá, dữ liệu và checkpoint tái lập đúng**,
không phải phương pháp đúng.

Không đọc từng con số trong bảng. Chỉ vào cột mR@20 hàng With rồi nói "34,7 so
với 35,5" là đủ.

## Slide 12 — Ba điều mang về (0,5 phút)

Đọc lướt, mỗi ý một câu. Không giải thích thêm — hết giờ.

---

## Câu hỏi có thể bị hỏi

**Che 90% nhãn thì còn gì để học?**
Che theo **bộ ba**, không theo mẫu. Một khung có nhiều cặp vật thể; che hết cặp
head vẫn còn cặp tail. Mất mát phân loại vật thể cũng không bị che.

**Khác gì focal loss?**
Focal loss giảm trọng số theo **độ tin cậy dự đoán**, liên tục và tự động.
ImparTail che theo **lớp**, nhị phân, theo lịch định trước. Bài báo **không so
trực tiếp với focal loss** — đó là một thiếu sót.

**Cải thiện có nằm trong nhiễu ngẫu nhiên không?**
Không kiểm chứng được — bài báo **không báo cáo độ lệch chuẩn**, không nói chạy
mấy seed. Với mức 12–22% thì khả năng là thật, nhưng đúng ra phải có nhiều seed.

**Nhóm có chạy lại không?**
Nhóm đã **thử thật trên Kaggle** và ghi lại chặn ở đâu, thay vì phỏng đoán.

Ba thứ tưởng là rào cản hoá ra không phải:
- **Dữ liệu**: Kaggle có sẵn `action-genome` (annotation) và
  `action-genome-frames-part-1..6` — video **đã tách khung sẵn**, đúng cấu trúc
  `<video>.mp4/<frame>.png` mà mã cần. Không phải tải video, không phải tách.
  Phải gắn **đủ cả 6 part** (~58 GB): lần đầu chỉ gắn 2 part, được 3.200 thư mục
  video nghe như thừa so với 1.814 video của tập test, nhưng đó là video **ngẫu
  nhiên trên toàn tập** — thiếu 1.224/1.814 video test, và chạy được 553/1.750
  video thì dừng.
- **`draw_rectangles`** (Cython): `build_ext` xong trong 3 giây.
- **Checkpoint**: tải được từ UT Dallas Box — 3 tệp, 1,1 GB, đúng định dạng
  `sttran_state_dict` và `faster_rcnn`.

**Rào cản kỹ thuật đã gỡ được hết.** Biên dịch CUDA extension từng là chỗ chặn:
mã viết cho torch 1.x, Kaggle cấp Tesla P100 (sm_60) mà torch cài sẵn 2.10 chỉ hỗ
trợ từ sm_70. Hạ xuống 2.4.1+cu121 rồi vá **51 chỗ dùng API đã bỏ**
(`.type()`, `.data<T>()`, header `THC/THC.h`) thì biên dịch sạch, `exit 0`, và
`nms` import được.

**Một trục trặc khi dựng môi trường:** ba tệp test đều
`from analysis.results import Result, ResultDetails, Metrics`. Lần clone đầu
nhóm dùng `--filter=blob:none` nên thiếu nội dung tệp và tưởng thư mục
`analysis/` không được commit. Kiểm lại trực tiếp trên GitHub thì **thư mục có
đủ 22 tệp Python**, cả `Result.py` lẫn `FirebaseService.py`. Đây là lỗi thao
tác của nhóm, không phải thiếu sót của tác giả.

Phát biểu thận trọng — mọi nhận định ở đây là mô tả **kho mã đã công bố**,
không kết luận tác giả chạy thí nghiệm thế nào.

**Và chạy xong thật.** Toàn bộ 1.750 video tập test, chế độ PredCls, checkpoint
`sttran_predcls` do tác giả phát hành, khoảng 1 giờ trên Tesla P100. Chín con số
mean Recall khớp Bảng 1 của bài báo:

| ràng buộc | mR@10 | mR@20 | mR@50 |
|---|---|---|---|
| With | 30,5 → **30,7** | 34,7 → **35,5** | 34,8 → **35,6** |
| No | 45,7 → **44,3** | 63,4 → **63,8** | 80,5 → **80,7** |
| Semi | 36,6 → **36,2** | 51,8 → **52,7** | 53,8 → **55,3** |

Lệch trung bình **0,74 điểm**, lớn nhất 1,52. Đây là con số của mô hình **nền**
STTran, không phải bản +ImparTail — nhóm chưa chạy bản có phương pháp. Nhưng nó
xác nhận được điều cần xác nhận: pipeline đánh giá, dữ liệu và checkpoint của
tác giả tái lập đúng. Để so sánh, mức cải thiện mà bài báo công bố cho
+ImparTail trung bình **16,6 điểm** — gấp 22 lần sai số tái lập này, nên sai số
đó không đe doạ kết luận của bài báo.

Điểm cộng của repo: giấy phép Apache 2.0 rõ ràng, có phát hành checkpoint — khá
hơn nhiều repo cùng lĩnh vực.

**Gắn được lên mô hình mới hơn không?**
Về nguyên tắc có, miễn mô hình dùng pipeline biểu diễn hướng-đối-tượng và có một
số hạng mất mát phân loại vị ngữ tách rời.

**Vì sao chỉ Action Genome?**
Đây là bộ chuẩn de-facto cho VidSGG và SGA. Nhưng chỉ một bộ dữ liệu cũng là
điểm yếu — không biết kết quả có chuyển sang bộ khác không.
