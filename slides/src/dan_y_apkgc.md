# Dàn ý trình bày — APKGC (15 phút)

Bài: *APKGC: Noise-enhanced Multi-Modal Knowledge Graph Completion with
Attention Penalty* — Y. Jian, X. Luo, Z. Li, M. Zhang, Y. Zhang, K. Xiao,
X. Hou (Hubei University). **AAAI 2025**, tr. 15005–15013.

Deck: `seminar/25C15053_Slides_APKGC.pdf` — 19 trang.

**Mạch trình bày.** Toàn bộ bài nói xoay quanh một câu hỏi: *bài báo đề xuất hai
thành phần — thành phần nào thực sự tạo ra kết quả?* Phần I trình bày phương
pháp và kết quả theo đúng cấu trúc bài báo. Phần II trả lời câu hỏi bằng thực
nghiệm tái lập. Không nêu kết luận sớm; kết quả chính xuất hiện ở trang 17.

**Ba nguyên tắc khi trình bày.**
1. Không đọc slide. Slide chứa hình và số liệu; lời nói đảm nhiệm phần dẫn dắt.
2. Mỗi trang kết thúc bằng một câu chuyển sang trang sau.
3. Nói "mã công bố làm X", không nói "tác giả đã làm X": ta chỉ quan sát được mã
   nguồn, không biết quy trình thực nghiệm của tác giả.

Ký hiệu: **[chỉ]** = chỉ vào đâu trên slide · lời trong ngoặc kép = câu nói ·
**→** = câu chuyển sang trang sau.

---

## Trang 1 — Bìa (0,25 phút)

> "Em xin trình bày bài APKGC, công bố tại AAAI 2025. Bài đề xuất hai thành phần
> mới cho bài toán hoàn thiện đồ thị tri thức đa phương thức. Em sẽ trình bày
> phương pháp, sau đó là kết quả tái lập nhằm đánh giá đóng góp của từng thành
> phần."

**→** "Trước hết là bối cảnh của bài toán."

## Trang 3 — Background: From KG to MMKGC (0,75 phút)

Ý chính: *định vị bài toán trong một hình; không giảng lại kiến thức đã học.*

**[chỉ]** bốn hình từ trái sang phải, mỗi hình một câu:

> "Đồ thị tri thức biểu diễn sự kiện dưới dạng bộ ba đầu, quan hệ, đuôi. Bài
> toán hoàn thiện đồ thị là dự đoán các cạnh còn thiếu. Đồ thị đa phương thức
> gắn thêm ảnh và văn bản cho mỗi thực thể. Kết hợp hai điều đó là bài toán
> APKGC giải quyết, trong khung đỏ."

**[chỉ]** dòng ⇒ dưới hình:

> "APKGC không sử dụng GNN. Mô hình mã hoá văn bản và ảnh bằng Transformer, sau
> đó chấm điểm bộ ba bằng RotatE. Trong RotatE, mỗi thực thể là một vector và mỗi
> quan hệ là một *phép quay*: quay vector đầu theo quan hệ, kết quả càng gần
> vector đuôi thì bộ ba càng có khả năng đúng."

**→** "Bài báo xác định hai hạn chế của các phương pháp hiện có."

## Trang 4 — Introduction (0,75 phút)

Không giảng lại đồ thị tri thức; đi thẳng vào hai hạn chế.

**[chỉ]** hai khối xanh.

> "Hạn chế thứ nhất: dữ liệu đa phương thức thực tế chứa nhiều nhiễu, trong khi
> các phương pháp trước chủ yếu tìm cách loại bỏ nhiễu. Hạn chế thứ hai: mô hình
> ngôn ngữ có hiện tượng *attention over-trust*, tức phân bổ attention quá mức
> vào những token không mang ngữ nghĩa."

**[chỉ]** dải token ở giữa, ô đỏ là dấu chấm:

> "Ví dụ với câu mô tả Dennis Hopper: attention tập trung vào dấu chấm, trong khi
> các token mang thông tin là *actor* và *awards*. Đây là hình minh hoạ khái
> niệm, không phải số đo."

**[chỉ]** dòng ⇒.

> "Hai hạn chế này độc lập với nhau, nên bài đề xuất hai cơ chế riêng biệt. Tính
> độc lập này cho phép đánh giá từng cơ chế một cách tách bạch ở phần sau."

**→** "Các nghiên cứu trước đã tiếp cận bài toán theo những hướng nào?"

## Trang 5 — Related Work (0,5 phút)

Ý chính: *APKGC phát triển tiếp một hướng nghiên cứu đã có.*

**[chỉ]** các nhãn màu bên trái biểu đồ, từ trên xuống:

> "Bài phân loại các phương pháp đa phương thức trước đó thành bốn hướng: hợp
> nhất phương thức, cải tiến lấy mẫu âm, kết hợp quyết định, và sử dụng nhiễu.
> Mẫu âm là bộ ba sai, tạo bằng cách thay đầu hoặc đuôi của một bộ ba đúng bằng
> thực thể ngẫu nhiên; mô hình học chấm bộ ba đúng cao hơn mẫu âm.
> APKGC thuộc hướng cuối, do SnAg khởi xướng, và bổ sung thêm cơ chế phạt
> attention."

**[chỉ]** cột APKGC dưới cùng, rồi dòng ⇒ dưới hình:

> "Trong bảng kết quả chính của bài, APKGC đạt MRR cao nhất, 0,374. Đường đứt
> nét là PairRE, phương pháp đơn phương thức tốt nhất, dùng làm mốc so sánh."

**→** "Cụ thể, đầu vào và đầu ra của mô hình là gì?"

## Trang 6 — Example: Input and Output (0,75 phút)

Đây là một bộ ba có thật trong DB15K, đồng thời là ví dụ trong phần case study
của bài. Dòng nguồn nhỏ dưới công thức ghi rõ: lấy từ bài APKGC [1].

> "Truy vấn có dạng: thực thể đầu là giải thưởng nam diễn viên phụ xuất sắc của
> Hiệp hội phê bình phim Hoa Kỳ, quan hệ là *award*, cần dự đoán thực thể đuôi."

**[chỉ]** ba phương thức bên trái, rồi bảng xếp hạng bên phải.

> "Mô hình không sinh ra một đáp án duy nhất, mà chấm điểm *toàn bộ* thực thể
> trong đồ thị rồi xếp hạng. Trong ví dụ này, đáp án đúng là Dennis Hopper, đứng
> hạng nhất."

> "Cả ba ứng viên đầu đều từng nhận giải thưởng này, ở các năm khác nhau. Nếu chỉ
> dựa vào cấu trúc đồ thị, ba ứng viên gần như không phân biệt được; thông tin
> từ ảnh và văn bản giúp phân biệt chúng."

**[chỉ]** công thức MRR: "Đáp án đúng ở hạng 1 được 1 điểm, ở hạng 2 được 0,5
điểm. Đây là thước đo chính xuyên suốt bài."

**→** "Tiếp theo là kiến trúc tổng thể của mô hình."

## Trang 7 — Overall Architecture (0,75 phút)

Đi theo mũi tên từ trái sang phải. **Không đọc từng hộp.**

> "Mô hình gồm ba module: lấy mẫu nhiễu theo phương thức, trích xuất đặc trưng
> thực thể, và chấm điểm bộ ba. Văn bản được mã hoá bằng BERT, ảnh bằng BEiT."

**[chỉ]** hai khối đỏ.

> "Hai khối đỏ là hai đóng góp của bài, và chúng nằm ở hai module khác nhau:
> nhiễu tác động ở bước *tạo đầu vào*, phạt attention tác động ở bước *trích
> xuất đặc trưng*. Vì tách rời, ta có thể đánh giá riêng từng thành phần."

**[chỉ]** dòng tham số nhỏ dưới hai khối đỏ.

> "Lưu ý thêm: tham số của khối nhiễu là *học được*, còn tham số của khối phạt
> là *cố định*. Điểm này sẽ được nhắc lại ở phần phương pháp."

**→** "Luồng dữ liệu qua mô hình có thể tóm tắt thành mười bước."

## Trang 8 — Data Flow: 10 Steps (1 phút)

Ý chính: *các công thức của bài tạo thành một chuỗi 10 bước, trong đó chỉ hai bước là mới.*

**[chỉ]** đi theo mũi tên từ ô 1 đến ô 10, đọc tên bước:

> "Ghép các công thức của bài lại, dữ liệu đi qua mười bước: tổng hợp thông tin,
> thêm nhiễu, tách token, tính attention, phạt attention, mã hoá, hợp nhất văn
> bản và ảnh, chuẩn hoá, chấm điểm, và tính hàm mất mát."

**[chỉ]** hai ô đỏ, ô 2 và ô 5:

> "Trong mười bước, chỉ hai ô đỏ là đóng góp của bài; tám bước còn lại kế thừa
> từ các phương pháp trước."

**[chỉ]** ô 9:

> "Bước chấm điểm sử dụng lại RotatE. Riêng RotatE đạt MRR 0,337 trên MKG-W,
> APKGC đạt 0,374. Phần chênh lệch này, theo bài, đến từ hai ô đỏ."

**[chỉ]** khung "Bước 9-10" ở dưới; nói lướt, không đọc công thức:

> "Hai bước cuối: công thức (19) chấm điểm bằng khoảng cách, bộ ba đúng có
> khoảng cách nhỏ. Hàm loss (20), dùng lại từ RotatE, kéo bộ ba đúng lại gần và
> đẩy bộ ba sai, tức mẫu âm, ra xa. Mọi tham số, kể cả θ_r của khối nhiễu, được
> học bằng cách giảm loss này."

Công thức đầy đủ của (20) nằm trong mục "Câu hỏi có thể gặp", nếu được hỏi.

Nếu được hỏi bước nào ứng với công thức số mấy, tra mục "Ghi chú" cuối dàn ý.

**→** "Đi vào chi tiết từng thành phần, bắt đầu với lấy mẫu nhiễu."

## Trang 9 — Method: Adaptive Noise Sampling (0,75 phút)

Ý chính: *hai phân phối nhiễu, và một quy tắc trộn nhiễu vào vector.*

> "Thay vì loại bỏ nhiễu, bài *chủ động thêm nhiễu* trong quá trình huấn luyện,
> nhằm tăng độ bền vững của mô hình trước dữ liệu nhiễu."

**[chỉ]** hai ô trên:

> "Bài dùng hai loại nhiễu. Nhiễu Gauss tập trung quanh trung bình, thỉnh thoảng
> có giá trị lớn. Nhiễu phân phối đều trải đều trong một khoảng xác định. Theo
> mục thiết lập của bài, DB15K dùng nhiễu Gauss, MKG-W dùng nhiễu phân phối
> đều."

**[chỉ]** ô đỏ ở dưới, công thức (5):

> "Quy tắc trộn: mỗi thực thể nhận một số ngẫu nhiên ρ. Nếu ρ vượt ngưỡng θ_n,
> vector của thực thể được trộn với nhiễu theo tỉ lệ θ_r; ngược lại giữ nguyên.
> Theo bài, θ_r là tham số *học được*: nó được cập nhật như mọi trọng số khác,
> bằng gradient của hàm mất mát qua lan truyền ngược."

**→** "Thành phần thứ hai là phạt attention."

## Trang 10 — Method: Attention Penalty (1 phút)

Ý chính: *token có attention vượt ngưỡng bị giảm, phần dư được phân bổ lại cho token lân cận.*

**[chỉ]** hai ô ① và ② ở trên, theo mũi tên:

> "Cơ chế gồm hai bước. Bước một, phạt: token có attention vượt ngưỡng chú ý φ,
> tức attention threshold, bị giảm theo tỉ lệ θ_p. Bước hai, phân bổ lại: phần
> vượt quá φ được chia cho k token lân cận."

**[chỉ]** biểu đồ, cột đỏ "for", rồi đường đứt nét 0,5:

> "Trong case study của bài, token *for*, một giới từ, nhận attention 0,66. Đó là
> hiện tượng over-trust. Cột xám là trước khi phạt, cột xanh là sau khi phạt:
> *for* giảm về 0,5, các token *award* và *supporting* tăng lên."

**[chỉ]** dòng ⇒, nối lại với trang 7:

> "Ba tham số của cơ chế này là ngưỡng, tỉ lệ phạt và kích thước cửa sổ, đều
> *cố định*. Mục RQ4 của bài cho thấy chúng phải được dò thủ công, khác với θ_r
> của khối nhiễu là tham số học được."

**→** "Tiếp theo là kết quả thực nghiệm bài báo công bố."

## Trang 11 — Experiments from the Paper (0,5 phút)

**[chỉ]** bảng trên.

> "Trên MKG-W, APKGC đạt MRR 0,374, cao nhất trong các phương pháp được so
> sánh."

**[chỉ]** ô công thức bên phải; nói lướt, không đọc công thức:

> "Hai độ đo: MRR là trung bình nghịch đảo hạng của đáp án đúng; Hits@K là tỉ lệ
> truy vấn có đáp án đúng nằm trong top K. Cả hai càng cao càng tốt."

**[chỉ]** bảng thiết lập bên dưới.

> "Bài báo huấn luyện trên RTX 4090; thực nghiệm tái lập chạy trên Kaggle, GPU
> Tesla T4."

**[chỉ]** dòng ⇒, nói chậm:

> "Bảng kết quả không báo cáo *độ lệch chuẩn* và không nêu số seed. Trên DB15K,
> APKGC chỉ hơn NativE 0,002; không có độ lệch chuẩn thì không xác định được
> chênh lệch này có vượt mức biến động ngẫu nhiên hay không."

**→** "Bài cũng phân tích độ nhạy với siêu tham số."

## Trang 12 — Experiments: Hyper-parameter Analysis (RQ4) (0,75 phút)

Ý chính: *kết quả của bài nhạy với hai siêu tham số của cơ chế phạt.*

**[chỉ]** hai heatmap, trục ngang là tỉ lệ phạt θ_p, trục dọc là cửa sổ k:

> "Đây là phân tích siêu tham số của chính bài báo, mục RQ4. Mỗi ô là một cấu
> hình của cửa sổ k và tỉ lệ phạt θ_p. Hàng dưới cùng, khung đỏ, là k bằng 0,
> tức không phạt."

**[chỉ]** heatmap bên trái: ô đậm nhất (0,364, k = 2, θ_p = 0,01), rồi ô nhạt nhất
(0,324, góc trên bên phải):

> "Trên DB15K, MRR thay đổi từ 0,324 đến 0,364 tuỳ cấu hình. Nghĩa là kết quả
> nhạy với hai siêu tham số này, và chúng phải được dò thủ công."

**→** "Từ đó, em tổng hợp các hạn chế của bài."

## Trang 13 — Limitations & Conclusion (0,5 phút)

> "Bài báo không có mục Limitations riêng. Em rút ra ba hạn chế."

**[chỉ]** lần lượt ba dòng, mỗi dòng một câu:

> "Thứ nhất, bài không báo cáo độ lệch chuẩn. Thứ hai, bài chưa tách bạch đóng
> góp của từng cơ chế. Thứ ba, ba tham số của cơ chế phạt phải dò thủ công, như
> vừa thấy ở heatmap."

**→** "Phần tiếp theo là thực nghiệm tái lập, nhằm kiểm tra các điểm này."

## Trang 15 — 1. Reproducing the Published Results (1,5 phút)

Ý chính: *tái lập được; dùng đúng cấu hình của bài thì kết quả sát số công bố.*

> "Phần tái lập gồm ba nội dung: tái lập kết quả trên MKG-W, tái lập trên DB15K,
> và kiểm tra thành phần phạt attention. Trước hết, mã công bố được chạy lại trên MKG-W
> với ba seed khác nhau, tức ba cách khởi tạo ngẫu nhiên."

**[chỉ]** ba thanh, từ trên xuống (thanh xám là "Tái lập - cấu hình mã công bố",
thanh cam là "Tái lập - cấu hình theo bài"):

> "Bài báo công bố MRR 0,374. Chạy mã công bố với ba seed, MRR đạt 0,3605. File
> chạy trong repo dùng batch 1024 và 32 mẫu âm, khác với 2048 và 64 ghi trong
> bài; chạy lại theo cấu hình của bài, MRR đạt 0,3691, sát với số công bố."

**[chỉ]** bảng nhỏ: "Các độ đo H@1 và H@10 cũng thấp hơn công bố, cùng chiều."

**→** "Tiếp theo là bộ dữ liệu thứ hai của bài, DB15K."

## Trang 16 — 2. Second Dataset: DB15K (0,75 phút)

Ý chính: *trên bộ dữ liệu thứ hai, kết quả tái lập cũng thấp hơn công bố, cùng chiều.*

**[chỉ]** đường cam, rồi đường đứt nét xanh:

> "Đây là MRR trên DB15K theo số epoch huấn luyện. Do giới hạn phiên 12 giờ của
> Kaggle, thực nghiệm dừng ở epoch 600. Đường học đã gần phẳng: 50 epoch cuối chỉ
> tăng 0,0001. MRR tái lập đạt 0,3469, thấp hơn mức công bố 0,364."

**[chỉ]** bảng: "Cả bốn độ đo đều thấp hơn công bố, cùng chiều với MKG-W. Kết quả
này dựa trên một seed."

**→** "Tiếp theo, em kiểm tra thành phần phạt attention."

## Trang 17 — 3. Attention Penalty: Four Implementations (1,25 phút) — TRỌNG TÂM

Nói chậm. Đây là kết quả chính.

Ý chính: *bốn biến thể cài đặt đều không cải thiện MRR; trong mã công bố, điều kiện phạt không được kích hoạt lần nào.*

**[chỉ]** bảng bốn biến thể:

> "Phạt attention được cài đặt theo bốn cách. A: đúng như mã công bố trên
> GitHub. B: đúng công thức (9)(10) viết trong bài. C: ngưỡng thích ứng do em đề
> xuất: trong mỗi batch, 10% hàng attention có entropy thấp nhất, tức đang dồn
> vào một token, bị phạt; ngưỡng không cố định mà tính lại theo từng batch. B2: cắt xác suất attention ở 0,5 như ví dụ *for* trong case study,
> là cách mạnh tay nhất. A, B, C cho MRR gần như trùng nhau; B2 còn thấp hơn.
> Không cách nào cải thiện MRR."

Không có trên slide; nói tiếp sau bảng:

> "Cơ chế phạt chỉ tác động khi điều kiện 'attention vượt ngưỡng' đúng. Em gắn
> bộ đếm vào mã công bố để đếm, qua 300 lần chạy mô hình, lệnh phạt được thực thi
> bao nhiêu lần: kết quả là không lần nào. Nghĩa là attention chưa từng bị thay
> đổi, nên bật hay tắt thành phần này cho cùng một kết quả."

**→** "Tổng hợp lại các kết quả."

## Trang 18 — Conclusion (0,5 phút)

Ý chính: *trả lời trực tiếp câu hỏi: kết quả có đến từ hai thành phần đề xuất không?*

**[chỉ]** lần lượt hai dòng, mỗi dòng một câu, nhìn người nghe:

> "Về tái lập: trên MKG-W, sau khi hiệu chỉnh cấu hình, MRR đạt 0,369 so với
> 0,374 công bố."

> "Về phạt attention: bốn cách cài đặt đều không làm MRR tăng."

Câu kết, nhìn người nghe:

> "Em xin cảm ơn thầy và các bạn."

## Trang 19 — References (0 phút)

Để sẵn trên màn hình khi hỏi đáp. Trên slide, mỗi số trích dẫn đều có tên nguồn
đứng cạnh. Sáu nguồn: [1] APKGC, [2] NativE, [3] MMKG (nguồn bộ DB15K),
[4] MMRNS (nguồn bộ MKG-W), [5] repo mã công bố, [6] RotatE (ICLR 2019), nguồn của
bước chấm điểm, được nhắc ở trang 3 và trang 8.

Dưới mỗi trang tái lập (15–17) có link notebook Kaggle đã chạy thí nghiệm đó; các
notebook để public, phiên bản đã chạy có log nằm trong tab Versions.

---

**Tổng: 12,25 phút** — Phần I 8,25 phút, Phần II 4 phút. Còn dư 2,75 phút so với giới
hạn 15 phút, dùng cho hỏi đáp hoặc nói chậm lại ở trang 17.

**Nếu bị ép giờ:** trang 17 rút còn 0,5 phút (chỉ đọc bảng bốn cách); trang 12 rút
còn 0,5 phút (chỉ nói dòng ⇒).

---

# Giải thích thuật ngữ — nếu được hỏi

| thuật ngữ | giải thích |
|---|---|
| seed | giá trị khởi tạo bộ sinh số ngẫu nhiên; đổi seed là đổi trọng số khởi tạo, thứ tự dữ liệu và nhiễu |
| độ lệch chuẩn (std) | mức biến động của kết quả khi chạy lại với các seed khác nhau |
| tái lập | chạy lại thực nghiệm để kiểm tra có thu được số liệu công bố hay không |
| MRR | Mean Reciprocal Rank: trung bình của 1/hạng đáp án đúng qua mọi truy vấn; hạng 1 được 1, hạng 2 được 0,5 |
| H@K (Hits@K) | tỉ lệ truy vấn có đáp án đúng nằm trong top K |
| KG, KGC | Knowledge Graph (đồ thị tri thức), Knowledge Graph Completion (hoàn thiện đồ thị tri thức) |
| MMKG | Multi-Modal Knowledge Graph: đồ thị tri thức có thêm ảnh và văn bản |
| nhiễu Gauss | nhiễu theo phân phối chuẩn: đa số giá trị gần trung bình, thỉnh thoảng có giá trị lớn |
| nhiễu phân phối đều | nhiễu có xác suất như nhau trong một khoảng xác định; bài gọi là "average" |
| MMKGC | Multi-Modal Knowledge Graph Completion: hoàn thiện đồ thị tri thức đa phương thức |
| GNN | Graph Neural Network: mạng nơ-ron học trên đồ thị bằng truyền thông điệp giữa các nút; APKGC không dùng |
| RotatE | hàm chấm điểm: quay vector đầu theo quan hệ rồi đo khoảng cách tới vector đuôi; APKGC dùng ở bước cuối |
| RQ4 | Research Question 4: câu hỏi nghiên cứu thứ tư trong bài, về ảnh hưởng của siêu tham số |
| ablation | loại bỏ lần lượt từng thành phần để đo đóng góp của nó |
| checkpoint | một phiên bản mô hình được lưu trong quá trình huấn luyện |
| tập test | tập dữ liệu dùng để báo cáo kết quả; không dùng để chọn mô hình |
| attention | trọng số mô hình phân bổ cho từng token khi mã hoá |
| batch | số mẫu đưa vào mô hình trong mỗi bước cập nhật |
| mẫu âm (negative sample) | bộ ba sai được tạo bằng cách thay đầu hoặc đuôi của bộ ba đúng bằng thực thể ngẫu nhiên; mô hình học chấm bộ ba đúng cao hơn mẫu âm. "32 mẫu âm" = mỗi bộ ba đúng so với 32 bộ ba sai (trang 5, 8, 15) |
| epoch | một lượt huấn luyện đi hết toàn bộ tập train (trang 16) |
| lượt chạy mô hình (forward) | một lần đưa một batch qua mô hình để tính đầu ra (trang 17) |
| softmax | hàm biến điểm attention thành phân phối xác suất có tổng bằng 1 |
| entropy (chuẩn hoá) | độ "phân tán" của phân phối attention: gần 0 = dồn vào một token, gần 1 = trải đều (trang 13, 17) |
| Transformer | kiến trúc mạng dùng cơ chế attention; BERT và BEiT đều là Transformer (trang 3, 7) |
| over-trust | hiện tượng attention dồn quá mức vào token không mang nghĩa (trang 4, 10) |
| baseline | phương pháp dùng làm mốc so sánh |
| SOTA | state of the art: kết quả tốt nhất hiện có |
| hàm loss | hàm đo mức sai của mô hình; huấn luyện là cập nhật tham số để giảm nó |
| margin (λ) | ngưỡng khoảng cách phân tách bộ ba đúng và mẫu âm trong loss (trang 8) |
| gradient | đạo hàm của hàm loss theo từng tham số; cho biết chỉnh tham số theo hướng nào để loss giảm |

---

# Cách đọc ký hiệu

| ký hiệu | đọc | nghĩa trong bài |
|---|---|---|
| θ | theta | θ_r: tỉ lệ nhiễu (học được); θ_n: ngưỡng lấy mẫu; θ_p: tỉ lệ phạt; $\theta^n_j$: trọng số chia lại |
| φ | phi | ngưỡng chú ý (attention threshold) ở công thức (9)(10) |
| ρ | rho | số ngẫu nhiên trong [0, 1] gán cho mỗi thực thể, công thức (5) |
| λ | lambda | margin trong hàm loss (20) |
| σ | sigma | hàm sigmoid |
| η | eta | tốc độ học (learning rate) |
| ∂L/∂θ | đạo hàm riêng của L theo theta | gradient dùng để cập nhật tham số |
| ‖·‖ | chuẩn (norm) | độ dài vector, dùng làm khoảng cách ở (19) |

---

# Các phương pháp khác có tên trên slide — nếu được hỏi

Chỉ cần trả lời một câu. Nguồn trích dẫn nằm trong bài APKGC [1].

| tên | xuất hiện ở | một câu giải thích |
|---|---|---|
| TransE | Q&A | coi quan hệ là phép tịnh tiến: bộ ba đúng khi h + r ≈ t |
| RotatE [6] | trang 3, 5, 8 | coi quan hệ là phép quay trong không gian số phức: quay h theo r thì ≈ t |
| PairRE | trang 5 | mỗi quan hệ có hai vector, lần lượt co giãn thực thể đầu và đuôi trước khi so khớp; phương pháp đơn phương thức tốt nhất trong bảng (0,344) |
| IKRL | trang 5 | một trong các phương pháp đầu tiên đưa ảnh vào biểu diễn thực thể, trên nền mô hình kiểu TransE |
| OTKGE | trang 5 | dùng optimal transport (vận chuyển tối ưu) để căn chỉnh và hợp nhất biểu diễn các phương thức |
| NativE [2] | trang 5, 11 | MMKGC "in the wild": hợp nhất thích ứng theo quan hệ và huấn luyện đối kháng giữa các phương thức, xử lý dữ liệu thiếu hoặc mất cân bằng phương thức |
| MANS | trang 5 | lấy mẫu âm có xét phương thức (modality-aware negative sampling) |
| MMRNS [4] | trang 5 | lấy mẫu âm tăng cường theo quan hệ, dùng thông tin đa phương thức để chọn mẫu âm khó; bài này cũng giới thiệu bộ MKG-W |
| IMF | trang 5 | hợp nhất đa phương thức tương tác, rồi kết hợp quyết định của từng phương thức |
| SnAg | trang 5 | khung biểu diễn thống nhất, đưa nhiễu vào đặc trưng phương thức khi huấn luyện; APKGC đi tiếp hướng này |
| OPERA | trang 4 | phương pháp giảm ảo giác cho mô hình ngôn ngữ đa phương thức lớn, đề xuất khái niệm over-trust penalty mà APKGC mượn ý tưởng |
| BERT | trang 7 | mô hình ngôn ngữ Transformer tiền huấn luyện; APKGC dùng để mã hoá văn bản |
| BEiT | trang 7 | Transformer cho ảnh, tiền huấn luyện bằng che và khôi phục mảnh ảnh; APKGC dùng để mã hoá ảnh |
| OpenKE | Q&A | thư viện mã nguồn mở cho mô hình nhúng đồ thị tri thức; mã APKGC và NativE xây trên nó |

Nếu bị hỏi sâu hơn về một phương pháp không phải APKGC: *"Em chỉ dùng số liệu
của phương pháp này từ bảng kết quả trong bài APKGC, không tái lập nó."*

---

# Ghi chú — những ý đã rút khỏi slide

Không bắt buộc trình bày. Dùng khi có câu hỏi chuyên sâu.

- **Trang 4.** Ví dụ over-trust mà bài dẫn là dấu chấm câu `[.]`, lấy từ OPERA
  (Huang et al., 2023).
- **Trang 5.** Nếu được hỏi về đường đứt nét: trên bảng kết quả chính của bài,
  10 trong 14 phương pháp đa phương thức không vượt PairRE (0,344). Bổ sung ảnh
  và văn bản không tự động cải thiện kết quả; cách khai thác chúng mới quyết định.
- **Trang 6.** Mô hình cũng dự đoán theo chiều ngược: (?, location, Saint Vincent
  and the Grenadines) → Caribbean Sea (case study của bài). Tức là mô hình dự
  đoán được cả thực thể đầu lẫn thực thể đuôi.
- **Trang 8.** Mười bước ứng với công thức của bài: bước 1 = (1)(2), bước 2 =
  (3)–(6), bước 3 = (7), bước 4 = (8), bước 5 = (9)(10), bước 6 = (11)(12),
  bước 7 = (13)–(17), bước 8 = (18), bước 9 = (19), bước 10 = (20).
- **Trang 9.** Công thức (5) xác định vector nào được thêm nhiễu, dựa trên việc so
  sánh một số ngẫu nhiên với ngưỡng lấy mẫu.
- **Trang 10.** Ký hiệu trong (9)(10), theo bài: chỉ số trên $p$ là vị trí token bị
  phạt; $n$ là vị trí các token lân cận, nằm trong cửa sổ $(p-k, p+k)$; $k$ là kích
  thước cửa sổ; chỉ số dưới $i$ và $j$ phân biệt giá trị *trước* và *sau* khi điều
  chỉnh (không phải chỉ số token); $\theta_p$ là tỉ lệ phạt; $\theta^n_j$ là trọng
  số chia lại cho token $n$. Dòng chú thích nhỏ dưới hai ô công thức ghi đúng
  các ký hiệu này.
- **Trang 11.** Bài báo cáo H@1 tăng 3,4% (MKG-W) và 3,3% (DB15K) so với
  baseline. Thực nghiệm tái lập chạy batch 1024 theo script, sau đó 2048 theo bài.
  Giới hạn phiên 12 giờ của Kaggle buộc dừng DB15K ở epoch 600.
- **Trang 12.** Trong heatmap DB15K, 5 trên 16 cấu hình có phạt (0,324; 0,329; 0,330;
  0,330; 0,332) cho MRR thấp hơn hàng không phạt (0,34). Ô tốt nhất (k = 2,
  θ_p = 0,01) cho 0,364. Chỉ nói nếu được hỏi.
- **Trang 13.** Future work của bài là bổ sung video và âm thanh: mở rộng dữ liệu,
  không giải quyết các hạn chế phương pháp nêu trên.
- **Trang 15.** Với cấu hình mã công bố, khoảng chênh 0,0135 lớn gấp khoảng 15 lần
  độ lệch chuẩn 0,0009, nên không phải biến động ngẫu nhiên. Hiệu chỉnh cấu hình
  theo bài thu hẹp được 0,0086, khoảng hai phần ba khoảng chênh.
- **Trang 15.** Ba nguyên nhân khả dĩ cho phần chênh 0,0049 còn lại, chưa loại
  trừ được: phiên bản bộ nhúng, số epoch thực chạy, cách chọn checkpoint.
- **Trang 15.** Ngoài batch và số mẫu âm, bài ghi MKG-W dùng nhiễu phân phối đều,
  còn mã công bố chỉ cài nhiễu Gauss; mọi lần chạy MKG-W đều dùng nhiễu Gauss.
- **Trang 17.** Không có lần chạy tắt phạt riêng. Nếu được hỏi: khi điều kiện phạt
  không đúng ở bất kỳ vị trí nào, mã công bố bỏ qua toàn bộ khối phạt và attention
  giống hệt bản không phạt; bộ đếm ghi 300 lần gọi forward, không phải toàn bộ
  quá trình huấn luyện.
- **Trang 17.** Biến thể B2 làm MRR giảm 0,0024, tức 2,6 lần độ lệch chuẩn, dưới
  ngưỡng 3 lần.
- **Trang 17.** Biến thể B *có* kích hoạt, nhưng tỉ lệ phạt chỉ 0,001 nên token có
  điểm lớn nhất chỉ giảm 0,1%; sau softmax, phân phối xác suất gần như không đổi.
  Tỉ lệ phần tử âm là 62,66% thay vì khoảng 50%, cho thấy các điểm attention có
  tương quan.
- **Trang 17.** Trước khi đo, mô phỏng giả định độc lập dự đoán điều kiện phạt được
  thỏa 18–26% số lượt; đo thực tế cho 0%. Giả định độc lập không đúng với dữ liệu
  thật.

---

# Câu hỏi có thể gặp

**Ngoài RotatE còn những cách tiếp cận KGC nào?** (chỉ trả lời nếu được hỏi)
Có bốn họ phương pháp chính:
- *Tịnh tiến / quay* (TransE, RotatE): mô hình hoá quan hệ như phép tịnh tiến hoặc
  phép quay; bộ ba đúng khi "đầu biến đổi theo quan hệ ≈ đuôi".
- *Phân rã tensor* (DistMult, ComplEx): xem đồ thị như một tensor ba chiều và phân
  rã thành tích các vector.
- *CNN* (ConvE): ghép vector đầu và quan hệ thành ma trận rồi áp dụng tích chập.
- *GNN*: cập nhật biểu diễn mỗi nút bằng thông tin từ các nút lân cận.
Câu chốt: *"Ở bước chấm điểm, APKGC thuộc họ thứ nhất và không dùng GNN."*

**Vì sao APKGC không dùng GNN?**
Bài tập trung khai thác văn bản và ảnh của từng thực thể, nên dùng mô hình ngôn
ngữ và thị giác (BERT, BEiT) thay vì truyền thông tin giữa các nút. Cấu trúc đồ
thị chỉ được khai thác ở bước chấm điểm qua RotatE.

**N_m khi nào là nhiễu Gauss, khi nào là nhiễu phân phối đều?**
Mục thiết lập của bài ghi: DB15K dùng nhiễu Gauss, MKG-W dùng nhiễu phân phối
đều ("average"). Tuy nhiên mã công bố chỉ cài nhiễu Gauss (`mean + std * randn`)
cho cả hai bộ. Do đó mọi lần chạy MKG-W trong phần tái lập đều dùng nhiễu Gauss;
đây là một điểm cấu hình không khớp, bên cạnh batch và số mẫu âm.

**θ_r "học được" nghĩa là gì?**
θ_r là một tham số của mô hình, nên ở mỗi bước huấn luyện, lan truyền ngược tính
gradient của hàm mất mát theo θ_r và cập nhật nó như mọi trọng số khác. Trong mã
công bố, θ_r là `mask_ratio` (khởi tạo 0,7), nằm trong phép trộn
`(1 − θ_r)·V_o + θ_r·N`, nên nhận được gradient. Ngưỡng θ_n (`noise_ratio`, 0,2)
chỉ xuất hiện trong phép so sánh `ρ < θ_n`; phép so sánh không khả vi, nên trong
mã công bố θ_n giữ nguyên 0,2. Đọc từ checkpoint tốt nhất của lần chạy mã công
bố (MKG-W, seed 42): θ_r giảm từ 0,7 xuống −0,0004, tức gần bằng 0; θ_n vẫn là
0,2. Với θ_r ≈ 0, phép trộn (1 − θ_r)·V_o + θ_r·N gần như giữ nguyên vector gốc:
đến cuối huấn luyện, mô hình đã tự "tắt" nhiễu. Đây là một lần chạy, một seed.

**Một seed có đủ để kết luận về bốn biến thể không?**
Với A, B, C thì đủ: chênh lệch lớn nhất 0,0001, nhỏ hơn độ lệch chuẩn 0,0009 tới
chín lần. Với B2 thì chưa: 2,6 lần, dưới ngưỡng 3 lần đặt trước.

**Có chạy ablation bỏ lấy mẫu nhiễu không?** (chỉ trả lời nếu được hỏi)
Có một lần chạy tắt nhiễu (cờ add\_noise = 0), MKG-W, seed 42: MRR 0,3655 so với 0,3629 của
bản đầy đủ. Chênh +0,0026, tức 2,9 lần độ lệch chuẩn, dưới ngưỡng 3 lần, và chỉ
một seed, nên em không đưa vào slide. Thêm nữa, mã công bố chỉ cài nhiễu Gauss
trong khi bài ghi MKG-W dùng nhiễu phân phối đều, nên phép so sánh không cùng
điều kiện với bài. Cả hai lần chạy (có và không nhiễu) dùng cùng cơ chế dừng sớm
của mã công bố và báo cáo checkpoint tốt nhất.

**Hàm loss của APKGC là gì?**
Công thức (20), dùng lại loss của RotatE (sigmoid-based, có self-adversarial
negative sampling). Vế thứ nhất, −log σ(λ − S), kéo khoảng cách của bộ ba đúng
xuống dưới margin λ (mã công bố đặt 12). Vế thứ hai đẩy khoảng cách của E mẫu âm
lên trên λ, với trọng số vᵢ lớn hơn cho mẫu âm khó (mô hình đang chấm gần đúng).
E là số mẫu âm cho mỗi bộ ba đúng: 64 theo bài, 32 trong file chạy của repo. Hai
thành phần mới của APKGC không nằm trong loss; chúng tác động lên đầu vào và bước
mã hoá.

**Token bị phạt có đúng là token vô nghĩa không, hay bị phạt nhầm?**
Bài chỉ có hai ví dụ case study, không đánh giá định lượng. Trong mã công bố, khối
có bước phạt chỉ nhận ba token mỗi thực thể (cấu trúc, ảnh, văn bản), nên thứ bị phạt
là cả một phương thức, không phải một từ như dấu chấm. Biến thể A không phạt lần nào;
B phạt quá nhỏ. Với C và B2, em chưa ghi lại vị trí bị phạt, nên chưa trả lời được
đúng hay nhầm.

**Loss lan truyền ngược tới bước nào?**
Không tới BERT và BEiT: đặc trưng tính sẵn và đóng băng. Loss cập nhật lớp chiếu văn
bản/ảnh, khối Transformer một lớp, vector thực thể và quan hệ, và θ_r (chỉ qua nhánh
nhiễu của vector thực thể). Bước phạt không có tham số, gradient đi xuyên qua.

**Hai thành phần mới dùng lúc train hay cả lúc test?**
Bài nói "trong quá trình huấn luyện". Trong mã công bố: nhiễu chỉ thêm khi train; bước
phạt nằm thẳng trong hàm attention nên chạy cả khi test. Mỗi bước train: lấy batch bộ
ba đúng và E mẫu âm, thêm nhiễu, qua Transformer có bước phạt, RotatE chấm điểm, tính
loss, lan truyền ngược, Adam cập nhật. Khi test: không nhiễu, chấm điểm mọi thực thể
rồi xếp hạng.

**Bài có điểm yếu gì về trình bày và phương pháp?** (chỉ trả lời nếu được hỏi)
- Công thức (7) đưa vector m_t vào tokenizer, trong khi tokenizer nhận chuỗi ký tự;
  mã công bố làm ngược lại (đặc trưng tính sẵn, nhiễu cộng vào vector).
- Bài ghi "bốn câu hỏi nghiên cứu" nhưng liệt kê năm; bảng ablation thành phần không
  ghi bộ dữ liệu (số khớp MKG-W); H@3 DB15K là 0,413 ở Bảng 2 nhưng 0,411 ở Bảng 3.
- Trên DB15K chỉ hơn NativE 0,002 MRR, thua ở H@10 (0,527 so với 0,529), không có độ
  lệch chuẩn hay kiểm định thống kê, không báo cáo thời gian huấn luyện.
- Công thức (9)(10) không nói phạt trước hay sau softmax; mã phạt trước softmax nên
  tổng xác suất vẫn bằng 1. Chia phần dư cho "token lân cận" không được biện luận;
  trong mã chỉ có 3 token và k = 1.
- Chọn nhiễu theo bộ dữ liệu không được giải thích. Nếu bị gợi ý là "cherry-picking":
  em chỉ nói được là bài không giải thích và không có thực nghiệm so sánh.

**Ngưỡng thích ứng (biến thể C) là gì?**
Thay vì so với một ngưỡng cố định (như φ trong bài, hay số 10 trong mã công
bố), biến thể C tính entropy chuẩn hoá H ∈ [0, 1] của phân phối attention trên
mỗi hàng (0 = dồn hết vào một token, 1 = trải đều). Ngưỡng τ là phân vị 10% của
H trong từng batch, nên không phải một con số cố định: hàng nào có H < τ, tức
thuộc 10% hàng "nhọn" nhất, thì phạt token có xác suất lớn nhất của hàng đó. Lý
do chọn cách này: nó không thêm siêu tham số phải dò theo từng bộ dữ liệu. Phần chia lại cho token lân cận giữ nguyên như bài. Kết quả: MRR
0,3630, không khác biến thể A.

**Vì sao phải tự cài đặt lại thay vì chỉ dùng mã công bố?**
Mã công bố có thể không phải phiên bản dùng để tạo số liệu trong bài. Cài đặt lại
từ chính công thức của bài giúp kết luận không phụ thuộc vào repo.

**Vì sao điều kiện phạt trong mã công bố không bao giờ được kích hoạt?**
Mã công bố so ngưỡng 10 với *tích* các điểm attention từ một vị trí đến cuối
hàng, chứ không so với điểm của chính token đó. Điểm lớn nhất đo được chỉ 10,8,
và 62,7% thừa số mang dấu âm, nên tích gần như không bao giờ vượt 10.

**Khác biệt phần cứng có giải thích được khoảng chênh không?**
Chưa kiểm chứng được vì không có RTX 4090. Tuy nhiên chỉ hiệu chỉnh batch và số
mẫu âm, trên cùng GPU Tesla T4, đã thu hẹp 64% khoảng chênh. Vì vậy phần lớn
khoảng chênh không nằm ở phần cứng; phần 0,005 còn lại chưa loại trừ được.
