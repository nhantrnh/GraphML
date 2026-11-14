# BÁO CÁO CHUYÊN ĐỀ

## ImparTail: Towards Unbiased and Robust Spatio-Temporal Scene Graph Generation and Anticipation

| | |
|---|---|
| **Hội nghị** | IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2025), **Highlight** (~3,7%) |
| **Tác giả** | Rohith Peddi, Saurabh, Ayush Abhay Shrivastava, Parag Singla, Vibhav Gogate — UT Dallas & IIT Delhi |
| **DOI** | 10.1109/cvpr52734.2025.00808 |
| **Mã nguồn** | https://github.com/rohithpeddi/ImparTail |
| **Môn học** | GraphML — HP3 |

---

## 1. Lý do chọn bài báo

**Chất lượng công bố.** CVPR 2025 chấp nhận bài ở dạng Highlight — khoảng 387
trong hơn 13.000 bài nộp, tức nhóm 3,7% được đánh giá cao nhất.

**Bài toán chưa quen thuộc.** Đồ thị cảnh không-thời gian (STSG) mở rộng đồ thị
tri thức sang miền video: nút là vật thể phát hiện được trong khung hình, cạnh
là quan hệ giữa chúng, và toàn bộ cấu trúc **thay đổi theo thời gian**. Đây là
phần chưa được học trong khoá, khác với đồ thị tri thức tĩnh ở Chủ đề 9.

**Khả năng tái lập.** Mã nguồn công khai kèm giấy phép Apache 2.0, và — điểm
hiếm trong lĩnh vực — tác giả phát hành **26 checkpoint** qua bốn đường dẫn
Box, phủ cả mô hình nền lẫn mô hình đã huấn luyện bằng phương pháp của họ, ở ba
chế độ và ba tỉ lệ che.

**Khả năng phản biện.** Bài có bốn đóng góp, trong đó một đóng góp — sinh mặt nạ
theo lịch (curriculum) — cho phép đối chiếu giữa điều bài báo mô tả và điều mã
nguồn thực hiện.

---

## 2. Kiến thức nền

### 2.1 Từ đồ thị tri thức sang đồ thị cảnh

Đồ thị tri thức lưu sự kiện dưới dạng bộ ba `(chủ thể, quan hệ, khách thể)`.
Đồ thị cảnh dùng cùng cấu trúc nhưng chủ thể và khách thể là **vật thể trong
ảnh**, còn quan hệ là tương tác giữa chúng — ví dụ
`(người, đang_cầm, cốc)`.

Đồ thị cảnh **không-thời gian** thêm chiều thời gian: mỗi khung hình có một đồ
thị, và các đồ thị này liên kết với nhau qua danh tính vật thể. Một vật thể
xuất hiện ở khung 10 và khung 50 phải được nhận là cùng một vật thể.

### 2.2 Hai tác vụ

| Tác vụ | Đầu vào | Đầu ra |
|---|---|---|
| **VidSGG** | video đã quan sát | đồ thị cho các khung đã thấy |
| **SGA** | phần đầu video | đồ thị cho các khung **chưa thấy** |

Bài báo xử lý cả hai. Báo cáo này tập trung vào VidSGG vì đó là phần nhóm tái
lập được.

### 2.3 Ba chế độ đánh giá

Quy ước của lĩnh vực, quyết định mô hình được cho sẵn bao nhiêu:

- **PredCls** — cho sẵn hộp bao và nhãn vật thể, chỉ phân loại quan hệ
- **SGCls** — cho sẵn hộp bao, phải đoán cả nhãn vật thể lẫn quan hệ
- **SGDet** — không cho gì, phải phát hiện vật thể từ đầu

### 2.4 Bộ dữ liệu Action Genome

Chuẩn của lĩnh vực. Chú thích quan hệ giữa **người** và vật thể trong video
Charades, chia thành ba nhóm vị ngữ:

| nhóm | số lớp | ví dụ |
|---|---|---|
| attention | 3 | `looking_at`, `not_looking_at`, `unsure` |
| spatial | 6 | `in_front_of`, `behind`, `above` |
| contacting | 17 | `holding`, `touching`, `eating` |

Nhóm đếm trực tiếp từ tệp chú thích: tập huấn luyện có **417.302** cặp
(người, vật) mang nhãn, phân bố cực kỳ lệch — `holding` xuất hiện **169.741**
lần trong khi `twisting` chỉ **93** lần, tỉ lệ **1.825:1**.

---

## 3. Vấn đề bài báo đặt ra

### 3.1 Phân bố đuôi dài

Khi một lớp chiếm gần 40% dữ liệu còn lớp khác chiếm 0,02%, mô hình tối ưu
theo độ chính xác tổng thể sẽ học cách **luôn đoán lớp phổ biến**. Chỉ số
**Recall@K** che giấu điều này vì nó tính trên toàn bộ dự đoán; **mean
Recall@K** — trung bình recall của từng lớp — mới lộ ra.

### 3.2 Thiếu bền vững trước nhiễu đầu vào

Mô hình huấn luyện trên video sạch có còn hoạt động khi video bị mờ, nhiễu, hay
thiếu sáng không? Bài báo đặt ra hai tác vụ mới — **Robust VidSGG** và
**Robust SGA** — với 16 kiểu nhiễu đầu vào.

---

## 4. Phương pháp

### 4.1 Masked loss — ý tưởng trung tâm

Thay vì đổi kiến trúc, bài báo đổi **hàm mục tiêu**. Công thức (5) của bài:

    L_ijk = (1 − m_ijk) · L_ijk = { L_ijk nếu m = 0 ; 0 nếu m = 1 }

kèm câu giải thích: *"A mask initialized i.e. setting m = 1 **voids the
contribution** of loss."* Nghĩa là nhãn bị che phải đóng góp **đúng bằng 0** vào
hàm mất mát — mô hình không học gì từ nó, chứ không phải học điều ngược lại.

Ý đồ: che bớt nhãn của **lớp phổ biến** để gradient dồn về lớp hiếm.

### 4.2 Sinh mặt nạ

Algorithm 1 của bài báo:

    Prob[rel] = 1/|P|
    Tar[rel] ~ Multinomial(N_target, Prob[rel])

Đây là phân bố **đều trên lớp vị ngữ**, không phải trên từng thể hiện. Hệ quả:
lớp phổ biến bị che nhiều hơn *trong kỳ vọng*, tức cơ chế đã cân bằng lớp một
cách **ngầm định**.

### 4.3 Curriculum

Bài báo mô tả tỉ lệ che tăng dần theo epoch: `R_m = e × R_s`. Ý tưởng là bắt
đầu với phân bố gốc rồi chuyển dần sang phân bố cân bằng.

---

## 5. Thực nghiệm — nhóm tái lập

### 5.1 Cấu hình

Chế độ **PredCls**, mô hình nền **STTran**, toàn bộ **1.750 video** tập kiểm
thử Action Genome, checkpoint do chính tác giả phát hành. Chạy trên Tesla P100
của Kaggle, khoảng một giờ.

### 5.2 Kết quả

Chín con số mean Recall so với Bảng 1 của bài báo:

| ràng buộc | mR@10 | mR@20 | mR@50 |
|---|---|---|---|
| With | 30,5 → **30,70** | 34,7 → **35,54** | 34,8 → **35,65** |
| No | 45,7 → **44,33** | 63,4 → **63,80** | 80,5 → **80,73** |
| Semi | 36,6 → **36,24** | 51,8 → **52,67** | 53,8 → **55,32** |

**Lệch trung bình 0,74 điểm, lớn nhất 1,52.**

Cần nói rõ phạm vi: đây là số của mô hình **nền** STTran, không phải bản
+ImparTail. Điều được xác nhận là **pipeline đánh giá, dữ liệu và checkpoint
của tác giả tái lập đúng** — không phải phương pháp đúng.

#### Vì sao R@50 gần như bằng R@20 — trần bão hoà của chỉ số

Ở ràng buộc *with*, R@20 = 64,94 nhưng R@50 = 64,96 — tăng vỏn vẹn **0,02
điểm** khi K tăng gấp 2,5 lần. Không phải nhiễu: đây là **trần bão hoà** do dữ
liệu, không do mô hình. Ở *semi* hiệu ứng nhẹ hơn (82,11 → 83,27), ở *no
constraint* thì không có (91,70 → 99,11).

Với *with*/*semi* constraint, mã đánh giá lấy `rel_scores.argmax(1)` — **đúng
một** vị ngữ mỗi cặp (`evaluation_recall.py:173-181`). Nên K không so với
không gian dự đoán mà so thẳng với **số nhãn đúng trong khung**.

Đếm trên tập train (198.973 khung có vật thể):

| | max | K=20 phủ hết |
|---|---|---|
| vật thể mỗi khung | 8 | 100% |
| **nhãn mỗi khung** | **29** | **99,84%** |

Chỗ dễ nhầm nằm ở đây: mỗi cặp mang **nhiều nhãn đồng thời** (attention +
spatial + contacting), nên số nhãn lớn hơn hẳn số vật thể. Chỉ **0,16% số
khung** (≈318 khung) có hơn 20 nhãn — và chúng tạo ra toàn bộ khoảng hở
64,94 → 64,96.

Suy ra một **tiên đoán kiểm chứng được**: vì K=50 phủ 100% số khung, R@50 phải
**bằng đúng** R@100 ở *with* và *semi*. Số tái lập xác nhận — cả hai lệch đúng
**0,0000**, trong khi *no constraint* lệch 0,76 điểm như dự kiến. QA khẳng
định với dung sai 5·10⁻³ (`qa/verify_numbers.py:227`).

Hệ quả cho việc đọc bảng: ở *with*/*semi*, **R@50 và R@100 không mang thêm
thông tin** so với R@20 — mọi so sánh giữa các phương pháp tại K lớn đều diễn
ra dưới trần này. Chỉ *no constraint* (xếp hạng 26 vị ngữ mỗi cặp) mới còn dư
địa: ở đó K=50 chỉ bão hoà tuyệt đối trên 34,2% số khung, K=100 trên 89,7%.

### 5.3 Sáu rào cản kỹ thuật

Ghi lại vì chúng phản ánh tình trạng tái lập của lĩnh vực:

1. **torch không hợp GPU.** Kaggle cấp Tesla P100 (sm_60), torch cài sẵn 2.10
   chỉ hỗ trợ từ sm_70. Phải hạ xuống 2.4.1+cu121.
2. **Header `THC/THC.h` bị gỡ** từ torch 1.11. Mã CUDA của fasterRCNN vẫn dùng.
3. **51 chỗ dùng API đã bỏ** — `.type()`, `.data<T>()`.
4. **`np.float`, `np.bool`** bị gỡ ở NumPy 1.24, còn 13 chỗ.
5. **Thiếu dữ liệu.** Phải gắn đủ **6 part** khung hình (~58 GB); gắn 2 part cho
   3.200 thư mục video nghe như dư so với 1.814 video test, nhưng đó là video
   ngẫu nhiên trên toàn tập — thiếu 1.224/1.814 video test, và lần chạy đầu
   dừng ở video 553/1.750.
6. **Tên checkpoint.** `stsg_base.py:51` tách tên theo
   `"_".join(split("_")[:-2])`, nên `sttran_predcls.tar` cho chuỗi rỗng — phải
   đổi thành `sttran_predcls_epoch_3.tar`.

Tất cả đều giải được. Không có rào cản nào là thật.

---

## 6. Đánh giá phản biện

### 6.1 Curriculum — đóng góp thứ tư không tạo ra số trong bảng chính

Ba nguồn độc lập trong chính kho mã và bài báo cùng chỉ một hướng:

| nguồn | nội dung |
|---|---|
| Remark trang 7 | *"a fixed Stochastic Masking Ratio of 0.9 achieved **similar performance**... we opted for a **fixed Masking Ratio of 0.9**"* |
| `train_sgg_methods.py:402` | `is_curriculum = False` |
| `scripts/*/train.txt` | `grep -rn curriculum scripts/` → **không kết quả** |

Curriculum chiếm một phần tư Hình 3 của bài báo nhưng số trong bảng chính đến
từ tỉ lệ che **cố định**. Câu hỏi để lại: đóng góp thật nằm ở *curriculum* hay
chỉ ở *việc che*?

Phát biểu thận trọng: đây là mô tả **kho mã đã công bố**, không kết luận tác giả
chạy thí nghiệm thế nào — rất có thể phiên bản họ chạy khác phiên bản commit.

### 6.2 Đánh đổi head–tail chỉ công bố trong phụ lục

Bảng 1 — bảng chính — có **13 lần `mR@`** và **0 lần `R@`** trần. Recall thường
chỉ xuất hiện ở Bảng 6 phụ lục, và ở đó:

| cấu hình | R@50 | mR@50 |
|---|---|---|
| SGCLS STTran S=10 | 46,5 → 41,1 (**−11,6%**) | 27,6 → 36,2 (+31,2%) |
| SGCLS DSGDetr S=10 | 49,4 → 42,0 (**−15,0%**) | 28,1 → 36,0 (+28,1%) |

Trên Bảng 3 (độ bền), nhóm đếm được **21/30 ô Recall giảm (70%)** trong khi
**0/90 ô mean Recall giảm** — đánh đổi có hệ thống, không phải nhiễu.

Phần Limitations của bài gọi mức này là *"although small"*. Thực tế là **6–17%**.

### 6.3 Nhưng phải công bằng với bài báo

Nhóm tính **trung bình điều hoà hR** — chỉ số cân bằng R và mR mà mã của chính
tác giả đã tính sẵn (`evaluation_recall.py:66-69`) nhưng bài báo không công bố:

| K | ΔR | ΔmR | **ΔhR** |
|---|---|---|---|
| 10 | −1,7 | +26,1 | **+17,7** |
| 20 | −3,0 | +25,2 | **+15,4** |
| 50 | 0,0 | +13,0 | **+7,4** |

hR tăng **7–18 điểm**. Phương pháp **có lợi thật**, không phải chỉ dịch chuyển
chỉ số. Nếu chỉ nêu mặt xấu thì đó là xuyên tạc.

### 6.4 Bốn hạn chế tác giả tự nêu

Phụ lục trang 15: chỉ thí nghiệm trên Action Genome; 16 kiểu nhiễu chưa phủ hết
thực tế; có đánh đổi head–tail; chỉ dùng recall/mean recall nên không đo được
quan hệ bậc cao.

---

## 7. Rà soát tài liệu

Nhóm khảo sát các công trình 2024–2026 để xem hạn chế nào đã được khắc phục.

| hạn chế | đã ai làm chưa |
|---|---|
| Chỉ Action Genome | **Chính nhóm tác giả** — ActionGenome4D (preprint 3/2026) nâng lên 4D |
| Đuôi dài | 7/8 bài cùng làm — VISA (CVPR 2025), STABILE (TMM 2025), TD²-Net (AAAI 2024) |
| Nhất quán thời gian | 6/8 bài **than phiền** nhưng **không bài nào** đề xuất chỉ số đo nó |
| Robustness sau ImparTail | **không tìm thấy** bài nối tiếp nào |

Hai kết quả âm đáng ghi: **không tồn tại** "Action Genome++" hay bản gán nhãn
lại nào, và **không bài nào** trong 304 công trình trích dẫn Action Genome phê
bình dữ liệu hoặc cách đánh giá của chính nó.

---

## 8. Hướng cải tiến đề xuất

### 8.1 Cài đặt mặt nạ lệch khỏi công thức (5)

Mã nhánh BCE (`stsg_base.py:305-318`) dựng vector đích toàn 0 rồi chỉ set 1 cho
nhãn **sống sót**. Với `nn.BCELoss()` reduction `'mean'` trên toàn bộ `max_len`
phần tử, nhãn dương bị che trở thành **nhãn âm** và **vẫn sinh gradient**.

Kiểm bằng đạo hàm dạng đóng của BCE (q = 0,6, tỉ lệ giữ 10%, 200.000 mẫu):

| | gradient |
|---|---|
| Đúng (y = 1) | **−1,667** — đẩy xác suất lên |
| Loại số hạng khỏi loss | −0,168 — cùng dấu |
| **Biến thành âm (mã công bố)** | **+2,084** — **ngược dấu** |

Không phép đổi thang nào sửa được dấu.

**Ngưỡng đổi dấu.** Mã cho `E[∂L/∂z] ∝ (q − p_c)` trong khi đúng ra phải là
`∝ (q − 1) < 0`. Suy ra cập nhật đảo chiều khi mô hình tin vào nhãn nhiều hơn
`p_c`. Với `holding`, `p_c = 0,016` — mô hình chỉ cần gán **>1,6%** xác suất cho
một quan hệ **có thật** là bị đẩy ngược.

### 8.2 Kết quả sửa

Nhóm vá theo đúng công thức (5) — `BCELoss(reduction='none')`, nhân mặt nạ theo
từng phần tử, chuẩn hoá theo số phần tử hiệu lực — rồi chạy hai nhánh cùng cấu
hình, khác đúng một cờ:

| ràng buộc | R@50 gốc → vá | mR@50 gốc → vá | ΔhR |
|---|---|---|---|
| with | 54,9 → **64,3** (+9,4) | 52,4 → 50,9 (−1,6) | **+3,2** |
| no | 99,6 → 99,6 (0,0) | 97,6 → 96,5 (−1,1) | −0,5 |
| semi | 21,7 → **33,1** (+11,4) | 13,8 → **22,2** (+8,4) | **+9,7** |

**ΔhR trung bình +4,11 điểm.** Ở semi-constraint **cả hai chỉ số cùng tăng**.

**Giới hạn phải nêu:** mỗi nhánh mới chạy **một seed, hai epoch** thay vì bốn
của tác giả. Con số tuyệt đối không so trực tiếp với bảng bài báo được, chỉ so
hai nhánh với nhau. Cần ít nhất ba seed trước khi coi là kết luận.

### 8.3 Hướng đã kiểm và loại

Ghi lại để không ai lặp lại:

- **Che theo tần suất lớp.** Kiểm giải tích cho thấy sơ đồ này **trùng 0,0%**
  với Algorithm 1 — `Prob[rel] = 1/|P|` đã cân bằng lớp ngầm định.
- **Giữ cặp bị vứt bằng mục tiêu nhị phân.** Khi mặt nạ che hết nhãn của một
  cặp, mã loại luôn cả cặp (`continue`) — ở tỉ lệ giữ 10% là **58%** số cặp.
  Nhưng **88,2%** cặp bị vứt thuộc 4 lớp phổ biến nhất, nên dạy "cặp này có
  quan hệ" sẽ củng cố đúng thiên lệch cần phá.

---

## 9. Kết luận

ImparTail đổi **hàm mục tiêu** thay vì kiến trúc — một phép nhân với mặt nạ nhị
phân, không thêm tham số. Cách tiếp cận này gọn và gắn được lên nhiều mô hình
nền, đó là điểm mạnh thật.

Nhóm tái lập được kết quả của mô hình nền với sai lệch **0,74 điểm**, xác nhận
pipeline đánh giá và checkpoint của tác giả hoạt động đúng. Sáu rào cản kỹ thuật
gặp phải đều giải được, không cái nào là thật.

Ba quan sát từ việc đọc mã đối chiếu bài báo: curriculum không tạo ra số trong
bảng chính; đánh đổi head–tail có thật ở mức 6–17% nhưng chỉ công bố trong phụ
lục; và cài đặt mặt nạ lệch khỏi công thức (5) theo hướng đổi dấu gradient. Sửa
điểm cuối cho **+4,11 điểm hR**, với Recall tăng tới **+11,4** điểm.

Cần nói cho đúng mức: những quan sát này **không** phủ nhận đóng góp của bài
báo. Tính trung bình điều hoà từ chính số liệu của họ cho thấy phương pháp tăng
**7–18 điểm** — nó có lợi thật. Điều báo cáo này chỉ ra là một phần cái giá
được báo cáo có thể đến từ cài đặt chứ không từ bản chất phương pháp.

---

## Tài liệu tham khảo chính

1. R. Peddi, Saurabh, A. A. Shrivastava, P. Singla, and V. Gogate, "Towards
   Unbiased and Robust Spatio-Temporal Scene Graph Generation and
   Anticipation," in *CVPR*, 2025. DOI 10.1109/cvpr52734.2025.00808
2. Y. Cong, W. Liao, H. Ackermann, B. Rosenhahn, and M. Y. Yang,
   "Spatial-Temporal Transformer for Dynamic Scene Graph Generation," in
   *ICCV*, 2021. DOI 10.1109/iccv48922.2021.01606
3. J. Ji, R. Krishna, L. Fei-Fei, and J. C. Niebles, "Action Genome: Actions as
   Compositions of Spatio-temporal Scene Graphs," in *CVPR*, 2020.
   DOI 10.1109/cvpr42600.2020.01025
4. S. Nag, K. Min, S. Tripathi, and A. K. Roy-Chowdhury, "Unbiased Scene Graph
   Generation in Videos," in *CVPR*, 2023. DOI 10.1109/cvpr52729.2023.02184
5. M. Grunde-McLaughlin, R. Krishna, and M. Agrawala, "AGQA: A Benchmark for
   Compositional Spatio-Temporal Reasoning," in *CVPR*, 2021.
   DOI 10.1109/cvpr46437.2021.01113
