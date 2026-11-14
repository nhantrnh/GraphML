# Trạng thái thực nghiệm — APKGC

Ghi lại chính xác những gì nhóm đã chạy, chưa chạy, và mức độ tin cậy của từng
kết quả. Cập nhật: 20/8/2026.

Nguyên tắc: mọi số trong cột "đã đo" đều do nhóm tự chạy và có log kiểm chứng
được. Số của tác giả luôn ghi rõ nguồn là bảng nào, trang nào.

---

## Phần 1 — Đã chạy, có số liệu

### 1.1 Huấn luyện lại APKGC trên MKG-W

Năm lần chạy chia hai đợt với **điều kiện khác nhau**, phải báo cáo tách biệt:

| Đợt | Seed | Epoch | Đánh giá mỗi | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---|---|---|---|---|---|
| 1 | 7 | 8000 | 100 | 0,3616 | 0,2942 | 0,3918 | 0,4853 |
| 1 | 42 | 8000 | 100 | 0,3641 | 0,2956 | 0,3954 | 0,4840 |
| 2 | 1234 | 2000 | 250 | 0,3610 | 0,2987 | 0,3864 | 0,4759 |
| 2 | 2024 | 2000 | 250 | 0,3611 | 0,2943 | 0,3907 | 0,4819 |
| 2 | 3407 | 2000 | 250 | 0,3595 | 0,2925 | 0,3917 | 0,4799 |

Trung bình đợt 1 (n=2): MRR 0,3629
Trung bình đợt 2 (n=3): MRR $0{,}3605 \pm 0{,}0009$

**Ước lượng độ lệch chuẩn lấy từ đợt hai**, vì đây là ba lần chạy duy nhất cùng
điều kiện. Gộp cả năm lần cho 0,0017, nhưng con số đó trộn lẫn nhiễu seed với
chênh lệch 0,0023 giữa hai đợt --- vốn đến từ khác biệt về số epoch và tần suất
đánh giá.

Số công bố trong bài báo (Bảng 2, tr. 15010): MRR 0,374 · H@1 0,306 · H@3 0,404 · H@10 0,501

**Thiết lập.** Kaggle, GPU Tesla P100, torch 2.4.1. Siêu tham số đọc trực tiếp
từ `scripts/run_base.sh` của tác giả: `dim=128`, `batch=1024`, `margin=12`,
`neg_num=32`, `lr=1e-4`, `epoch=8000`, early stopping sau 10 lần đánh giá không
cải thiện. Chỉ thay đổi tham số `-seed`.

**Mức độ tin cậy: cao.** Chạy mã gốc, dữ liệu gốc, GPU thật. Log lưu tại
`results/MKG-W_seed7.log` chứa thanh tiến trình tqdm kèm dấu thời gian, cảnh báo
của PyTorch, và phần in ra từ thư viện C++ `Base.so` theo đúng định dạng của
`Test.h`:

```
averaged(filter):  0.210617   951.168030   0.321947   0.229293   0.151614
Epoch 1499 | loss: 0.517856:  19%|█▊  | 1499/8000 [2:37:24<11:22:38, 6.30s/it]
INFO - 08/20/26 08:42:20 - 2:39:12 - mrr:0.3615906536579132, ...
```

---

### 1.1b Thí nghiệm tách nguyên nhân: cấu hình bài báo

Chạy thêm một lần với `batch = 2048`, `neg_num = 64` (đúng tr. 15010), seed 42.

Lần chạy này dùng trần 2000 epoch, đánh giá mỗi 250 --- **cùng điều kiện đợt hai**,
nên đối chứng là đợt hai chứ không phải trung bình gộp.

| Chỉ số | Đợt 2 (1024/32) | Bài báo (2048/64) | Công bố | Thu hẹp |
|---|---|---|---|---|
| MRR | 0,3605 | 0,3691 | 0,3740 | 64% |
| Hits@1 | 0,2952 | 0,3028 | 0,3060 | 70% |
| Hits@3 | 0,3896 | 0,3997 | 0,4040 | 70% |
| Hits@10 | 0,4792 | 0,4874 | 0,5010 | 37% |

**Mức độ tin cậy: cao** về mặt số liệu (chạy thật, log kiểm chứng được), nhưng
**chỉ một seed** nên chưa có độ lệch chuẩn riêng cho cấu hình này.

Kết luận: giả thuyết cấu hình lệch được xác nhận --- chạy đúng cấu hình ghi
trong bài báo thu hẹp 64% khoảng lệch MRR. Riêng cấu hình đóng góp +0,0086 MRR,
gấp gần mười lần độ lệch chuẩn giữa các seed. Phần còn lại 0,0049 MRR chưa giải
thích được.

---

### 1.2 Audit tham số trên mô hình thật

Chạy `audit_apkgc_real.py` --- nạp lớp `APKGC` từ repo gốc, dùng **bộ nhúng thật**
của MKG-W (visual 383-d, textual 384-d), chạy 5 lượt forward/backward.

```
Tổng số tham số      : 37
Học được bình thường : 17
Đóng băng có chủ ý   :  7
Không nhận gradient  : 13
```

Mười ba tham số không nhận gradient chia ba nhóm:

**Nhóm một --- lỗi thực sự (1 tham số).** `noise_ratio` khai báo `nn.Parameter`
giống hệt `mask_ratio` ở dòng liền kề, nhưng chỉ tham gia phép so sánh
`torch.rand(n) < self.noise_ratio` nên không có đường gradient; `mask_ratio`
tham gia phép nhân nên học được. Bài báo mô tả tham số này là *"adaptively tuned
during the training process"* (Eq. 5).

**Nhóm hai --- nhánh không kích hoạt (8 tham số).** `weight_raw`, `comb_proj`,
`ent_attn`, `img_proj_2`, `text_proj_2` --- thuộc các cấu hình mà `run.sh` không
dùng tới.

**Nhóm ba --- lớp con không chạy (4 tham số).** Khối feed-forward của
`fusion_layer`. Với `use_intermediate=0` (giá trị dùng cho cả hai bộ dữ liệu),
`BertLayer.forward` thoát ngay sau attention.

**Mức độ tin cậy: cao.** Mã gốc, dữ liệu gốc. Ban đầu chạy với vector ngẫu nhiên
rồi chạy lại với bộ nhúng thật --- kết quả trùng khớp, đúng như dự đoán vì đường
đi của gradient do cấu trúc mã quyết định.

Tái lập: `python audit_apkgc_real.py --repo <APKGC> --dataset MKG-W`

---

### 1.3 Audit tĩnh năm repo MMKGC

Chạy `audit_static.py` --- phân tích cây cú pháp, không cần cài đặt gì.

| Repo | Tổng `nn.Parameter` | Có gradient | Nghi vấn | Không thấy dùng | Đóng băng |
|---|---|---|---|---|---|
| APKGC | 17 | 1 | **1** | 1 | 14 |
| AdaMF-MAT | 41 | 5 | 0 | 0 | 36 |
| MyGO | 29 | 14 | 0 | 15 | 0 |
| MoMoK | 11 | 5 | 1\* | 5 | 0 |
| DHNS | 14 | 0 | 0 | 0 | 14 |

\* Kiểm tra thủ công cho thấy đây là báo nhầm: `models/GAT.py` không nằm trên
đường chạy chính của MoMoK.

**Kết luận sau khi loại báo nhầm: chỉ một trường hợp thực sự**, ở APKGC. Hiện
tượng này **không phổ biến** trong lĩnh vực như giả thuyết ban đầu của nhóm.

**Mức độ tin cậy: trung bình.** Phân tích tĩnh không theo được luồng gọi hàm
giữa các tệp nên có thể bỏ sót hoặc báo nhầm; mọi trường hợp phát hiện cần xác
minh bằng audit động.

---

### 1.4 Đo chi phí tính toán

| Khâu | Thời gian | Cách đo |
|---|---|---|
| Huấn luyện | 8,4 giây mỗi epoch | MPS, đo trực tiếp |
| Đánh giá | 21,3 phút mỗi lần | 40 bước đầu rồi ngoại suy cho 4.274 bước |

`run_link_prediction` xếp hạng tuần tự 4.274 bộ ba kiểm thử theo hai chiều, mỗi
lần trên 15.000 thực thể, không gộp lô. Với cấu hình gốc (8000 epoch, đánh giá
mỗi 100), riêng khâu đánh giá cần khoảng 28 giờ nếu chạy hết.

**Mức độ tin cậy: cao** cho phần huấn luyện (đo trực tiếp), **trung bình** cho
phần đánh giá (ngoại suy từ 40 bước đầu).

---

### 1.5 Số chiều đặc trưng

Đọc trực tiếp từ tệp `.pth` tải về từ Google Drive của tác giả AdaMF-MAT:

| Tệp | Kích thước tensor | Dung lượng |
|---|---|---|
| DB15K-visual.pth | (12842, 4096) | 200 MB |
| DB15K-textual.pth | (12842, 768) | 37 MB |
| MKG-W-visual.pth | (15000, 383) | 21 MB |
| MKG-W-textual.pth | (15000, 384) | 21 MB |
| MKG-Y-visual.pth | (15000, 383) | 21 MB |
| MKG-Y-textual.pth | (15000, 384) | 21 MB |

Chênh lệch giữa DB15K và hai bộ còn lại là hơn mười lần về số chiều thị giác,
bắt nguồn từ việc dùng encoder khác nhau (VGG cho DB15K, BEiT cho MKG-W và
MKG-Y --- AdaMF-MAT nói rõ điều này trong bài).

**Mức độ tin cậy: cao.** Đọc trực tiếp từ tệp.

---

### 1.6 Kiểm chứng cơ chế Concrete relaxation

| Biến thể | Nhận gradient | Sau 300 bước |
|---|---|---|
| Bernoulli cố định (như bản gốc) | không | 0,2000 → 0,2000 |
| Concrete relaxation | có | 0,2000 → 0,0919 |
| Straight-through | có | 0,2000 → 0,0909 |

**Mức độ tin cậy: thấp hơn các mục trên.** Đây là **mô phỏng cơ chế** trong một
module độc lập (`concrete_noise.py`), **không phải** sửa trực tiếp mã APKGC rồi
huấn luyện lại. Nó chứng minh kỹ thuật khả thi về mặt gradient, nhưng chưa cho
biết hiệu quả thực tế trên bài toán.

### Kiểm chứng tái lập (26/08/2026) ✅

Chạy lại seed 42 với đúng cấu hình đợt một, phiên Kaggle độc lập, sau sáu ngày.

| Chỉ số | 20/08 | 26/08 | Lệch |
|---|---|---|---|
| MRR | 0,3641 | 0,3641 | +0,00001 |
| Hits@1 | 0,2956 | 0,2956 | +0,00002 |
| Hits@3 | 0,3954 | 0,3954 | +0,00001 |
| Hits@10 | 0,4840 | 0,4840 | −0,00003 |
| Thời gian | 158 phút | 159 phút | |

Cả hai lần dừng sớm ở **epoch 1499** sau 15 lần đánh giá (14 lần giữa chừng
cộng một lần cuối trên checkpoint tốt nhất), dù trần đặt 8000.

**Hai hệ quả.** Thứ nhất, mọi con số trong báo cáo tái lập được — không phải kết
quả một lần chạy may mắn. Thứ hai, độ lệch 0,0135 MRR so với số công bố không đến
từ dao động khi chạy, vì dao động đó **bằng không**; nguyên nhân phải nằm ở cấu
hình, phần cứng, hoặc phiên bản thư viện.

Điều này cũng cho thấy trần 8000 epoch trong `scripts/run_base.sh` **không bao giờ
được chạm tới** với cấu hình mặc định — early stopping cắt ở khoảng một phần năm.
Thí nghiệm "chạy đủ 8000 epoch" vì thế không còn ý nghĩa và đã được gỡ khỏi danh
sách chưa chạy.

Nhật ký: `experiments/results/apkgc_res3/seed42.log` · kernel `nhantrnh/apkgc-remaining-exp`

---

## Phần 2 — Chưa chạy

| Thí nghiệm | Ước tính | Ghi chú |
|---|---|---|
| Nhiều seed với cấu hình bài báo | ~15 giờ | Hiện chỉ có một seed cho cấu hình 2048/64 |
| Ablation tự chạy (bỏ noise, đổi fusion) | ~6 giờ | Xác minh Bảng 4 của tác giả |
| Bộ dữ liệu DB15K | ~4 giờ | Chưa chạy tập nào ngoài MKG-W |
| Bộ dữ liệu MKG-Y | ~4 giờ | Bài báo không dùng, nhưng đáng chạy để kiểm tra tổng quát hoá |
| Quét siêu tham số (Hình 3) | rất lâu | 16 cấu hình × 2,6 giờ, vượt xa quota một tuần |

**Về quota Kaggle.** Giới hạn 30 giờ mỗi tuần. Một lần chạy gộp cả năm seed
trong một kernel bị cắt ở mốc 12 giờ mà chưa xong seed đầu tiên. Bài học: tách
mỗi seed một kernel ngay từ đầu.

**Về tính đồng nhất điều kiện.** Vì phải rút ngắn epoch cho vừa phiên Kaggle, ba
seed đợt hai chạy khác điều kiện với hai seed đợt một. Ban đầu nhóm gộp cả năm
vào một mẫu và báo cáo độ lệch chuẩn 0,0017; rà soát lại cho thấy con số đó trộn
lẫn nhiễu seed với chênh lệch giữa hai đợt. Đã sửa thành báo cáo tách biệt. Bài
học: cố định toàn bộ điều kiện trước khi bắt đầu, đừng đổi giữa chừng.

---

## Phần 3 — Số nào là của tác giả

Để tránh nhầm lẫn, đây là những bảng trong slide và báo cáo lấy từ bài báo, nhóm
**chưa** tự đo lại:

| Nội dung | Nguồn |
|---|---|
| Bảng so sánh 19 mô hình nền | Bảng 2, tr. 15010 |
| Ablation theo mô thái | Bảng 3, tr. 15010 |
| Ablation theo thành phần | Bảng 4, tr. 15011 |
| Quét siêu tham số $k$ và $\theta_p$ | Hình 3, tr. 15011 |
| Nghiên cứu trường hợp Dennis Hopper | Hình 4, tr. 15011 |

Slide và báo cáo đều ghi rõ nguồn cho từng bảng.

---

## Phần 4 — Câu hỏi mở

**Khoảng lệch so với số công bố đến từ đâu? --- đã trả lời một phần.**

Thí nghiệm tách nguyên nhân (mục 1.1b) xác nhận: **cấu hình không khớp là nguyên
nhân chính**, chiếm 64% khoảng lệch MRR và 70% ở Hits@1 lẫn Hits@3.

Còn lại 0,0049 MRR chưa giải thích được, vẫn gấp khoảng ba lần độ lệch chuẩn giữa
các seed. Ba nguyên nhân chưa tách:

1. **Phần cứng.** Bài báo dùng RTX 4090, nhóm dùng Tesla P100.
2. **Phiên bản thư viện.** Phải hạ xuống torch 2.4.1 vì bản mặc định trên Kaggle
   không hỗ trợ kiến trúc sm_60.
3. **Số epoch bị giới hạn.** Chạy 2000 epoch với đánh giá mỗi 250, thay vì 8000
   và 100 như script gốc.

Đáng chú ý: Hits@10 chỉ thu hẹp 37%, thấp hơn hẳn ba chỉ số còn lại --- gợi ý phần
lệch còn lại ảnh hưởng tới đuôi phân phối xếp hạng nhiều hơn phần đầu.

---

## Phần 5 — Giới hạn diễn giải

Mọi quan sát trong tài liệu này nói về **những gì được công bố trong kho mã và
bài báo**, trên phần cứng và phiên bản thư viện cụ thể mà nhóm dùng.

Không thể suy ra từ đây rằng tác giả chỉ chạy một cấu hình, hay chưa từng đo độ
lệch chuẩn. Nhóm nghiên cứu hoàn toàn có thể đã thử nhiều seed và nhiều cấu hình
rồi chỉ công bố một script.

Phát biểu kiểm chứng được: *bài báo không báo cáo độ lệch chuẩn*, và *mã nguồn
công bố có những đặc điểm khiến người ngoài khó chạy lại*. Đây là vấn đề về cách
báo cáo và chia sẻ, không phải kết luận về phương pháp nghiên cứu của tác giả.

---

## Phụ lục — Danh sách tệp

**Mã kiểm thử**

| Tệp | Chức năng |
|---|---|
| `audit_gradient.py` | Bộ khung audit tham số, dùng chung |
| `audit_apkgc_real.py` | Chạy audit trên mô hình APKGC thật |
| `audit_static.py` | Phân tích tĩnh, chạy được trên mọi repo |
| `concrete_noise.py` | Mô phỏng ba biến thể lấy mẫu nhiễu |
| `pilot_seed_study.py` | Chạy nhiều seed cho AdaMF-MAT (chưa dùng tới) |

**Notebook**

| Tệp | Chức năng |
|---|---|
| `kaggle_apkgc_seeds.ipynb` | Chạy nhiều seed, đã dùng cho seed 42 và 7 |
| `kaggle_remaining_exp.ipynb` | Các thí nghiệm còn lại, có công tắc chọn phần chạy |
| `APKGC_Colab.ipynb` | Bản cho Google Colab |

**Kết quả và tài liệu**

| Tệp | Nội dung |
|---|---|
| `seed_results.md` | Kết quả hai seed kèm phân tích |
| `audit_result.md` | Kết quả audit đầy đủ, phân loại ba nhóm |
| `audit_5repos.md` | So sánh năm repo |
| `repo_audit.md` | Khảo sát mã nguồn năm phương pháp |
| `published_gaps.json` | Khoảng cách trong bài báo, có ghi nguồn từng số |
