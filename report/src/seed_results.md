# Kết quả chạy lại APKGC trên MKG-W

Chạy trên Kaggle, GPU Tesla P100, mỗi seed một phiên riêng. Siêu tham số đọc
trực tiếp từ `scripts/run_base.sh` của tác giả; chỉ thay đổi tham số `-seed`.

---

## Số liệu thu được

| Seed | MRR | Hits@1 | Hits@3 | Hits@10 | MR | Thời gian |
|---|---|---|---|---|---|---|
| 42 | 0,3641 | 0,2956 | 0,3954 | 0,4840 | 778,0 | 158 phút |
| 7 | 0,3616 | 0,2942 | 0,3918 | 0,4853 | 796,8 | 159 phút |
| **Trung bình** | **0,3629 ± 0,0018** | 0,2949 ± 0,0010 | 0,3936 ± 0,0026 | 0,4846 ± 0,0009 | | |
| Bài báo công bố | 0,374 | 0,306 | 0,404 | 0,501 | | |

Cả hai lần chạy đều dừng nhờ early stopping trước khi hết 8000 epoch.

---

## Hai quan sát

### Mô hình ổn định giữa các seed

Độ lệch chuẩn MRR giữa hai lần chạy là **0,0018**, biên độ 0,0025. Con số này
nhỏ hơn nhiều so với khoảng cách 0,008 mà bài báo dùng để tuyên bố vượt NativE
trên MKG-W.

Đây là kết quả ngược với giả thuyết ban đầu của nhóm. Trước khi chạy, nhóm dự
đoán mức cải thiện dưới một điểm phần trăm có thể nằm trong nhiễu ngẫu nhiên.
Số liệu cho thấy không phải vậy: APKGC cho kết quả nhất quán giữa các seed.

Cần lưu ý hai seed là quá ít để ước lượng độ lệch chuẩn một cách tin cậy. Con
số 0,0018 chỉ mang tính chỉ báo.

### Kết quả đo được thấp hơn số công bố khoảng 0,011 MRR

Chênh lệch xuất hiện đều ở cả bốn chỉ số và ở cả hai seed. Nhóm **chưa xác định
được nguyên nhân**, và có ít nhất bốn cách giải thích chưa loại trừ được:

**Cấu hình không khớp.** Bài báo (tr. 15010) ghi batch size 2048 và 64 mẫu âm.
Script `scripts/run_base.sh` — file mà `run.sh` thực sự gọi — đặt 1024 và 32.
Lần chạy này dùng giá trị trong script. Đây là cách giải thích đơn giản nhất và
nhiều khả năng nhất.

**Phần cứng khác.** Bài báo dùng RTX 4090, lần chạy này dùng Tesla P100.

**Phiên bản thư viện khác.** Phải dùng torch 2.4.1 vì bản mặc định trên Kaggle
không hỗ trợ kiến trúc GPU được cấp.

**Early stopping.** Cả hai lần đều dừng sớm, có thể trước khi mô hình đạt đỉnh.

Muốn tách bạch, cần chạy lại với đúng cấu hình trong bài báo (batch 2048, 64 mẫu
âm) trên cùng phần cứng. Nhóm chưa thực hiện bước này.

---

## Những trở ngại gặp phải khi chạy lại

Ghi lại đầy đủ vì đây là phần công việc thực tế, và cũng cho thấy khoảng cách
giữa mã nguồn công bố và khả năng chạy lại của người ngoài.

**Thiếu mã nguồn C++.** `mmkgc/make.sh` cần thư mục `mmkgc/base/` chứa `Base.cpp`
và các header, nhưng thư mục này không có trong repo. File `Base.so` được commit
sẵn là bản biên dịch cho Linux x86-64 nên không chạy trên kiến trúc khác.

Khắc phục: `mmkgc/` là bản sửa đổi của OpenKE, nên lấy `base/` từ repo gốc. Chữ
ký hàm `sampling()` khớp đúng 11 tham số mà `TrainDataLoader.py` khai báo, và
biên dịch thành công trên cả ARM64 lẫn x86-64.

**Gọi `.cuda()` cứng ở 17 vị trí.** Không chạy được nếu không có GPU NVIDIA. Đã
thay bằng một module chọn thiết bị.

**Thiếu phụ thuộc.** `requirement.txt` không liệt kê `scikit-learn` và `scipy`,
nhưng `Tester.py` cần chúng.

**Đường dẫn lưu nằm ngoài dự án.** `run_KGC_train.py` ghi kết quả ra `../../data/`.

**Sập khi chạy dưới 100 epoch.** `best_model_wts` chỉ được tạo tại mốc lưu
checkpoint nhưng lại được truy cập vô điều kiện ở cuối chương trình.

**Ràng buộc phiên bản torch quá lỏng.** Repo ghim `torch>=1.6.0` không có cận
trên. Bản torch mặc định trên Kaggle hiện nay không còn hỗ trợ kiến trúc sm_60
của Tesla P100, nên phải hạ xuống 2.4.1.

---

## Chi phí tính toán

Đo trên phần cứng tương đương:

| Khâu | Thời gian |
|---|---|
| Huấn luyện | khoảng 8,4 giây mỗi epoch |
| Đánh giá | khoảng 21 phút mỗi lần |

`run_link_prediction` xếp hạng tuần tự 4.274 bộ ba kiểm thử theo hai chiều, mỗi
lần trên 15.000 thực thể, không gộp lô. Với cấu hình gốc — 8000 epoch, đánh giá
mỗi 100 epoch — riêng khâu đánh giá cần khoảng 28 giờ nếu chạy hết.

Thực tế cả hai seed dừng ở khoảng 158 phút nhờ early stopping.

Một lần chạy gộp cả năm seed trong một kernel đã bị Kaggle cắt khi hết giới hạn
12 giờ mà chưa xong seed đầu tiên, nên phải tách thành các kernel riêng.

---

## Cách tái lập

Notebook `kaggle_apkgc_seeds.ipynb` tự xử lý toàn bộ các trở ngại nêu trên: bổ
sung `mmkgc/base/`, biên dịch lại `Base.so`, thay các lời gọi `.cuda()`, cài phụ
thuộc còn thiếu, tải bộ nhúng, và hạ phiên bản torch khi GPU không tương thích.
