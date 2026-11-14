# Khảo sát mã nguồn năm phương pháp MMKGC

Đo trực tiếp trên bản clone nguyên gốc từ GitHub, ngày 20/8/2026.

Mọi con số trong tài liệu này đều lấy bằng lệnh chạy được, ghi kèm bên dưới từng
bảng để người đọc tự kiểm chứng.

---

## 1. Cách xử lý random seed

| Phương pháp | Venue | Tham số dòng lệnh | Đặt seed trong mã | Ghi chú |
|---|---|---|---|---|
| **APKGC** | AAAI 2025 | `-seed`, mặc định 42 | `torch.manual_seed(args.seed)` | Hoạt động đầy đủ |
| **AdaMF-MAT** | LREC-COLING 2024 | `-seed`, mặc định 42 | `torch.manual_seed(args.seed)` | Hoạt động, nhưng các script trong `scripts/` không truyền tham số này |
| **MoMoK** | ICLR 2025 | Không có | `np.random.seed`, `torch.manual_seed` đọc từ dict cấu hình | Giá trị 10010 nằm trong `train.py:25`, đổi được nhưng phải sửa mã |
| **MyGO** | AAAI 2025 | Không có | `torch.manual_seed(2024)` cố định trong mã | `train_mygo_fgc.py:25-27`, không đổi được từ ngoài |
| **DHNS** | DASFAA 2025 | Không có | Không có lời gọi nào | Không cố định được tính ngẫu nhiên |

```bash
# Tham số dòng lệnh
grep -rn "add_argument.*seed" --include="*.py" <repo>
# Nơi đặt seed
grep -rn "manual_seed\|random.seed" --include="*.py" <repo>
```

**Nhận xét.** Chỉ hai trong năm phương pháp cho phép đổi seed từ dòng lệnh. Với ba
phương pháp còn lại, muốn chạy nhiều seed thì phải sửa mã nguồn. Riêng DHNS không
đặt seed ở đâu cả, nên hai lần chạy cùng cấu hình sẽ cho kết quả khác nhau.

Điều này nói lên rằng **hạ tầng để chạy nhiều seed chưa sẵn có** trong phần lớn mã
nguồn được công bố. Nó không nói lên tác giả đã hay chưa chạy nhiều seed — kho mã
chỉ phản ánh những gì được công bố.

---

## 2. Mức độ ràng buộc vào CUDA

| Phương pháp | Số lời gọi `.cuda()` cứng |
|---|---|
| MyGO | 39 |
| AdaMF-MAT | 28 |
| APKGC | 17 |
| DHNS | 6 |
| MoMoK | 0 |

```bash
grep -rn "\.cuda()" --include="*.py" <repo> | wc -l
```

Bốn trong năm phương pháp gọi `.cuda()` trực tiếp thay vì dùng biến thiết bị. Hệ quả
thực tế: mã không chạy được trên máy không có GPU NVIDIA nếu không sửa. MoMoK là
ngoại lệ, dùng cách viết linh hoạt hơn.

---

## 3. Khung mã nền

| Phương pháp | Dựa trên OpenKE (`mmkgc/`) | Có `base/` để biên dịch lại |
|---|---|---|
| AdaMF-MAT | Có | Có |
| DHNS | Có (nhánh từ AdaMF-MAT) | Có |
| APKGC | Có | **Không** — chỉ có `Base.so` dựng sẵn |
| MyGO | Không, mã riêng | — |
| MoMoK | Không, mã riêng | — |

Ba phương pháp dùng chung khung OpenKE nên định dạng dữ liệu và quy trình đánh giá
tương thích với nhau. Hai phương pháp còn lại viết lại từ đầu; MyGO còn dùng biểu
diễn dạng token thay vì vector nên không dùng chung được bộ nhúng.

### Trường hợp APKGC: thiếu mã nguồn C++

`mmkgc/make.sh` cần thư mục `mmkgc/base/` chứa `Base.cpp` và các header, nhưng thư
mục này không có trong repo. File `Base.so` được commit sẵn là bản biên dịch cho
Linux x86-64:

```
$ file mmkgc/release/Base.so
ELF 64-bit LSB shared object, x86-64
```

Hệ quả: trên máy kiến trúc khác (Apple Silicon, ARM server) không chạy được, và
không thể biên dịch lại vì thiếu mã nguồn.

Cách khắc phục đã kiểm chứng: lấy `base/` từ [OpenKE](https://github.com/thunlp/OpenKE)
gốc. Chữ ký hàm `sampling()` trong `Base.cpp` khớp đúng 11 tham số mà
`TrainDataLoader.py` của APKGC khai báo, và biên dịch thành công trên ARM64.

---

## 4. Số chiều đặc trưng

Đọc trực tiếp từ các tệp `.pth` tải về từ Google Drive của tác giả AdaMF-MAT:

| Bộ dữ liệu | Thị giác | Văn bản | Số thực thể |
|---|---|---|---|
| DB15K | **4096** | 768 | 12.842 |
| MKG-W | **383** | 384 | 15.000 |
| MKG-Y | **383** | 384 | 15.000 |

```python
import torch
t = torch.load("embeddings/DB15K-visual.pth", map_location="cpu")
print(t.shape)   # torch.Size([12842, 4096])
```

Chênh lệch giữa DB15K và hai bộ còn lại là hơn mười lần về số chiều thị giác. Điều
này bắt nguồn từ việc dùng encoder khác nhau — AdaMF-MAT nói rõ trong bài:

> *"We employ VGG for DB15K and BEiT for MKG-W/MKG-Y as PVEs"*

SNAG cũng mô tả tương tự. Con số 383 là bất thường: trích bằng BEiT thông thường sẽ
ra 384 hoặc 768. Việc con số này lặp lại y hệt trong bảng của nhiều bài cho thấy các
nhóm dùng chung một tệp đặc trưng có nguồn từ MMRNS (ACM MM 2022).

---

## 5. Kết quả audit tham số trên APKGC

Chạy `audit_apkgc_real.py` trên mô hình thật, cấu hình mặc định của `run.sh`:

```
Tổng số tham số      : 37
Học được bình thường : 17
Đóng băng có chủ ý   : 7
Không nhận gradient  : 13
```

Mười ba tham số không nhận gradient chia làm ba nhóm với ba nguyên nhân khác nhau:

**Nhóm một — một tham số.** `noise_ratio` khai báo `nn.Parameter` giống hệt
`mask_ratio` ở dòng liền kề, nhưng chỉ tham gia phép so sánh
`torch.rand(n) < self.noise_ratio` nên không có đường gradient. `mask_ratio` tham gia
phép nhân nên học được bình thường. Bài báo mô tả tham số này là *"adaptively tuned
during the training process"* (Eq. 5).

**Nhóm hai — tám tham số.** Thuộc các nhánh mà cấu hình công bố không kích hoạt:
`weight_raw` (chỉ dùng khi `joint_way=learnable_weight`), `comb_proj` (chỉ dùng khi
`cat_mlp`), `ent_attn` (chỉ dùng khi `atten_weight`), `img_proj_2`/`text_proj_2` (chỉ
dùng khi `num_proj=2`).

**Nhóm ba — bốn tham số.** Khối feed-forward của `fusion_layer`. Với
`use_intermediate=0` — giá trị dùng cho cả hai bộ dữ liệu trong `run.sh` — hàm
`BertLayer.forward` thoát ngay sau attention:

```python
def forward(self, hidden_states, output_attentions=False):
    self_attention_outputs = self.attention(hidden_states, ...)
    if not self.config.use_intermediate:
        return (self_attention_outputs[0], self_attention_outputs[1])
```

Nên khối này thực tế là một lớp self-attention đơn, không có feed-forward và không có
residual kèm layer-norm như một khối Transformer đầy đủ.

Ngoài ra, công thức tương quan chéo giữa các mô thái (Eq. 14 trong bài) không xuất
hiện trong mã nguồn:

```bash
$ grep -rn "theta_ab\|modal_corr\|correlation" mmkgc/module/model/APKGC.py
(không có kết quả)
```

---

## 6. Khác biệt giữa cấu hình công bố và cấu hình trong mã

Với APKGC, ba nguồn mô tả siêu tham số không khớp nhau:

| Nguồn | Batch size | Số mẫu âm |
|---|---|---|
| Bài báo, tr. 15010 | 2048 | 64 |
| README của repo | 2048 | 64 |
| `scripts/run_base.sh` — file mà `run.sh` thực sự gọi | **1024** | **32** |

```bash
$ cat run.sh
bash scripts/run_base.sh 0 DB15K 2 0 Mformer_hd_graph 0.2 0.7 epoch 1 2 K001
bash scripts/run_base.sh 0 MKG-W 1 0 Mformer_hd_mean 0.2 0.7 epoch 1 2 W001

$ head -8 scripts/run_base.sh
DATA=$2
EMB_DIM=128
NUM_BATCH=1024
MARGIN=12
LR=1e-4
LRG=1e-4
NEG_NUM=32
EPOCH=8000
```

Chạy theo đúng hướng dẫn sẽ dùng batch size và số mẫu âm bằng một nửa giá trị công bố.

Một lỗi liên đới: hàm `add_noise_to_embeddings` nhận đối số `noise_ratio` nhưng thân
hàm dùng `self.noise_ratio`, nên cờ `-noise_ratio` trên dòng lệnh không có tác dụng.
Điều này ảnh hưởng tới phần khảo sát siêu tham số (Hình 3), vốn truyền tham số đó qua
`scripts/run_base.sh`.

---

## 7. Thời gian huấn luyện đo được

Đo trên Apple M-series, backend MPS, bộ MKG-W:

| Phương pháp | Giây mỗi epoch |
|---|---|
| APKGC | 8,4 |
| AdaMF-MAT | 160–183 |

AdaMF-MAT nặng hơn khoảng hai mươi lần do có thêm mạng sinh đối kháng. Với 1000 epoch
theo script tác giả, năm lần chạy AdaMF-MAT cần khoảng mười ngày trên phần cứng này —
không khả thi, phải dùng GPU NVIDIA.

Đây là lý do thí nghiệm thăm dò chọn APKGC làm mô hình khảo sát.

---

## 8. Giới hạn của khảo sát này

Tất cả quan sát trên đều nói về **những gì được công bố trong kho mã và bài báo**.

Không thể suy ra từ đây rằng tác giả chỉ chạy một cấu hình, hay chưa từng đo độ lệch
chuẩn. Nhóm nghiên cứu hoàn toàn có thể đã thử nhiều seed và nhiều cấu hình rồi chỉ
công bố một script.

Phát biểu kiểm chứng được là: *bài báo không báo cáo độ lệch chuẩn, và mã nguồn công
bố không có sẵn hạ tầng để chạy nhiều seed*. Đây là vấn đề về cách báo cáo và chia sẻ,
không phải kết luận về phương pháp nghiên cứu của tác giả.
