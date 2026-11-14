# Bộ thuyết trình APKGC — chính xác, nhưng không nộp

Gom ngày 03/09/2026, **không phải vì sai** mà vì đổi phân công.

Ban đầu APKGC được chuẩn bị làm bài thuyết trình seminar. Sau đó xác định lại:
học phần cần **một bài thuyết trình và một bài báo cáo** trên hai bài báo khác nhau.

| Bài | Vai trò mới |
|---|---|
| **ImparTail** (CVPR 2025 Highlight) | thuyết trình |
| **APKGC** (AAAI 2025 Oral) | báo cáo |

Lý do gán như vậy: công sức chạy lại thực nghiệm của APKGC (5 lần chạy, phân tích
nhiễu seed, khảo sát 13 repo, bộ ViMMKG) là ~15 trang số liệu — chỉ vừa vào báo
cáo. ImparTail giàu hình hơn nên hợp làm bài nói.

**Đính chính (10/09/2026).** Lý do ban đầu ghi ở đây là "ImparTail gần như không
chạy lại được". Thử thật trên Kaggle cho thấy điều đó **sai**: dữ liệu có sẵn
(3.200 video đã tách khung), checkpoint tải được (1,2 GB), và CUDA extension
biên dịch được sau khi vá 51 chỗ API cũ. Chặn cuối là thư mục `analysis/` không
có trong kho mã công bố. Việc phân công bài vẫn giữ, nhưng lý do phải nêu đúng.

## Nội dung ở đây

| Tệp | Nội dung |
|---|---|
| `slides.pdf` / `slides.tex` | 28 trang, bộ slide APKGC hoàn chỉnh |
| `DanYTrinhBay.pdf` / `dan_y*.{md,tex}` | Kịch bản trình bày 60 phút cho APKGC |

**Số liệu trong đây vẫn đúng** — đều lấy từ `experiments/results/`. Dùng lại được
nếu cần trình bày APKGC ở dịp khác, hoặc lấy hình minh hoạ cho báo cáo.
