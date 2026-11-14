# `skills/` — bộ skill Claude Code dùng lại được

Thư mục này **không phục vụ bài nộp**. Nó chứa công cụ tách ra từ chính dự án
này để dùng cho project sau.

## Cài

```bash
cp -R skills/repo-freshness ~/.claude/skills/
```

Claude Code nhận skill mới **ngay trong phiên đang chạy**, không cần khởi động
lại. Kiểm tra bằng `/skills` hoặc gõ `/repo-freshness`.

Muốn chỉ dùng cho một project thì chép vào `.claude/skills/` của project đó
thay vì `~/.claude/skills/`.

## `repo-freshness`

Bắt tài liệu lệch khỏi mã và dữ liệu nó mô tả — loại lỗi **không** gây lỗi
biên dịch, không làm test đỏ, nên không công cụ thường nào bắt được. PDF vẫn
đẹp, script vẫn chạy, chỉ nội dung sai.

| File | Việc |
|---|---|
| `SKILL.md` | 8 loại drift, cách rà, quy tắc phòng |
| `scripts/check_repo.py` | tự động bắt loại 1, 5, 6, 7 |
| `scripts/verify_numbers.py` | mẫu — điền một `chk()` cho mỗi số tài liệu trích |
| `scripts/validate_skill.py` | kiểm skill có đúng Agent Skills spec |
| `reference/manual-audit.md` | loại 3, 4, 8 — máy không bắt được, phải đọc |
| `reference/adopting.md` | gắn vào repo mới, kèm pre-commit hook |

### Dùng ngay không cần cài

```bash
python skills/repo-freshness/scripts/check_repo.py --root /đường/dẫn/repo
python skills/repo-freshness/scripts/validate_skill.py ~/.claude/skills/*/
```

### Nó bắt được gì trong chính dự án này

- `research/README.md` ghi hai thí nghiệm "chưa chạy" sau khi cả hai đã chạy
  xong trên Kaggle — và một lần sửa trước đó của tôi **im lặng không ăn** vì
  chuỗi thay thế không khớp
- Ngưỡng thống kê ghi 17.5 ở một file, 18.0 ở file khác
- Báo cáo khai "chưa đánh giá tiếng Việt không dấu" trong khi EXP-2 đã đánh giá
- `verify_report_numbers.py` hardcode `/Users/nhantt/...`, chỉ chạy được một máy

## Vì sao tách khỏi `scripts/`

`scripts/check_repo.py` là bản **đã cấu hình cho repo này**. Bản trong
`skills/` là bản gốc dùng lại được: khối cấu hình tách riêng ở đầu file, và có
cờ `--root` để chạy từ repo khác. Sửa mã ở đây thì nhớ đồng bộ sang
`scripts/`, và ngược lại.
