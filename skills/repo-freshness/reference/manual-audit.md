# Rà thủ công — ba loại drift máy không bắt được

`check_repo.py` bắt được thứ so sánh cơ học. Ba loại dưới đây cần đọc, và
chúng là loại tốn thời gian nhất khi lọt.

## Loại 3 — cùng đại lượng, hai tài liệu ghi khác nhau

Xảy ra khi sửa một tài liệu mà quên tài liệu kia. Cách tìm:

```bash
# Với mỗi con số đầu mục, đếm nó xuất hiện ở đâu
for n in 18.0 4.3 61.03 15.98; do
  printf "%-8s" "$n"
  for f in README.md docs/report.tex NOTES.md; do
    printf " %s:%s" "$(basename $f)" "$(grep -c -- "$n" "$f" 2>/dev/null)"
  done; echo
done
```

Con số xuất hiện ở đúng **một** file thì hoặc ổn, hoặc là dấu hiệu file kia
còn giá trị cũ. Phải mở ra xem, không suy đoán.

**Bẫy đã gặp thật.** Ngưỡng phát hiện ghép cặp ghi 17.5 ở một nơi, 18.0 ở nơi
khác. Cả hai đều là số **thật** — nhưng của hai cặp phương pháp khác nhau
(18.0 cho A–B, 14.7 cho C–A). Không phải một cái sai, mà là **thiếu ngữ cảnh**.
Sửa đúng là nói rõ cặp nào, không phải chọn bừa một số.

## Loại 4 — số đúng, đại lượng sai

Nguy hiểm nhất, vì mọi kiểm tra số học đều **pass**.

Ví dụ thật: tài liệu viết *"đạt 81% của headroom"*. Con số 81% đúng — nhưng nó
là `X/Trần`, còn "headroom" nghĩa là khoảng `Trần − Cơ sở`, và phần thu hồi
được của khoảng đó là **−8%**. Số đúng, chữ sai, kết luận sai hẳn.

Quy trình kiểm:

1. Liệt kê mọi tỉ số và phần trăm trong tài liệu.
2. Với từng cái, hỏi: **mẫu số là gì?**
3. Đọc câu văn: nó có gọi đúng mẫu số đó không?
4. Nếu tài liệu dùng thuật ngữ chuyên ngành (headroom, recall, coverage),
   kiểm định nghĩa chuẩn của thuật ngữ, đừng dựa vào cảm giác.

## Loại 8 — ghép hai nguồn khác cấu hình

Xảy ra khi một script đọc kết quả có sẵn rồi tính thêm thứ khác trên cùng dữ
liệu — nhưng bằng cấu hình khác.

Ví dụ thật: script tính đặc trưng bằng `bm25` rồi ghép với điểm F1 vốn được
sinh bằng `dual`. Hai ngữ cảnh khác nhau ⇒ tương quan thu được vô nghĩa.
Không có exception nào, chỉ ra một con số trông hợp lý.

Cách phòng:

- Script ghép dữ liệu phải **đọc cấu hình từ file nguồn**, không dùng mặc định
  riêng. Mặc định phải là `None` rồi suy ra, chứ không phải một giá trị cứng.
- Nếu người dùng ghi đè, **in cảnh báo to**.
- Viết test khoá lại điều đó — nó không tự lộ ra.

## Thứ tự rà cho hiệu quả

1. Chạy máy trước (`check_repo.py`, `verify_numbers.py`) — vài giây, dọn hết
   phần cơ học.
2. Loại 4 — đọc từng tỉ số, hỏi mẫu số.
3. Loại 3 — grep chéo các số đầu mục.
4. Loại 8 — chỉ với script đọc kết quả của script khác.
