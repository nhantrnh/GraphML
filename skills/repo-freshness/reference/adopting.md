# Đưa vào một repo mới

## Bước 1 — chép hai script

```bash
mkdir -p scripts
cp ~/.claude/skills/repo-freshness/scripts/check_repo.py scripts/
cp ~/.claude/skills/repo-freshness/scripts/verify_numbers.py scripts/
```

## Bước 2 — sửa khối cấu hình đầu `check_repo.py`

Chỉ khối này phụ thuộc repo:

```python
CODE_DIRS     = ("src/mypkg", "scripts", "tests")
RESULTS_DIR   = "results"
DOCS          = ("README.md", "docs/report.md")
INVENTORY_DOC = "README.md"
```

Rồi điền hai bảng, mỗi dòng là một khẳng định tài liệu đang nói:

```python
CLAIMS_VS_RESULTS = {          # "chưa chạy" mà kết quả đã có
    "README.md": [("train.py", "run_001.json")],
}
CONFIG_CLAIMS = [              # tài liệu tả cấu hình, file kết quả là bằng chứng
    ("run_001.json", "model", "resnet50"),
]
```

## Bước 3 — điền `verify_numbers.py`

Một `chk(...)` cho mỗi con số tài liệu trích. Nhãn phải chỉ được **chỗ** con
số nằm, để khi lệch thì biết sửa đâu:

```python
d = load("run_001")
chk("README bảng 1, dòng accuracy", 94.2, d["summary"]["acc"])
```

## Bước 4 — chạy tự động

Cách rẻ nhất, không cần CI:

```bash
cat > .git/hooks/pre-commit <<'EOF'
#!/bin/sh
python scripts/check_repo.py --strict || exit 1
python scripts/verify_numbers.py || exit 1
EOF
chmod +x .git/hooks/pre-commit
```

Nếu có CI, thêm một job chạy đúng hai lệnh đó.

## Quy tắc giữ cho nó sống

Bản thân bộ kiểm tra cũng có thể lỗi thời. Ba thói quen giữ nó đúng:

1. **Viết số vào tài liệu ⇒ thêm `chk()` ngay**, không để sau.
2. **Thêm thí nghiệm ⇒ thêm dòng vào `CLAIMS_VS_RESULTS`.**
3. **Sau mỗi lần sửa bằng script, chạy lại checker.** Một phép thay chuỗi
   không khớp sẽ **im lặng không làm gì** — đã xảy ra thật, và checker là thứ
   duy nhất phát hiện.

## Khi nào KHÔNG cần

Repo không có số liệu, không có thư mục kết quả, tài liệu chỉ mô tả API — thì
`check_repo.py` gần như không có gì để bắt. Dùng linter markdown và kiểm tra
link là đủ.
