"""Đối chiếu mọi con số trong tài liệu với dữ liệu gốc — MẪU để chép đi.

VÌ SAO. Một con số bị chép sai vào tài liệu không gây lỗi gì cả: file vẫn
biên dịch, script vẫn chạy. Nó chỉ sai. Cách duy nhất bắt được là tính lại
từ dữ liệu gốc rồi so.

CÁCH DÙNG. File này KHÔNG chạy được ngay — nó là mẫu. Chép vào repo, rồi
điền phần CHECKS bên dưới bằng chính những con số tài liệu của bạn đang
trích. Mỗi dòng `chk(...)` là một khẳng định: "tài liệu nói X, dữ liệu phải
cho X".

Quy tắc khi thêm check:

  - Thêm check NGAY khi viết một con số vào tài liệu, không để sau.
  - Dung sai đủ chặt để bắt lỗi chép nhầm, đủ lỏng cho làm tròn (thường 0.02).
  - Nhãn phải nói rõ con số nằm ở ĐÂU trong tài liệu, để khi lệch thì biết
    sửa chỗ nào.

    python scripts/verify_numbers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"          # ← sửa nếu repo để kết quả chỗ khác

checks: list[tuple[bool, str, float, float | None]] = []


def load(name: str) -> dict | None:
    """Nạp một file kết quả; trả None nếu chưa có (để báo thay vì nổ)."""
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def chk(label: str, claimed: float, actual: float | None,
        tol: float = 0.02) -> None:
    """So một con số trong tài liệu với giá trị tính từ dữ liệu.

    `actual=None` nghĩa là file kết quả chưa tồn tại — tính là lệch, vì tài
    liệu đang trích một con số không có nguồn.
    """
    ok = actual is not None and abs(claimed - actual) <= tol
    checks.append((ok, label, claimed, actual))


# ══════════════════════════════════════════════════════════════════
# CHECKS — điền phần này. Ví dụ:
#
#   d = load("main_results")
#   chk("Bảng 1, dòng Oracle, cột F1", 61.03, d["summary"]["oracle"]["f1"])
#   chk("Mục 4.1, tỉ lệ giữ", 0.206, d["summary"]["itercomp"]["ratio"], 0.002)
# ══════════════════════════════════════════════════════════════════


def main() -> int:
    if not checks:
        print("Chưa có check nào — xem hướng dẫn ở đầu file.")
        return 0
    bad = [c for c in checks if not c[0]]
    print(f"{len(checks) - len(bad)}/{len(checks)} con số khớp\n")
    for _, label, claimed, actual in bad:
        got = "thiếu file" if actual is None else round(actual, 4)
        print(f"  ✗ {label}: tài liệu={claimed}  dữ liệu={got}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
