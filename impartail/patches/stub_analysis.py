#!/usr/bin/env python3
"""Tạo module `analysis` còn thiếu trong kho mã ImparTail.

Ba tệp test (`test_sgg_base.py`, `test_sga_base.py`, `test_easg_base.py`) đều
import từ `analysis.results`, nhưng **thư mục `analysis/` không có trong kho mã
đã công bố** và cũng không nằm trong `.gitignore`. Thiếu nó thì mọi script đánh
giá dừng ngay ở dòng import, trước khi nạp được mô hình.

Kiểm ở máy cho thấy `Result`, `ResultDetails`, `Metrics` chỉ được dùng bên trong
`_prepare_metrics_from_stats()` và `_publish_results_to_firebase()` — cả hai đều
thuộc nhánh đẩy kết quả lên Firebase, mà mọi lời gọi nhánh đó đã bị tác giả
comment (`test_sgg_base.py:158`, `test_sga_base.py:210`).

Nên module giả chỉ cần thoả import. Nó **không** thay đổi con số nào của phép
đánh giá; nếu có mã nào thật sự dùng tới thì các lớp này vẫn ghi nhận thuộc tính
bình thường thay vì lỗi âm thầm.

    python stub_analysis.py <đường dẫn ImparTail>
"""
import os, sys

RESULT_PY = '''"""Bù cho module `analysis.results` không có trong kho mã công bố.

Chỉ đủ để thoả import. Xem impartail/patches/stub_analysis.py để biết vì sao.
"""


class _Bag:
    """Nhận mọi keyword, giữ lại, và cho gán thêm thuộc tính về sau."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, name):
        # phương thức add_* / set_* gọi được mà không nổ
        if name.startswith(("add_", "set_", "publish_")):
            return lambda *a, **k: None
        raise AttributeError(name)

    def __repr__(self):
        return f"{type(self).__name__}({self.__dict__})"


class Metrics(_Bag):
    pass


class ResultDetails(_Bag):
    pass


class Result(_Bag):
    pass
'''

FIREBASE_PY = '''"""Bù cho `analysis.results.FirebaseService`.

Dịch vụ thật đẩy kết quả lên Firebase và cần `pyrebase`, thư viện kéo theo một
bản `pyOpenSSL` cũ không chạy được với `cryptography` hiện tại. Mọi lời gọi nó
trong kho mã đã bị comment sẵn, nên bản giả này chỉ để import không lỗi.
"""


class FirebaseService:
    def __init__(self, *a, **k):
        raise RuntimeError(
            "FirebaseService bị vô hiệu hoá: chỉ dùng để đẩy kết quả lên cloud, "
            "không liên quan tới đánh giá."
        )
'''


def main(root):
    d = os.path.join(root, "analysis", "results")
    if os.path.isdir(os.path.join(root, "analysis")) and not os.environ.get("FORCE"):
        print("  analysis/ đã tồn tại — không ghi đè")
        return 0
    os.makedirs(d, exist_ok=True)
    for pkg in (os.path.join(root, "analysis"), d):
        open(os.path.join(pkg, "__init__.py"), "w").close()
    open(os.path.join(d, "Result.py"), "w", encoding="utf-8").write(RESULT_PY)
    open(os.path.join(d, "FirebaseService.py"), "w", encoding="utf-8").write(FIREBASE_PY)
    print(f"  ✓ tạo analysis/results/ với Result.py và FirebaseService.py")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
