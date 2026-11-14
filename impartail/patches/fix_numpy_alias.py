#!/usr/bin/env python3
"""Thay các bí danh numpy đã bị gỡ (np.float, np.bool, …).

NumPy gỡ hẳn `np.float`, `np.int`, `np.bool`, `np.object`, `np.str` ở bản 1.24
(deprecated từ 1.20). Mã ImparTail viết trước đó nên còn dùng, và script đánh giá
dừng ở:

    AttributeError: module 'numpy' has no attribute 'float'

Đây chỉ là bí danh của kiểu dựng sẵn Python — thay bằng chính kiểu đó là tương
đương tuyệt đối, không đổi ngữ nghĩa hay độ chính xác.

Quét toàn repo cho ra 13 chỗ trong 8 tệp, gồm cả `.pyx` (phải vá TRƯỚC khi
cythonize, vì lỗi chỉ lộ ra lúc import module đã biên dịch).

    python fix_numpy_alias.py <đường dẫn ImparTail>
"""
import os, re, sys

#: np.float -> float. Không đụng np.float32/float64 nhờ ranh giới \b và (?!\d).
ALIASES = {"float": "float", "int": "int", "bool": "bool",
           "object": "object", "str": "str", "long": "int", "complex": "complex"}
PAT = re.compile(r"\bnp\.(" + "|".join(ALIASES) + r")\b(?!\d|_)")


def patch(path):
    try:
        src = open(path, encoding="utf-8").read()
    except (OSError, UnicodeDecodeError):
        return 0
    out = PAT.sub(lambda m: ALIASES[m.group(1)], src)
    if out == src:
        return 0
    open(path, "w", encoding="utf-8").write(out)
    n = len(PAT.findall(src))
    print(f"  ✓ {os.path.relpath(path):52} {n} chỗ")
    return n


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    total = files = 0
    for dp, dn, fn in os.walk(root):
        if ".git" in dp:
            continue
        for f in fn:
            if f.endswith((".py", ".pyx")):
                k = patch(os.path.join(dp, f))
                total += k
                files += 1 if k else 0
    print(f"  vá {total} chỗ trong {files} tệp")
    left = []
    for dp, dn, fn in os.walk(root):
        if ".git" in dp:
            continue
        for f in fn:
            if not f.endswith((".py", ".pyx")):
                continue
            p = os.path.join(dp, f)
            try:
                for i, l in enumerate(open(p, encoding="utf-8", errors="ignore"), 1):
                    if PAT.search(l):
                        left.append(f"{os.path.relpath(p, root)}:{i}")
            except OSError:
                pass
    print("  còn sót: " + (", ".join(left[:6]) if left else "không"))
