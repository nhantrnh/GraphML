#!/usr/bin/env python3
"""Tính hR@K cho dòng VidSGG unbiased — chỉ số không bài nào báo cáo.

hR@K = trung bình điều hoà của R@K và mR@K, do Iterative Scene Graph Generation
(Khandelwal & Sigal, NeurIPS 2022, DOI 10.52202/068431-1764) đề xuất đúng vì lý
do này. Nguyên văn: "one can trade-off R@K for mR@K by using long tail
recognition techniques... This trade-off is seldom analyzed or reported, making
it difficult to compare performance across methods."

Các bài VidSGG unbiased phần lớn chỉ công bố mR@K, nên không so được cái giá
phải trả. Script gom những cặp (R, mR) tìm được và tính hR.

Nguồn từng con số ghi ngay trong bảng NGUON bên dưới — không chép tay con số
nào mà không nêu xuất xứ. Ô nào bài báo không công bố R@K thì để None và script
báo rõ là KHÔNG TÍNH ĐƯỢC, thay vì đoán.
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (nhãn, R@50, mR@50, nguồn) — R@50 = None nghĩa là bài báo không công bố
NGUON = [
    # ImparTail, Bảng 6 phụ lục (with-constraint) — chỗ DUY NHẤT trong bài có
    # R@K song song mR@K cho cấu hình chính.
    ("STTran  SGCLS  (nền)",        46.50, 27.60, "ImparTail Bảng 6"),
    ("  +ImparTail S=70",           43.30, 28.80, "ImparTail Bảng 6"),
    ("  +ImparTail S=40",           43.80, 30.70, "ImparTail Bảng 6"),
    ("  +ImparTail S=10",           41.10, 36.20, "ImparTail Bảng 6"),
    ("DSGDetr SGCLS  (nền)",        49.40, 28.10, "ImparTail Bảng 6"),
    ("  +ImparTail S=70",           47.40, 30.30, "ImparTail Bảng 6"),
    ("  +ImparTail S=40",           42.40, 30.80, "ImparTail Bảng 6"),
    ("  +ImparTail S=10",           42.00, 36.00, "ImparTail Bảng 6"),
    # Bảng 1 (bảng chính) chỉ có mR@K — đây chính là vấn đề.
    ("STTran  PREDCLS (nền)",        None, 34.80, "ImparTail Bảng 1 (thiếu R@K)"),
    ("  +ImparTail",                 None, 52.90, "ImparTail Bảng 1 (thiếu R@K)"),
]


def hr(r, m):
    """Trung bình điều hoà. Trả None nếu thiếu dữ liệu hoặc mẫu số 0."""
    if r is None or m is None:
        return None
    if r + m == 0:
        return None
    return 2 * r * m / (r + m)


def doc_ket_qua_tai_lap():
    """Số nhóm tự đo được: PredCls/STTran nền, cả ba kiểu ràng buộc."""
    p = os.path.join(ROOT, "experiments/results/impartail_repro/sttran_predcls_v8.csv")
    if not os.path.isfile(p):
        return []
    with open(p, newline="") as f:
        v = [float(x) for x in list(csv.reader(f))[1][1:]]
    ra = []
    for i, ten in enumerate(("with", "no", "semi")):
        b = i * 12
        # R@50 ở vị trí 2, mR@50 ở vị trí 6, hR@50 ở vị trí 10 trong mỗi nhóm
        ra.append((f"STTran PredCls {ten}", v[b + 2] * 100, v[b + 6] * 100,
                   "nhóm tái lập", v[b + 10] * 100))
    return ra


def main():
    print(__doc__.split("\n\n")[0])
    print("\n@50, đơn vị %\n")
    print(f"{'cấu hình':<26}{'R':>8}{'mR':>8}{'hR':>8}   nguồn")
    print("-" * 74)

    thieu = 0
    for ten, r, m, ng in NGUON:
        h = hr(r, m)
        if h is None:
            thieu += 1
            print(f"{ten:<26}{'—':>8}{m:>8.1f}{'KHÔNG TÍNH ĐƯỢC':>8}   {ng}")
        else:
            print(f"{ten:<26}{r:>8.1f}{m:>8.1f}{h:>8.1f}   {ng}")

    tl = doc_ket_qua_tai_lap()
    if tl:
        print()
        for ten, r, m, ng, h_ma in tl:
            h = hr(r, m)
            # Đối chiếu với hR do chính mã tác giả tính (evaluation_recall.py:66-69)
            khop = "✓" if abs(h - h_ma) < 0.05 else f"✗ mã={h_ma:.1f}"
            print(f"{ten:<26}{r:>8.1f}{m:>8.1f}{h:>8.1f}   {ng} {khop}")

    print("-" * 74)
    if thieu:
        print(f"\n{thieu} cấu hình KHÔNG tính được hR vì bài báo chỉ công bố mR@K.")
        print("Đó chính là luận điểm: không có R@K thì không ai biết cái giá phải trả.")

    # Điểm mà chỉ hR mới lộ ra: với DSGDetr, che 40% cho hR THẤP HƠN che 70%,
    # tức tổng hoà đi lùi dù mR vẫn tăng. Nhìn riêng mR thì không thấy được.
    print("\nĐiều chỉ hR mới cho thấy (DSGDetr SGCLS):")
    d = {t.strip(): (r, m) for t, r, m, _ in NGUON if r is not None}
    for nhan in ("+ImparTail S=70", "+ImparTail S=40"):
        r, m = d[nhan]
        print(f"  {nhan:<20} mR {m:.1f}  nhưng hR {hr(r, m):.1f}")
    print("  Che 40% cho mR cao hơn che 70% (30,8 > 30,3) nhưng hR lại THẤP hơn")
    print("  (35,7 < 37,0): phần R mất đi nhiều hơn phần mR thu được. Chỉ báo cáo")
    print("  mR thì cấu hình này trông như tiến bộ.")


if __name__ == "__main__":
    main()
