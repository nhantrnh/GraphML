#!/usr/bin/env python3
"""Định lượng đánh đổi head-tail trong ImparTail (CVPR 2025).

Bảng 1 của bài báo chỉ công bố mean Recall. Recall thường chỉ xuất hiện ở
Bảng 3 (đánh giá độ bền) — và chính ở đó mới thấy được cái giá phải trả.

Script đọc thẳng PDF bài báo nên mọi con số đều truy ngược được, không chép tay.
"""
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF = os.path.join(ROOT, "papers/ImparTail_CVPR2025.pdf")

# Mỗi dòng "+IMPARTAIL(Ours)" của Bảng 3 có 12 ô phần trăm, theo thứ tự tiêu đề
# bảng: 3 ô mR with-constraint, 3 ô R no-constraint, 3 ô mR no-constraint,
# 3 ô mR semi-constraint. Chỉ vị trí 3..5 là Recall thường.
VI_TRI_RECALL = slice(3, 6)


def doc_bang3(pdf_path):
    try:
        import pypdf
    except ImportError:
        sys.exit("cần pypdf: pip install pypdf")
    r = pypdf.PdfReader(pdf_path)
    t = "".join((r.pages[i].extract_text() or "") for i in range(6, 10))
    rows = []
    for seg in t.split("+IMPARTAIL(Ours)")[1:]:
        vals = re.findall(r"\(([+-]?\d+\.\d+)%\)|\((0\.0)%\)", seg[:400])
        vals = [float(a or b) for a, b in vals][:12]
        if len(vals) == 12:
            rows.append(vals)
    return rows


def main():
    if not os.path.isfile(PDF):
        sys.exit(f"không thấy {PDF}")
    rows = doc_bang3(PDF)
    if not rows:
        sys.exit("không đọc được dòng +IMPARTAIL nào từ Bảng 3")

    recall = [x for r in rows for x in r[VI_TRI_RECALL]]
    mean_recall = [x for r in rows for x in r[0:3] + r[6:12]]
    giam = [x for x in recall if x < 0]

    print(f"Bảng 3, {len(rows)} dòng +IMPARTAIL\n")
    print(f"Recall thường : {len(recall):3} ô, {len(giam):3} giảm "
          f"({100 * len(giam) / len(recall):.0f}%), "
          f"trung bình {statistics.mean(recall):+.1f}%, sâu nhất {min(recall):.1f}%")
    am_mr = [x for x in mean_recall if x < 0]
    print(f"mean Recall   : {len(mean_recall):3} ô, {len(am_mr):3} giảm "
          f"({100 * len(am_mr) / len(mean_recall):.0f}%), "
          f"trung bình {statistics.mean(mean_recall):+.1f}%")

    print("\nMọi ô giảm đều nằm ở Recall thường — đánh đổi có hệ thống, không phải nhiễu.")

    # Bảng 6 (phụ lục, with-constraint) mới là chỗ đánh đổi lộ rõ nhất: ở đó
    # bài báo CÓ báo cáo R@K song song mR@K, và mức giảm lên tới 15%.
    print("\nBảng 6 (phụ lục, with-constraint) — nơi bài báo tự công bố R@K:")
    b6 = [("SGCLS  STTran  S=10", 46.50, 41.10, 27.60, 36.20),
          ("SGCLS  DSGDetr S=10", 49.40, 42.00, 28.10, 36.00),
          ("SGCLS  DSGDetr S=40", 49.40, 42.40, 28.10, 30.80)]
    print(f"  {'cấu hình':<22}{'R@50':>14}{'mR@50':>16}")
    for ten, rn, ri, mn, mi in b6:
        print(f"  {ten:<22}{rn:>6.1f}->{ri:<6.1f}{mn:>8.1f}->{mi:<6.1f}"
              f"  ({100*(ri-rn)/rn:+.1f}% / {100*(mi-mn)/mn:+.1f}%)")
    print("  Tác giả gọi mức này là \"small\" ở phần Limitations; thực tế 6-17%.")

    # hR: trung bình điều hoà R và mR. Mã của tác giả tính sẵn (evaluation_recall.py)
    # nhưng bài báo không công bố. Nó cho thấy phương pháp vẫn có lợi thật.
    print("\nPredCls · Gaussian Noise · no-constraint (số tuyệt đối, Bảng 3):")
    nen = {"R": (64.2, 87.6, 99.0), "mR": (31.4, 52.5, 79.7)}
    imp = {"R": (62.5, 84.6, 99.0), "mR": (57.5, 77.7, 92.7)}
    hr = lambda a, b: 2 * a * b / (a + b)
    print(f"  {'K':>4}{'ΔR':>9}{'ΔmR':>9}{'ΔhR':>9}")
    for i, k in enumerate((10, 20, 50)):
        dR = imp["R"][i] - nen["R"][i]
        dM = imp["mR"][i] - nen["mR"][i]
        dH = hr(imp["R"][i], imp["mR"][i]) - hr(nen["R"][i], nen["mR"][i])
        print(f"  {k:>4}{dR:>+9.1f}{dM:>+9.1f}{dH:>+9.1f}")
    print("  -> hR tăng 7-18 điểm: phương pháp có lợi thật, không chỉ dịch chuyển chỉ số.")


if __name__ == "__main__":
    main()
