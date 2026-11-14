#!/usr/bin/env python3
"""So hai nhánh huấn luyện: mặt nạ theo mã gốc và theo công thức (5).

Hai nhánh chạy cùng cấu hình, cùng số epoch, cùng dữ liệu — khác duy nhất cờ
`--true_masked_bce`. Chi tiết bản vá: impartail/patches/fix_masked_bce.py.

Giả thuyết cần kiểm: phần lớn đánh đổi head-tail mà bài báo gọi là "although
small" (thực tế 6-17%, Bảng 6) là hệ quả của cài đặt, không phải bản chất
phương pháp. Nếu đúng thì nhánh vá giữ được mR mà không mất R, tức hR cao hơn.

Đọc CSV do chính test_sgg_methods.py sinh ra; thứ tự 36 cột theo
test_sgg_base.py:186-193 (with -> no -> semi, mỗi nhóm R/mR/hR @10,20,50,100).
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THUMUC = os.path.join(ROOT, "experiments/results/impartail_repro")
NHANH = {"goc": "sttran_partial10_goc.csv", "va": "sttran_partial10_va.csv"}
RANG_BUOC = ("with", "no", "semi")


def doc(path):
    """Trả {ràng buộc: {K: (R, mR, hR)}} theo phần trăm."""
    with open(path, newline="") as f:
        hang = list(csv.reader(f))
    v = [float(x) for x in hang[-1][1:]]
    if len(v) != 36:
        sys.exit(f"{path}: cần 36 cột, có {len(v)}")
    ra = {}
    for i, ten in enumerate(RANG_BUOC):
        b = i * 12
        ra[ten] = {k: (v[b + j] * 100, v[b + 4 + j] * 100, v[b + 8 + j] * 100)
                   for j, k in enumerate((10, 20, 50, 100))}
    return ra


def main():
    thieu = [n for n, f in NHANH.items() if not os.path.isfile(os.path.join(THUMUC, f))]
    if thieu:
        print("Chưa có kết quả cho nhánh: " + ", ".join(thieu))
        print(f"Chờ kernel Kaggle xong rồi đặt CSV vào {THUMUC}/")
        return 0

    d = {n: doc(os.path.join(THUMUC, f)) for n, f in NHANH.items()}

    print(__doc__.split("\n\n")[0])
    print("\n@50, đơn vị %. Δ > 0 nghĩa là nhánh vá tốt hơn.\n")
    print(f"{'ràng buộc':<10}{'R gốc':>8}{'R vá':>8}{'ΔR':>8}"
          f"{'mR gốc':>9}{'mR vá':>8}{'ΔmR':>8}"
          f"{'hR gốc':>9}{'hR vá':>8}{'ΔhR':>8}")
    print("-" * 78)

    tong_dhr = []
    for rb in RANG_BUOC:
        rg, mg, hg = d["goc"][rb][50]
        rv, mv, hv = d["va"][rb][50]
        tong_dhr.append(hv - hg)
        print(f"{rb:<10}{rg:>8.1f}{rv:>8.1f}{rv - rg:>+8.1f}"
              f"{mg:>9.1f}{mv:>8.1f}{mv - mg:>+8.1f}"
              f"{hg:>9.1f}{hv:>8.1f}{hv - hg:>+8.1f}")

    print("-" * 78)
    tb = sum(tong_dhr) / len(tong_dhr)
    print(f"\nΔhR trung bình: {tb:+.2f} điểm")

    # Đọc kết quả trung thực: nêu cả khả năng giả thuyết sai.
    if tb > 0.5:
        print("→ Nhánh vá cho tổng hoà tốt hơn: ủng hộ giả thuyết rằng một phần")
        print("  đánh đổi head-tail đến từ cài đặt chứ không từ phương pháp.")
    elif tb < -0.5:
        print("→ Nhánh vá cho tổng hoà KÉM hơn. Giả thuyết không đứng vững:")
        print("  cách áp mặt nạ của mã gốc, dù lệch công thức (5), lại có lợi.")
        print("  Đây vẫn là phát hiện đáng báo cáo — nhưng phải báo cáo đúng chiều.")
    else:
        print("→ Hai nhánh gần như tương đương (|ΔhR| < 0,5 điểm). Sai lệch cài đặt")
        print("  có thật về mặt toán học nhưng không đổi kết quả ở quy mô này.")
        print("  Cần thêm seed trước khi kết luận, vì chênh lệch nằm trong mức nhiễu.")

    print("\nCẢNH BÁO: mỗi nhánh mới chạy MỘT lần. Bài báo gốc cũng không báo cáo")
    print("độ lệch chuẩn hay số seed, nên đừng lặp lại lỗi đó — cần ít nhất 3 seed")
    print("trước khi đưa con số vào bài viết.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
