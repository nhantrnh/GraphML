#!/usr/bin/env python3
"""Cận CHẶT và ĐẠT ĐƯỢC cho mức đảo thứ hạng của mean Recall.

Khác với các công cụ khác trong thư mục (đo, đếm, đối chiếu mã), đây là một
định lý nhỏ có chứng minh và có cấu trúc đạt cận.

────────────────────────────────────────────────────────────────────────
ĐỊNH LÝ. Cho tập C lớp chia thành T lớp mang nghĩa và N = C − T lớp phủ
định. Với hai mô hình A, B mà A tốt hơn B đúng δ trên MỌI lớp mang nghĩa:

    δ < N / T   ⟺   tồn tại cấu hình recall trên lớp phủ định khiến
                    mR(B) > mR(A) khi tính trên đủ C lớp.

Cận N/T là CHẶT: đạt được khi B đạt recall 1 và A đạt 0 trên cả N lớp
phủ định.

CHỨNG MINH. Gọi a, b là recall trung bình trên T lớp mang nghĩa, với
b = a − δ; gọi a_N, b_N là recall trung bình trên N lớp phủ định. Theo
định nghĩa,

    mR(A) = (T·a + N·a_N)/C,    mR(B) = (T·b + N·b_N)/C

B thắng khi mR(B) > mR(A), tức

    T·(a − δ) + N·b_N  >  T·a + N·a_N
    N·(b_N − a_N)      >  T·δ

Vì recall ∈ [0,1] nên b_N − a_N ≤ 1, cho δ < N/T. Ngược lại, với mọi
δ < N/T, chọn a_N = 0 và b_N = 1 thoả bất đẳng thức. ∎
────────────────────────────────────────────────────────────────────────

Điều khiến cận này có sức nặng, chứ không chỉ là một khả năng trên giấy:
lớp phủ định là lớp ĐÔNG NHẤT trong nhóm của nó, nên recall cao trên chúng
đạt được bởi một bộ phân loại TẦM THƯỜNG — luôn đoán lớp đa số.
"""
import json
import os
import sys
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "experiments/results/impartail_repro/ag_predicate_freq.json")

# Chỉ số theo dataloader/base_ag_dataset.py:69-83
AM = {
    "attention_relationship": ["not_looking_at", "unsure"],
    "contacting_relationship": ["not_contacting", "other_relationship"],
}


def main():
    if not os.path.isfile(DATA):
        sys.exit(f"không thấy {DATA}")
    f = json.load(open(DATA))["tan_suat"]

    C = sum(len(c) for c in f.values())
    N = sum(len(v) for v in AM.values())
    T = C - N

    print(__doc__.split("────")[1].strip())
    print("\n" + "=" * 60)
    print(f"\nÁP VÀO ACTION GENOME: C = {C}, N = {N}, T = {T}\n")
    can = Fraction(N, T)
    print(f"  δ_max = N/T = {N}/{T} = {float(can):.4f}"
          f"  ({float(can)*100:.1f} điểm recall)")

    # Kiểm cấu trúc đạt cận
    a = 0.50
    b = a - float(can) + 1e-9
    mA = (T * a + N * 0.0) / C
    mB = (T * b + N * 1.0) / C
    print(f"\n  Kiểm cấu trúc đạt cận:")
    print(f"    A: lớp thật {a:.4f}, lớp phủ định 0,0  ->  mR = {mA:.6f}")
    print(f"    B: lớp thật {b:.4f}, lớp phủ định 1,0  ->  mR = {mB:.6f}")
    print(f"    B thắng dù tệ hơn {float(can)*100:.1f} điểm ở MỌI lớp thật:"
          f" {mB > mA}")

    # Vì sao cận đạt được trong thực tế, không chỉ trên giấy
    print("\n" + "=" * 60)
    print("\nVÌ SAO CẬN NÀY KHÔNG CHỈ LÀ KHẢ NĂNG TRÊN GIẤY\n")
    for nhom, lop_am in AM.items():
        c = f[nhom]
        tong = sum(c.values())
        dong_nhat = max(c, key=c.get)
        print(f"  {nhom} ({len(c)} lớp, {tong:,} nhãn)")
        for k, v in sorted(c.items(), key=lambda x: -x[1])[:3]:
            dau = " <- phủ định" if k in lop_am else ""
            print(f"    {k:<22}{v:>9,}  ({100*v/tong:>4.1f}%){dau}")
        if dong_nhat in lop_am:
            print(f"    => lớp ĐÔNG NHẤT là lớp phủ định. Bộ phân loại luôn")
            print(f"       đoán lớp đa số đạt recall 1,0 trên nó mà không cần")
            print(f"       học gì cả.")
        print()

    print("KẾT LUẬN")
    print(f"  Một mô hình có thể tệ hơn tới {float(can)*100:.1f} điểm recall trên")
    print(f"  MỌI lớp mang nghĩa mà vẫn thắng theo mean Recall, và cách đạt")
    print(f"  điều đó là hành vi tầm thường: thiên về lớp đa số.")
    print(f"\n  Đây chính là hành vi mà chỉ số mean Recall được tạo ra để ngăn.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
