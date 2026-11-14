#!/usr/bin/env python3
"""Điểm cân bằng TOÀN CỤC — mở rộng Định lý 4 sang cả mẫu âm.

Định lý 4 (four_theorems.py) phát biểu ŷ* = p_c. Đó là điểm cân bằng **có
điều kiện trên mẫu dương**: E[∂L/∂z | y=1] = ŷ − p_c = 0.

Nhưng bài toán là đa nhãn. Cùng một logit của lớp c còn nhận gradient từ mọi
cặp mà lớp c KHÔNG xuất hiện, ở đó y = 0 và ∂L/∂z = ŷ − 0 = ŷ > 0, tức luôn
kéo xuống. Gọi π = n_c / N_cặp là tỉ lệ cặp thực sự mang nhãn c:

    E[∂L/∂z] = π·(ŷ − p_c) + (1 − π)·ŷ = ŷ − π·p_c

nên điểm cân bằng toàn cục là

    ŷ*_toàn cục = π · p_c        (thấp hơn p_c một hệ số π)

HỆ QUẢ BẤT NGỜ. Algorithm 1 đặt quota = ρ·N/|P| cho MỌI lớp, nên với lớp
không bị clamp (n_c > quota):

    π · p_c = (n_c / N_cặp) · (quota / n_c) = quota / N_cặp

n_c **triệt tiêu**. Mọi lớp không bị clamp hội tụ về CÙNG một điểm cân bằng,
bất kể tần suất.

Đây là nghịch lý đáng nêu: cơ chế nhằm **cân bằng** các lớp thực ra ép chúng
về cùng một mức tin cậy — nhưng mức đó được đặt bởi quota chứ không bởi dữ
liệu, và các lớp hiếm (bị clamp) lại rơi xuống THẤP HƠN.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FREQ = os.path.join(ROOT, "experiments/results/impartail_repro/ag_predicate_freq.json")
NHOM = "contacting_relationship"
TI_LE_GIU = 0.10


def main():
    if not os.path.isfile(FREQ):
        sys.exit(f"không thấy {FREQ}")
    d = json.load(open(FREQ))
    c = d["tan_suat"][NHOM]
    n_tong, n_lop, n_cap = sum(c.values()), len(c), d["tong_cap"]
    quota = n_tong * TI_LE_GIU / n_lop

    print(__doc__.split("\n\n")[0])
    print(f"\nρ = {TI_LE_GIU:.0%} · quota = {quota:,.1f} · N_cặp = {n_cap:,}\n")
    print(f"{'lớp':<22}{'n_c':>9}{'π':>9}{'p_c':>8}{'ŷ* có ĐK':>10}{'ŷ* toàn cục':>13}")
    print("-" * 72)

    nhom_deu = []
    for k, v in sorted(c.items(), key=lambda x: -x[1]):
        pi = v / n_cap
        pc = min(quota, v) / v
        tc = pi * pc
        bi_clamp = v <= quota
        if not bi_clamp:
            nhom_deu.append(tc)
        print(f"{k:<22}{v:>9,}{pi:>9.4f}{pc:>8.3f}{pc:>10.3f}{tc:>13.6f}"
              + ("  clamp" if bi_clamp else ""))

    print("-" * 72)
    if nhom_deu:
        lech = max(nhom_deu) - min(nhom_deu)
        print(f"\n{len(nhom_deu)} lớp không bị clamp đều hội tụ về ŷ* = "
              f"{nhom_deu[0]:.6f} (chênh lệch {lech:.2e})")
        print(f"  = quota / N_cặp = {quota:,.1f} / {n_cap:,} = {quota/n_cap:.6f}")

    print("\nHAI HỆ QUẢ CHO BÀI VIẾT")
    print("  1. Mức dìm SÂU HƠN Định lý 4 dự đoán, thêm hệ số π. Với `holding`,")
    print(f"     ŷ* đi từ {min(quota,c['holding'])/c['holding']:.4f} (có điều kiện)")
    print(f"     xuống {quota/n_cap:.4f} (toàn cục) — sâu thêm khoảng"
          f" {(min(quota,c['holding'])/c['holding'])/(quota/n_cap):.1f} lần.")
    print("  2. Mọi lớp không bị clamp về CÙNG một điểm cân bằng — cơ chế nhằm")
    print("     cân bằng lớp lại đặt mức tin cậy theo quota chứ không theo dữ")
    print("     liệu, còn lớp hiếm bị clamp thì rơi xuống thấp hơn nữa.")

    print("\nLƯU Ý KHI PHÁT BIỂU")
    print("  Đây là phân tích điểm cân bằng của kỳ vọng gradient, giả định mặt")
    print("  nạ độc lập với tham số. Nó nói về hướng mà hàm mất mát kéo logit,")
    print("  không phải giá trị cuối cùng sau khi huấn luyện — mô hình còn chịu")
    print("  chuẩn hoá, ràng buộc kiến trúc, và tương quan giữa các lớp.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
