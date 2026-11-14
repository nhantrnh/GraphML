#!/usr/bin/env python3
"""Tiên đoán định lượng của Định lý 4, kiểm được trên bốn checkpoint.

ĐỊNH LÝ 4 (four_theorems.py). Với BCE, đạo hàm theo logit là ∂L/∂z = ŷ − y.
Mặt nạ giữ nhãn dương với xác suất p_c, biến thành âm với xác suất 1 − p_c:

    E[∂L/∂z] = p_c(ŷ − 1) + (1 − p_c)(ŷ − 0) = ŷ − p_c

nên điểm cân bằng ổn định bị chặn cứng tại ŷ* = p_c.

VÌ SAO ĐÂY LÀ TIÊN ĐOÁN CHỨ KHÔNG PHẢI QUAN SÁT. ImparTail phát hành
checkpoint ở ba tỉ lệ giữ nhãn (70%, 40%, 10%) cộng baseline 100%. Mỗi tỉ lệ
cho một giá trị p_c khác nhau cho từng lớp, theo Algorithm 1:

    p_c = min(quota, n_c) / n_c,     quota = ρ · N / |P|

Định lý do đó tiên đoán một ĐƯỜNG CONG, không phải một điểm: recall của lớp
head phải tụt theo p_c khi ρ giảm, còn lớp hiếm (p_c = 1 ở mọi ρ) thì không
đổi. Bốn checkpoint cho bốn điểm trên đường cong đó.

Nếu số liệu khớp xu hướng này, định lý được xác nhận định lượng. Nếu không
khớp, định lý sai — và đó cũng là kết quả đáng báo cáo.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FREQ = os.path.join(ROOT, "experiments/results/impartail_repro/ag_predicate_freq.json")
NHOM = "contacting_relationship"
TI_LE = (1.00, 0.70, 0.40, 0.10)      # baseline + ba checkpoint ImparTail


def p_c(n_c, n_tong, n_lop, giu):
    """Xác suất giữ nhãn của lớp c theo Algorithm 1 (ag_dataset.py:58-84)."""
    quota = n_tong * giu / n_lop
    return min(quota, n_c) / n_c


def main():
    if not os.path.isfile(FREQ):
        sys.exit(f"không thấy {FREQ}")
    c = json.load(open(FREQ))["tan_suat"][NHOM]
    n_tong, n_lop = sum(c.values()), len(c)

    print(__doc__.split("\n\n")[0])
    print(f"\nNhóm {NHOM}: {n_lop} lớp, {n_tong:,} nhãn\n")
    print(f"{'lớp':<22}{'n_c':>9}" +
          "".join(f"{f'ρ={int(g*100)}%':>10}" for g in TI_LE))
    print("-" * (31 + 10 * len(TI_LE)))

    for k, v in sorted(c.items(), key=lambda x: -x[1]):
        row = f"{k:<22}{v:>9,}"
        for giu in TI_LE:
            row += f"{p_c(v, n_tong, n_lop, giu):>10.3f}"
        print(row)

    print("\nTIÊN ĐOÁN")
    print("  1. Lớp head (p_c nhỏ và giảm theo ρ): recall phải TỤT khi ρ giảm.")
    hold = c["holding"]
    print(f"     `holding`: p_c đi từ {p_c(hold,n_tong,n_lop,1.0):.3f}"
          f" xuống {p_c(hold,n_tong,n_lop,0.10):.3f} — giảm"
          f" {p_c(hold,n_tong,n_lop,1.0)/p_c(hold,n_tong,n_lop,0.10):.0f} lần.")

    bat_bien = [k for k, v in c.items()
                if all(p_c(v, n_tong, n_lop, g) >= 1.0 for g in TI_LE)]
    print(f"  2. {len(bat_bien)} lớp có p_c = 1 ở MỌI ρ, nên recall KHÔNG được")
    print(f"     đổi theo ρ: {', '.join(sorted(bat_bien))}")
    print("     Đây là nhóm đối chứng nội tại — nếu chúng cũng đổi thì có yếu")
    print("     tố khác đang tác động, không phải cơ chế trong định lý.")

    print("\n  3. Mức tụt phải tỉ lệ với log(p_c), không phải tuyến tính theo ρ.")
    print("     So đường cong, không so hai điểm đầu cuối.")

    print("\nDỮ LIỆU CẦN: recall từng lớp của bốn checkpoint")
    print("  sttran_predcls (ρ=100%) · partial_70 · partial_40 · partial_10")
    print("  -> kernel impartail-main-table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ══════════════════════════════════════════════════════════════════
# KẾT QUẢ (kernel impartail-main-table v2, 14/09/2026) — TIÊN ĐOÁN HỎNG
# ══════════════════════════════════════════════════════════════════
#
# Nhóm đối chứng KHÔNG kiểm được gì, vì cả 4 lớp có ΣTP = 0 ở MỌI ρ:
#
#     lớp                    |F_c|   ΣGT    ΣTP (ρ=10/40/70/100)
#     have_it_on_the_back      161    161    0  0  0  0
#     twisting                  27     27    0  0  0  0
#     wiping                   246    255    0  0  0  0
#     writing_on               277    277    0  0  0  0
#
# Recall của chúng bất biến vì luôn bằng 0, KHÔNG phải vì cơ chế trong định lý
# giữ chúng ổn định. Một đại lượng hằng-0 tương thích với mọi lý thuyết, nên nó
# không phân biệt được giả thuyết nào — đây là đối chứng RỖNG, không phải đối
# chứng ĐẠT. Bản kiểm tự động ban đầu in "đối chứng ĐẠT" vì chỉ đo biên độ
# dao động; phải đọc thêm ΣTP mới thấy.
#
# Tiên đoán 1 (lớp head TỤT khi ρ giảm) cũng SAI ở 2/3 lớp:
#
#     lớp              ρ=10%   ρ=100%   chiều
#     holding          42.19    75.58   TĂNG theo ρ  (ngược tiên đoán)
#     not_contacting   50.61    90.57   TĂNG theo ρ  (ngược tiên đoán)
#     touching         60.94    18.47   giảm theo ρ  (đúng tiên đoán)
#
# Ngoài ra `in_front_of` có ΣTP = 0 ở ρ=40/70/100 nhưng 63,98% ở ρ=10 — lớp
# phổ biến nhất tập dữ liệu (277.032 nhãn) mà recall sụp về 0, dấu hiệu của
# vấn đề khác chứ không phải hiệu ứng đang xét.
#
# KẾT LUẬN: dữ liệu KHÔNG ủng hộ Định lý 4 theo dạng đã phát biểu. Các
# checkpoint này khác nhau cả ở số epoch (partial_10 dừng ở epoch 0, ba cái
# còn lại epoch 3), nên ρ không phải biến duy nhất thay đổi — thiết kế thí
# nghiệm chưa tách được biến. Cần checkpoint cùng số epoch trước khi phát
# biểu lại bất cứ điều gì.

# ══════════════════════════════════════════════════════════════════
# PHƯƠNG ÁN B — đổi nhóm đối chứng, không tốn thêm giờ GPU
# ══════════════════════════════════════════════════════════════════
#
# Nhóm đối chứng cũ rỗng, nên thử tìm nhóm khác trong dữ liệu đã có: lớp vừa
# có p_c = 1 ở MỌI ρ, vừa có TP > 0. KHÔNG CÓ lớp nào thoả. Bốn lớp p_c = 1
# đúng là bốn lớp hiếm nhất, và mô hình chưa từng đoán đúng chúng lần nào.
#
# Nên bỏ tiêu chí "bất biến" và kiểm thẳng tiên đoán cốt lõi — recall phải
# TĂNG theo p_c — bằng tương quan hạng Spearman trên 13 lớp còn dùng được
# (loại 4 lớp hằng-0 vì chúng tương thích với mọi lý thuyết):
#
#     ĐÚNG  4/13      SAI  9/13      ρ trung bình  -0,308
#     P(>= 4/13 đúng nếu chỉ do ngẫu nhiên) = 0,954
#
# Dấu trung bình ÂM: recall có xu hướng GIẢM khi p_c tăng, ngược hẳn tiên đoán.
#
# Kiểm lại sau khi loại biến epoch (chỉ 3 checkpoint cùng epoch 3, bỏ ρ=10 vốn
# dừng ở epoch 0): ĐÚNG 4/13, SAI 9/13, ρ tb -0,385 — KHÔNG đổi. Vậy kết quả
# âm không phải do thiết kế chưa tách biến, mà là kết luận thật.
#
# KẾT LUẬN PHƯƠNG ÁN B: Định lý 4 theo dạng đã phát biểu KHÔNG được dữ liệu
# ủng hộ, và việc train lại checkpoint cùng epoch (23 giờ GPU) sẽ không cứu
# được — biến epoch đã được loại mà kết quả giữ nguyên.
