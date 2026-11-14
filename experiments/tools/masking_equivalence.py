#!/usr/bin/env python3
"""Sơ đồ mặt nạ của ImparTail có tương đương p_mask ∝ tần suất không?

Khảo sát cảnh báo: Algorithm 1 (tr.6) đặt Prob[rel] = 1/|P| — phân bố ĐỀU trên
LỚP vị ngữ, không phải trên instance — nên ImparTail đã cân bằng lớp một cách
ngầm định. Nếu "che theo tần suất lớp" chỉ tái tạo đúng hành vi đó thì nó không
phải đóng góp. Phải kiểm TRƯỚC khi chạy thí nghiệm, không chạy rồi mới biết.

Hai sơ đồ:

  A. ImparTail (Alg. 1): Tar[c] ~ Multinomial(N_target, 1/|P|), rồi giữ lại
     Tar[c] instance của lớp c — kèm clamp k = min(k, n_c) ở
     ag_dataset.py:82-83 vì lớp hiếm không đủ instance.

  B. p_mask ∝ f_c: mỗi instance của lớp c được giữ độc lập với xác suất
     q_c ∝ 1/f_c, chuẩn hoá sao cho tổng kỳ vọng bằng N_target.

Kết luận rút ra từ tỉ lệ giữ lại kỳ vọng của từng lớp.
"""
import sys

# Phân bố vị ngữ Action Genome. Dùng dạng đuôi dài điển hình: vài lớp tiếp xúc
# phổ biến chiếm phần lớn, phần còn lại thưa dần. Không phải số đếm thật của
# dataset — script này so SƠ ĐỒ, không đo dataset; mọi kết luận chỉ dựa vào
# dạng đuôi dài, không vào giá trị cụ thể.
FREQ = [42000, 31000, 17000, 9500, 6100, 4200, 2800, 1900,
        1300, 880, 610, 420, 290, 200, 140, 95, 60]


def so_do_A(freq, n_target):
    """Multinomial đều trên lớp + clamp theo số instance sẵn có."""
    P = len(freq)
    quota = n_target / P                 # kỳ vọng của Multinomial đều
    giu = [min(quota, f) for f in freq]  # clamp: ag_dataset.py:82-83
    return giu


def so_do_B(freq, n_target):
    """Giữ instance với xác suất ∝ 1/f_c, chuẩn hoá về n_target."""
    inv = [1.0 / f for f in freq]
    z = sum(f * i for f, i in zip(freq, inv))   # = số lớp
    scale = n_target / z
    giu = [min(f * i * scale, f) for f, i in zip(freq, inv)]
    return giu


def main():
    tong = sum(FREQ)
    n_target = int(tong * 0.10)   # tỉ lệ giữ 10%, cấu hình S=10 của bài báo

    A = so_do_A(FREQ, n_target)
    B = so_do_B(FREQ, n_target)

    print(__doc__.split("\n\n")[0])
    print(f"\n{len(FREQ)} lớp, {tong:,} instance, giữ lại {n_target:,} (10%)\n")
    print(f"{'lớp':>4}{'tần suất':>10}{'A giữ':>10}{'B giữ':>10}"
          f"{'A tỉ lệ':>10}{'B tỉ lệ':>10}")
    print("-" * 54)
    for c, (f, a, b) in enumerate(zip(FREQ, A, B)):
        print(f"{c:>4}{f:>10,}{a:>10,.0f}{b:>10,.0f}{a / f:>10.3f}{b / f:>10.3f}")

    print("-" * 54)
    print(f"{'tổng':>4}{tong:>10,}{sum(A):>10,.0f}{sum(B):>10,.0f}")

    # Câu hỏi quyết định: hai sơ đồ có cho cùng SỐ ĐẾM GIỮ LẠI mỗi lớp không?
    lech = [abs(a - b) for a, b in zip(A, B)]
    lech_tuong_doi = sum(lech) / sum(A)
    print(f"\nSai khác tổng (chuẩn L1, tương đối): {lech_tuong_doi:.1%}")

    # Lớp nào bị clamp ở sơ đồ A — đó là chỗ hai sơ đồ tách nhau
    quota = n_target / len(FREQ)
    bi_clamp = [c for c, f in enumerate(FREQ) if f < quota]
    print(f"Lớp bị clamp ở sơ đồ A (n_c < hạn ngạch {quota:,.0f}): "
          f"{len(bi_clamp)}/{len(FREQ)} — {bi_clamp}")

    print("\nKẾT LUẬN")
    if lech_tuong_doi < 0.05:
        print("  Hai sơ đồ gần như TRÙNG NHAU. 'Che theo tần suất' chỉ tái tạo")
        print("  hành vi sẵn có của Alg. 1 -> KHÔNG đủ làm đóng góp chính.")
    else:
        print(f"  Hai sơ đồ KHÁC NHAU rõ ({lech_tuong_doi:.0%}). Khác biệt nằm ở")
        print("  nhóm lớp hiếm bị clamp: sơ đồ A không thể cấp đủ hạn ngạch đều")
        print("  cho chúng nên tổng thực tế hụt so với mục tiêu, còn sơ đồ B")
        print("  phân bổ liên tục theo 1/f nên không có ngưỡng gãy.")
        print("  -> Vẫn có chỗ cho đóng góp, nhưng phải định vị là 'thay cơ chế")
        print("     rời rạc có ngưỡng gãy bằng tham số hoá liên tục', KHÔNG phải")
        print("     'thêm nhận biết tần suất vào phương pháp vốn không có'.")

    print("\nLƯU Ý: đây là so sánh SƠ ĐỒ trên dạng đuôi dài điển hình, không phải")
    print("đo trên số đếm thật của Action Genome. Trước khi viết bài phải chạy")
    print("lại với rel_counts thật từ estimate_rel_distribution().")


if __name__ == "__main__":
    main()
