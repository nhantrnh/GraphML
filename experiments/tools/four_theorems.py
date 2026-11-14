#!/usr/bin/env python3
"""Bốn dẫn xuất giải tích dạng đóng cho đề cương mR@K.

Mỗi mệnh đề dưới đây có chứng minh ba dòng và được kiểm lại bằng số. Chúng
thay thế các lập luận bằng ví dụ hoặc Monte Carlo trong bản trước.
"""
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FREQ = os.path.join(ROOT, "experiments/results/impartail_repro/ag_predicate_freq.json")
OBJ = os.path.join(ROOT, "experiments/results/impartail_repro/ag_objects_per_frame.json")

C_TOTAL, N_AM = 26, 4
T_THAT = C_TOTAL - N_AM


def dinh_ly_1():
    """Hệ số phồng biên độ — phần tiến bộ đến từ lớp phủ định."""
    print("ĐỊNH LÝ 1 (Hệ số phồng biên độ)\n")
    print("  Mức tăng báo cáo trên đủ C lớp phân rã thành")
    print(f"      Δ_rep = (T/C)·Δ_T + (N/C)·Δ_N = ({T_THAT}/{C_TOTAL})·Δ_T"
          f" + ({N_AM}/{C_TOTAL})·Δ_N")
    print("  Tỉ lệ tiến bộ đến từ lớp phủ định:")
    print(f"      SR = (N/C)·Δ_N / Δ_rep = 1 / (1 + (T/N)·Δ_T/Δ_N)")
    print(f"         = 1 / (1 + {T_THAT/N_AM:.1f}·Δ_T/Δ_N)\n")

    print(f"  {'Δ_T':>6}{'Δ_N':>7}{'SR trực tiếp':>15}{'công thức':>12}")
    print("  " + "-" * 42)
    for dT, dN in ((1.0, 5.0), (2.0, 2.0), (0.5, 10.0), (3.0, 1.0)):
        rep = (T_THAT * dT + N_AM * dN) / C_TOTAL
        truc = (N_AM * dN / C_TOTAL) / rep
        ct = 1 / (1 + (T_THAT / N_AM) * (dT / dN))
        print(f"  {dT:>6.1f}{dN:>7.1f}{truc:>15.4f}{ct:>12.4f}")
    print("\n  Dùng thế nào: điền SR cho từng mô hình trong bảng thực nghiệm.")
    print("  SR > 0,5 ở bất kỳ mô hình nào nghĩa là hơn một nửa thành tích")
    print("  công bố đến từ việc tối ưu các vị ngữ phủ định.")
    print("\n  ĐIỀU KIỆN DÙNG ĐƯỢC. Đẳng thức đúng với mọi Δ, nhưng cách đọc")
    print("  SR như 'tỉ lệ tiến bộ' chỉ có nghĩa khi Δ_T và Δ_N CÙNG DẤU.")
    print("  Trái dấu thì SR ra ngoài [0,1] — ví dụ Δ_T=−1, Δ_N=5 cho")
    print("  SR = −10,0 — và lúc đó phải báo cáo Δ_T, Δ_N riêng thay vì SR.")
    print("  Trên cặp nhánh mặt nạ (decisive_test.py) hai đại lượng cùng dấu")
    print("  âm, SR = 0,992 nằm trong khoảng hợp lệ. Nhưng trên cặp HAI")
    print("  PHƯƠNG PHÁP (impartail-ranking, 15/09) thì TRÁI DẤU:")
    print("      Δ_T = +1,35 (22 lớp mang nghĩa)   Δ_N = −3,29 (4 lớp âm)")
    print("      SR = −0,796  -> ngoài [0,1], không đọc được là 'tỉ lệ'")
    print("  Ở đó 4 lớp phủ định KHÔNG thổi phồng biên mà DÌM nó: DSG-DETR")
    print("  hơn STTran +1,35 trên lớp mang nghĩa, nhưng mR đủ 26 chỉ cho")
    print("  +0,64 vì STTran ăn điểm ở lớp âm (mR_N 48,47 vs 45,18).")
    print("  Chiều của sai lệch phụ thuộc cặp mô hình, không cố định.")


def dinh_ly_2():
    """Nguyên lý Dirichlet dưới ràng buộc."""
    print("\n\nĐỊNH LÝ 2 (Chặn ứng viên dưới with-constraint)\n")
    print("  Với mỗi cặp (u,v), bộ dự đoán lấy argmax trên tập vị ngữ")
    print("  (evaluation_recall.py:180), nên sinh ĐÚNG một dự đoán mỗi cặp.")
    print("  Recall bão hoà khi K phủ hết số NHÃN ĐÚNG trong khung, m_f:")
    print("      R@K ≡ R@(max_f m_f)  với mọi K ≥ max_f m_f")
    print("  Lưu ý m_f (số nhãn) ≠ n_f (số cặp): mỗi cặp mang đồng thời một")
    print("  vị ngữ attention, một spatial và một contacting.\n")

    if not os.path.isfile(OBJ):
        print("  (thiếu ag_objects_per_frame.json — bỏ qua phần số)")
        return
    o = json.load(open(OBJ))
    tong = o["tong_khung"]
    # Chặn phải so với số NHÃN mỗi khung, không phải số VẬT THỂ. Mỗi cặp mang
    # đồng thời một vị ngữ attention, một spatial và một contacting, nên số
    # nhãn lớn hơn hẳn số vật thể (max 29 so với max 8). Bản đầu của định lý
    # này dùng vat_moi_khung và kết luận R@K ≡ R@8, sai: số tái lập cho
    # R@10 = 62,03 nhưng R@20 = 64,94 ở with-constraint.
    vat = {int(k): c for k, c in o["vat_moi_khung"].items()}
    nhan = {int(k): c for k, c in o["nhan_moi_khung"].items()}
    print(f"  max vật thể mỗi khung : {max(vat)}")
    print(f"  max NHÃN  mỗi khung   : {max(nhan)}   <- đây mới là chặn đúng\n")
    for n in (10, 20, 50):
        p = sum(c for k, c in nhan.items() if k <= n) / tong
        print(f"  khung có mọi nhãn trong top-{n:>2}: {100*p:>6.2f}%")
    n_max = max(nhan)
    print(f"\n  => R@K ≡ R@{n_max} với mọi K ≥ {n_max}, KHÔNG phải K ≥ {max(vat)}.")
    print(f"     K = 20 đã phủ 99,84% số khung nên R@50 và R@100 gần như")
    print(f"     không thêm thông tin, nhưng R@10 -> R@20 vẫn tăng thật")
    print(f"     (62,03 -> 64,94 ở with-constraint).")


def dinh_ly_3():
    """Phân rã hiệp phương sai cho sai lệch macro-frame vs global-ratio."""
    print("\n\nĐỊNH LÝ 3 (Phân rã hiệp phương sai)\n")
    print("  Đặt R_{f,c} = TP_{f,c}/GT_{f,c} và w_{f,c} = GT_{f,c}/Σ_f GT_{f,c}.")
    print("  Khi đó sai lệch giữa hai định nghĩa bằng ĐÚNG")
    print("      R_c^global − R̄_c^macro = |F_c| · Cov(R_{f,c}, w_{f,c})\n")
    print("  Cov < 0 (đoán trúng ở khung ít mẫu) -> macro bị THỔI PHỒNG.")
    print("  Cov > 0 (đoán trúng ở khung nhiều mẫu) -> macro bị DÌM.\n")

    random.seed(1)
    print(f"  {'|F|':>5}{'R_global':>11}{'R_macro':>10}{'hiệu':>10}{'|F|·Cov':>10}")
    print("  " + "-" * 46)
    for _ in range(4):
        F = random.randint(4, 9)
        gt = [random.randint(1, 12) for _ in range(F)]
        tp = [random.randint(0, g) for g in gt]
        Rf = [t / g for t, g in zip(tp, gt)]
        S = sum(gt)
        w = [g / S for g in gt]
        Rg, Rm = sum(tp) / S, sum(Rf) / F
        mR, mw = sum(Rf) / F, sum(w) / F
        cov = sum((Rf[i] - mR) * (w[i] - mw) for i in range(F)) / F
        print(f"  {F:>5}{Rg:>11.4f}{Rm:>10.4f}{Rg-Rm:>+10.4f}{F*cov:>+10.4f}")
    print("\n  Khớp chính xác trong sai số máy.")
    print("  Thí nghiệm cần chạy: tính ΔR_c cho cả 26 lớp trên checkpoint thật,")
    print("  rồi vẽ ΔR_c theo tần suất lớp. Nếu lớp đuôi có ΔR_c > 0 thì chỉ số")
    print("  hiện tại đang trợ giá cho chính nhóm lớp nó định đo.")


def dinh_ly_4():
    """Điểm cân bằng gradient dưới che nhãn lệch."""
    print("\n\nĐỊNH LÝ 4 (Điểm cân bằng gradient)\n")
    print("  Với BCE, đạo hàm theo LOGIT là ∂L/∂z = ŷ − y. Mặt nạ giữ nhãn")
    print("  dương với xác suất p_c, biến thành âm với xác suất 1 − p_c:")
    print("      E[∂L/∂z] = p_c(ŷ − 1) + (1 − p_c)(ŷ − 0) = ŷ − p_c\n")
    print("  Hệ quả: điểm cân bằng ổn định bị chặn cứng tại ŷ* = p_c.")
    print("    ŷ < p_c -> gradient âm -> tăng logit (học đúng)")
    print("    ŷ > p_c -> gradient dương -> GIẢM logit (phạt vì đoán đúng)\n")

    random.seed(0)
    N = 200000
    print(f"  {'ŷ':>7}{'p_c':>8}{'Monte Carlo':>14}{'giải tích':>12}{'':4}chiều")
    print("  " + "-" * 50)
    for yh, pc in ((0.60, 0.016), (0.010, 0.016), (0.30, 0.30)):
        mc = sum(yh - (1.0 if random.random() < pc else 0.0)
                 for _ in range(N)) / N
        chieu = "kéo XUỐNG" if yh > pc else "kéo LÊN"
        print(f"  {yh:>7.3f}{pc:>8.3f}{mc:>+14.5f}{yh-pc:>+12.5f}    {chieu}")

    if os.path.isfile(FREQ):
        f = json.load(open(FREQ))["tan_suat"]["contacting_relationship"]
        n = sum(f.values())
        quota = n * 0.10 / len(f)
        pc = min(quota, f["holding"]) / f["holding"]
        print(f"\n  Với lớp `holding`, p_c = {pc:.3f}: mô hình chỉ cần gán xác")
        print(f"  suất > {100*pc:.1f}% cho một quan hệ CÓ THẬT là bị kéo giật lùi.")
        print("  Đây là dẫn xuất giải thích vì sao nhánh --true_masked_bce cho")
        print("  Recall tăng +9,4 và +11,4 điểm.")


def main():
    print(__doc__.strip())
    print("\n" + "=" * 58)
    dinh_ly_1()
    dinh_ly_2()
    dinh_ly_3()
    dinh_ly_4()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
