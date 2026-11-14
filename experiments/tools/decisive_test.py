#!/usr/bin/env python3
"""Phép thử quyết định: loại 4 lớp phủ định rồi tính lại mean Recall.

Lý thuyết (rank_reversal_bound.py) nói thứ hạng CÓ THỂ đảo khi δ < N/T.
Thí nghiệm này kiểm xem nó có đảo THẬT trên hai checkpoint đã huấn luyện.

KẾT QUẢ: thứ hạng KHÔNG đảo (0/3 ràng buộc). Nhưng chiến lược thoát hiểm cho
kết quả mạnh hơn — xem phần co biên độ bên dưới.

Dữ liệu: experiments/results/impartail_repro/per_class_recall.json, do
impartail/patches/dump_per_class_recall.py sinh ra trên Kaggle. Hai nhánh chỉ
khác nhau cờ --true_masked_bce.

Chỉ số lớp phủ định theo dataloader/base_ag_dataset.py:69-83:
    1 not_looking_at · 2 unsure · 17 not_contacting · 18 other_relationship
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "experiments/results/impartail_repro/per_class_recall.json")
AM = {1, 2, 17, 18}
C_TOTAL = 26
N_AM = len(AM)
T_THAT = C_TOTAL - N_AM
RB = ("with", "no", "semi")


def tach(pc):
    """Trả (mR đủ C, mR trên T lớp thật, mR trên N lớp phủ định), đơn vị %."""
    du = sum(pc) / len(pc)
    mT = sum(v for i, v in enumerate(pc) if i not in AM) / T_THAT
    mN = sum(v for i, v in enumerate(pc) if i in AM) / N_AM
    return du * 100, mT * 100, mN * 100


def main():
    if not os.path.isfile(DATA):
        sys.exit(f"không thấy {DATA}")
    d = json.load(open(DATA))

    print(__doc__.split("\n\n")[0])
    print("\nPHẦN 1 — Thứ hạng có đảo không?\n")
    print(f"{'ràng buộc':<8}{'nhánh':<6}{'mR đủ 26':>10}{'mR bỏ 4 âm':>13}{'lệch':>9}")
    print("-" * 48)
    bang = {}
    for rb in RB:
        for nh in ("goc", "va"):
            du, mT, mN = tach(d[nh][rb]["20"])
            bang[(rb, nh)] = (du, mT, mN)
            print(f"{rb:<8}{nh:<6}{du:>10.2f}{mT:>13.2f}{mT-du:>+9.2f}")

    dao = 0
    print()
    for rb in RB:
        gd, gT, _ = bang[(rb, "goc")]
        vd, vT, _ = bang[(rb, "va")]
        td = "goc" if gd > vd else "va"
        tb = "goc" if gT > vT else "va"
        if td != tb:
            dao += 1
        print(f"  {rb:<6} đủ 26: {td:<4} | bỏ 4 âm: {tb:<4}"
              f"{'   <<< ĐẢO' if td != tb else ''}")
    print(f"\n  => {dao}/3 ràng buộc bị đảo thứ hạng.")

    # ── Chiến lược thoát hiểm: co biên độ và tỉ lệ tiến bộ giả ──────
    print("\n\nPHẦN 2 — Co biên độ (kết luận không phụ thuộc việc có đảo)\n")
    print(f"{'ràng buộc':<8}{'Δ đủ 26':>10}{'Δ 22 lớp':>11}{'co':>8}"
          f"{'Δ_T':>9}{'Δ_N':>9}{'SR':>8}")
    print("-" * 63)
    for rb in RB:
        g = bang[(rb, "goc")]
        v = bang[(rb, "va")]
        b_du, b_T = v[0] - g[0], v[1] - g[1]
        dT, dN = v[1] - g[1], v[2] - g[2]
        rep = (T_THAT * dT + N_AM * dN) / C_TOTAL
        SR = (N_AM * dN / C_TOTAL) / rep if rep else float("nan")
        co = (1 - abs(b_T) / abs(b_du)) * 100 if b_du else float("nan")
        print(f"{rb:<8}{b_du:>+10.2f}{b_T:>+11.2f}{co:>7.0f}%"
              f"{dT:>+9.2f}{dN:>+9.2f}{SR:>8.3f}")

    print("\nĐỌC KẾT QUẢ")
    print("  with-constraint: biên độ −1,59 điểm giữa hai nhánh co lại còn")
    print("  −0,01 khi bỏ 4 lớp phủ định — tức 99% chênh lệch đến từ chúng")
    print("  (Δ_T = −0,01 so với Δ_N = −10,23, SR = 0,992).")
    print("\n  Nói cách khác: hai nhánh gần như KHÔNG khác nhau trên 22 lớp")
    print("  mang nghĩa; toàn bộ khác biệt nằm ở nhóm lớp phủ định.")

    print("\nGIỮ ĐÚNG MỨC")
    print("  Đây là hai checkpoint của CÙNG một phương pháp khác nhau ở cách")
    print("  cài mặt nạ, không phải hai phương pháp cạnh tranh. Muốn kết luận")
    print("  về bảng xếp hạng SOTA thì phải chạy trên STTran, FloCoDe,")
    print("  ImparTail, VISA — đó là thí nghiệm còn lại.")
    print("\n  Mỗi nhánh cũng mới chạy MỘT seed, hai epoch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
