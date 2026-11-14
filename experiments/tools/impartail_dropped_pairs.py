#!/usr/bin/env python3
"""Bao nhiêu cặp (người, vật) biến mất hoàn toàn khỏi hàm mất mát?

Khi mặt nạ che HẾT nhãn của một cặp, mã công bố loại luôn cả cặp đó:

    gt_masked = gt[mask == 1]
    if gt_masked.shape[0] == 0:
        continue                      # stsg_base.py:297 (MLM) và :312 (BCE)

`continue` bỏ qua cặp, nên nó không vào `filtered_labels` lẫn
`filtered_distribution`. Mất không chỉ NHÃN bị che, mà mất cả thông tin "hai
vật thể này CÓ quan hệ" — thứ đúng bất kể quan hệ là gì.

Số đếm lấy từ chính tệp annotation của Action Genome (đếm đúng cách
estimate_rel_distribution() đếm, ag_dataset.py:41-52), không phải phân bố giả
định. Dữ liệu: experiments/results/impartail_repro/ag_predicate_freq.json.

Đây là mô tả mã đã công bố, không phải kết luận về cách tác giả chạy thí nghiệm.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "experiments/results/impartail_repro/ag_predicate_freq.json")

# Cấu hình chính của bài báo giữ lại 10% chú thích (S=10, scripts/sgg/partial/train.txt)
TI_LE_GIU = 0.10


def ti_le_giu_tung_nhom(tan_suat):
    """Alg.1 đặt Prob[rel]=1/|P| nên mỗi lớp nhận cùng hạn ngạch, clamp theo n_c."""
    ra = {}
    for nhom, c in tan_suat.items():
        n = sum(c.values())
        quota = n * TI_LE_GIU / len(c)
        ra[nhom] = (sum(min(quota, v) for v in c.values()) / n, quota, c)
    return ra


def main():
    if not os.path.isfile(DATA):
        sys.exit(f"không thấy {DATA} — chạy notebook impartail-predicate-freq trước")
    d = json.load(open(DATA))
    tan_suat, nhan_moi_cap, tong_cap = d["tan_suat"], d["nhan_moi_cap"], d["tong_cap"]

    print(__doc__.split("\n\n")[0])
    print(f"\nTập train: {tong_cap:,} cặp (người, vật) có chú thích")
    print(f"Tỉ lệ giữ lại: {TI_LE_GIU:.0%} (cấu hình S=10 của bài báo)\n")

    tl = ti_le_giu_tung_nhom(tan_suat)
    print(f"{'nhóm':<26}{'lớp':>5}{'giữ lại':>10}{'lớp giữ trọn':>14}")
    print("-" * 55)
    for nhom, (tb, quota, c) in tl.items():
        tron = sum(1 for v in c.values() if quota >= v)
        print(f"{nhom:<26}{len(c):>5}{tb:>10.3f}{tron:>8}/{len(c):<5}")

    # Mỗi cặp luôn mang ít nhất một nhãn ở CẢ BA nhóm, nên tối thiểu 3 nhãn.
    p_che = [1 - tl[n][0] for n in tan_suat]
    p3 = p_che[0] * p_che[1] * p_che[2]
    n3 = nhan_moi_cap.get("3", 0)

    print(f"\nPhân bố số nhãn mỗi cặp:")
    for k in sorted(nhan_moi_cap, key=int):
        v = nhan_moi_cap[k]
        print(f"  {k:>2} nhãn: {v:>9,} cặp ({100 * v / tong_cap:>5.1f}%)")

    print(f"\nP(che hết cả 3 nhãn của một cặp 3-nhãn) = {p3:.3f}")
    print(f"-> khoảng {p3 * n3:,.0f} cặp bị loại, tức {100 * p3 * n3 / tong_cap:.0f}% "
          f"tổng số cặp biến mất khỏi gradient.")

    print("\nVì sao đáng quan tâm:")
    print("  - Cặp bị loại vẫn là dữ liệu ĐÚNG: ta biết chắc hai vật thể có quan hệ.")
    print("  - Mặt nạ vốn để 'không học từ nhãn head', nhưng hệ quả phụ là")
    print("    'không học từ cặp đó nữa', kể cả phần thông tin không liên quan head.")

    # ── Cặp bị vứt lệch về đâu? Câu này quyết định mọi cách khắc phục ──
    print("\n" + "=" * 55)
    print("Cặp bị vứt lệch về lớp nào?\n")
    for nhom, (tb, quota, c) in tl.items():
        xep = sorted(c.items(), key=lambda x: -x[1])
        mat = {k: v * (1 - min(quota, v) / v) for k, v in c.items()}
        tong_mat = sum(mat.values())
        top3 = sum(mat[k] for k, _ in xep[:3])
        print(f"  {nhom:<26} 3 lớp phổ biến nhất chiếm "
              f"{100 * top3 / tong_mat:.1f}% số nhãn bị che")

    print("\nHỆ QUẢ — bác bỏ cách khắc phục ngây thơ:")
    print("  Ý tưởng 'giữ cặp lại, chỉ học CÓ quan hệ mà không học quan hệ nào'")
    print("  nghe hợp lý nhưng KHÔNG dùng được: nhãn nhị phân tự nó không thiên")
    print("  lệch, song tập cặp được cứu thì lệch mạnh về head — nó củng cố đúng")
    print("  prior mà phương pháp đang cố phá. Mỗi cặp cũng chỉ cho 1 bit, và cơ")
    print("  chế đã có tiền lệ: Graph R-CNN (ECCV 2018) huấn luyện BCE trên")
    print("  relation proposal; RS-Net (preprint) có đầu nhị phân relatedness phụ.")

    print("\nHướng còn đứng vững: fallback mức NHÓM thay vì nhị phân.")
    print("  Xem group_fallback_viability.py — spatial và contacting dùng sigmoid")
    print("  độc lập nên giám sát mức nhóm có gradient thật; attention đi qua")
    print("  CrossEntropyLoss nên marginalize ở đó là phép toán rỗng.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
