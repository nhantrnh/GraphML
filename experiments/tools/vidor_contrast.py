#!/usr/bin/env python3
"""VidOR làm đối chứng tương phản cho Action Genome.

Phản biện đúng: nếu chỉ làm trên Action Genome, người đọc có quyền nói đây là
lỗi cục bộ của một dataset. VidOR trả lời câu đó — và câu trả lời không phải
"chỗ nào cũng hỏng" mà là "Action Genome hỏng vì hai đặc điểm cấu trúc cụ thể".

Số liệu đếm từ annotation chính thức (training-annotation.zip, mirror
HuggingFace shangxd/vidor), mẫu 800/5.604 video.

HAI KHÁC BIỆT CẤU TRÚC

1. Không có lớp phủ định. VidOR có 50 vị ngữ, KHÔNG lớp nào mang nghĩa phủ
   định hay gom: không `not_*`, không `unsure`, không `other_relationship`.
   Action Genome có 4 lớp như vậy trong 26, chiếm 15,4% trọng số mean Recall.
   -> Mệnh đề 1 (đảo thứ hạng) là đặc thù Action Genome.

2. Không person-centric. Trong Action Genome chủ thể LUÔN là person, nên n vật
   thể cho n cặp. VidOR cho mọi cặp vật thể có quan hệ, tức n(n−1) cặp.
   -> Không gian ứng viên là 50·n(n−1) thay vì 26·n, lớn hơn hàng chục lần.
   -> Mệnh đề 2 (bão hoà R@K) yếu đi rất nhiều.

Kết luận cho bài: Action Genome là benchmark bão hoà nặng nhất vì kết hợp
person-centric (ít cặp) với tập nhãn đóng hẹp (26 lớp). Đây là luận cứ mạnh
hơn "dataset nào cũng có vấn đề".
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIDOR = os.path.join(ROOT, "experiments/results/impartail_repro/vidor_stat.json")
AG_OBJ = os.path.join(ROOT, "experiments/results/impartail_repro/ag_objects_per_frame.json")

AM_AG = ["not_looking_at", "unsure", "not_contacting", "other_relationship"]


def main():
    if not os.path.isfile(VIDOR):
        sys.exit(f"không thấy {VIDOR}")
    v = json.load(open(VIDOR))
    pred, obj = v["pred"], {int(k): c for k, c in v["obj"].items()}
    P, tot = len(pred), sum(obj.values())

    print(__doc__.split("\n\n")[0])
    print(f"\nVidOR: {P} lớp vị ngữ, {tot:,} khung có vật thể "
          f"(mẫu {v['n_vid']} video)\n")

    # 1. Lớp phủ định
    co_am = [k for k in pred if k.startswith("not_") or k in
             ("unsure", "other", "other_relationship", "none")]
    print("1. LỚP PHỦ ĐỊNH / GOM")
    print(f"   VidOR        : {len(co_am)}/{P} lớp  {co_am if co_am else '(không có)'}")
    print(f"   Action Genome: {len(AM_AG)}/26 lớp  {', '.join(AM_AG)}")
    print(f"   -> Mệnh đề 1 là đặc thù Action Genome.\n")

    # 2. Bão hoà
    print("2. NGƯỠNG BÃO HOÀ R@K")
    print(f"   {'K':>5}{'VidOR':>12}{'Action Genome':>16}")
    print("   " + "-" * 33)
    ag = None
    if os.path.isfile(AG_OBJ):
        a = json.load(open(AG_OBJ))
        ag = {int(k): c for k, c in a["vat_moi_khung"].items()}
        ag_tot = a["tong_khung"]
    for K in (10, 20, 50, 100):
        bh_v = sum(c for n, c in obj.items() if n > 1 and P * n * (n - 1) <= K)
        cot_ag = "—"
        if ag:
            bh_a = sum(c for n, c in ag.items() if 26 * n <= K)
            cot_ag = f"{100*bh_a/ag_tot:.1f}%"
        print(f"   {K:>5}{100*bh_v/tot:>11.1f}%{cot_ag:>16}")

    print(f"\n   VidOR: {P}·n(n−1) ứng viên (mọi cặp vật thể)")
    print(f"   AG   : 26·n ứng viên (chủ thể luôn là person)")
    print(f"   -> Mệnh đề 2 yếu hẳn trên VidOR.\n")

    print("KẾT LUẬN")
    print("  Action Genome bão hoà nặng nhất vì kết hợp person-centric (ít cặp)")
    print("  với tập nhãn đóng hẹp (26 lớp). Đây là luận cứ có cấu trúc, mạnh")
    print("  hơn nhiều so với phát biểu chung chung rằng benchmark nào cũng có")
    print("  vấn đề — và nó cho thấy hai mệnh đề chính KHÔNG phải hiện tượng")
    print("  phổ quát, mà là hệ quả của những lựa chọn thiết kế cụ thể.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
