#!/usr/bin/env python3
"""R@K trên Action Genome bão hoà ở đâu — mệnh đề chứng minh được.

Ở chế độ no-constraint (evaluation_recall.py:173-178), mã xếp hạng TOÀN BỘ 26
vị ngữ cho mỗi cặp (người, vật). Với khung hình có n vật thể nhìn thấy được:

    số ứng viên = 26·n        số triplet đúng ≈ 3·n

(mỗi cặp mang ít nhất một nhãn ở cả ba nhóm attention/spatial/contacting, và
77,6% cặp mang đúng ba nhãn — đếm được từ annotation).

MỆNH ĐỀ. Khi K ≥ 26·n, tập K ứng viên hàng đầu chứa TOÀN BỘ không gian dự
đoán, nên R@K ≡ 1 bất kể mô hình. Chỉ số không còn phân biệt được gì.

Đây là mệnh đề về chỉ số, đúng cho mọi mô hình, không cần thí nghiệm. Số khung
hình lấy từ chính tệp annotation, không phải giả định.

Không tồn tại định lý nào trong văn liệu về bão hoà Recall@K — đừng trích kiểu
đó. Nền gần nhất cho "khả năng phân biệt của chỉ số" là Buckley & Voorhees,
SIGIR 2000 (10.1145/345508.345543): "Measures that depend on a relatively few
highly ranked documents have higher error rates."
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "experiments/results/impartail_repro/ag_objects_per_frame.json")
N_PRED = 26          # len(AG_all_predicates)


def main():
    if not os.path.isfile(DATA):
        sys.exit(f"không thấy {DATA} — chạy notebook impartail-obj-per-frame trước")
    d = json.load(open(DATA))
    v = {int(k): c for k, c in d["vat_moi_khung"].items()}
    tong = d["tong_khung"]

    print(__doc__.split("\n\n")[0])
    print(f"\nTập train: {tong:,} khung hình có vật thể\n")
    print(f"{'vật/khung':>10}{'số khung':>11}{'tỉ lệ':>8}{'ứng viên':>10}{'GT':>6}")
    print("-" * 46)
    for n in sorted(v):
        print(f"{n:>10}{v[n]:>11,}{100*v[n]/tong:>7.1f}%{N_PRED*n:>10}{3*n:>6}")

    print(f"\n{'K':>5}{'bão hoà tuyệt đối':>20}{'K ≥ nửa không gian':>21}")
    print("-" * 46)
    for K in (10, 20, 50, 100):
        bh = sum(c for n, c in v.items() if N_PRED * n <= K)
        gan = sum(c for n, c in v.items() if N_PRED * n <= 2 * K)
        print(f"{K:>5}{100*bh/tong:>19.1f}%{100*gan/tong:>20.1f}%")

    bh50 = sum(c for n, c in v.items() if N_PRED * n <= 50) / tong
    bh100 = sum(c for n, c in v.items() if N_PRED * n <= 100) / tong
    print(f"\nR@50 vô nghĩa trên {bh50:.1%} số khung; R@100 trên {bh100:.1%}.")
    print("Bài báo báo cáo R@50 và R@100 như hai con số độc lập, nhưng trên")
    print("phần lớn dữ liệu chúng đo cùng một thứ: toàn bộ không gian dự đoán.")

    print("\nHệ quả cho cách đọc bảng kết quả: chênh lệch ở R@50/R@100 giữa các")
    print("phương pháp phần lớn đến từ thiểu số khung hình nhiều vật thể, chứ")
    print("không phải từ toàn tập. Chỉ R@10 và R@20 còn phân biệt được trên")
    print("đa số dữ liệu.")

    # ── PHẠM VI: mệnh đề này KHÔNG áp dụng đồng đều cho ba subtask ──
    print("\n" + "=" * 56)
    print("PHẠM VI ÁP DỤNG — phải nêu rõ, nếu không sẽ bị bác\n")
    print("  PredCls / SGCls: chủ thể luôn là person nên n vật -> n cặp")
    print(f"                   -> {N_PRED}n ứng viên. Mệnh đề áp dụng ĐẦY ĐỦ.")
    M = 20
    print(f"  SGDet          : M hộp phát hiện -> M(M-1) cặp. Với M={M} là")
    print(f"                   {N_PRED*M*(M-1):,} ứng viên -> K=100 KHÔNG bão hoà.")
    print("\n  Nói cách khác: mệnh đề tác động mạnh nhất lên PredCls và SGCls —")
    print("  vốn là hai subtask chiếm phần lớn bảng báo cáo của dòng unbiased")
    print("  SGG — nhưng không phủ SGDet. Nêu rõ giới hạn này trong bài.")

    # ── Chế độ ràng buộc cũng đổi không gian ứng viên ──────────────
    print("\n" + "=" * 56)
    print("CHẾ ĐỘ RÀNG BUỘC — bão hoà còn SỚM hơn ở with/semi\n")
    print("  evaluation_recall.py:173-181 rẽ hai nhánh:")
    print("    no-constraint  : argsort_desc(overall_scores)[:100]")
    print(f"                     -> tối đa {N_PRED} vị ngữ mỗi cặp = {N_PRED}n ứng viên")
    print("    with/semi      : rel_scores.argmax(1)")
    print("                     -> ĐÚNG 1 vị ngữ mỗi cặp = n ứng viên\n")
    print(f"  {'K':>5}{'no-constraint':>18}{'with/semi':>14}")
    print("  " + "-" * 37)
    for K in (10, 20, 50, 100):
        bh_no = sum(c for n, c in v.items() if N_PRED * n <= K)
        bh_wc = sum(c for n, c in v.items() if n <= K)   # chỉ n ứng viên
        print(f"  {K:>5}{100*bh_no/tong:>17.1f}%{100*bh_wc/tong:>13.1f}%")
    print("\n  Ở with/semi-constraint, số ứng viên chỉ bằng số cặp, nên K=10 đã")
    print("  phủ toàn bộ không gian trên gần như MỌI khung hình. Điều này giải")
    print("  thích vì sao R@20, R@50, R@100 trùng nhau ở hai chế độ đó.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
