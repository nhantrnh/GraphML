#!/usr/bin/env python3
"""Phân bố vị ngữ Action Genome có tuân Zipf không? — KHÔNG.

Đây là phép thử cổng (gate test) cho một hướng lý thuyết: nếu tần suất lớp
tuân luật luỹ thừa `p_c ∝ c^(-s)` thì có thể mượn nền infinite urn scheme
(Gnedin, Hansen & Pitman, *Probability Surveys* 4, 2007, `10.1214/07-PS092`)
— Proposition 23 cho tương đương Zipf ⟺ biến thiên chính quy, Corollary 21
cho Heaps' law — để suy ra tính chất của mean Recall theo (C, s).

Nếu KHÔNG tuân, toàn bộ nền đó vô dụng và hướng phải bỏ. Phép thử này tốn
vài phút; đi sai hướng tốn vài tháng.

Cách kiểm: so hai mô hình trên cùng dữ liệu.
  Zipf (luật luỹ thừa):  log f = log C − s·log(hạng)   -> tuyến tính trên log-log
  Hàm mũ:                log f = log C − λ·hạng        -> tuyến tính trên log-linear

KẾT QUẢ: cả ba nhóm vị ngữ đều khớp HÀM MŨ tốt hơn Zipf. Hướng Zipf bị loại.
"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "experiments/results/impartail_repro/ag_predicate_freq.json")


def fit(xs, logx):
    """Hồi quy tuyến tính log(tần suất) theo logx(hạng). Trả (độ dốc, R²)."""
    n = len(xs)
    lf = [math.log(v) for v in xs]
    lr = [logx(i) for i in range(n)]
    mx, my = sum(lr) / n, sum(lf) / n
    den = sum((r - mx) ** 2 for r in lr)
    if den == 0:
        return None, 0.0
    s = -sum((lr[i] - mx) * (lf[i] - my) for i in range(n)) / den
    b = my + s * mx
    res = sum((lf[i] - (b - s * lr[i])) ** 2 for i in range(n))
    tot = sum((f - my) ** 2 for f in lf)
    return s, 1 - res / tot


def main():
    if not os.path.isfile(DATA):
        sys.exit(f"không thấy {DATA}")
    d = json.load(open(DATA))["tan_suat"]

    print(__doc__.split("\n\n")[0])
    print(f"\n{'nhóm':<26}{'lớp':>5}{'s (Zipf)':>10}{'R² Zipf':>10}{'R² mũ':>9}")
    print("-" * 60)
    thang_mu = 0
    for nhom, c in d.items():
        xs = sorted(c.values(), reverse=True)
        s_z, r_z = fit(xs, lambda i: math.log(i + 1))
        _, r_e = fit(xs, lambda i: i)
        if r_e > r_z:
            thang_mu += 1
        print(f"{nhom:<26}{len(xs):>5}{s_z:>10.3f}{r_z:>10.4f}{r_e:>9.4f}")

    print("-" * 60)
    print(f"\n{thang_mu}/{len(d)} nhóm khớp hàm mũ tốt hơn Zipf.")

    if thang_mu == len(d):
        print("\nKẾT LUẬN: phân bố KHÔNG tuân luật luỹ thừa.")
        print("  -> Nền infinite urn scheme (Gnedin-Hansen-Pitman) giả định")
        print("     power-law, nên KHÔNG áp dụng được. Hướng Zipf bị loại.")
        print("  -> Mọi bound long-tail hiện có đều phụ thuộc n_j chứ không")
        print("     phụ thuộc tham số Zipf, nên cũng không mượn được.")
    else:
        print("\nMột phần nhóm vẫn khớp Zipf — cần xem kỹ từng nhóm.")

    # Sai lệch có hệ thống ở hai đầu là dấu hiệu điển hình của "không phải Zipf"
    c = d["contacting_relationship"]
    xs = sorted(c.values(), reverse=True)
    s_z, _ = fit(xs, lambda i: math.log(i + 1))
    lf = [math.log(v) for v in xs]
    n = len(xs)
    lr = [math.log(i + 1) for i in range(n)]
    b = sum(lf) / n + s_z * sum(lr) / n
    print(f"\nSai lệch ở hai đầu (contacting, s = {s_z:.2f}):")
    for i in (0, n - 1):
        pred = math.exp(b - s_z * lr[i])
        print(f"  hạng {i+1:>2}: thật {xs[i]:>8,} | Zipf dự đoán {pred:>9,.0f}"
              f" | tỉ lệ {xs[i]/pred:.2f}")
    print("  Lệch mạnh ở CẢ HAI đầu -> không phải nhiễu mẫu, mà sai dạng mô hình.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
