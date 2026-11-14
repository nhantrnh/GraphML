#!/usr/bin/env python3
"""Đo mức thổi phồng MRR do chọn checkpoint trên TẬP TEST.

APKGC (Trainer.py:85) tạo dataloader đánh giá từ tập test, rồi giữ checkpoint có
MRR cao nhất. Đây là chọn mô hình trên chính tập dùng để báo cáo.

Vết đánh giá lưu trong experiments/results/*/partial_*.json có dạng:
    [đánh giá lúc huấn luyện × k] + [đánh giá lại checkpoint tốt nhất]
nên phần tử cuối luôn bằng max của phần đầu. Mức thổi phồng là

    max(các lần huấn luyện) − lần huấn luyện cuối cùng

tức chênh lệch giữa "báo cáo đỉnh tìm được trên test" và "báo cáo checkpoint cuối".

Cũng đối chiếu với thống kê cực trị: nếu k lần đánh giá chỉ là nhiễu iid quanh
cùng một giá trị thì kỳ vọng max − cuối ≈ σ·sqrt(2·ln k). Đo được lớn hơn nhiều
nghĩa là mô hình đạt đỉnh giữa chừng rồi đi xuống, nên chọn trên test bắt được cả
đỉnh thật lẫn nhiễu.
"""
import glob, json, math, os, statistics, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEED_SD = 0.0009          # tự đo, đợt 2, n=3
PUB_GAP = 0.008           # khoảng cách APKGC–NativE trên MKG-W theo bài báo


def traces():
    for f in sorted(glob.glob(os.path.join(ROOT, "experiments/results/*/partial_*.json"))):
        d = json.load(open(f))
        m = [e["mrr"] for e in d["evals"]]
        if len(m) < 3:
            continue
        train, final = m[:-1], m[-1]
        # bất biến: phần tử cuối là đánh giá lại checkpoint tốt nhất
        if abs(max(train) - final) > 1e-6:
            print(f"  BỎ QUA {d['tag']}: cuối {final:.4f} != max {max(train):.4f}", file=sys.stderr)
            continue
        yield d["tag"], train


def main():
    rows = []
    for tag, tr in traces():
        rows.append(dict(tag=tag, k=len(tr), best=max(tr), last=tr[-1],
                         inflation=max(tr) - tr[-1], peak_at=tr.index(max(tr)) + 1))
    if not rows:
        print("không có vết đánh giá nào"); return 1

    print(f"{'run':<15}{'k':>3}{'đỉnh':>9}{'cuối':>9}{'thổi phồng':>12}{'đỉnh ở lần':>12}")
    for r in rows:
        print(f"{r['tag']:<15}{r['k']:>3}{r['best']:>9.4f}{r['last']:>9.4f}"
              f"{r['inflation']:>+12.4f}{r['peak_at']:>7}/{r['k']}")

    inf = [r["inflation"] for r in rows]
    mean = statistics.mean(inf)
    print(f"\ntrung bình      {mean:+.4f}   (khoảng {min(inf):+.4f} … {max(inf):+.4f})")
    print(f"  = {mean/SEED_SD:.1f}× độ lệch chuẩn giữa seed ({SEED_SD})")
    print(f"  = {mean/PUB_GAP*100:.0f}% khoảng cách APKGC–NativE đã công bố ({PUB_GAP})")

    print(f"\nđối chiếu thống kê cực trị (nhiễu iid, σ={SEED_SD}):")
    ratios = []
    for r in rows:
        pred = SEED_SD * math.sqrt(2 * math.log(r["k"]))
        ratios.append(r["inflation"] / pred)
        print(f"  {r['tag']:<15}k={r['k']:>2}  lý thuyết {pred:.4f}  đo {r['inflation']:.4f}"
              f"  ({r['inflation']/pred:.1f}×)")
    print(f"\n  trung vị {statistics.median(ratios):.1f}× so với dự đoán nhiễu thuần.")
    print("  Lớn hơn 1 nghĩa là các lần đánh giá KHÔNG phải nhiễu quanh một giá trị:")
    print("  mô hình đạt đỉnh giữa chừng rồi đi xuống, nên chọn trên test bắt được")
    print("  cả phần đỉnh thật lẫn phần nhiễu.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
