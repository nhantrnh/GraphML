#!/usr/bin/env python3
"""Đối chiếu kết quả tái lập ImparTail với Bảng 1 của bài báo.

Chạy trên Kaggle (Tesla P100), checkpoint `sttran_predcls` do tác giả phát hành,
toàn bộ 1.750 video của tập test Action Genome, chế độ PredCls.

Thứ tự 36 cột CSV lấy từ test_sgg_base.py:186-193 — ba nhóm 12 cột theo đúng
thứ tự _collate_evaluation_stats() ghi ra: with → no → semi constraint, mỗi nhóm
là R@{10,20,50,100}, mR@{10,20,50,100}, hR@{10,20,50,100}.

Số của bài báo: Bảng 1 "Mean Recall Results for VidSGG", hàng PREDCLS / STTran —
tức mô hình NỀN, không phải bản +IMPARTAIL.
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV = os.path.join(ROOT, "experiments/results/impartail_repro/sttran_predcls_v8.csv")

# Bảng 1, hàng PREDCLS / STTran [7]: mR@10, mR@20, mR@50
PAPER = {"With": (30.5, 34.7, 34.8),
         "No":   (45.7, 63.4, 80.5),
         "Semi": (36.6, 51.8, 53.8)}

# Cùng hàng, dòng +IMPARTAIL — để thấy khoảng cách mà phương pháp tạo ra lớn hơn
# hẳn sai số tái lập, nên kết luận của bài báo không bị sai số này đe doạ.
PAPER_IMPARTAIL = {"With": (44.0, 52.7, 52.9),
                   "No":   (65.5, 82.0, 93.0),
                   "Semi": (47.7, 69.7, 73.4)}


def doc_ket_qua(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    v = [float(x) for x in rows[1][1:]]
    if len(v) != 36:
        sys.exit(f"cần 36 cột, tệp có {len(v)}")
    out = {}
    for i, ten in enumerate(["With", "No", "Semi"]):
        b = i * 12
        # mR@10, mR@20, mR@50 nằm ở vị trí 4,5,6 trong mỗi nhóm 12
        out[ten] = tuple(v[b + 4 + j] * 100 for j in range(3))
    return out


def main():
    if not os.path.isfile(CSV):
        sys.exit(f"không thấy {CSV}")
    mine = doc_ket_qua(CSV)

    print("PredCls · STTran · mean Recall (%) — tái lập so với Bảng 1\n")
    print(f"{'ràng buộc':<12}{'chỉ số':<9}{'bài báo':>9}{'ta':>9}{'lệch':>9}")
    print("-" * 48)
    lech = []
    for ten in ("With", "No", "Semi"):
        for j, k in enumerate((10, 20, 50)):
            b, m = PAPER[ten][j], mine[ten][j]
            lech.append(abs(m - b))
            print(f"{ten:<12}{'mR@' + str(k):<9}{b:>9.1f}{m:>9.2f}{m - b:>+9.2f}")

    print("-" * 48)
    print(f"lệch trung bình {sum(lech) / len(lech):.2f} điểm, lớn nhất {max(lech):.2f} điểm")

    # Sai số tái lập phải nhỏ so với hiệu ứng mà bài báo công bố, nếu không thì
    # nó không nói lên được điều gì về tính đúng đắn của kết luận.
    hieu_ung = [PAPER_IMPARTAIL[t][j] - PAPER[t][j] for t in PAPER for j in range(3)]
    tb_hieu_ung = sum(hieu_ung) / len(hieu_ung)
    print(f"mức cải thiện +ImparTail công bố: trung bình {tb_hieu_ung:.1f} điểm "
          f"— gấp {tb_hieu_ung / (sum(lech) / len(lech)):.0f} lần sai số tái lập")


if __name__ == "__main__":
    main()
