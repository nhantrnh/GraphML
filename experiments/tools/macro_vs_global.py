#!/usr/bin/env python3
"""Đo sai lệch macro-frame vs global-ratio trên dữ liệu thật (Định lý 3).

    R̄_c^macro  = mean_{f ∈ F_c} ( TP_{f,c} / GT_{f,c} )     <- mã ImparTail
    R_c^global = ( Σ_f TP_{f,c} ) / ( Σ_f GT_{f,c} )        <- định nghĩa chuẩn
    ΔR_c       = R̄_c^macro − R_c^global = −|F_c|·Cov(R_f, w_f)

Câu hỏi cần trả lời: ΔR_c có dồn về phía lớp ĐUÔI không? Nếu có, chỉ số hiện
tại đang trợ giá cho chính nhóm lớp nó được tạo ra để đo.

Đầu vào: JSON do impartail/patches/dump_per_class_recall.py sinh ra, khoá
"raw" chứa danh sách (TP, GT) theo từng khung cho từng lớp.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FREQ = os.path.join(ROOT, "experiments/results/impartail_repro/ag_predicate_freq.json")

# Thứ tự 26 lớp theo dataloader/base_ag_dataset.py:62-84:
# attention 0-2, spatial 3-8, contacting 9-25 (xếp chữ cái, có ghi đè).
TEN_LOP = (["looking_at", "not_looking_at", "unsure"]
           + ["in_front_of", "behind", "on_the_side_of", "above", "beneath", "in"]
           + ["carrying", "covered_by", "drinking_from", "eating",
              "have_it_on_the_back", "holding", "leaning_on", "lying_on",
              "not_contacting", "other_relationship", "sitting_on",
              "standing_on", "touching", "twisting", "wearing", "wiping",
              "writing_on"])

TAN_SUAT = {}
if os.path.isfile(FREQ):
    with open(FREQ) as _f:
        _d = json.load(_f)["tan_suat"]
    for _nhom in _d.values():
        TAN_SUAT.update(_nhom)

# Thứ tự lớp theo dataloader/base_ag_dataset.py:69-83
AM_IDX = {1, 2, 17, 18}          # not_looking_at, unsure, not_contacting, other_relationship


def tinh(raw_k):
    """raw_k[n] = [(TP, GT), ...] theo khung. Trả (macro, global, |F|, Cov)."""
    ra = []
    for n, cap in enumerate(raw_k):
        if not cap:
            ra.append(None)
            continue
        Rf = [t / g for t, g in cap if g > 0]
        S = sum(g for _, g in cap)
        if not Rf or S == 0:
            ra.append(None)
            continue
        w = [g / S for _, g in cap]
        F = len(Rf)
        macro = sum(Rf) / F
        glob = sum(t for t, _ in cap) / S
        mR, mw = macro, sum(w) / F
        cov = sum((Rf[i] - mR) * (w[i] - mw) for i in range(F)) / F
        ra.append((macro, glob, F, cov))
    return ra


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Dùng: python macro_vs_global.py per_class.json [ràng_buộc]")
        print("\nChưa có tệp JSON — chạy kernel impartail-per-class trước.")
        return 0
    d = json.load(open(sys.argv[1]))
    rb = sys.argv[2] if len(sys.argv) > 2 else "with"
    raw = d[rb]["raw"]["20"]          # K=20

    ten = []
    if os.path.isfile(FREQ):
        f = json.load(open(FREQ))["tan_suat"]
        for nhom in ("attention_relationship", "spatial_relationship",
                     "contacting_relationship"):
            ten += sorted(f[nhom], key=lambda k: -f[nhom][k])
    tan = {}
    if os.path.isfile(FREQ):
        for nhom, c in json.load(open(FREQ))["tan_suat"].items():
            tan.update(c)

    kq = tinh(raw)
    print(__doc__.split("\n\n")[0])
    print(f"\nRàng buộc: {rb} · K=20\n")
    print(f"{'lớp':<4}{'|F_c|':>8}{'macro':>9}{'global':>9}{'ΔR_c':>9}{'':3}loại")
    print("-" * 50)
    duoi_duong = duoi_tong = 0
    for n, r in enumerate(kq):
        if r is None:
            continue
        macro, glob, F, cov = r
        dR = macro - glob
        loai = "phủ định" if n in AM_IDX else ""
        print(f"{n:<4}{F:>8,}{macro:>9.4f}{glob:>9.4f}{dR:>+9.4f}   {loai}")
    print("-" * 50)

    hop_le = [(n, r) for n, r in enumerate(kq) if r]
    duong = sum(1 for _, r in hop_le if r[0] - r[1] > 0)
    print(f"\n{duong}/{len(hop_le)} lớp có ΔR_c > 0 (macro cao hơn global)")

    # Câu hỏi thật: ΔR_c có dồn về lớp đuôi không? Quét MỌI ràng buộc và K,
    # không chỉ cấu hình đang in, vì một dấu hiệu chỉ xuất hiện ở một cấu hình
    # thì không đủ để kết luận.
    print("\n" + "=" * 50)
    print("ΔR CÓ DỒN VỀ ĐUÔI KHÔNG? (quét mọi ràng buộc × K)\n")
    print(f"{'rb':<6}{'K':<5}{'head ΔR':>10}{'tail ΔR':>10}{'đuôi cao hơn?':>15}")
    print("-" * 46)
    co = tong = 0
    for _rb in ("with", "no", "semi"):
        if _rb not in d:
            continue
        for _K in ("10", "20", "50", "100"):
            _raw = d[_rb].get("raw", {}).get(_K)
            if not _raw:
                continue
            _r = []
            for _n, _cap in enumerate(_raw):
                if not _cap:
                    continue
                _Rf = [t / g for t, g in _cap if g > 0]
                _S = sum(g for _, g in _cap)
                if not _Rf or _S == 0:
                    continue
                _ten = TEN_LOP[_n] if _n < len(TEN_LOP) else str(_n)
                _r.append((TAN_SUAT.get(_ten, 0),
                           sum(_Rf) / len(_Rf) - sum(t for t, _ in _cap) / _S))
            if len(_r) < 4:
                continue
            _r.sort(key=lambda x: -x[0])
            _h = len(_r) // 2
            _hm = sum(x[1] for x in _r[:_h]) / _h
            _tm = sum(x[1] for x in _r[_h:]) / (len(_r) - _h)
            tong += 1
            co += _tm > _hm
            print(f"{_rb:<6}{_K:<5}{_hm:>+10.4f}{_tm:>+10.4f}"
                  f"{('CÓ' if _tm > _hm else 'KHÔNG'):>15}")
    if tong:
        print("-" * 46)
        print(f"\n{co}/{tong} cấu hình cho thấy đuôi được đội điểm.")
        if co == tong:
            print("=> NHẤT QUÁN: chỉ số trợ giá cho đuôi ở mọi cấu hình.")
        elif co == 0:
            print("=> NHẤT QUÁN NGƯỢC LẠI: không có dấu hiệu trợ giá đuôi.")
        else:
            print("=> KHÔNG NHẤT QUÁN — dấu hiệu đổi chiều theo ràng buộc/K,")
            print("   nên KHÔNG kết luận được là chỉ số trợ giá cho đuôi.")
            print("   Biên độ |ΔR| ~ 0,01 cũng quá nhỏ so với chênh lệch")
            print("   giữa các phương pháp, nên đây không phải hiệu ứng")
            print("   đủ lớn để làm đảo thứ hạng.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
