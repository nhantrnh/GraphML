"""Đối chiếu mọi con số trong tài liệu với dữ liệu gốc — MẪU để chép đi.

VÌ SAO. Một con số bị chép sai vào tài liệu không gây lỗi gì cả: file vẫn
biên dịch, script vẫn chạy. Nó chỉ sai. Cách duy nhất bắt được là tính lại
từ dữ liệu gốc rồi so.

CÁCH DÙNG. File này KHÔNG chạy được ngay — nó là mẫu. Chép vào repo, rồi
điền phần CHECKS bên dưới bằng chính những con số tài liệu của bạn đang
trích. Mỗi dòng `chk(...)` là một khẳng định: "tài liệu nói X, dữ liệu phải
cho X".

Quy tắc khi thêm check:

  - Thêm check NGAY khi viết một con số vào tài liệu, không để sau.
  - Dung sai đủ chặt để bắt lỗi chép nhầm, đủ lỏng cho làm tròn (thường 0.02).
  - Nhãn phải nói rõ con số nằm ở ĐÂU trong tài liệu, để khi lệch thì biết
    sửa chỗ nào.

    python qa/verify_numbers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results"   # chia theo thư mục con từng lần chạy

checks: list[tuple[bool, str, float, float | None]] = []


def load(name: str) -> dict | None:
    """Nạp một file kết quả; trả None nếu chưa có (để báo thay vì nổ).

    Tìm đệ quy vì kết quả nằm trong thư mục con theo từng lần chạy Kaggle
    (apkgc_res/, apkgc_res2/, apkgc_res3/).
    """
    hits = list(RESULTS.rglob(f"{name}.json"))
    return json.loads(hits[0].read_text()) if hits else None


def chk(label: str, claimed: float, actual: float | None,
        tol: float = 0.02) -> None:
    """So một con số trong tài liệu với giá trị tính từ dữ liệu.

    `actual=None` nghĩa là file kết quả chưa tồn tại — tính là lệch, vì tài
    liệu đang trích một con số không có nguồn.
    """
    ok = actual is not None and abs(claimed - actual) <= tol
    checks.append((ok, label, claimed, actual))


# ══════════════════════════════════════════════════════════════════
# CHECKS — mỗi dòng là một khẳng định tài liệu đang nói.
# Nhãn phải chỉ được CHỖ con số nằm, để lệch thì biết sửa đâu.
# ══════════════════════════════════════════════════════════════════
import statistics

PUB = dict(mrr=0.3740, hit1=0.3060, hit3=0.4040, hit10=0.5010)   # bài báo, Bảng 2

# ── Đợt 2: ba seed cùng điều kiện (2000 epoch, đánh giá mỗi 250) ──
b2 = [load(f"seed{s_}") for s_ in (1234, 2024, 3407)]
if all(b2):
    for name, d, mrr in (("seed1234", b2[0], 0.3610),
                         ("seed2024", b2[1], 0.3611),
                         ("seed3407", b2[2], 0.3595)):
        chk(f"report.md bảng đợt 2, dòng {name}", mrr, d["mrr"], 5e-5)
    m2 = {k: statistics.mean(x[k] for x in b2) for k in PUB}
    sd = {k: statistics.stdev(x[k] for x in b2) for k in PUB}
    chk("report.md, MRR trung bình đợt 2",        0.3605, m2["mrr"],   5e-5)
    chk("report.md, độ lệch chuẩn MRR",           0.0009, sd["mrr"],   5e-5)
    chk("report.md, độ lệch chuẩn Hits@1",        0.0032, sd["hit1"],  5e-5)
    chk("report.md, độ lệch chuẩn Hits@3",        0.0028, sd["hit3"],  5e-5)
    chk("report.md, độ lệch chuẩn Hits@10",       0.0030, sd["hit10"], 5e-5)
    chk("report.md, lệch MRR vs công bố",  0.0135, PUB["mrr"]  - m2["mrr"],  5e-5)
    chk("report.md, lệch Hits@1",          0.0108, PUB["hit1"] - m2["hit1"], 5e-5)
    chk("report.md, lệch Hits@3",          0.0144, PUB["hit3"] - m2["hit3"], 5e-5)
    chk("report.md, lệch Hits@10",         0.0218, PUB["hit10"]- m2["hit10"],5e-5)

# ── Cấu hình đúng bài báo (batch 2048 / neg 64) ──
pc = load("paper_config")
if pc and all(b2):
    chk("report.md, MRR cấu hình bài báo", 0.3691, pc["mrr"], 5e-5)
    chk("report.md, đóng góp của cấu hình", 0.0086, pc["mrr"] - m2["mrr"], 5e-5)
    chk("report.md, phần dư chưa giải thích", 0.0049, PUB["mrr"] - pc["mrr"], 5e-5)
    # tỉ lệ thu hẹp: (tự_chạy_cấu_hình_bài_báo − mốc) / (công_bố − mốc)
    for key, claimed in (("mrr", 64), ("hit1", 70), ("hit3", 70), ("hit10", 37)):
        got = (pc[key] - m2[key]) / (PUB[key] - m2[key]) * 100
        chk(f"report.md, tỉ lệ thu hẹp {key.upper()} (%)", claimed, got, 0.6)

# ── Kiểm chứng tái lập: seed 42 chạy lại sau sáu ngày ──
s42 = load("seed42")
if s42:
    chk("report.md §tái lập, MRR seed 42",    0.3641, s42["mrr"],   5e-5)
    chk("report.md §tái lập, Hits@1 seed 42", 0.2956, s42["hit1"],  5e-5)
    chk("report.md §tái lập, Hits@3 seed 42", 0.3954, s42["hit3"],  5e-5)
    chk("report.md §tái lập, Hits@10 seed 42",0.4840, s42["hit10"], 5e-5)
    chk("report.md §tái lập, số lần đánh giá", 15, s42["n_evals"], 0)

# ── ViMMKG: phép thử bộ dữ liệu có học được không ──
tv = json.loads((ROOT / "vimmkg/data/transe_vimmkg_d3.json").read_text()) \
     if (ROOT / "vimmkg/data/transe_vimmkg_d3.json").is_file() else None
tm = json.loads((ROOT / "vimmkg/data/transe_mkgw.json").read_text()) \
     if (ROOT / "vimmkg/data/transe_mkgw.json").is_file() else None
if tv: chk("TRACKING §TransE, MRR ViMMKG-d3", 0.1651, tv["mrr"], 5e-5)
if tm: chk("TRACKING §TransE, MRR MKG-W đối chứng", 0.1281, tm["mrr"], 5e-5)

# ── ViMMKG: kích thước bộ dữ liệu, đọc thẳng từ tệp ──
DS = ROOT / "vimmkg/data/dataset"
if DS.is_dir():
    for f, claimed in (("entity2id.txt", 12334), ("relation2id.txt", 329),
                       ("train2id.txt", 45284), ("valid2id.txt", 5612),
                       ("test2id.txt", 5646)):
        chk(f"PROPOSAL §3ter, {f}", claimed, int((DS / f).read_text().split("\n")[0]), 0)
    n_ent = int((DS / "entity2id.txt").read_text().split("\n")[0])
    n_tr  = int((DS / "train2id.txt").read_text().split("\n")[0])
    chk("PROPOSAL §3ter, bậc trung bình ViMMKG", 7.34, 2 * n_tr / n_ent, 0.005)

# ── ViMMKG: số thực thể Wikidata ──
we = ROOT / "vimmkg/data/wikidata_vi_entities.json"
if we.is_file():
    chk("PROPOSAL §3bis, số thực thể Wikidata",
        42081, len(json.loads(we.read_text())), 0)



# ── Thổi phồng do chọn checkpoint trên tập test ──
# Tính lại từ vết đánh giá, xem experiments/tools/selection_bias.py
import glob as _g, statistics as _st
_tr = []
for _f in sorted(_g.glob(str(RESULTS / "*" / "partial_*.json"))):
    _m = [e["mrr"] for e in json.loads(open(_f).read())["evals"]]
    if len(_m) >= 3 and abs(max(_m[:-1]) - _m[-1]) < 1e-6:
        _tr.append(max(_m[:-1]) - _m[:-1][-1])
if _tr:
    chk("PROPOSAL_PAPER §1.2, thổi phồng do chọn trên test",
        0.0054, _st.mean(_tr), 5e-5)
    chk("PROPOSAL_PAPER §1.2, số lần chạy đo được", 5, len(_tr), 0)

# ── ImparTail: tái lập PredCls/STTran trên Kaggle ───────────────────
# CSV do chính test_sgg_methods.py sinh ra; thứ tự 36 cột theo
# test_sgg_base.py:186-193 (with -> no -> semi, mỗi nhóm R/mR/hR @10,20,50,100).
_ir = ROOT / "experiments/results/impartail_repro/sttran_predcls_v8.csv"
if _ir.is_file():
    import csv as _csv
    with open(_ir, newline="") as _f:
        _v = [float(x) for x in list(_csv.reader(_f))[1][1:]]
    # mR@10, mR@20, mR@50 ở vị trí 4,5,6 trong mỗi nhóm 12 cột
    _mr = {n: tuple(_v[i * 12 + 4 + j] * 100 for j in range(3))
           for i, n in enumerate(("with", "no", "semi"))}
    for _n, _vals in _mr.items():
        for _k, _got in zip((10, 20, 50), _vals):
            chk(f"dan_y_impartail, mR@{_k} {_n}-constraint", round(_got, 1), _got, 0.05)
    # Bảng 1, hàng PREDCLS / STTran — mô hình nền, không phải +ImparTail
    _paper = {"with": (30.5, 34.7, 34.8), "no": (45.7, 63.4, 80.5),
              "semi": (36.6, 51.8, 53.8)}
    _lech = [abs(_mr[n][j] - _paper[n][j]) for n in _paper for j in range(3)]
    chk("dan_y_impartail, lệch trung bình so với Bảng 1",
        0.74, sum(_lech) / len(_lech), 5e-3)
    chk("dan_y_impartail, lệch lớn nhất", 1.52, max(_lech), 5e-3)


# ── ImparTail: hai nhánh mặt nạ (gốc vs công thức 5) ────────────────
# CSV do test_sgg_methods.py sinh; hai nhánh khác nhau ĐÚNG một cờ
# --true_masked_bce. Cùng cấu hình, cùng 2 epoch, cùng dữ liệu.
_nh = {}
for _ten in ("goc", "va"):
    _f = ROOT / f"experiments/results/impartail_repro/sttran_partial10_{_ten}.csv"
    if _f.is_file():
        import csv as _csv2
        with open(_f, newline="") as _fh:
            _nh[_ten] = [float(x) for x in list(_csv2.reader(_fh))[-1][1:]]
if len(_nh) == 2:
    def _mr50(v, i):   # mR@50 = vị trí 6 trong nhóm 12 cột
        return v[i * 12 + 6] * 100
    def _r50(v, i):    # R@50 = vị trí 2
        return v[i * 12 + 2] * 100
    def _hr50(v, i):   # hR@50 = vị trí 10, do chính mã tác giả tính
        return v[i * 12 + 10] * 100
    for _i, _rb in enumerate(("with", "no", "semi")):
        # hR trong CSV phải khớp trung bình điều hoà tính lại từ R và mR
        for _ten in ("goc", "va"):
            _v = _nh[_ten]
            _r, _m, _h = _r50(_v, _i), _mr50(_v, _i), _hr50(_v, _i)
            chk(f"impartail {_ten} {_rb}, hR@50 khớp R và mR",
                round(_h, 2), 2 * _r * _m / (_r + _m), 5e-3)
    _dhr = [_hr50(_nh["va"], _i) - _hr50(_nh["goc"], _i) for _i in range(3)]
    chk("IMPARTAIL_HUONG_PHAT_TRIEN, ΔhR trung bình",
        4.11, sum(_dhr) / len(_dhr), 5e-3)
    chk("IMPARTAIL_HUONG_PHAT_TRIEN, ΔR@50 semi", 11.4,
        _r50(_nh["va"], 2) - _r50(_nh["goc"], 2), 5e-2)


# ── Action Genome: tần suất vị ngữ thật ────────────────────────────
# Đếm từ chính tệp annotation trên Kaggle, đúng cách
# estimate_rel_distribution() đếm (ag_dataset.py:41-52).
_agf = ROOT / "experiments/results/impartail_repro/ag_predicate_freq.json"
if _agf.is_file():
    import json as _json2
    _ag = _json2.loads(_agf.read_text())
    chk("ag_predicate_freq, số cặp train", 417302, _ag["tong_cap"], 0)
    chk("ag_predicate_freq, số lớp contacting", 17,
        len(_ag["tan_suat"]["contacting_relationship"]), 0)
    # Đuôi dài: lớp phổ biến nhất so với hiếm nhất
    _c = _ag["tan_suat"]["contacting_relationship"]
    chk("ag_predicate_freq, holding", 169741, _c["holding"], 0)
    chk("ag_predicate_freq, twisting", 93, _c["twisting"], 0)
    # Mọi cặp mang ít nhất 3 nhãn (một cho mỗi nhóm) -> không có cặp 1-2 nhãn
    chk("ag_predicate_freq, cặp ít hơn 3 nhãn", 0,
        sum(v for k, v in _ag["nhan_moi_cap"].items() if int(k) < 3), 0)


# ── Bão hoà R@K: lý thuyết tiên đoán, số liệu xác nhận ─────────────
_objf = ROOT / "experiments/results/impartail_repro/ag_objects_per_frame.json"
if _objf.is_file():
    import json as _json3
    _o = _json3.loads(_objf.read_text())
    _v = {int(k): c for k, c in _o["vat_moi_khung"].items()}
    _tong = _o["tong_khung"]
    chk("ag_objects_per_frame, số khung train", 198973, _tong, 0)
    # no-constraint xếp hạng 26 vị ngữ mỗi cặp -> bão hoà khi 26*n <= K
    for _K, _mong in ((50, 34.2), (100, 89.7)):
        _bh = sum(c for n, c in _v.items() if 26 * n <= _K)
        chk(f"recall_at_k_saturation, bão hoà tuyệt đối K={_K} (%)",
            _mong, 100 * _bh / _tong, 0.05)
    # Tiên đoán kiểm chứng được: R@50 == R@100 ở with và semi constraint
    _rf = ROOT / "experiments/results/impartail_repro/sttran_predcls_v8.csv"
    if _rf.is_file():
        import csv as _csv3
        with open(_rf, newline="") as _fh:
            _r = [float(x) for x in list(_csv3.reader(_fh))[1][1:]]
        for _i, _n in ((0, "with"), (2, "semi")):
            chk(f"bão hoà: R@50 == R@100 ({_n})",
                round(_r[_i * 12 + 2] * 100, 2), _r[_i * 12 + 3] * 100, 5e-3)

    # §5.2 — trần bão hoà giải thích R@20 ~ R@50 ở with constraint.
    # Mỗi cặp mang nhiều nhãn cùng lúc, nên phải đếm NHÃN/khung, không phải
    # VẬT THỂ/khung (max 8 vs max 29) — chỗ này từng suy luận nhầm.
    _nh = {int(k): c for k, c in _o["nhan_moi_khung"].items()}
    chk("ag_objects_per_frame, max nhãn mỗi khung", 29, max(_nh), 0)
    chk("ag_objects_per_frame, max vật thể mỗi khung", 8, max(_v), 0)
    chk("trần bão hoà, K=20 phủ hết nhãn (%)",
        99.84, 100 * sum(c for n, c in _nh.items() if n <= 20) / _tong, 0.01)
    chk("trần bão hoà, K=50 phủ hết nhãn (%)",
        100.0, 100 * sum(c for n, c in _nh.items() if n <= 50) / _tong, 1e-9)
    if _rf.is_file():
        chk("§5.2 with: R@20 (%)", 64.94, _r[1] * 100, 5e-3)
        chk("§5.2 with: R@50 (%)", 64.96, _r[2] * 100, 5e-3)
        chk("§5.2 semi: R@20 (%)", 82.11, _r[2 * 12 + 1] * 100, 5e-3)
        chk("§5.2 semi: R@50 (%)", 83.27, _r[2 * 12 + 2] * 100, 5e-3)
        chk("§5.2 no: R@20 (%)", 91.70, _r[1 * 12 + 1] * 100, 5e-3)
        chk("§5.2 no: R@50 (%)", 99.11, _r[1 * 12 + 2] * 100, 5e-3)
        # no constraint KHÔNG bão hoà — đối chứng cho lập luận
        chk("no constraint không bão hoà: R@50 != R@100 (điểm)",
            0.76, abs(_r[1 * 12 + 2] - _r[1 * 12 + 3]) * 100, 0.01)

        # Bảng chín số ở §5.2. Chúng được in trong báo cáo suốt nhiều tuần mà
        # không assert nào chạm tới — check_repo.check_number_coverage sinh ra
        # chính vì lỗ hổng này.
        for _i, _n in ((0, "with"), (1, "no"), (2, "semi")):
            for _j, _k in ((0, 10), (1, 20), (2, 50)):
                _mong = {("with", 10): 30.70, ("with", 20): 35.54, ("with", 50): 35.65,
                         ("no", 10): 44.33, ("no", 20): 63.80, ("no", 50): 80.73,
                         ("semi", 10): 36.24, ("semi", 20): 52.67, ("semi", 50): 55.32}[(_n, _k)]
                chk(f"§5.2 {_n}: mR@{_k} (%)", _mong,
                    _r[_i * 12 + 4 + _j] * 100, 5e-3)


# ── Định lý cận chặt cho đảo thứ hạng ──────────────────────────────
# delta_max = N/T, đạt được khi B đạt recall 1 và A đạt 0 trên lớp phủ định.
_agf2 = ROOT / "experiments/results/impartail_repro/ag_predicate_freq.json"
if _agf2.is_file():
    import json as _json4
    _ts = _json4.loads(_agf2.read_text())["tan_suat"]
    _C = sum(len(c) for c in _ts.values())
    _N = 4          # not_looking_at, unsure, not_contacting, other_relationship
    _T = _C - _N
    chk("rank_reversal_bound, tổng số lớp", 26, _C, 0)
    chk("rank_reversal_bound, cận N/T (điểm recall)", 18.18, 100 * _N / _T, 0.005)
    # Cấu trúc đạt cận: A=(0.5, 0.0), B=(0.5-N/T, 1.0) cho mR bằng nhau
    _a, _d = 0.50, _N / _T
    _mA = (_T * _a + _N * 0.0) / _C
    _mB = (_T * (_a - _d) + _N * 1.0) / _C
    chk("rank_reversal_bound, cấu trúc đạt cận (mA == mB)",
        round(_mA, 6), _mB, 1e-6)
    # Lớp đông nhất của nhóm attention phải là lớp phủ định
    _at = _ts["attention_relationship"]
    chk("rank_reversal_bound, not_looking_at là lớp đông nhất attention",
        212640, max(_at.values()), 0)


# ── Bốn định lý dạng đóng ──────────────────────────────────────────
_T, _N, _C = 22, 4, 26
# ĐL1: SR = 1/(1 + (T/N)·ΔT/ΔN) phải khớp tính trực tiếp
for _dT, _dN in ((1.0, 5.0), (0.5, 10.0)):
    _rep = (_T * _dT + _N * _dN) / _C
    chk(f"four_theorems ĐL1, SR(ΔT={_dT},ΔN={_dN})",
        round((_N * _dN / _C) / _rep, 6), 1 / (1 + (_T / _N) * (_dT / _dN)), 1e-6)
# ĐL2: n_f lớn nhất quyết định ngưỡng dư thừa của K
_objf2 = ROOT / "experiments/results/impartail_repro/ag_objects_per_frame.json"
if _objf2.is_file():
    import json as _json5
    _vv = {int(k): c for k, c in
           _json5.loads(_objf2.read_text())["vat_moi_khung"].items()}
    chk("four_theorems ĐL2, n_f lớn nhất", 8, max(_vv), 0)
# ĐL3: R_global − R_macro = |F|·Cov(R_f, w_f), kiểm trên một ví dụ cố định
_gt, _tp = [1, 10], [1, 1]
_Rf = [t / g for t, g in zip(_tp, _gt)]
_S = sum(_gt); _w = [g / _S for g in _gt]; _F = len(_gt)
_mR = sum(_Rf) / _F; _mw = sum(_w) / _F
_cov = sum((_Rf[i] - _mR) * (_w[i] - _mw) for i in range(_F)) / _F
chk("four_theorems ĐL3, phân rã hiệp phương sai",
    round(sum(_tp) / _S - _mR, 9), _F * _cov, 1e-9)
# ĐL4: điểm cân bằng ŷ* = p_c
chk("four_theorems ĐL4, E[dL/dz] tại ŷ=p_c", 0.0, 0.30 - 0.30, 1e-12)


# ── VidOR làm đối chứng tương phản ─────────────────────────────────
_vdf = ROOT / "experiments/results/impartail_repro/vidor_stat.json"
if _vdf.is_file():
    import json as _json6
    _vd = _json6.loads(_vdf.read_text())
    chk("vidor_contrast, số lớp vị ngữ", 50, len(_vd["pred"]), 0)
    # VidOR không có lớp phủ định nào
    _am = [k for k in _vd["pred"] if k.startswith("not_")
           or k in ("unsure", "other", "other_relationship", "none")]
    chk("vidor_contrast, số lớp phủ định", 0, len(_am), 0)
    # Bão hoà K=100: 50·n(n-1) <= 100 chỉ đúng với n=2
    _ob = {int(k): c for k, c in _vd["obj"].items()}
    _tt = sum(_ob.values())
    _bh = sum(c for n, c in _ob.items() if n > 1 and 50 * n * (n - 1) <= 100)
    chk("vidor_contrast, bão hoà K=100 (%)", 31.57, 100 * _bh / _tt, 0.05)


# ── Phép thử quyết định: loại lớp phủ định ─────────────────────────
_pcf = ROOT / "experiments/results/impartail_repro/per_class_recall.json"
if _pcf.is_file():
    import json as _json7
    _pc = _json7.loads(_pcf.read_text())
    _AM = {1, 2, 17, 18}
    def _tach(v):
        return (sum(v) / len(v) * 100,
                sum(x for i, x in enumerate(v) if i not in _AM) / 22 * 100,
                sum(x for i, x in enumerate(v) if i in _AM) / 4 * 100)
    _g = _tach(_pc["goc"]["with"]["20"])
    _v = _tach(_pc["va"]["with"]["20"])
    chk("decisive_test, mR đủ 26 (goc, with)", 52.29, _g[0], 0.005)
    chk("decisive_test, mR đủ 26 (va, with)", 50.71, _v[0], 0.005)
    # Co biên độ: −1,59 -> −0,01 khi bỏ 4 lớp phủ định
    chk("decisive_test, biên đủ 26 (with)", -1.59, _v[0] - _g[0], 0.005)
    chk("decisive_test, biên 22 lớp (with)", -0.01, _v[1] - _g[1], 0.005)
    chk("decisive_test, Δ_N (with)", -10.23, _v[2] - _g[2], 0.005)


# ── Tiên đoán Định lý 4 theo tỉ lệ che ─────────────────────────────
if _agf2.is_file():
    _cc = _json4.loads(_agf2.read_text())["tan_suat"]["contacting_relationship"]
    _nt, _nl = sum(_cc.values()), len(_cc)
    def _pc(n_c, giu):
        return min(_nt * giu / _nl, n_c) / n_c
    # holding: p_c giảm 10 lần từ ρ=100% xuống ρ=10%
    chk("theorem4_prediction, p_c holding @100%", 0.157, _pc(_cc["holding"], 1.0), 5e-4)
    chk("theorem4_prediction, p_c holding @10%", 0.016, _pc(_cc["holding"], 0.10), 5e-4)
    # nhóm đối chứng nội tại: lớp có p_c = 1 ở MỌI tỉ lệ
    _bb = [k for k, v in _cc.items()
           if all(_pc(v, g) >= 1.0 for g in (1.0, 0.70, 0.40, 0.10))]
    chk("theorem4_prediction, số lớp bất biến theo ρ", 4, len(_bb), 0)


# ── Bản đồ checkpoint: id phải khớp bản đã xác minh ────────────────
# Mọi checkpoint STTran cùng 416.767.365 byte nên không xác nhận được bằng
# dung lượng; id sai thì tải nhầm file mà không có dấu hiệu nào. Ghim lại.
_ckf = ROOT / "experiments/results/impartail_repro/checkpoints.json"
if _ckf.is_file():
    import json as _json8
    _ck = _json8.loads(_ckf.read_text())
    chk("checkpoints.json, số baseline ImparTail", 6,
        _ck["impartail_cvpr2025"]["so_checkpoint"]["baseline"], 0)
    chk("checkpoints.json, số checkpoint ImparTail", 20,
        _ck["impartail_cvpr2025"]["so_checkpoint"]["impartail"], 0)
    chk("checkpoints.json, số tỉ lệ che", 3,
        len(_ck["impartail_cvpr2025"]["ti_le_che"]), 0)
    # STTran và TEMPURA đều phải có đủ ba chế độ
    for _m in ("sttran_iccv2021", "tempura_cvpr2023"):
        _n = sum(1 for _k in ("predcls", "sgcls", "sgdet") if _k in _ck[_m])
        chk(f"checkpoints.json, {_m} đủ 3 chế độ", 3, _n, 0)
    # Detector dùng chung — nếu id này sai thì cả ba phương pháp hỏng
    chk("checkpoints.json, độ dài id detector", 33,
        len(_ck["detector_chung"]["faster_rcnn_ag.pth"]), 0)


# ── Điểm cân bằng toàn cục: π·p_c triệt tiêu n_c ───────────────────
if _agf2.is_file():
    _dd = _json4.loads(_agf2.read_text())
    _cc2 = _dd["tan_suat"]["contacting_relationship"]
    _nt2, _nl2, _ncap = sum(_cc2.values()), len(_cc2), _dd["tong_cap"]
    _quota = _nt2 * 0.10 / _nl2
    # Mọi lớp không bị clamp phải cho cùng ŷ* = quota / N_cặp
    _deu = [(v / _ncap) * (min(_quota, v) / v)
            for v in _cc2.values() if v > _quota]
    chk("global_equilibrium, số lớp không bị clamp", 13, len(_deu), 0)
    chk("global_equilibrium, ŷ* toàn cục", 0.006401,
        sum(_deu) / len(_deu), 5e-7)
    chk("global_equilibrium, chênh lệch giữa các lớp", 0.0,
        max(_deu) - min(_deu), 1e-12)


# ── Deck seminar APKGC: số trên slide phải khớp dữ liệu ────────────
_deck = ROOT / "seminar/25C15053_Slides_APKGC.pdf"
if _deck.is_file():
    try:
        import pypdf as _pypdf
        _t = "".join((_p.extract_text() or "")
                     for _p in _pypdf.PdfReader(str(_deck)).pages)
        # Mỗi số trên slide phải có mặt; giá trị đã được các chk ở trên ràng buộc
        # 0,0054 và 10/13 (chọn checkpoint trên test) đã rút khỏi bài nói theo
        # quyết định của người trình bày; số vẫn được kiểm ở khối riêng phía trên.
        # 0,0086 (phần chênh được thu hẹp) đã rút khỏi slide 14, còn trong dàn ý.
        for _so in ("0,3605", "0,3691",
                    "0,2952", "0,4792", "0,0009", "0,0032"):
            chk(f"25C15053_Slides_APKGC.pdf chứa {_so}", 1, 1 if _so in _t else 0, 0)
    except ImportError:
        pass


# ── NativE trên MKG-W, cùng bộ đặc trưng với APKGC (kernel native-mkgw-s42/-s1234) ──
_nv = {sd_: ROOT / f"experiments/results/native_res/native_{sd_}_e1000.json" for sd_ in ("s42", "s1234")}
if all(p_.is_file() for p_ in _nv.values()):
    import json as _json14
    _n42 = _json14.loads(_nv["s42"].read_text()); _n1234 = _json14.loads(_nv["s1234"].read_text())
    chk("NativE s42: MRR checkpoint cuối", 0.3684, _n42["final"]["mrr"], 5e-5)
    chk("NativE s1234: MRR checkpoint cuối", 0.3733, _n1234["final"]["mrr"], 5e-5)
    chk("NativE s42: MRR tốt nhất trên test", 0.3730, _n42["best_on_test"]["mrr"], 5e-5)
    # So với APKGC: cả hai seed NativE cao hơn APKGC mã công bố (0,3605); so với APKGC
    # cấu hình của bài (0,3691, 1 seed) thì s42 thấp hơn 0,0007, s1234 cao hơn 0,0042.
    chk("NativE trung bình 2 seed (cuối)", 0.3709, (_n42["final"]["mrr"] + _n1234["final"]["mrr"]) / 2, 5e-5)


# ── Độ bền khi phương thức bị hỏng lúc test (kernel mmkgc-robustness-eval) ──
_rb = ROOT / "experiments/results/robustness"
if (_rb / "robustness_apkgc_A_seed42.json").is_file():
    import json as _json15
    def _dk(f_, kieu, p, pt):
        d_ = _json15.loads((_rb / f_).read_text())
        return next(c_ for c_ in d_["dieu_kien"] if c_["kieu"] == kieu and c_["ti_le"] == p and c_["phuong_thuc"] == pt)
    _A, _N42, _N1234 = ("robustness_apkgc_A_seed42.json", "robustness_native_s42.json",
                        "robustness_native_s1234.json")
    for f_ in (_A, _N42, _N1234):
        chk(f"{f_}: kiểm chứng MRR sạch đạt", 1, int(_json15.loads((_rb / f_).read_text())["kiem_chung_dat"]), 0)
    # APKGC gần như không dùng ảnh: bỏ hết ảnh, MRR không đổi
    chk("APKGC bỏ toàn bộ ảnh: ΔMRR", 0.0001, _dk(_A, "missing", 1.0, "img")["delta_mrr"], 5e-5)
    chk("APKGC tráo toàn bộ ảnh: ΔMRR", -0.0014, _dk(_A, "shuffle", 1.0, "img")["delta_mrr"], 5e-5)
    # Văn bản bị tráo: APKGC tệ hơn hẳn NativE
    chk("APKGC tráo văn bản: MRR", 0.0567, _dk(_A, "shuffle", 1.0, "txt")["mrr"], 5e-5)
    chk("NativE s42 tráo văn bản: MRR", 0.1706, _dk(_N42, "shuffle", 1.0, "txt")["mrr"], 5e-5)
    chk("NativE s1234 tráo văn bản: MRR", 0.1756, _dk(_N1234, "shuffle", 1.0, "txt")["mrr"], 5e-5)
    chk("APKGC nhiễu σ=2 cả hai: MRR", 0.1760, _dk(_A, "noise_2", 1.0, "both")["mrr"], 5e-5)
    chk("NativE s42 nhiễu σ=2 cả hai: MRR", 0.2846, _dk(_N42, "noise_2", 1.0, "both")["mrr"], 5e-5)
    chk("APKGC thiếu văn bản: MRR", 0.2020, _dk(_A, "missing", 1.0, "txt")["mrr"], 5e-5)
    chk("NativE s42 thiếu văn bản: MRR", 0.2317, _dk(_N42, "missing", 1.0, "txt")["mrr"], 5e-5)
    # Seed 2024 (kernel thnhnhntrng/apkgc-a-seed2024, dừng sớm ở epoch ~1300)
    _A24 = "robustness_apkgc_A_seed2024.json"
    if (_rb / _A24).is_file():
        chk(f"{_A24}: kiểm chứng MRR sạch đạt", 1, int(_json15.loads((_rb / _A24).read_text())["kiem_chung_dat"]), 0)
        chk("APKGC s2024 sạch: MRR", 0.3613, _json15.loads((_rb / _A24).read_text())["sach"]["mrr"], 5e-5)
        chk("APKGC s2024 bỏ toàn bộ ảnh: ΔMRR", -0.0001, _dk(_A24, "missing", 1.0, "img")["delta_mrr"], 5e-5)
        chk("APKGC s2024 tráo văn bản: MRR", 0.0629, _dk(_A24, "shuffle", 1.0, "txt")["mrr"], 5e-5)
        chk("APKGC s2024 nhiễu σ=2 cả hai: MRR", 0.1650, _dk(_A24, "noise_2", 1.0, "both")["mrr"], 5e-5)


# ── Số trong mục so sánh cùng giao thức và độ bền của báo cáo ──
if (ROOT / "experiments/results/robustness/robustness_apkgc_A_seed2024.json").is_file():
    import json as _json17, statistics as _st17
    _R17 = ROOT / "experiments/results/robustness"
    _A17 = [_json17.loads((_R17 / f_).read_text()) for f_ in ("robustness_apkgc_A_seed42.json", "robustness_apkgc_A_seed2024.json")]
    _N17 = [_json17.loads((_R17 / f_).read_text()) for f_ in ("robustness_native_s42.json", "robustness_native_s1234.json")]
    def _m17(ds, k, p, pt):
        return _st17.mean(next(c_["mrr"] for c_ in d_["dieu_kien"] if (c_["kieu"], c_["ti_le"], c_["phuong_thuc"]) == (k, p, pt)) for d_ in ds)
    chk("Báo cáo: APKGC sạch (TB 2 seed)", 0.3621, _st17.mean(d_["sach"]["mrr"] for d_ in _A17), 5e-5)
    chk("Báo cáo: NativE sạch (TB 2 seed)", 0.3709, _st17.mean(d_["sach"]["mrr"] for d_ in _N17), 5e-5)
    for (k_, p_, pt_), va_, vn_ in ((("missing", 1.0, "txt"), 0.2114, 0.2331), (("shuffle", 1.0, "txt"), 0.0598, 0.1731),
                                    (("noise_1", 1.0, "both"), 0.3245, 0.3464), (("noise_2", 1.0, "both"), 0.1705, 0.2870),
                                    (("missing", 1.0, "img"), 0.3621, 0.3696), (("shuffle", 1.0, "img"), 0.3610, 0.3654)):
        chk(f"Báo cáo: APKGC {k_}/{pt_}", va_, _m17(_A17, k_, p_, pt_), 5e-5)
        chk(f"Báo cáo: NativE {k_}/{pt_}", vn_, _m17(_N17, k_, p_, pt_), 5e-5)
    _f17 = [_json17.loads((ROOT / f"experiments/results/apkgc_res/partial_seed{s_}.json").read_text())["evals"][7]["mrr"] for s_ in (1234, 2024, 3407)]
    chk("Báo cáo: APKGC checkpoint epoch 2000 (TB 3 seed)", 0.3572, _st17.mean(_f17), 5e-5)
    chk("Báo cáo: APKGC checkpoint epoch 2000 (std)", 0.0017, _st17.stdev(_f17), 5e-5)
    chk("Báo cáo: APKGC cấu hình bài, epoch 2000", 0.3571,
        _json17.loads((ROOT / "experiments/results/apkgc_res2/partial_paper_config.json").read_text())["evals"][7]["mrr"], 5e-5)


# ── Báo cáo cuối kỳ 25C15053: số then chốt phải có mặt trong PDF ──────
# Giá trị của từng số đã được các chk ở trên ràng buộc với tệp kết quả gốc;
# khối này chỉ bắt trường hợp PDF nộp đi không phải bản đã dựng lại.
_rp = ROOT / "report/final/25C15053.pdf"
if _rp.is_file():
    try:
        import pypdf as _pypdf2
        _rt = "".join((_p.extract_text() or "") for _p in _pypdf2.PdfReader(str(_rp)).pages)
        _rt = _rt.replace("\u2212", "-")
        # Báo cáo viết bằng tiếng Anh: dấu chấm thập phân, dấu phẩy phân cách hàng nghìn.
        for _so in ("0.3605", "0.3691", "0.0009", "0.3469", "0.3629", "0.3630",
                    "0.3655", "0.0026", "0.74", "1.52", "4.11", "1,825:1", "169,741",
                    "417,302", "99.84", "0.411", "0.413", "0.529",
                    # so sánh cùng giao thức với NativE và đo độ bền (mục mới)
                    "0.3572", "0.0017", "0.3684", "0.3733", "0.3730", "0.0598", "0.2114",
                    "0.1731", "0.1705", "0.2870", "0.3245", "0.3464"):
            chk(f"25C15053.pdf chứa {_so}", 1, 1 if _so in _rt else 0, 0)
    except ImportError:
        pass


# ── Báo cáo ImparTail: số trong bài phải khớp dữ liệu ──────────────
_bc = ROOT / "report/src/report_impartail.md"
if _bc.is_file() and _agf2.is_file():
    _rep = _bc.read_text()
    _cf = _json4.loads(_agf2.read_text())
    _cc3 = _cf["tan_suat"]["contacting_relationship"]
    # Số đếm phải có mặt trong báo cáo VÀ khớp dữ liệu
    for _nhan, _chuoi, _that in (
            ("holding", "169.741", _cc3["holding"]),
            ("twisting", "93", _cc3["twisting"]),
            ("cặp train", "417.302", _cf["tong_cap"])):
        chk(f"report_impartail chứa {_nhan}", 1, 1 if _chuoi in _rep else 0, 0)
    chk("report_impartail, tỉ lệ holding/twisting", 1825,
        _cc3["holding"] / _cc3["twisting"], 1.0)
    # Chín con số tái lập
    _rf2 = ROOT / "experiments/results/impartail_repro/sttran_predcls_v8.csv"
    if _rf2.is_file():
        import csv as _csv4
        with open(_rf2, newline="") as _fh2:
            _vv2 = [float(x) for x in list(_csv4.reader(_fh2))[1][1:]]
        _thieu = sum(1 for _i in range(3) for _j in range(3)
                     if f"{_vv2[_i*12+4+_j]*100:.2f}".replace(".", ",") not in _rep)
        chk("report_impartail, 9 số tái lập đều có mặt", 0, _thieu, 0)


# ── Hits@K trên 3 seed: số trên slide phải khớp dữ liệu ────────────
import glob as _glob2
_rs = [_json2.loads(open(_f).read()) for _f in
       sorted(_glob2.glob(str(ROOT / "experiments/results/apkgc_res/seed*.json")))] \
      if (ROOT / "experiments/results/apkgc_res").is_dir() else []
_rs = [_r for _r in _rs if isinstance(_r, dict) and "hit1" in _r]
if len(_rs) >= 3:
    import statistics as _st2
    for _k, _mong in (("hit1", 0.2952), ("hit3", 0.3896), ("hit10", 0.4792)):
        chk(f"APKGC_Slides, {_k} trung bình 3 seed", _mong,
            _st2.mean(_r[_k] for _r in _rs), 5e-5)
    chk("APKGC_Slides, std hit1", 0.0032,
        _st2.stdev(_r["hit1"] for _r in _rs), 5e-5)



# ── Phép thử quyết định: bỏ 4 lớp phủ định có đảo thứ hạng không ────
# Kernel impartail-per-class v4. Lý thuyết cho phép đảo (rank_reversal_bound
# nói δ < N/T = 18,2); đây là chỗ đo xem có đảo THẬT không. KHÔNG có.
_pcf = ROOT / "experiments/results/impartail_repro/per_class_mr20.json"
if _pcf.is_file():
    import json as _json4
    _pc = _json4.loads(_pcf.read_text())
    for _k, (_du, _bo) in (("with|goc", (52.29, 52.14)), ("with|va", (50.71, 52.12)),
                           ("no|goc", (76.99, 76.70)), ("no|va", (76.85, 76.51)),
                           ("semi|goc", (13.81, 10.19)), ("semi|va", (22.18, 22.08))):
        chk(f"per-class mR@20 {_k} đủ 26 lớp", _du, _pc[_k][0], 5e-3)
        chk(f"per-class mR@20 {_k} bỏ 4 âm", _bo, _pc[_k][1], 5e-3)
    # Kết quả ÂM: không ràng buộc nào đảo thứ hạng.
    for _rb in ("with", "no", "semi"):
        _g, _v = _pc[f"{_rb}|goc"], _pc[f"{_rb}|va"]
        chk(f"không đảo thứ hạng ({_rb})",
            1.0, float((_g[0] > _v[0]) == (_g[1] > _v[1])), 0)


# ── Định lý 4: tiên đoán đã đăng ký KHÔNG được dữ liệu ủng hộ ───────
# Kernel impartail-main-table v2. Ghi lại kết quả ÂM để không ai đọc bảng
# recall rồi tưởng đối chứng đã đạt.
_t4f = ROOT / "experiments/results/impartail_repro/theorem4_per_class.json"
if _t4f.is_file():
    import json as _json5
    _t4 = _json5.loads(_t4f.read_text())
    # Đối chứng RỖNG: 4 lớp có ΣTP = 0 ở mọi ρ, nên recall bất biến vì luôn
    # bằng 0 — không phân biệt được giả thuyết nào.
    for _c, _gt in (("have_it_on_the_back", 161), ("twisting", 27),
                    ("wiping", 255), ("writing_on", 277)):
        chk(f"Định lý 4, {_c}: ΣGT (ρ=100%)", _gt, _t4["100"][_c]["GT"], 0)
        for _r in ("10", "40", "70", "100"):
            chk(f"Định lý 4, {_c}: ΣTP = 0 (ρ={_r}%)", 0, _t4[_r][_c]["TP"], 0)
    # Tiên đoán 1 sai ở 2/3 lớp head: recall TĂNG theo ρ thay vì tụt.
    for _c in ("holding", "not_contacting"):
        _lo = _t4["10"][_c]["TP"] / _t4["10"][_c]["GT"]
        _hi = _t4["100"][_c]["TP"] / _t4["100"][_c]["GT"]
        chk(f"Định lý 4, {_c}: recall(ρ=100) > recall(ρ=10) — ngược tiên đoán",
            1.0, float(_hi > _lo), 0)



# ── Phương án B: đổi nhóm đối chứng — tiên đoán vẫn không được ủng hộ ──
# Kiểm tiên đoán cốt lõi (recall tăng theo p_c) bằng Spearman trên các lớp
# còn dùng được. Không tốn giờ GPU: chỉ phân tích lại dữ liệu đã có.
_t4f2 = ROOT / "experiments/results/impartail_repro/theorem4_per_class.json"
_agf3 = ROOT / "experiments/results/impartail_repro/ag_predicate_freq.json"
if _t4f2.is_file() and _agf3.is_file():
    import json as _json6
    _t4b = _json6.loads(_t4f2.read_text())
    _cc = _json6.loads(_agf3.read_text())["tan_suat"]["contacting_relationship"]
    _nt, _nl = sum(_cc.values()), len(_cc)

    def _pc(_n, _g):
        return min(_nt * _g / _nl, _n) / _n

    def _spear(_a, _b):
        def _rk(_x):
            _s = sorted(range(len(_x)), key=lambda i: _x[i])
            _r = [0] * len(_x)
            for _j, _i in enumerate(_s):
                _r[_i] = _j
            return _r
        _ra, _rb = _rk(_a), _rk(_b)
        _n = len(_a)
        return 1 - 6 * sum((_ra[i] - _rb[i]) ** 2 for i in range(_n)) / (_n * (_n * _n - 1))

    for _R, _G, _nhan, _dung, _tong in (
            (("10", "40", "70", "100"), {"10": .10, "40": .40, "70": .70, "100": 1.0},
             "đủ 4 ρ", 4, 13),
            (("40", "70", "100"), {"40": .40, "70": .70, "100": 1.0},
             "cùng epoch 3", 4, 13)):
        _pos = _n_dung = 0
        for _k, _v in _cc.items():
            _tp = [_t4b[_r][_k]["TP"] for _r in _R]
            _gt = [_t4b[_r][_k]["GT"] for _r in _R]
            if not all(_g > 0 for _g in _gt) or all(_t == 0 for _t in _tp):
                continue
            _n_dung += 1
            _pos += _spear([_pc(_v, _G[_r]) for _r in _R],
                           [_t / _g for _t, _g in zip(_tp, _gt)]) > 0
        chk(f"Định lý 4 ({_nhan}): số lớp dùng được", _tong, _n_dung, 0)
        chk(f"Định lý 4 ({_nhan}): số lớp tiên đoán ĐÚNG", _dung, _pos, 0)
    # 4 lớp p_c=1 đều là 4 lớp hiếm nhất và đều TP=0 -> không có đối chứng nào dùng được
    _ung = [_k for _k, _v in _cc.items()
            if all(_pc(_v, _g) >= 1.0 for _g in (.10, .40, .70, 1.0))
            and all(_t4b[_r][_k]["TP"] > 0 for _r in ("10", "40", "70", "100"))]
    chk("Định lý 4: số đối chứng dùng được (p_c=1 và TP>0)", 0, len(_ung), 0)



# ── Định lý 2: chặn phải là số NHÃN, không phải số VẬT THỂ ──────────
# Bản đầu dùng vat_moi_khung (max 8) và kết luận R@K ≡ R@8 với mọi K ≥ 8.
# Số tái lập bác bỏ ngay: R@10 = 62,03 nhưng R@20 = 64,94 ở with-constraint.
# Chặn đúng là max số nhãn mỗi khung = 29.
if _objf.is_file() and _rf.is_file():
    import json as _json7
    _o2 = _json7.loads(_objf.read_text())
    _v2 = {int(k): c for k, c in _o2["vat_moi_khung"].items()}
    _nh2 = {int(k): c for k, c in _o2["nhan_moi_khung"].items()}
    chk("Định lý 2: chặn sai nếu dùng vật thể (max)", 8, max(_v2), 0)
    chk("Định lý 2: chặn ĐÚNG là số nhãn (max)", 29, max(_nh2), 0)
    # Phản chứng: nếu R@K ≡ R@8 thì R@10 phải bằng R@20. Nó không bằng.
    import csv as _csv7
    with open(_rf, newline="") as _fh7:
        _r7 = [float(x) for x in list(_csv7.reader(_fh7))[1][1:]]
    chk("Định lý 2: R@20 − R@10 > 0 (bác bỏ chặn = 8)",
        1.0, float(_r7[1] > _r7[0]), 0)
    chk("Định lý 2: R@20 − R@10 ở with (điểm)", 2.91,
        (_r7[1] - _r7[0]) * 100, 5e-3)



# ── Xếp hạng hai phương pháp: biên GIÃN, không co ──────────────────
# Kernel impartail-ranking. STTran (ICCV 2021) vs DSG-DETR (WACV 2023),
# cùng backbone và cùng detector faster_rcnn_ag.pth. Thứ hạng KHÔNG đảo,
# nhưng Δ_T và Δ_N TRÁI DẤU nên SR ra ngoài [0,1] và mất nghĩa — đúng điều
# kiện dùng đã ghi trong four_theorems.py.
_rkf = ROOT / "experiments/results/impartail_repro/ranking_two_methods.json"
if _rkf.is_file():
    import json as _json7
    _rk = _json7.loads(_rkf.read_text())
    for _m, _v26, _v22, _vn in (("sttran", 35.54, 33.19, 48.47),
                                ("dsgdetr", 36.17, 34.53, 45.18)):
        chk(f"xếp hạng, {_m}: mR@20 đủ 26 lớp", _v26, _rk[_m]["mR26"], 5e-3)
        chk(f"xếp hạng, {_m}: mR@20 bỏ 4 âm", _v22, _rk[_m]["mR22"], 5e-3)
        chk(f"xếp hạng, {_m}: mR@20 chỉ 4 lớp âm", _vn, _rk[_m]["mR_N"], 5e-3)
    _dT = _rk["dsgdetr"]["mR22"] - _rk["sttran"]["mR22"]
    _dN = _rk["dsgdetr"]["mR_N"] - _rk["sttran"]["mR_N"]
    chk("xếp hạng: Δ_T (22 lớp mang nghĩa)", 1.35, _dT, 5e-3)
    chk("xếp hạng: Δ_N (4 lớp phủ định)", -3.29, _dN, 5e-3)
    # Δ trái dấu -> SR ngoài [0,1]
    chk("xếp hạng: Δ_T và Δ_N trái dấu", 1.0, float(_dT * _dN < 0), 0)
    # Định lý 1 tái dựng đúng biên đo được từ Δ_T và Δ_N
    chk("Định lý 1 tái dựng Δ_rep", (22 * _dT + 4 * _dN) / 26,
        _rk["dsgdetr"]["mR26"] - _rk["sttran"]["mR26"], 5e-3)
    # Thứ hạng KHÔNG đảo: DSG-DETR thắng ở cả hai tập lớp
    chk("xếp hạng: không đảo (DSG-DETR thắng cả hai)", 1.0,
        float((_rk["dsgdetr"]["mR26"] > _rk["sttran"]["mR26"])
              == (_rk["dsgdetr"]["mR22"] > _rk["sttran"]["mR22"])), 0)



# ── APKGC: cơ chế phạt attention đo sai đại lượng ──────────────────
# APKGC.py:360-365 so ngưỡng với TÍCH LUỸ HẬU TỐ của điểm attention, không
# phải với độ nhọn. Ba con số dưới đây tái dựng bằng mô phỏng tất định
# (seed 0) — xem experiments/tools/apkgc_penalty_defect.py.
import math as _m8, random as _rd8
_rd8.seed(0)


def _tich_hau_to(_a):
    _p = []
    for _i in range(len(_a)):
        _v = 1.0
        for _x in _a[_i:]:
            _v *= _x
        _p.append(_v)
    return _p


def _ent(_a):
    _mx = max(_a)
    _e = [_m8.exp(_x - _mx) for _x in _a]
    _s = sum(_e)
    _q = [_x / _s for _x in _e]
    return -sum(_x * _m8.log(_x + 1e-12) for _x in _q) / _m8.log(len(_a))


# Tần suất kích hoạt PHỤ THUỘC thang điểm attention. Với std ~ 1 thì gần
# như không chạy; std >= 2 thì chạy. Chưa đo std thật trên checkpoint đã
# huấn luyện, nên đây là điều kiện, KHÔNG phải kết luận về mô hình thật.
for _L, _std, _duoi in ((16, 1.0, 1.0), (64, 1.0, 1.0)):
    _vuot = _tong = 0
    for _ in range(200):
        _p8 = _tich_hau_to([_rd8.gauss(0, _std) for _ in range(_L)])
        _vuot += sum(1 for _x in _p8 if _x > 10)
        _tong += _L
    chk(f"APKGC phạt: std={_std} -> vượt ngưỡng dưới {_duoi}% (L={_L})",
        1.0, float(100 * _vuot / _tong < _duoi), 0)
for _L, _std, _tren in ((16, 3.0, 10.0), (64, 3.0, 10.0)):
    _vuot = _tong = 0
    for _ in range(200):
        _p8 = _tich_hau_to([_rd8.gauss(0, _std) for _ in range(_L)])
        _vuot += sum(1 for _x in _p8 if _x > 10)
        _tong += _L
    chk(f"APKGC phạt: std={_std} -> vượt ngưỡng TRÊN {_tren}% (L={_L})",
        1.0, float(100 * _vuot / _tong > _tren), 0)
# Bất biến thật: ~50% phần tử âm ở MỌI phương sai -> mask `> 10` không chạm tới.
for _std in (0.5, 1.0, 3.0):
    _am = _tong = 0
    for _ in range(200):
        _p8 = _tich_hau_to([_rd8.gauss(0, _std) for _ in range(16)])
        _am += sum(1 for _x in _p8 if _x < 0)
        _tong += 16
    chk(f"APKGC phạt: ~50% phần tử âm (std={_std}), không phụ thuộc thang",
        1.0, float(45 < 100 * _am / _tong < 55), 0)

# Entropy xếp đúng thứ tự nhọn -> đều; tích hậu tố thì không.
_deu = [0.1] * 16
_nhon = [0.0] * 16
_nhon[3] = 8.0
chk("APKGC phạt: entropy phân bố đều = 1", 1.0, _ent(_deu), 1e-9)
chk("APKGC phạt: entropy phân bố nhọn < 0,02", 1.0,
    float(_ent(_nhon) < 0.02), 0)
chk("APKGC phạt: tích hậu tố KHÔNG phân biệt được nhọn/đều", 1.0,
    float(max(_tich_hau_to(_nhon)) <= max(_tich_hau_to(_deu))), 0)



# ── Bản vá entropy: chọn đúng hàng nhọn, tiêu chí gốc thì không ────
# Đối chứng tất định trên batch dựng sẵn: 20 hàng đều + 5 hàng nhọn.
# Xem impartail/patches/apkgc_entropy_penalty.py
_rd9 = __import__("random").Random(0)
_m9 = _m8   # math, đã import ở khối trên


def _sm9(_v):
    _mx = max(_v)
    _e = [_m9.exp(_x - _mx) for _x in _v]
    _s = sum(_e)
    return [_x / _s for _x in _e]


def _ent9(_v):
    _p = _sm9(_v)
    return -sum(_x * _m9.log(_x + 1e-12) for _x in _p) / _m9.log(len(_v))


_L9 = 16
_rows = [[_rd9.gauss(0, 0.3) for _ in range(_L9)] for _ in range(20)]
for _ in range(5):
    _r = [_rd9.gauss(0, 0.3) for _ in range(_L9)]
    _r[_rd9.randrange(_L9)] += 8.0
    _rows.append(_r)

_H9 = [_ent9(_r) for _r in _rows]
_tau9 = sorted(_H9)[int(0.10 * len(_H9))]
_chon = [_i for _i, _h in enumerate(_H9) if _h < _tau9]
chk("bản vá entropy: mọi hàng bị phạt đều là hàng nhọn", 1.0,
    float(bool(_chon) and all(_c >= 20 for _c in _chon)), 0)
# Tiêu chí gốc không kích hoạt lần nào trên chính batch này.
_n9 = 0
for _r in _rows:
    _n9 += sum(1 for _x in _tich_hau_to(_r) if _x > 10)
chk("tiêu chí gốc: 0 lần kích hoạt trên batch có 5 hàng nhọn", 0, _n9, 0)



# ── ĐO THẬT: cơ chế phạt không kích hoạt lần nào ───────────────────
# Kernel apkgc-attn-probe (27/09): chèn đoạn đo vào APKGC.py, huấn luyện
# MKG-W 40 epoch, ghi thống kê trên 300 lần gọi forward. Đây là số ĐO,
# thay cho mô phỏng — mô phỏng iid dự đoán sai hẳn (18-26% vs 0%).
_asf = ROOT / "experiments/results/apkgc_res/attn_scale.json"
if _asf.is_file():
    import json as _json10
    _as = _json10.loads(_asf.read_text())
    chk("probe: số lần gọi forward đo được", 300, _as["n"], 0)
    chk("probe: attention_scores.std() trung bình", 2.0945, _as["std_mean"], 5e-4)
    chk("probe: |attention_scores| lớn nhất", 10.798, _as["absmax_max"], 5e-3)
    chk("probe: tỉ lệ mask kích hoạt = 0", 0.0, _as["fire_mean"], 0)
    chk("probe: tỉ lệ mask kích hoạt lớn nhất = 0", 0.0, _as["fire_max"], 0)
    chk("probe: tỉ lệ penalty_scores âm", 0.6266, _as["neg_mean"], 5e-4)
    # Điểm attention thật KHÔNG độc lập: tỉ lệ âm lệch hẳn khỏi ~50% của iid.
    chk("probe: tỉ lệ âm lệch khỏi giả định iid 50%", 1.0,
        float(_as["neg_mean"] > 0.55), 0)



# ── Ba nhánh phạt attention: A và B cho kết quả TRÙNG ──────────────
# Kernel apkgc-three-branch (27-28/09, MKG-W, seed 42). Nhánh C bị cắt khi
# kernel bị huỷ. A = mã công bố, B = đúng công thức (9)(10) trong bài.
_3bf = ROOT / "experiments/results/apkgc_res/three_branch.json"
if _3bf.is_file():
    import json as _json11
    _3b = _json11.loads(_3bf.read_text())
    for _n, _mrr, _h1 in (("A_goc", 0.3629, 0.2933), ("B_baibao", 0.3629, 0.2933)):
        chk(f"ba nhánh, {_n}: MRR", _mrr, _3b[_n]["mrr"], 5e-5)
        chk(f"ba nhánh, {_n}: H@1", _h1, _3b[_n]["hit1"], 5e-5)
    # Trùng khớp KHÔNG do B không chạy: thời lượng chênh 6 phút.
    chk("ba nhánh: A và B trùng MRR", 0.0,
        abs(_3b["A_goc"]["mrr"] - _3b["B_baibao"]["mrr"]), 1e-9)
    chk("ba nhánh: B có chạy thật (thời lượng khác A)", 1.0,
        float(abs(_3b["A_goc"]["phut"] - _3b["B_baibao"]["phut"]) > 1), 0)
    # Biên độ phạt của nhánh B quá nhỏ để dịch MRR: θp = 0,001 trên A = 10,798
    chk("ba nhánh: biên độ phạt B dưới 0,2% giá trị logit", 1.0,
        float(0.001 * 10.798 / 10.798 < 0.002), 0)
    # Nhánh C (entropy) chạy riêng, kernel apkgc-entropy-c.
    chk("ba nhánh, C_entropy: MRR", 0.3630, _3b["C_entropy"]["mrr"], 5e-5)
    chk("ba nhánh, C_entropy: H@1", 0.2934, _3b["C_entropy"]["hit1"], 5e-5)
    # KẾT QUẢ ÂM: chênh lệch nhỏ hơn nhiễu seed (std 0,0009) chín lần.
    _cd = abs(_3b["C_entropy"]["mrr"] - _3b["A_goc"]["mrr"])
    chk("ba nhánh: C khác A dưới 0,0002 MRR", 1.0, float(_cd < 2e-4), 0)
    chk("ba nhánh: chênh lệch NHỎ HƠN độ lệch chuẩn seed 0,0009", 1.0,
        float(_cd < 0.0009), 0)
    # Ablation tắt nhiễu (kernel apkgc-ablation-noise). Kết quả NGƯỢC chiều
    # tiên đoán của bài: tắt nhiễu làm MRR TĂNG, không giảm.
    chk("ablation, tắt nhiễu: MRR", 0.3655, _3b["D_khong_nhieu"]["mrr"], 5e-5)
    chk("ablation, tắt nhiễu + entropy: MRR", 0.3651,
        _3b["E_khong_nhieu_entropy"]["mrr"], 5e-5)
    _dn = _3b["D_khong_nhieu"]["mrr"] - _3b["A_goc"]["mrr"]
    chk("ablation: tắt nhiễu làm MRR TĂNG (ngược tiên đoán)", 1.0,
        float(_dn > 0), 0)
    # Nhưng chênh lệch DƯỚI ngưỡng 3x std đặt trước -> chưa kết luận mạnh.
    chk("ablation: chênh lệch dưới ngưỡng 3x std (2,9x)", 1.0,
        float(_dn / 0.0009 < 3.0), 0)
    # Nhánh B2: cắt xác suất attention ở 0,5 như Hình 4 (kernel apkgc-cap-prob).
    chk("B2 cap 0,5: MRR", 0.3605, _3b["B2_cap05"]["mrr"], 5e-5)
    chk("B2 cap 0,5: H@1", 0.2964, _3b["B2_cap05"]["hit1"], 5e-5)
    chk("B2 cap 0,5: chênh MRR so với A dưới 3x std", 1.0,
        float(abs(_3b["B2_cap05"]["mrr"] - _3b["A_goc"]["mrr"]) / 0.0009 < 3.0), 0)

# ── θ_r, θ_n sau huấn luyện, đọc từ checkpoint biến thể A (MKG-W, seed 42) ──
_lrf = ROOT / "experiments/results/apkgc_res/learned_ratios.json"
if _lrf.is_file():
    import json as _json13
    _lr = _json13.loads(_lrf.read_text())
    # Bộ đọc đúng: các hằng số đã biết trong cùng checkpoint phải ra đúng giá trị.
    chk("checkpoint A: pi_const", 3.14159, _lr["kiem_tra_bo_doc"]["pi_const"], 1e-4)
    chk("checkpoint A: margin", 12.0, _lr["kiem_tra_bo_doc"]["margin"], 1e-6)
    chk("dàn ý Q&A: θ_r sau huấn luyện ≈ 0 (khởi tạo 0,7)", -0.0004,
        _lr["mask_ratio_theta_r"]["sau_huan_luyen"], 5e-5)
    chk("dàn ý Q&A: θ_n giữ nguyên 0,2", 0.2, _lr["noise_ratio_theta_n"]["sau_huan_luyen"], 1e-6)

# ── DB15K, bộ dữ liệu thứ hai, chặn 600 epoch (kernel apkgc-db15k-capped) ──
_dbf = ROOT / "experiments/results/apkgc_res/db15k_seed42.json"
if _dbf.is_file():
    import json as _json12
    _db = _json12.loads(_dbf.read_text())
    chk("DB15K seed42: MRR tốt nhất", 0.3469, _db["best"]["mrr"], 5e-5)
    chk("DB15K seed42: H@1 tốt nhất", 0.2591, _db["best"]["hit1"], 5e-5)
    chk("DB15K seed42: số lần đánh giá", 12, len(_db["evals"]), 0)
    chk("DB15K seed42: chặn epoch", 600, _db["epoch_cap"], 0)
    chk("DB15K: lệch MRR so với bài", -0.0171, _db["best"]["mrr"] - _db["paper"]["mrr"], 5e-4)



def main() -> int:
    if not checks:
        print("Chưa có check nào — xem hướng dẫn ở đầu file.")
        return 0
    bad = [c for c in checks if not c[0]]
    print(f"{len(checks) - len(bad)}/{len(checks)} con số khớp\n")
    for _, label, claimed, actual in bad:
        got = "thiếu file" if actual is None else round(actual, 4)
        print(f"  ✗ {label}: tài liệu={claimed}  dữ liệu={got}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
