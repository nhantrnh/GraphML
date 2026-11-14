#!/usr/bin/env python3
"""Khảo sát: repo MMKGC nào chọn checkpoint báo cáo trên TẬP TEST?

Chọn mô hình trên chính tập dùng để báo cáo làm phồng kết quả. Mức thổi phồng đo
được trên APKGC là +0,0054 MRR, bằng 6× nhiễu giữa seed — xem selection_bias.py.

Script này clone một loạt repo MMKGC rồi trích **bằng chứng** cho ba câu hỏi:
  1. Có đánh giá giữa lúc huấn luyện không?
  2. Có theo dõi checkpoint "tốt nhất" không?
  3. Nếu có thì so trên split nào?

Cố ý KHÔNG tự kết luận. Nhận dạng theo mẫu chuỗi sẽ sai — ví dụ `get_best_threshlod`
là ngưỡng phân loại bộ ba, không phải chọn checkpoint; còn MoMoK đặt tên biến là
`val_metrics` nhưng truyền `'test'` vào hàm. Script in bằng chứng ra để người đọc
tự phân loại.

    python survey_selection.py --clone     # tải repo về ~/.cache/mmkgc_survey
    python survey_selection.py             # quét những repo đã có
"""
import argparse, glob, os, re, subprocess, sys

CACHE = os.path.expanduser("~/.cache/mmkgc_survey")

REPOS = [
    "zjunlp/MKGformer", "quqxui/MMRNS", "OreOZhao/MoSE4MKGC", "OreOZhao/CMR",
    "1994L/MKGC", "ADMIS-TONGJI/C2RS", "zjukg/NATIVE", "zjukg/SNAG", "zjukg/MANS",
    "zjukg/AdaMF-MAT", "zjukg/MyGO", "ngl567/DHNS", "HubuKG/APKGC",
]

BEST  = re.compile(r"best[_a-z]*|max_mrr|highest", re.I)
ASSIGN= re.compile(r"[<>]|=\s*[^=]")
#: `get_best_threshlod` là ngưỡng phân loại bộ ba của OpenKE, không liên quan
#: tới chọn checkpoint. Bốn repo họ OpenKE đều có nó và đều KHÔNG chọn checkpoint.
NOISE = re.compile(r"get_best_thresh?lod|best_threshold", re.I)


def clone():
    os.makedirs(CACHE, exist_ok=True)
    for r in REPOS:
        dst = os.path.join(CACHE, r.split("/")[-1])
        if os.path.isdir(dst):
            print(f"  đã có   {r}"); continue
        ok = subprocess.run(["git", "clone", "-q", "--depth", "1",
                             f"https://github.com/{r}.git", dst],
                            capture_output=True).returncode == 0
        print(f"  {'✓' if ok else '✗'} {r}")


def evidence(root):
    out = []
    for f in glob.glob(f"{root}/**/*.py", recursive=True):
        if "/.git/" in f or "__pycache__" in f:
            continue
        try:
            L = open(f, encoding="utf-8", errors="ignore").read().split("\n")
        except OSError:
            continue
        for i, l in enumerate(L):
            if not (BEST.search(l) and ASSIGN.search(l)) or NOISE.search(l):
                continue
            ctx = "\n".join(L[max(0, i - 12):i + 3]).lower()
            out.append(dict(
                file=os.path.relpath(f, root), line=i + 1, code=l.strip()[:76],
                test=bool(re.search(r"\btest\b|['\"]test['\"]|\.test\b", ctx)),
                valid=bool(re.search(r"\bvalid\w*\b|\bdev\b", ctx))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone", action="store_true", help="tải các repo về trước")
    a = ap.parse_args()
    if a.clone:
        clone(); print()
    dirs = sorted(d for d in glob.glob(f"{CACHE}/*") if os.path.isdir(d))
    if not dirs:
        print(f"chưa có repo nào trong {CACHE} — chạy lại với --clone"); return 1
    for d in dirs:
        ev = evidence(d)
        name = os.path.basename(d)
        if not ev:
            print(f"\n{name:<14} không theo dõi checkpoint 'tốt nhất' "
                  f"(chỉ chọn trên test mới là vấn đề, nên đây là KHÔNG mắc lỗi)")
            continue
        t = sum(e["test"] and not e["valid"] for e in ev)
        v = sum(e["valid"] for e in ev)
        print(f"\n{name:<14} {len(ev)} dòng | ngữ cảnh có test: {t} | có valid: {v}")
        for e in ev[:4]:
            tag = "TEST" if e["test"] and not e["valid"] else ("valid" if e["valid"] else "?")
            print(f"    [{tag:>5}] {e['file']}:{e['line']}  {e['code']}")
    print("\nĐọc bằng chứng rồi tự phân loại — đừng tin nhãn tự động.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
