#!/usr/bin/env python3
"""Chuyển chọn checkpoint từ TẬP TEST sang TẬP VALID cho các repo MMKGC.

Đây là bước chuẩn bị cho thí nghiệm quyết định: chạy lại các phương pháp dưới quy
trình đúng rồi xem thứ hạng công bố có đổi không.

Mỗi bản vá được ghi rõ: sửa tệp nào, dòng nào, vì sao. Sau khi vá, script tự đọc
lại tệp để xác nhận thay đổi đã ăn — thay chuỗi không khớp thì im lặng không làm
gì, và đã có lần bị như vậy.

    python make_valid_selection.py --repo <đường dẫn> --model apkgc|mygo|c2rs|momok
    python make_valid_selection.py --repo <đường dẫn> --model apkgc --check
"""
import argparse, os, re, sys

# ── APKGC ────────────────────────────────────────────────────────────
# Trainer.py dựng loader chọn checkpoint bằng TestDataLoader, mà lớp này gọi
# importTestFiles() trong C++ và hàm đó đọc test2id.txt. Không có cờ nào đổi split.
#
# May là Setting.h có sẵn hàm ngoại setTestPath(). Gọi nó TRƯỚC importTestFiles()
# là loader đọc valid2id.txt. Không phải tráo tệp, không phải sửa C++.
APKGC_LOADER = (
    '\t\tself.in_path = in_path\n',
    '\t\tself.in_path = in_path\n'
    '\t\t# split="valid" trỏ loader sang valid2id.txt để chọn checkpoint không\n'
    '\t\t# đụng tập test. Mặc định giữ nguyên hành vi cũ.\n'
    '\t\tself.split = split\n')
APKGC_SIG = (
    "def __init__(self, in_path = \"./\", sampling_mode = 'link', type_constrain = True):",
    "def __init__(self, in_path = \"./\", sampling_mode = 'link', type_constrain = True, split = 'test'):")
APKGC_READ = (
    "\t\tself.lib.randReset()\n\t\tself.lib.importTestFiles()",
    "\t\tself.lib.randReset()\n"
    "\t\tif self.split == 'valid':\n"
    "\t\t\t_p = os.path.join(self.in_path, 'valid2id.txt')\n"
    "\t\t\tself.lib.setTestPath(ctypes.create_string_buffer(_p.encode(), len(_p) * 2))\n"
    "\t\tself.lib.importTestFiles()")
APKGC_TRAINER = (
    'self.test_dataloader = TestDataLoader("./benchmarks/" + args.dataset + \'/\', "link")',
    'self.test_dataloader = TestDataLoader("./benchmarks/" + args.dataset + \'/\', "link",\n'
    '                                              split=getattr(args, "select_on", "test"))')

# ── MyGO / C2RS ──────────────────────────────────────────────────────
# Cùng một khuôn mã: đánh giá trên KG.test rồi giữ best_mrr. Đổi vòng lặp chọn
# sang KG.valid; phần lưu checkpoint giữ nguyên nên vẫn đánh giá lại trên test sau.
KGTEST = ("for triplet in tqdm(KG.test)", "for triplet in tqdm(KG.valid)")

# ── MoMoK ────────────────────────────────────────────────────────────
MOMOK = ("corpus.get_validation_pred(model, 'test')",
         "corpus.get_validation_pred(model, 'valid')")


def patch(path, pairs, label):
    if not os.path.isfile(path):
        print(f"  ✗ không thấy {path}"); return False
    s = open(path, encoding="utf-8").read()
    done, skipped = 0, 0
    for old, new in pairs:
        if new in s:
            skipped += 1; continue
        if old not in s:
            print(f"  ✗ {label}: không khớp {old[:52]!r}"); return False
        s = s.replace(old, new, 1); done += 1
    open(path, "w", encoding="utf-8").write(s)
    # đọc lại để chắc chắn đã ăn
    after = open(path, encoding="utf-8").read()
    ok = all(new in after for _, new in pairs)
    print(f"  {'✓' if ok else '✗'} {label}: vá {done}, đã có sẵn {skipped}")
    return ok


def apkgc(root):
    tdl = os.path.join(root, "mmkgc/data/TestDataLoader.py")
    s = open(tdl, encoding="utf-8").read()
    if "import os" not in s.split("class")[0]:
        s = "import os\n" + s
        open(tdl, "w", encoding="utf-8").write(s)
    ok = patch(tdl, [APKGC_SIG, APKGC_LOADER, APKGC_READ], "TestDataLoader")
    ok &= patch(os.path.join(root, "mmkgc/config/Trainer.py"), [APKGC_TRAINER], "Trainer")
    print("\n  Dùng: thêm -select_on=valid vào dòng lệnh (mặc định vẫn là test).")
    print("  Sau khi huấn luyện, đánh giá checkpoint bằng run_KGC_test.py trên tập test thật.")
    return ok


def mygo_like(root, fname):
    return patch(os.path.join(root, fname), [KGTEST], fname)


def momok(root):
    ok = patch(os.path.join(root, "train.py"), [MOMOK], "train.py")
    print("\n  CẢNH BÁO: MoMoK KHÔNG lưu checkpoint — nó chỉ giữ giá trị lớn nhất của")
    print("  từng chỉ số qua các epoch. Đổi sang 'valid' làm việc chọn đúng hơn, nhưng")
    print("  vẫn không có mô hình nào để đánh giá lại trên test. Muốn so sánh công bằng")
    print("  thì phải THÊM phần lưu checkpoint — đây là sửa mã, không phải đổi cấu hình.")
    return ok


def check(root):
    hits = []
    for dp, _, fn in os.walk(root):
        if "/.git/" in dp or "__pycache__" in dp: continue
        for f in fn:
            if not f.endswith(".py"): continue
            p = os.path.join(dp, f)
            try: L = open(p, encoding="utf-8", errors="ignore").read().split("\n")
            except OSError: continue
            for i, l in enumerate(L):
                if re.search(r"tqdm\(KG\.test\)|get_validation_pred\([^)]*['\"]test['\"]|TestDataLoader\(", l):
                    hits.append(f"    {os.path.relpath(p, root)}:{i+1}  {l.strip()[:64]}")
    print("  còn trỏ vào test:" if hits else "  không còn chỗ nào trỏ vào test")
    print("\n".join(hits[:8]))


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--repo", required=True)
    a.add_argument("--model", choices=["apkgc", "mygo", "c2rs", "momok"])
    a.add_argument("--check", action="store_true", help="chỉ liệt kê chỗ còn dùng test")
    a = a.parse_args()
    if a.check:
        check(a.repo); sys.exit(0)
    if not a.model:
        sys.exit("cần --model hoặc --check")
    fn = dict(apkgc=apkgc,
              mygo=lambda r: mygo_like(r, "train_mygo_fgc.py"),
              c2rs=lambda r: mygo_like(r, "train_c2rs.py"),
              momok=momok)[a.model]
    sys.exit(0 if fn(a.repo) else 1)
