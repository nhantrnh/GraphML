"""Cổng kiểm tra repo — bắt tài liệu lỗi thời TRƯỚC khi người đọc bắt được.

VÌ SAO TỒN TẠI. Dự án này đã gặp tám loại lỗi tài liệu, và **không loại nào gây
lỗi biên dịch hay lỗi chạy**. PDF vẫn đẹp, script vẫn chạy, chỉ là nội dung sai:

  1. README nói "chưa chạy" trong khi `results/` đã có file
  2. Số trong tài liệu lệch số trong `results/`
  3. Cùng một đại lượng, hai tài liệu ghi hai con số
  4. Số đúng nhưng mô tả sai ĐẠI LƯỢNG ("81% headroom" vs "81% điểm Oracle")
  5. Cấu hình ghi sai so với `config` trong chính file kết quả
  6. README liệt kê file đã xoá, hoặc thiếu file mới thêm
  7. `requirements.txt` thiếu package mà mã thực sự import
  8. Script ghép hai nguồn dữ liệu với cấu hình khác nhau

Loại 2 do `verify_report_numbers.py` lo. Loại 8 do `tests/` lo. Script này lo
phần còn lại — những thứ so được **tự động** giữa tài liệu và hiện trạng repo.

    python qa/check_repo.py          # báo cáo
    python qa/check_repo.py --strict # thoát mã 1 nếu có lỗi (dùng cho CI)
"""
from __future__ import annotations

import argparse
import ast
import json
import fnmatch
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ══════════════════════════════════════════════════════════════════
# CẤU HÌNH — chỉ phần này cần sửa khi dùng cho repo khác.
# ══════════════════════════════════════════════════════════════════

#: Thư mục chứa mã Python cần soi import và đối chiếu với README.
# impartail/patches nằm ngoài danh sách này suốt một thời gian, nên bốn bản vá
# không được README nhắc mà checker vẫn báo sạch — đúng loại lỗ hổng mà file
# này tồn tại để bịt.
CODE_DIRS = ("vimmkg/src", "qa", "experiments/tools", "impartail/patches")

#: Thư mục chứa kết quả thí nghiệm (JSON có khoá "config").
RESULTS_DIR = "experiments/results"

#: Các tài liệu mô tả repo. Dùng để bắt file mồ côi và tham chiếu chết.
DOCS = ("report/src/report.md", "report/src/TRANG_THAI_THUC_NGHIEM.md",
        "report/src/seed_results.md", "vimmkg/README.md", "README.md")

#: Tài liệu nào chịu trách nhiệm liệt kê file mã.
INVENTORY_DOC = "README.md"

#: (mức, thông điệp). "loi" chặn; "canh_bao" chỉ nhắc.
Issue = tuple[str, str]
issues: list[Issue] = []


def loi(msg: str) -> None:
    issues.append(("loi", msg))


def canh_bao(msg: str) -> None:
    issues.append(("canh_bao", msg))


def read(rel: str) -> str:
    """Đọc tài liệu, chuẩn hoá cách LaTeX viết số.

    Tài liệu tiếng Việt biên dịch bằng XeLaTeX viết dấu thập phân là `0{,}0009`
    để LaTeX không chèn khoảng trắng sau dấu phẩy. Không chuẩn hoá thì mọi phép
    tìm số đều trượt và checker báo động giả — đã gặp thật với `0,0009`.
    """
    p = ROOT / rel
    if not p.is_file():
        # Trả "" ở đây thì mọi re.search bên dưới trượt và checker đẻ ra cảnh
        # báo giả "tài liệu còn số cũ" — đã bị đúng vậy sau lần refactor cấu
        # trúc repo: SHARED_NUMBERS còn ghi "report.md" trong khi tệp đã dời
        # sang report/src/report.md. Đường dẫn sai phải là lỗi, không im lặng.
        loi(f"{rel}: không tồn tại — SHARED_NUMBERS trỏ sai đường dẫn?")
        return ""
    t = p.read_text()
    return t.replace("{,}", ",").replace("{.}", ".")


# ── 6. README có khớp cây thư mục thật không ────────────────────────
def check_file_inventory() -> None:
    """Mọi script/module phải được README nhắc tới, và ngược lại.

    Bắt cả hai chiều: file mới thêm mà quên ghi, và file đã xoá mà README
    vẫn liệt kê — cái thứ hai nguy hiểm hơn vì người đọc sẽ đi tìm.
    """
    readme = read(INVENTORY_DOC)
    for d in CODE_DIRS:
        for f in sorted((ROOT / d).glob("*.py")):
            if f.name.startswith("_"):
                continue
            if f.name not in readme:
                canh_bao(f"{INVENTORY_DOC} không nhắc {d}/{f.name}")

    # chieu nguoc: README nhac file khong ton tai
    for m in re.finditer(r"[\w/-]+\.(?:py|ipynb|tex|sh)", readme):  # tên kernel có dấu gạch nối
        rel = m.group(0)
        if rel.startswith(("http", "www")):
            continue
        hits = list(ROOT.rglob(Path(rel).name))
        if not hits:
            loi(f"{INVENTORY_DOC} nhắc `{rel}` nhưng file KHÔNG tồn tại")


# ── 1. Tài liệu nói "chưa chạy" mà results/ đã có ───────────────────
#: Câu chữ trong tài liệu -> file kết quả chứng minh nó đã chạy.
CLAIMS_VS_RESULTS = {
    # (tài liệu, script, file kết quả): tài liệu nói "chưa chạy" mà kết quả đã có
    "report/src/TRANG_THAI_THUC_NGHIEM.md": [
        ("Chạy đủ 8000 epoch", "seed42.json"),
    ],
}

def check_stale_status() -> None:
    for doc, pairs in CLAIMS_VS_RESULTS.items():
        text = read(doc)
        for script, result in pairs:
            if not list((ROOT / RESULTS_DIR).rglob(result)):
                continue
            for line in text.splitlines():
                if script in line and any(w in line for w in STALE_WORDS):
                    loi(f"{doc}: dòng nói `{script}` chưa chạy, nhưng "
                        f"{RESULTS_DIR}/{result} đã tồn tại")


# ── 5. Cấu hình trong tài liệu vs config trong file kết quả ─────────
#: (file kết quả, khoá config, giá trị tài liệu ĐANG ghi ở đâu đó)
CONFIG_CLAIMS = [
    # (file kết quả, khoá, giá trị tài liệu khẳng định)
    ("seed42.json",       "epoch",      8000),
    ("seed42.json",       "save_steps",  100),
    ("seed1234.json",     "epoch",      2000),
    ("seed1234.json",     "save_steps",  250),
    ("paper_config.json", "epoch",      2000),
]

def check_result_configs() -> None:
    for fname, key, expected in CONFIG_CLAIMS:
        cands = list((ROOT / RESULTS_DIR).rglob(fname))
        p = cands[0] if cands else ROOT / RESULTS_DIR / fname
        if not p.is_file():
            canh_bao(f"{RESULTS_DIR}/{fname} không có — bỏ qua kiểm config")
            continue
        raw = json.loads(p.read_text())
        # repo này ghi siêu tham số PHẲNG ở gốc; repo khác lồng trong "config"
        cfg = raw.get("config", raw)
        got = cfg.get(key)
        if got != expected:
            loi(f"{RESULTS_DIR}/{fname}: config[{key}] = {got!r}, "
                f"tài liệu giả định {expected!r}")


# ── 3. Cùng đại lượng, nhiều tài liệu — có khớp không ───────────────
#: nhãn -> (regex, các file phải đồng ý)
SHARED_NUMBERS = {
    # nhãn: (chuỗi phải xuất hiện, các tài liệu phải cùng ghi)
    "MRR trung bình đợt 2":
        ("0,3605", ("report/src/report.md", "report/src/TRANG_THAI_THUC_NGHIEM.md")),
    "độ lệch chuẩn seed":
        ("0,0009", ("report/src/report.md", "report/src/TRANG_THAI_THUC_NGHIEM.md")),
    "MRR cấu hình bài báo":
        ("0,3691", ("report/src/report.md", "report/src/TRANG_THAI_THUC_NGHIEM.md")),
    "số thực thể ViMMKG":
        ("12.334", ("research/PROPOSAL_ViMMKG.md", "research/TRACKING_ViMMKG.md")),
    "số thực thể Wikidata":
        ("42.081", ("research/PROPOSAL_ViMMKG.md", "research/TRACKING_ViMMKG.md")),
}

def check_shared_numbers() -> None:
    for label, (pat, files) in SHARED_NUMBERS.items():
        missing = [f for f in files if not re.search(pat, read(f))]
        if missing:
            canh_bao(f"'{label}': không thấy trong {', '.join(missing)} "
                     f"— có thể một tài liệu còn số cũ")


# ── 6b. Số trong bảng kết quả có được verify_numbers.py khẳng định không ──
#: Tài liệu nào chịu kiểm độ phủ, và ngưỡng tối thiểu (%).
# Ngưỡng đặt ngay DƯỚI mức hiện tại, nên nó bắt TỤT LÙI chứ không đòi 100%.
# Lý do có kiểm này: verify_numbers.py chỉ khẳng định những con số ai đó nhớ
# thêm vào. Câu "max n_f trên train là 8 nên K=20 phủ hết" sai nhiều tuần vì
# không gì đối chiếu nó với dữ liệu — đó là lỗi LẬP LUẬN, mà lập luận bao giờ
# cũng neo vào một con số. Giữ độ phủ không tụt là cách rẻ nhất để chặn.
COVERAGE_DOCS = {
    "report/src/report_impartail.md": 40,
    "report/src/report.md": 40,
}


def _table_numbers(doc: str) -> set[str]:
    """Số thập phân kiểu Việt (dấu phẩy) nằm trong bảng Markdown.

    Chỉ lấy trong bảng vì đó là nơi đặt kết quả thí nghiệm. Số trong văn xuôi
    phần lớn là phiên bản thư viện, DOI, hay số đếm — nhiễu, không phải kết quả.
    """
    t = read(doc)
    if not t:
        return set()
    t = re.sub(r"```.*?```", "", t, flags=re.S)
    out: set[str] = set()
    for ln in t.split("\n"):
        ln = ln.strip()
        if not ln.startswith("|") or ln.count("|") < 3:
            continue
        if re.match(r"^[\s|:\-]+$", ln):        # dòng kẻ ngang
            continue
        ln = re.sub(r"`[^`]*`", "", ln)          # mã nội tuyến
        # (?<![\d.,]) ... (?![\d,]) loại "1.825" (phân cách nghìn) và DOI.
        out |= set(re.findall(r"(?<![\d.,])\d{1,3},\d+(?![\d,])", ln))
    return out


def check_number_coverage() -> None:
    qa_src = (ROOT / "qa/verify_numbers.py")
    if not qa_src.is_file():
        loi("qa/verify_numbers.py: không tồn tại")
        return
    src = qa_src.read_text()
    for doc, nguong in COVERAGE_DOCS.items():
        nums = _table_numbers(doc)
        if not nums:
            continue
        thieu = sorted(n for n in nums if n.replace(",", ".") not in src)
        phu = 100 * (len(nums) - len(thieu)) / len(nums)
        if phu < nguong:
            vd = ", ".join(thieu[:8])
            canh_bao(f"{doc}: chỉ {phu:.0f}% số trong bảng được "
                     f"verify_numbers.py khẳng định (ngưỡng {nguong}%) "
                     f"— {len(thieu)} số chưa có assert, vd: {vd}")


# ── 7. requirements.txt vs import thật trong mã ─────────────────────
#: import name -> tên package trên PyPI, khi hai cái khác nhau.
PKG_ALIAS = {
    "sklearn": "scikit-learn", "yaml": "PyYAML", "PIL": "Pillow",
    "rank_bm25": "rank-bm25", "huggingface_hub": "huggingface-hub",
    "dotenv": "python-dotenv", "llmlingua": "llmlingua",
}
# "args", "mmkgc": module của repo tác giả (APKGC, NativE), nạp bằng sys.path lúc chạy.
STDLIB_OK = set(sys.stdlib_module_names) | {"itercomp", "args", "mmkgc"}


def _local_modules() -> set[str]:
    """Module do chính repo định nghĩa — không phải dependency ngoài.

    `research/exp1_oracle_stopping.py` làm `from measure_stop_error import ...`
    vì hai file cùng thư mục. Đó không phải package trên PyPI.
    """
    out = {"itercomp"}
    for d in CODE_DIRS:
        out |= {f.stem for f in (ROOT / d).glob("*.py")}
    return out


def _is_optional_import(node: ast.AST, tree: ast.AST) -> bool:
    """Import nằm trong try/except ImportError -> có fallback, không bắt buộc.

    `scorer.py` thử FlagEmbedding rồi lùi về transformers thuần khi phiên bản
    không tương thích. Bắt buộc nó trong requirements là sai.
    """
    for parent in ast.walk(tree):
        if not isinstance(parent, ast.Try):
            continue
        if not any(n is node for n in ast.walk(parent.body[0].parent
                                                if False else parent)):
            continue
        for h in parent.handlers:
            if h.type is None:                       # bare except
                return True
            names = ([h.type] if isinstance(h.type, ast.Name)
                     else h.type.elts if isinstance(h.type, ast.Tuple) else [])
            for n in names:
                # Exception cũng bắt ImportError, nên nó cũng là fallback hợp lệ
                if isinstance(n, ast.Name) and n.id in {
                        "ImportError", "ModuleNotFoundError", "Exception",
                        "BaseException"}:
                    return True
    return False


def check_requirements() -> None:
    req = read("requirements.txt").lower()
    skip = STDLIB_OK | _local_modules()
    seen: set[str] = set()
    for d in CODE_DIRS:
        for f in (ROOT / d).glob("*.py"):
            try:
                tree = ast.parse(f.read_text())
            except SyntaxError:
                loi(f"{d}/{f.name}: lỗi cú pháp Python")
                continue
            for node in ast.walk(tree):
                if _is_optional_import(node, tree):
                    continue
                if isinstance(node, ast.Import):
                    seen |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    seen.add((node.module or "").split(".")[0])
    for mod in sorted(seen - skip):
        if not mod:
            continue
        pkg = PKG_ALIAS.get(mod, mod)
        if pkg.lower() not in req:
            loi(f"requirements.txt thiếu `{pkg}` (mã có `import {mod}`)")


# ── Kết quả mới mà chưa tài liệu nào nhắc ───────────────────────────
def check_orphan_results() -> None:
    """File kết quả mà KHÔNG tài liệu nào nhắc VÀ không script nào đọc.

    Chỉ cảnh báo khi cả hai đều không — một file được `verify_numbers` đọc thì
    đã có ràng buộc rồi, dù tài liệu không gọi tên nó. Nếu cảnh báo cả những
    file đó thì danh sách dài tới mức không ai đọc, và cảnh báo thật bị chìm.
    """
    docs = " ".join(read(f) for f in DOCS)
    code = " ".join(f.read_text() for d in CODE_DIRS
                    for f in (ROOT / d).glob("*.py") if f.is_file())
    # Script có thể nạp kết quả bằng mẫu glob thay vì gọi tên từng tệp —
    # selection_bias.py:28 đọc "experiments/results/*/partial_*.json". So
    # nguyên văn p.stem sẽ không khớp và báo là tệp mồ côi 5 vết đánh giá
    # chính là dữ liệu gốc của con số thổi phồng +0,0054 trong báo cáo.
    # Chỉ quét mã SINH ra/ĐỌC kết quả, không quét chính qa/: check_repo.py có
    # rglob("*.json") và regex sẽ nhặt luôn mẫu "*.json" đó — mẫu này khớp mọi
    # tệp, làm kiểm tra âm thầm luôn xanh. Đã dính đúng bẫy này khi viết hàm.
    doc_code = " ".join(f.read_text() for d in CODE_DIRS if d != "qa"
                        for f in (ROOT / d).glob("*.py") if f.is_file())
    globs = [g for g in re.findall(r'["\']([^"\']*\*[^"\']*\.json)["\']', doc_code)
             # mẫu phải nêu tên cụ thể hơn "mọi tệp json", nếu không nó vô hiệu
             # hoá kiểm tra thay vì thu hẹp nó
             if g.replace("*", "").strip("/") not in ("", ".json")]

    def duoc_glob_bat(path: Path) -> bool:
        rel = path.relative_to(ROOT).as_posix()
        return any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(path.name, g)
                   for g in globs)

    for p in sorted((ROOT / RESULTS_DIR).rglob("*.json")):
        if p.stem in docs or p.stem in code or duoc_glob_bat(p):
            continue
        canh_bao(f"{p.relative_to(ROOT)} không tài liệu nào nhắc, "
                 f"không script nào đọc — còn cần không?")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true",
                    help="thoát mã 1 nếu có lỗi (cho CI / pre-commit)")
    ap.add_argument("--root", type=Path, default=None,
                    help="gốc repo cần kiểm; mặc định suy từ vị trí script")
    args = ap.parse_args()
    if args.root:
        global ROOT
        ROOT = args.root.resolve()
        if not ROOT.is_dir():
            raise SystemExit(f"không thấy thư mục: {ROOT}")

    for fn in (check_file_inventory, check_stale_status, check_result_configs,
               check_shared_numbers, check_number_coverage, check_requirements,
               check_orphan_results):
        fn()

    errs = [m for lv, m in issues if lv == "loi"]
    warns = [m for lv, m in issues if lv == "canh_bao"]

    if errs:
        print(f"LỖI ({len(errs)}):")
        for m in errs:
            print(f"  ✗ {m}")
    if warns:
        print(f"\nCẢNH BÁO ({len(warns)}):")
        for m in warns:
            print(f"  ! {m}")
    if not issues:
        print("Không phát hiện vấn đề nào.")
    else:
        print(f"\n{len(errs)} lỗi, {len(warns)} cảnh báo.")
    return 1 if (errs and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
