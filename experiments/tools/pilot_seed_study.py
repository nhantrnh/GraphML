"""Thí nghiệm thăm dò: đo độ phân tán của MRR qua nhiều seed.

Mục đích: trả lời một câu hỏi duy nhất trước khi đầu tư vào một nghiên cứu lớn —

    Độ lệch chuẩn của MRR qua các seed có lớn hơn khoảng cách mà các bài báo
    MMKGC dùng để tuyên bố vượt SOTA hay không?

Các bài gần đây báo cáo mức cải thiện quanh 0.002–0.010 MRR mà không kèm ước
lượng phương sai. Nếu độ lệch chuẩn giữa các seed vượt ngưỡng đó, phần lớn
tuyên bố "vượt SOTA" trong lĩnh vực này không kiểm chứng được.

Ngưỡng quyết định:

    std > 0.005   luận điểm đứng vững, nên tiến hành nghiên cứu đầy đủ
    std < 0.002   cải tiến là thật, cần đổi hướng
    ở giữa        cần thêm seed hoặc thêm mô hình để kết luận

Cách chạy:

    python pilot_seed_study.py --repo ../AdaMF-MAT --dataset MKG-W --seeds 5

Script này không sửa mã của repo đích. Nó gọi `run_adamf_mat.py` như người dùng
bình thường, chỉ thay đổi `-seed`, rồi trích số liệu từ stdout.
"""

import argparse
import json
import os
import os.path as osp
import re
import statistics
import subprocess
import sys
import time
from typing import Dict, List, Optional

# Mẫu bắt kết quả trong stdout của OpenKE/mmkgc.
# Tester in ra dạng:  no type constraint results:
#                     metric:  MRR   MR   hit@10 hit@3 hit@1
#                     l(raw):  ...
METRIC_LINE = re.compile(
    r"^\s*(l|r|averaged)\(filter\):\s*"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
    re.MULTILINE,
)


def load_published_gaps() -> List[Dict]:
    """Nạp các khoảng cách MRR đã công bố từ published_gaps.json.

    Chỉ trả về những mục đã được đánh dấu `verified`, tức là đã đọc trực tiếp
    từ bảng trong bài báo gốc. Các mục chờ xác minh nằm ở khoá riêng và không
    được dùng cho phần đối chiếu.
    """
    path = osp.join(osp.dirname(osp.abspath(__file__)), "published_gaps.json")
    if not osp.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [g for g in data.get("gaps", []) if g.get("verified")]


def parse_metrics(stdout: str) -> Optional[Dict[str, float]]:
    """Trích MRR/MR/Hits từ stdout của mmkgc/base/Test.h.

    Định dạng do C++ in ra (Test.h, dòng 268-271):

        metric:            MRR      MR      hit@10   hit@3   hit@1
        l(filter):         %f       %f      %f       %f      %f
        r(filter):         %f       %f      %f       %f      %f
        averaged(filter):  %f       %f      %f       %f      %f

    Chỉ chấp nhận dòng `averaged(filter)` — đây là con số mà mọi bài báo
    trong lĩnh vực báo cáo. Nếu chỉ có l và r mà thiếu dòng averaged, hàm
    tự tính trung bình thay vì lấy nhầm một chiều.
    """
    sides: Dict[str, Dict[str, float]] = {}

    for m in METRIC_LINE.finditer(stdout):
        vals = dict(
            mrr=float(m.group(2)), mr=float(m.group(3)),
            hit10=float(m.group(4)), hit3=float(m.group(5)),
            hit1=float(m.group(6)),
        )
        sides[m.group(1)] = vals

    if "averaged" in sides:
        return sides["averaged"]

    if "l" in sides and "r" in sides:
        keys = ("mrr", "mr", "hit10", "hit3", "hit1")
        return {k: (sides["l"][k] + sides["r"][k]) / 2.0 for k in keys}

    return None


SCRIPT_FOR = {
    "MKG-W": "run_mkgw.sh",
    "MKG-Y": "run_mkgy.sh",
    "DB15K": "run_db15k.sh",
}

SHELL_VAR = re.compile(r"^\s*([A-Z_][A-Z0-9_]*)=(\S+)\s*$", re.MULTILINE)


def read_author_config(repo: str, dataset: str) -> Dict[str, str]:
    """Đọc siêu tham số trực tiếp từ script của tác giả.

    Không chép cứng giá trị nào vào đây: nếu tác giả cập nhật script, thí
    nghiệm thăm dò tự động dùng theo. Điều này cũng loại bỏ nguy cơ báo cáo
    một cấu hình khác với cấu hình gốc.
    """
    path = osp.join(repo, "scripts", SCRIPT_FOR[dataset])
    if not osp.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy script gốc: {path}. "
            "Thí nghiệm cần đọc siêu tham số từ đây thay vì tự đặt.")

    with open(path, encoding="utf-8") as f:
        text = f.read()

    cfg = dict(SHELL_VAR.findall(text))

    required = ["EMB_DIM", "NUM_BATCH", "MARGIN", "LR", "NEG_NUM", "EPOCH"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"{path} thiếu biến: {', '.join(missing)}")

    if cfg.get("DATA") != dataset:
        raise ValueError(
            f"{path} khai báo DATA={cfg.get('DATA')} nhưng đang chạy {dataset}")

    return cfg


def run_one(repo: str, dataset: str, seed: int, cfg: Dict[str, str],
            epoch: Optional[int], extra: List[str],
            timeout: int) -> Optional[Dict[str, float]]:
    """Chạy một lần huấn luyện + đánh giá với seed cho trước."""
    save = osp.join(repo, "checkpoint", f"pilot-{dataset}-seed{seed}")
    os.makedirs(osp.dirname(save), exist_ok=True)

    n_epoch = epoch if epoch is not None else int(cfg["EPOCH"])

    cmd = [
        sys.executable, "run_adamf_mat.py",
        f"-dataset={dataset}",
        f"-batch_size={cfg['NUM_BATCH']}",
        f"-margin={cfg['MARGIN']}",
        f"-epoch={n_epoch}",
        f"-dim={cfg['EMB_DIM']}",
        f"-neg_num={cfg['NEG_NUM']}",
        f"-learning_rate={cfg['LR']}",
        "-mu=0",
        f"-seed={seed}",
        f"-save={save}",
    ] + extra

    print(f"  seed {seed}: đang chạy ...", flush=True)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=repo, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  seed {seed}: quá thời gian ({timeout}s)")
        return None

    dt = time.time() - t0
    out = proc.stdout + "\n" + proc.stderr
    metrics = parse_metrics(out)

    if metrics is None:
        print(f"  seed {seed}: không trích được kết quả sau {dt:.0f}s")
        tail = "\n".join(out.strip().splitlines()[-15:])
        print("  --- đuôi log ---")
        print("  " + tail.replace("\n", "\n  "))
        return None

    metrics["seed"] = seed
    metrics["seconds"] = round(dt, 1)
    print(f"  seed {seed}: MRR = {metrics['mrr']:.4f}  "
          f"H@1 = {metrics['hit1']:.4f}  ({dt:.0f}s)")
    return metrics


def report(results: List[Dict[str, float]], dataset: str) -> None:
    if len(results) < 2:
        print("\nKhông đủ kết quả để tính độ phân tán.")
        return

    print()
    print("=" * 72)
    print(f"KẾT QUẢ — {dataset}, {len(results)} seed")
    print("=" * 72)
    print(f"  {'seed':>6}  {'MRR':>8}  {'H@1':>8}  {'H@3':>8}  {'H@10':>8}")
    for r in results:
        print(f"  {r['seed']:>6}  {r['mrr']:>8.4f}  {r['hit1']:>8.4f}  "
              f"{r['hit3']:>8.4f}  {r['hit10']:>8.4f}")

    print()
    stats = {}
    for key in ("mrr", "hit1", "hit3", "hit10"):
        vals = [r[key] for r in results]
        stats[key] = dict(
            mean=statistics.mean(vals),
            std=statistics.stdev(vals),
            lo=min(vals), hi=max(vals),
            spread=max(vals) - min(vals),
        )
        s = stats[key]
        print(f"  {key.upper():>6}: {s['mean']:.4f} ± {s['std']:.4f}   "
              f"[{s['lo']:.4f}, {s['hi']:.4f}]   biên độ {s['spread']:.4f}")

    mrr_std = stats["mrr"]["std"]
    print()
    print("-" * 72)
    print("DIỄN GIẢI")
    print("-" * 72)
    print(f"  Độ lệch chuẩn MRR đo được: {mrr_std:.4f}")
    print()
    print("  Đối chiếu với khoảng cách giữa các phương pháp trong bảng kết quả")
    print("  đã công bố. Mỗi con số dưới đây lấy từ bảng của bài báo tương ứng;")
    print("  xem published_gaps.json để biết nguồn từng dòng.")
    print()

    gaps = load_published_gaps()
    if not gaps:
        print("    (chưa nạp được published_gaps.json — bỏ qua phần đối chiếu)")
    else:
        for g in gaps:
            if g["dataset"] != dataset:
                continue
            verdict = "nằm trong nhiễu" if mrr_std >= g["gap"] else "vượt nhiễu"
            print(f"    {g['label']:<34} {g['gap']:.4f}   -> {verdict}")
            print(f"      nguồn: {g['source']}")

    print()
    if mrr_std > 0.005:
        print("  KẾT LUẬN: độ phân tán lớn hơn phần lớn khoảng cách được công bố.")
        print("  Luận điểm của nghiên cứu đứng vững — nên tiến hành đầy đủ.")
    elif mrr_std < 0.002:
        print("  KẾT LUẬN: mô hình rất ổn định giữa các seed. Các cải tiến được")
        print("  công bố nhiều khả năng là thật. Nên chuyển trọng tâm sang phần")
        print("  audit tham số và tính không đồng nhất của encoder.")
    else:
        print("  KẾT LUẬN: chưa dứt khoát. Cần thêm seed, hoặc lặp lại trên mô")
        print("  hình và bộ dữ liệu khác trước khi kết luận.")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True,
                    help="Thư mục repo AdaMF-MAT")
    ap.add_argument("--dataset", default="MKG-W",
                    choices=["MKG-W", "MKG-Y", "DB15K"])
    ap.add_argument("--seeds", type=int, default=5,
                    help="Số seed cần chạy")
    ap.add_argument("--epoch", type=int, default=None,
                    help="Ghi đè số epoch. Mặc định dùng giá trị trong script "
                         "của tác giả.")
    ap.add_argument("--timeout", type=int, default=14400,
                    help="Giới hạn thời gian mỗi lần chạy, tính bằng giây")
    ap.add_argument("--out", default="pilot_results.json")
    args, extra = ap.parse_known_args()

    repo = osp.abspath(args.repo)
    if not osp.exists(osp.join(repo, "run_adamf_mat.py")):
        sys.exit(f"Không tìm thấy run_adamf_mat.py trong {repo}")

    emb_dir = osp.join(repo, "embeddings")
    needed = [f"{args.dataset}-visual.pth", f"{args.dataset}-textual.pth"]
    missing = [f for f in needed if not osp.exists(osp.join(emb_dir, f))]
    if missing:
        sys.exit(
            "Thiếu bộ nhúng: " + ", ".join(missing) + "\n"
            "Tải từ Google Drive của tác giả và đặt vào " + emb_dir + "\n"
            "https://drive.google.com/drive/folders/1UJSfnb8DEx2s-k8zaQx1fWUw5f45GBpI"
        )

    seeds = [42, 1234, 2024, 3407, 7, 100, 2025, 999][:args.seeds]

    cfg = read_author_config(repo, args.dataset)
    n_epoch = args.epoch if args.epoch is not None else int(cfg["EPOCH"])

    print("=" * 72)
    print("THÍ NGHIỆM THĂM DÒ — ĐỘ PHÂN TÁN THEO SEED")
    print("=" * 72)
    print(f"  Mô hình   : AdaMF-MAT")
    print(f"  Dữ liệu   : {args.dataset}")
    print(f"  Cấu hình  : đọc từ scripts/{SCRIPT_FOR[args.dataset]}")
    print(f"              dim={cfg['EMB_DIM']} batch={cfg['NUM_BATCH']} "
          f"margin={cfg['MARGIN']} neg={cfg['NEG_NUM']} lr={cfg['LR']}")
    print(f"  Số epoch  : {n_epoch}" +
          ("" if args.epoch is None else "  (ghi đè từ dòng lệnh)"))
    print(f"  Các seed  : {seeds}")
    print()

    results = []
    for sd in seeds:
        r = run_one(repo, args.dataset, sd, cfg, args.epoch, extra, args.timeout)
        if r:
            results.append(r)

    report(results, args.dataset)

    if results:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"dataset": args.dataset, "epoch": args.epoch,
                       "runs": results}, f, indent=2)
        print(f"  Đã ghi kết quả: {args.out}")


if __name__ == "__main__":
    main()
