#!/usr/bin/env python3
"""Đưa các lần chạy tái lập APKGC lên Weights & Biases từ vết đã lưu.

Đề bài yêu cầu ghi nhận quá trình tái lập bằng W&B và gắn link công khai vào
báo cáo. Các lần chạy được thực hiện trên Kaggle và ghi ra hai dạng vết:

  * partial_<tag>.json  — danh sách `evals`, mỗi phần tử một lần đánh giá
                          (mrr, mr, hit1, hit3, hit10, elapsed);
  * <tag>.log           — stdout của run_KGC_train.py, mỗi lần đánh giá in
                          một dòng `INFO - ... - mrr:0.34,\t mr:805,\t hit10:...`.

Script này đọc cả hai, tạo MỘT run W&B cho MỖI lần chạy, ghi từng lần đánh
giá với step = chỉ số đánh giá (và epoch = step × save_steps nếu biết), rồi
đặt summary = checkpoint tốt nhất. Đường cong trên W&B do đó khớp 1:1 với vết
gốc — cách ghi này được nêu rõ trong báo cáo: run được đẩy lên từ vết đã lưu,
không phải log trực tiếp lúc huấn luyện.

Cần: WANDB_API_KEY trong môi trường (hoặc đã `wandb login`).
Dùng: python wandb_upload.py [--project apkgc-repro] [--entity <user>] [--dry]
"""
import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "experiments/results")

DONG_EVAL = re.compile(
    r"mrr:([\d.]+),\s*mr:([\d.]+),\s*hit10:([\d.]+),\s*hit3:([\d.]+),\s*hit1:([\d.]+)")

# (tệp, nhãn, cấu hình) — cấu hình ghi vào run.config để so được trên W&B
NGUON = [
    ("apkgc_res/partial_seed1234.json", "mkgw-seed1234",
     dict(dataset="MKG-W", seed=1234, batch=1024, neg=32, save_steps=100, nhanh="goc")),
    ("apkgc_res/partial_seed2024.json", "mkgw-seed2024",
     dict(dataset="MKG-W", seed=2024, batch=1024, neg=32, save_steps=100, nhanh="goc")),
    ("apkgc_res/partial_seed3407.json", "mkgw-seed3407",
     dict(dataset="MKG-W", seed=3407, batch=1024, neg=32, save_steps=100, nhanh="goc")),
    ("apkgc_res2/partial_paper_config.json", "mkgw-paper-config",
     dict(dataset="MKG-W", seed=1234, batch=2048, neg=64, save_steps=100, nhanh="goc")),
    ("apkgc_res3/partial_seed42.json", "mkgw-seed42",
     dict(dataset="MKG-W", seed=42, batch=1024, neg=32, save_steps=20, nhanh="goc")),
    ("apkgc_res/db15k_seed42.json", "db15k-seed42-cap600",
     dict(dataset="DB15K", seed=42, batch=1024, neg=32, save_steps=50, nhanh="goc",
          epoch_cap=600)),
]


def doc_partial(p):
    d = json.load(open(p))
    return [dict(mrr=e["mrr"], mr=e["mr"], hit1=e["hit1"], hit3=e["hit3"],
                 hit10=e["hit10"], elapsed_s=e.get("elapsed")) for e in d["evals"]]


def doc_log(p):
    out = []
    for m in DONG_EVAL.finditer(open(p, encoding="utf-8", errors="replace").read()):
        mrr, mr, h10, h3, h1 = map(float, m.groups())
        out.append(dict(mrr=mrr, mr=mr, hit1=h1, hit3=h3, hit10=h10, elapsed_s=None))
    return out


def day_len(evals, ten, cfg, project, entity, dry):
    best = max(evals, key=lambda e: e["mrr"])
    if dry:
        print(f"  [dry] {ten:<22} {len(evals):>2} eval  best mrr={best['mrr']:.4f}")
        return None
    import wandb
    run = wandb.init(project=project, entity=entity, name=ten, config=cfg,
                     reinit=True, notes="Đẩy lên từ vết đã lưu (partial JSON / log Kaggle).")
    ss = cfg.get("save_steps")
    for i, e in enumerate(evals, 1):
        rec = {k: v for k, v in e.items() if v is not None}
        rec["eval_idx"] = i
        if ss:
            rec["epoch"] = i * ss
        wandb.log(rec, step=i)
    for k in ("mrr", "hit1", "hit3", "hit10"):
        run.summary[f"best_{k}"] = best[k]
    run.summary["n_evals"] = len(evals)
    url = run.url
    run.finish()
    print(f"  ✓ {ten:<22} {len(evals):>2} eval  best mrr={best['mrr']:.4f}  {url}")
    return url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="apkgc-repro")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--dry", action="store_true", help="chỉ liệt kê, không đẩy")
    ap.add_argument("--extra-logs", nargs="*", default=[],
                    help="thêm log Kaggle: dạng ten=duong/dan.log[:dataset:seed:nhanh]")
    a = ap.parse_args()

    if not a.dry and not os.environ.get("WANDB_API_KEY") and \
            not os.path.exists(os.path.expanduser("~/.netrc")):
        sys.exit("Thiếu WANDB_API_KEY (hoặc chưa `wandb login`). Chạy --dry để xem trước.")

    viec = []
    for rel, ten, cfg in NGUON:
        p = os.path.join(RES, rel)
        if os.path.isfile(p):
            viec.append((ten, doc_partial(p), cfg))
        else:
            print(f"  - thiếu {rel}, bỏ qua")
    for spec in a.extra_logs:
        ten, rest = spec.split("=", 1)
        parts = rest.split(":")
        p = parts[0]
        cfg = dict(dataset=parts[1] if len(parts) > 1 else "?",
                   seed=int(parts[2]) if len(parts) > 2 else None,
                   nhanh=parts[3] if len(parts) > 3 else "?")
        ev = doc_log(p)
        if ev:
            viec.append((ten, ev, cfg))
        else:
            print(f"  - {p}: không thấy dòng eval nào")

    print(f"{len(viec)} run sẽ đẩy lên project '{a.project}'\n")
    urls = {}
    for ten, ev, cfg in viec:
        u = day_len(ev, ten, cfg, a.project, a.entity, a.dry)
        if u:
            urls[ten] = u
    if urls:
        out = os.path.join(RES, "apkgc_res/wandb_runs.json")
        json.dump(urls, open(out, "w"), indent=1, ensure_ascii=False)
        print(f"\nđã ghi {out}")
        print("link project:", urls[next(iter(urls))].rsplit("/runs/", 1)[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
