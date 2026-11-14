#!/usr/bin/env python3
"""Đưa so sánh cùng giao thức APKGC–NativE và đo độ bền lên W&B (project apkgc-repro).

Bổ sung cho `wandb_upload.py` và `wandb_upload_extra.py`. Mỗi thực nghiệm một run,
summary = số trong tệp kết quả gốc; các run có vết đánh giá theo epoch thì log
từng điểm để W&B vẽ đường học:

  * mkgw-native-s42 / -s1234      NativE huấn luyện lại với đặc trưng của APKGC,
                                  1000 epoch, đánh giá mỗi 100 epoch (native_res/)
  * mkgw-apkgc-A-seed2024         APKGC biến thể A, seed 2024, trần 2000 epoch (apkgc_res5/)
  * mkgw-robust-apkgc-s42 / -s2024, mkgw-robust-native-s42 / -s1234
                                  30 điều kiện làm hỏng đặc trưng lúc test, mỗi điều kiện
                                  một khoá summary và một bảng (robustness/)

Run được dựng từ tệp kết quả đã lưu, không phải log trực tiếp lúc huấn luyện.

Cần: đăng nhập W&B (~/.netrc hoặc WANDB_API_KEY).
Dùng: python wandb_upload_robustness.py --entity nhantrnh-graphml [--dry]
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "experiments/results")
PROJECT = "apkgc-repro"


def doc(p):
    return json.load(open(os.path.join(RES, p), encoding="utf-8"))


def native_runs():
    runs = []
    for sd in ("s42", "s1234"):
        f = f"native_res/native_{sd}_e1000.json"
        d = doc(f)
        cfg = dict(model="NativE", dataset="MKG-W", seed=d["seed"], epoch=d["epoch"],
                   emb_source=d["emb_source"], config="scripts/run_mkgw.sh của tác giả",
                   nguon=f"experiments/results/{f}")
        sm = {f"final/{k}": v for k, v in d["final"].items()}
        sm.update({f"best_on_test/{k}": v for k, v in d["best_on_test"].items()})
        runs.append((f"mkgw-native-{sd}", cfg, sm, d["evals"]))
    f = "apkgc_res5/A_goc_seed2024.json"
    d = doc(f)
    runs.append(("mkgw-apkgc-A-seed2024",
                 dict(model="APKGC", bien_the="A_goc", dataset="MKG-W", seed=d["seed"],
                      epoch_cap=d["epoch_cap"], config=d["config_source"],
                      nguon=f"experiments/results/{f}"),
                 {k: d[k] for k in ("mrr", "mr", "hit1", "hit3", "hit10", "seconds")}, None))
    return runs


def robust_runs():
    runs = []
    for ten, f in (("mkgw-robust-apkgc-s42", "robustness_apkgc_A_seed42.json"),
                   ("mkgw-robust-apkgc-s2024", "robustness_apkgc_A_seed2024.json"),
                   ("mkgw-robust-native-s42", "robustness_native_s42.json"),
                   ("mkgw-robust-native-s1234", "robustness_native_s1234.json")):
        d = doc(f"robustness/{f}")
        cfg = dict(model=d["model"], checkpoint=d["checkpoint"], dataset=d["dataset"],
                   cong_cu="experiments/tools/apkgc_robustness.py",
                   kiem_chung_mrr_sach=d["kiem_chung_mrr"], kiem_chung_dat=d["kiem_chung_dat"],
                   nguon=f"experiments/results/robustness/{f}")
        sm = {f"clean/{k}": v for k, v in d["sach"].items() if k != "giay"}
        for c in d["dieu_kien"]:
            khoa = f"{c['kieu']}/p{c['ti_le']}/{c['phuong_thuc']}"
            sm[f"{khoa}/mrr"] = c["mrr"]
            sm[f"{khoa}/delta_mrr"] = c["delta_mrr"]
        runs.append((ten, cfg, sm, d["dieu_kien"]))
    return runs


def day(entity, runs, dry):
    urls = {}
    for ten, cfg, sm, vet in runs:
        if dry:
            print(f"  [dry] {PROJECT}/{ten}: {len(sm)} số, vết {len(vet) if vet else 0}")
            continue
        import wandb
        r = wandb.init(project=PROJECT, entity=entity, name=ten, config=cfg, reinit=True)
        if vet and "epoch" in vet[0]:                 # đường học: log theo epoch
            for e in vet:
                r.log({k: v for k, v in e.items() if k != "epoch"}, step=e["epoch"])
        elif vet:                                     # điều kiện làm hỏng: bảng
            cot = ["kieu", "ti_le", "phuong_thuc", "mrr", "hit1", "hit3", "hit10", "delta_mrr", "so_seed"]
            r.log({"dieu_kien": wandb.Table(columns=cot, data=[[c.get(k) for k in cot] for c in vet])})
        for k, v in sm.items():
            r.summary[k] = v
        urls[ten] = r.url
        r.finish()
    return urls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    urls = day(a.entity, native_runs() + robust_runs(), a.dry)
    if urls:
        out = os.path.join(RES, "robustness/wandb_runs.json")
        json.dump(urls, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print("đã ghi", out)


if __name__ == "__main__":
    main()
