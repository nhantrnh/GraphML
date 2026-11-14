#!/usr/bin/env python3
"""Đưa các thực nghiệm chưa có trên W&B lên, và đặt project ở chế độ công khai.

`wandb_upload.py` đã đẩy các lần tái lập APKGC có vết đánh giá theo thời gian
(3 seed, cấu hình theo bài, seed 42, DB15K, B2). Tệp này bổ sung những thực
nghiệm chỉ còn số tổng kết, mỗi thực nghiệm một run, summary = số trong tệp
kết quả gốc:

  apkgc-repro
    * mkgw-penalty-A / -B / -C   bốn biến thể phạt attention (three_branch.json)
    * mkgw-no-noise              ablation tắt nhiễu, seed 42 (three_branch.json)
    * mkgw-attn-probe            bộ đếm lệnh phạt, 300 lần forward (attn_scale.json)
    * mkgw-learned-ratios        θ_r, θ_n đọc từ checkpoint A (learned_ratios.json)

  impartail-repro
    * sttran-predcls-eval        đánh giá checkpoint STTran, 3 ràng buộc
    * sttran-train-original      huấn luyện mã gốc
    * sttran-train-patched       huấn luyện bản vá mặt nạ theo công thức (5)

Run được dựng từ tệp kết quả đã lưu, không phải log trực tiếp lúc huấn luyện;
báo cáo ghi rõ điều này.

Cần: WANDB_API_KEY trong môi trường.
Dùng: python wandb_upload_extra.py --entity nhantrnh-graphml [--dry]
"""
import argparse
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "experiments/results")


def doc(p):
    return json.load(open(os.path.join(RES, p), encoding="utf-8"))


def csv_sttran(p):
    """CSV của ImparTail: 36 cột = 3 ràng buộc (with, no, semi) × 12 chỉ số."""
    with open(os.path.join(RES, p), encoding="utf-8") as f:
        hang = list(csv.reader(f))
    ten, so = hang[0][1:], [float(x) for x in hang[1][1:]]
    out = {}
    for i, (t, v) in enumerate(zip(ten, so)):
        rb = ("with", "no", "semi")[i // 12]
        out[f"{rb}/{t}"] = round(100 * v, 2)
    return out


def apkgc_runs():
    tb = doc("apkgc_res/three_branch.json")
    co_chung = dict(dataset="MKG-W", seed=tb["seed"], batch=1024, neg=32,
                    nguon="experiments/results/apkgc_res/three_branch.json")
    runs = []
    for ten, khoa, mo_ta in (
            ("mkgw-penalty-A", "A_goc", "mã công bố"),
            ("mkgw-penalty-B", "B_baibao", "công thức (9)(10) của bài"),
            ("mkgw-penalty-C", "C_entropy", "ngưỡng entropy, 10% hàng thấp nhất"),
            ("mkgw-no-noise", "D_khong_nhieu", "add_noise=0, phạt theo mã công bố")):
        sm = {k: v for k, v in tb[khoa].items() if isinstance(v, (int, float))}
        runs.append((ten, dict(co_chung, bien_the=khoa, mo_ta=mo_ta), sm))
    at = doc("apkgc_res/attn_scale.json")
    runs.append(("mkgw-attn-probe",
                 dict(dataset="MKG-W", mo_ta="đếm số lần điều kiện phạt đúng",
                      nguon="experiments/results/apkgc_res/attn_scale.json"),
                 {k: v for k, v in at.items() if isinstance(v, (int, float))}))
    lr = doc("apkgc_res/learned_ratios.json")
    runs.append(("mkgw-learned-ratios",
                 dict(dataset="MKG-W", seed=lr["seed"], nguon=lr["nguon"]),
                 {"theta_r_init": lr["mask_ratio_theta_r"]["khoi_tao"],
                  "theta_r_final": lr["mask_ratio_theta_r"]["sau_huan_luyen"],
                  "theta_n_init": lr["noise_ratio_theta_n"]["khoi_tao"],
                  "theta_n_final": lr["noise_ratio_theta_n"]["sau_huan_luyen"]}))
    return runs


def impartail_runs():
    co_chung = dict(dataset="Action Genome", mode="PredCls", model="STTran")
    return [
        ("sttran-predcls-eval",
         dict(co_chung, mo_ta="checkpoint STTran của tác giả, 1.750 video test",
              nguon="experiments/results/impartail_repro/sttran_predcls_v8.csv"),
         csv_sttran("impartail_repro/sttran_predcls_v8.csv")),
        ("sttran-train-original",
         dict(co_chung, mo_ta="huấn luyện bằng mã gốc, 1 seed, 2 epoch",
              nguon="experiments/results/impartail_repro/sttran_partial10_goc.csv"),
         csv_sttran("impartail_repro/sttran_partial10_goc.csv")),
        ("sttran-train-patched",
         dict(co_chung, mo_ta="mặt nạ BCE vá theo công thức (5), 1 seed, 2 epoch",
              nguon="experiments/results/impartail_repro/sttran_partial10_va.csv"),
         csv_sttran("impartail_repro/sttran_partial10_va.csv")),
    ]


def day(project, entity, runs, dry):
    urls = {}
    for ten, cfg, sm in runs:
        if dry:
            print(f"  [dry] {project}/{ten}: {len(sm)} số"); continue
        import wandb
        r = wandb.init(project=project, entity=entity, name=ten, config=cfg, reinit=True)
        for k, v in sm.items():
            r.summary[k] = v
        urls[ten] = r.url
        r.finish()
    return urls


def cong_khai(project, entity):
    """Đặt project ở chế độ ai cũng xem được (access = USER_READ), gọi thẳng GraphQL."""
    import requests
    q = ("mutation($n:String!,$e:String!){ upsertModel(input:{name:$n, entityName:$e, "
         "access:\"USER_READ\"}){ model{ name access } } }")
    r = requests.post("https://api.wandb.ai/graphql", auth=("api", os.environ["WANDB_API_KEY"]),
                      json={"query": q, "variables": {"n": project, "e": entity}}, timeout=60)
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--only", choices=["apkgc-repro", "impartail-repro"], default=None,
                    help="chỉ đẩy một project (tránh tạo run trùng khi chạy lại)")
    a = ap.parse_args()
    for proj, runs in (("apkgc-repro", apkgc_runs()), ("impartail-repro", impartail_runs())):
        if a.only and proj != a.only:
            continue
        day(proj, a.entity, runs, a.dry)
        if not a.dry:
            print(proj, cong_khai(proj, a.entity))
    if not a.dry:
        ghi_link(a.entity)


def ghi_link(entity):
    """Ghi link mọi run của hai project ra wandb_runs_all.json để báo cáo trích."""
    import wandb
    api = wandb.Api()
    ket_qua = {}
    for proj in ("apkgc-repro", "impartail-repro"):
        ket_qua[proj] = {r.name: r.url for r in api.runs(f"{entity}/{proj}")}
    out = os.path.join(RES, "wandb_runs_all.json")
    json.dump(ket_qua, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("đã ghi", out)


if __name__ == "__main__":
    main()
