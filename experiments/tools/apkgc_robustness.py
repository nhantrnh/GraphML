#!/usr/bin/env python3
"""Đánh giá độ bền của checkpoint MMKGC (APKGC, NativE) khi phương thức bị hỏng lúc test.

VÌ SAO. Bài APKGC nói lấy mẫu nhiễu làm mô hình "bền vững" hơn, nhưng chỉ đánh
giá trên tập test sạch. Công cụ này làm hỏng đặc trưng ảnh/văn bản lúc test rồi
đo lại MRR, Hits@K, không huấn luyện lại. NativE được đo cùng cách để so sánh.

CÁCH TÍNH NHANH MÀ VẪN ĐÚNG. Vector hợp nhất của một thực thể chỉ phụ thuộc vào
ba vector của chính nó (APKGC), hoặc thêm cổng của quan hệ (NativE: `rel_gate`).
Nên tính vector hợp nhất cho mọi thực thể một lần (APKGC) hoặc một lần cho mỗi
quan hệ (NativE), rồi chấm RotatE cho mọi truy vấn bằng phép toán ma trận. Xếp
hạng theo đúng `mmkgc/base/Test.h` của cả hai repo: đếm số ứng viên có khoảng
cách NHỎ HƠN HẲN đáp án, bỏ qua bộ ba đã biết (train + valid + test), hạng =
đếm + 1; MRR, Hits@K trung bình trên hai chiều đầu/đuôi. Điều kiện sạch phải
cho lại đúng MRR của checkpoint — tham số `--expect` kiểm chuyện đó.

CÁC KIỂU LÀM HỎNG (áp lên bảng đặc trưng đóng băng `img_embeddings`,
`text_embeddings`; vector cấu trúc không đổi):
  missing  thay đặc trưng bằng vector trung bình của các thực thể có phương thức
           đó — đúng cách mã APKGC tự điền cho thực thể thiếu phương thức
  shuffle  hoán vị đặc trưng giữa các thực thể được chọn
  noise_σ  cộng nhiễu Gauss σ × std theo từng chiều
mỗi kiểu với tỉ lệ thực thể bị hỏng p ∈ {0,5; 1}. Kiểu ngẫu nhiên lặp nhiều seed.

Dùng (mỗi mô hình một tiến trình vì hai repo cùng tên gói `mmkgc`):
    python apkgc_robustness.py --model apkgc --repo <APKGC> --ckpt <ckpt> \\
        --missing <missing.json> --expect 0.3629 --out <json>
    python apkgc_robustness.py --model native --repo <NATIVE> --ckpt <ckpt> \\
        --missing <missing.json> --expect 0.3684 --out <json>
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch

# Cờ dựng APKGC: giống lệnh huấn luyện của biến thể A (log seed42).
APKGC_ARGV = ["-dataset=MKG-W", "-num_proj=1", "-use_intermediate=0",
              "-joint_way=Mformer_hd_mean", "-dim=128", "-margin=12",
              "-noise_update=epoch", "-noise_ratio=0.2", "-mask_ratio=0.7",
              "-use_pool=1", "-add_noise=1", "-num_hidden_layers=1",
              "-num_attention_heads=2"]
# NativE trên MKG-W: scripts/run_mkgw.sh của tác giả.
NATIVE_DIM, NATIVE_MARGIN = 250, 4.0


def doc_bo_ba(path):
    with open(path) as f:
        n = int(f.readline())
        out = [tuple(map(int, f.readline().split())) for _ in range(n)]
    return [(h, r, t) for h, t, r in out]          # tệp OpenKE ghi h t r


def dung_mo_hinh(kind, repo, ckpt, device):
    """Dựng mô hình bằng mã của chính repo, nạp checkpoint; đặc trưng lấy từ checkpoint."""
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    # NativE lưu cả bộ bọc NegativeSamplingGP: khoá mô hình có tiền tố "model.",
    # còn "loss.*" và hằng số ngoài cùng không thuộc mô hình.
    if any(k.startswith("model.") for k in sd):
        sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    img, txt = sd["img_embeddings.weight"], sd["text_embeddings.weight"]
    n_ent, n_rel = sd["ent_embeddings.weight"].shape[0], sd["rel_embeddings.weight"].shape[0]
    sys.path.insert(0, repo)
    cwd = os.getcwd()
    os.chdir(repo)                                    # APKGC đọc ./embeddings/missing.json
    try:
        if kind == "apkgc":
            from args import get_args
            from mmkgc.module.model.APKGC import APKGC
            sys.argv = ["x"] + APKGC_ARGV
            args = get_args()
            m = APKGC(args=args, ent_tot=n_ent, rel_tot=n_rel, dim=args.dim,
                      margin=args.margin, epsilon=2.0, img_emb=img.clone(), text_emb=txt.clone())
        else:
            from mmkgc.module.model import AdvRelRotatE
            m = AdvRelRotatE(ent_tot=n_ent, rel_tot=n_rel, dim=NATIVE_DIM,
                             margin=NATIVE_MARGIN, epsilon=2.0, img_emb=img.clone(), text_emb=txt.clone())
    finally:
        os.chdir(cwd)
    thieu, thua = m.load_state_dict(sd, strict=False)
    # APKGC: hai bảng nhiễu tạo trong update_noise(), không tồn tại lúc dựng.
    assert not thieu, f"checkpoint thiếu khoá: {thieu}"
    assert set(thua) <= {"text_embeddings_noise.weight", "img_embeddings_noise.weight"}, thua
    return m.to(device).eval()


@torch.no_grad()
def vector_hop_nhat(kind, m, img_w, txt_w, rel, device, lo=4096):
    """Vector hợp nhất của mọi thực thể (với NativE: cho quan hệ `rel`)."""
    n = m.ent_embeddings.weight.shape[0]
    ra = []
    for s in range(0, n, lo):
        idx = torch.arange(s, min(n, s + lo), device=device)
        es = m.ent_embeddings(idx)
        ev = m.img_proj(img_w[idx])
        et = m.text_proj(txt_w[idx])
        if kind == "apkgc":
            ra.append(m.get_joint_embeddings(es, ev, et))
        else:
            rg = m.rel_gate(torch.full_like(idx, rel))
            ra.append(m.get_joint_embeddings(es, ev, et, rg))
    return torch.cat(ra)


@torch.no_grad()
def xep_hang(kind, m, img_w, txt_w, test, da_biet_t, da_biet_h, device, lo=16):
    """Hạng lọc của đáp án cho cả hai chiều, theo quy tắc của Test.h."""
    pi = m.pi_const.item()
    rng = m.rel_embedding_range.item()
    theo_qh = defaultdict(list)
    for x in test:
        theo_qh[x[1]].append(x)
    J_chung = vector_hop_nhat(kind, m, img_w, txt_w, None, device) if kind == "apkgc" else None
    hang = []
    for rel, ds in theo_qh.items():
        J = J_chung if J_chung is not None else vector_hop_nhat(kind, m, img_w, txt_w, rel, device)
        re_all, im_all = torch.chunk(J, 2, dim=-1)
        for s in range(0, len(ds), lo):
            lo_bt = ds[s:s + lo]
            h = torch.tensor([x[0] for x in lo_bt], device=device)
            r = torch.tensor([x[1] for x in lo_bt], device=device)
            t = torch.tensor([x[2] for x in lo_bt], device=device)
            ph = m.rel_embeddings(r) / (rng / pi)
            cr, sr = torch.cos(ph), torch.sin(ph)
            rh, ih = re_all[h], im_all[h]
            rt, it = re_all[t], im_all[t]
            # đuôi: |h∘r − t'| ; đầu: |conj(r)∘t − h'|
            for qr, qi, dung, loai in (
                    (rh * cr - ih * sr, rh * sr + ih * cr, t,
                     [da_biet_t[(a, rel)] for a, _, _ in lo_bt]),
                    (cr * rt + sr * it, cr * it - sr * rt, h,
                     [da_biet_h[(rel, c)] for _, _, c in lo_bt])):
                d = torch.sqrt((qr[:, None, :] - re_all[None]) ** 2 +
                               (qi[:, None, :] - im_all[None]) ** 2).sum(-1)     # [B, N]
                hon = d < d.gather(1, dung[:, None])
                for i, ung_vien in enumerate(loai):      # bỏ ứng viên tạo thành bộ ba đã biết
                    hon[i, list(ung_vien)] = False
                hang.extend((hon.sum(1) + 1).tolist())
    return hang


def chi_so(hang):
    n = len(hang)
    return dict(mrr=sum(1 / x for x in hang) / n, mr=sum(hang) / n,
                hit1=sum(x <= 1 for x in hang) / n, hit3=sum(x <= 3 for x in hang) / n,
                hit10=sum(x <= 10 for x in hang) / n)


def lam_hong(w, kieu, p, sd, mean):
    w = w.clone()
    g = torch.Generator().manual_seed(sd)
    n = w.shape[0]
    chon = torch.randperm(n, generator=g)[: int(round(p * n))].to(w.device)
    if kieu == "missing":
        w[chon] = mean
    elif kieu == "shuffle":
        w[chon] = w[chon[torch.randperm(len(chon), generator=g).to(w.device)]]
    elif kieu.startswith("noise"):
        sigma = float(kieu.split("_")[1])
        std = w.std(0, keepdim=True)
        w[chon] = w[chon] + sigma * std * torch.randn(len(chon), w.shape[1], generator=g).to(w.device)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["apkgc", "native"], required=True)
    ap.add_argument("--repo", required=True, help="thư mục repo của mô hình (chứa mmkgc/ và benchmarks/)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--missing", required=True, help="missing.json: chỉ số thực thể thiếu ảnh/văn bản")
    ap.add_argument("--expect", type=float, help="MRR sạch phải khớp (sai số 5e-4)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3, help="số seed cho kiểu làm hỏng ngẫu nhiên")
    ap.add_argument("--budget", type=float, default=2.5, help="giờ tối đa; quá thì dừng, kết quả đã ghi giữ nguyên")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_bat_dau = time.time()

    m = dung_mo_hinh(a.model, a.repo, a.ckpt, device)
    bm = os.path.join(a.repo, "benchmarks/MKG-W")
    test = doc_bo_ba(f"{bm}/test2id.txt")
    da_biet_t, da_biet_h = defaultdict(set), defaultdict(set)
    for f in ("train2id.txt", "valid2id.txt", "test2id.txt"):
        for h, r, t in doc_bo_ba(f"{bm}/{f}"):
            da_biet_t[(h, r)].add(t)
            da_biet_h[(r, t)].add(h)

    img0, txt0 = m.img_embeddings.weight.data, m.text_embeddings.weight.data
    ms = json.load(open(a.missing))
    thieu_img = set(ms["MKG-W-visual.pth"]); thieu_txt = set(ms["MKG-W-textual.pth"])
    co_img = torch.tensor([i for i in range(img0.shape[0]) if i not in thieu_img], device=device)
    co_txt = torch.tensor([i for i in range(txt0.shape[0]) if i not in thieu_txt], device=device)
    img_mean, txt_mean = img0[co_img].mean(0), txt0[co_txt].mean(0)

    def danh_gia(img_w, txt_w):
        t0 = time.time()
        kq = chi_so(xep_hang(a.model, m, img_w, txt_w, test, da_biet_t, da_biet_h, device))
        kq["giay"] = round(time.time() - t0, 1)
        return kq

    sach = danh_gia(img0, txt0)
    print(f"[{a.model}] sạch:", {k: round(v, 4) for k, v in sach.items()}, flush=True)
    ket_qua = {"model": a.model, "checkpoint": os.path.basename(a.ckpt), "dataset": "MKG-W",
               "sach": sach, "kiem_chung_mrr": a.expect, "dieu_kien": []}
    if a.expect is not None:
        ket_qua["kiem_chung_dat"] = abs(sach["mrr"] - a.expect) < 5e-4
        assert ket_qua["kiem_chung_dat"], f"MRR sạch {sach['mrr']:.4f} lệch {a.expect}: bộ đánh giá sai"
    json.dump(ket_qua, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    for kieu in ("missing", "shuffle", "noise_0.5", "noise_1", "noise_2"):
        for p in (1.0, 0.5):
            for pt in ("img", "txt", "both"):
                if time.time() - t_bat_dau > a.budget * 3600:
                    print("hết ngân sách thời gian, dừng", flush=True)
                    return
                tat_dinh = kieu == "missing" and p == 1.0
                lan = []
                for sd in range(1 if tat_dinh else a.seeds):
                    img_w = lam_hong(img0, kieu, p, sd, img_mean) if pt in ("img", "both") else img0
                    txt_w = lam_hong(txt0, kieu, p, 1000 + sd, txt_mean) if pt in ("txt", "both") else txt0
                    lan.append(danh_gia(img_w, txt_w))
                tb = {k: sum(x[k] for x in lan) / len(lan) for k in ("mrr", "hit1", "hit3", "hit10")}
                if len(lan) > 1:
                    mu = tb["mrr"]
                    tb["mrr_std"] = (sum((x["mrr"] - mu) ** 2 for x in lan) / (len(lan) - 1)) ** 0.5
                tb.update(kieu=kieu, ti_le=p, phuong_thuc=pt, so_seed=len(lan),
                          delta_mrr=tb["mrr"] - sach["mrr"], giay=sum(x["giay"] for x in lan))
                ket_qua["dieu_kien"].append(tb)
                print(f"[{a.model}] {kieu:10s} p={p:<4} {pt:5s} MRR={tb['mrr']:.4f} "
                      f"Δ={tb['delta_mrr']:+.4f} ({tb['giay']:.0f}s)", flush=True)
                json.dump(ket_qua, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
