#!/usr/bin/env python3
"""Trích đặc trưng thị giác (CLIP ViT-B/32) và văn bản (multilingual-e5-large)
cho ViMMKG, xuất đúng định dạng .pth mà APKGC nạp.

Thực thể thiếu mô thái để nguyên vector 0 — đúng như MKG-W, vốn có 537 hàng ảnh
và 851 hàng văn bản toàn 0.
"""
import torch, os, json, glob, time, argparse
from PIL import Image

D = os.path.expanduser("~/.cache/hanviet")

def ent_list(ds):
    L = open(f"{D}/{ds}/entity2id.txt").read().split("\n")[1:]
    out = [None] * int(open(f"{D}/{ds}/entity2id.txt").readline())
    for l in L:
        if not l.strip(): continue
        q, i = l.rsplit("\t", 1) if "\t" in l else l.rsplit(" ", 1)
        out[int(i)] = q
    assert all(x is not None for x in out), "entity2id.txt có lỗ hổng chỉ số"
    return out

@torch.no_grad()
def visual(ents, bs=64):
    from transformers import CLIPModel, CLIPImageProcessor
    m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    p = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    out = torch.zeros(len(ents), m.config.projection_dim)
    idx, ims = [], []
    n_ok = n_bad = 0
    t0 = time.time()
    def flush():
        nonlocal idx, ims
        if not ims: return
        x = p(images=ims, return_tensors="pt")
        out[idx] = m.get_image_features(**x).pooler_output
        idx, ims = [], []
    for i, q in enumerate(ents):
        f = f"{D}/images/{q}.jpg"
        if not os.path.exists(f): continue
        try:
            ims.append(Image.open(f).convert("RGB")); idx.append(i); n_ok += 1
        except Exception:
            n_bad += 1; continue
        if len(ims) >= bs:
            flush()
            if n_ok % 1024 < bs:
                print(f"  ảnh {n_ok:>6}/{len(ents)}  {(time.time()-t0)/60:.1f}ph", flush=True)
    flush()
    print(f"  ảnh: {n_ok:,} trích được, {n_bad} hỏng, {len(ents)-n_ok:,} để vector 0")
    return out

@torch.no_grad()
def textual(ents, bs=16, maxlen=256):
    from transformers import AutoTokenizer, AutoModel
    tk = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
    m = AutoModel.from_pretrained("intfloat/multilingual-e5-large").eval()
    ex = json.load(open(f"{D}/viwiki_extracts.json"))
    ti = json.load(open(f"{D}/viwiki_titles.json"))
    out = torch.zeros(len(ents), m.config.hidden_size)
    idx, txt = [], []
    n_ok = 0; t0 = time.time()
    def flush():
        nonlocal idx, txt
        if not txt: return
        b = tk(txt, return_tensors="pt", padding=True, truncation=True, max_length=maxlen)
        h = m(**b).last_hidden_state
        msk = b["attention_mask"].unsqueeze(-1).float()
        out[idx] = (h * msk).sum(1) / msk.sum(1)
        idx, txt = [], []
    for i, q in enumerate(ents):
        t = ex.get(q)
        if not t:
            lb = (ti.get(q) or {}).get("label")   # dự phòng: dùng nhãn
            if not lb: continue
            t = lb
        # e5 yêu cầu tiền tố
        txt.append("passage: " + t[:2000]); idx.append(i); n_ok += 1
        if len(txt) >= bs:
            flush()
            if n_ok % 1024 < bs:
                print(f"  vbản {n_ok:>6}/{len(ents)}  {(time.time()-t0)/60:.1f}ph", flush=True)
    flush()
    print(f"  văn bản: {n_ok:,} trích được, {len(ents)-n_ok:,} để vector 0")
    return out

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--dataset", default="ViMMKG_d3")
    a.add_argument("--out", default="/Users/nhantt/Downloads/MS_HCMUS/Study/HP3/GraphML/upstream/APKGC/embeddings")
    a.add_argument("--name", default="ViMMKG")
    a.add_argument("--only", choices=["visual","textual"], default=None)
    a = a.parse_args()
    ents = ent_list(a.dataset); print(f"{len(ents):,} thực thể")
    os.makedirs(a.out, exist_ok=True)
    if a.only in (None, "visual"):
        v = visual(ents); torch.save(v, f"{a.out}/{a.name}-visual.pth")
        print(f"-> {a.name}-visual.pth {tuple(v.shape)}  hàng 0: {(v.abs().sum(1)==0).sum().item():,}")
    if a.only in (None, "textual"):
        t = textual(ents); torch.save(t, f"{a.out}/{a.name}-textual.pth")
        print(f"-> {a.name}-textual.pth {tuple(t.shape)}  hàng 0: {(t.abs().sum(1)==0).sum().item():,}")
