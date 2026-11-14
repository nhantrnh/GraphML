#!/usr/bin/env python3
"""Dựng bộ ViMMKG theo đúng định dạng OpenKE mà APKGC/MyGO/MoMoK/AdaMF-MAT/DHNS dùng.

Chạy sau khi triples.py đã tạo triples_raw.json.
Chia train/valid/test theo tỉ lệ 8/1/1, KHÔNG rò rỉ: mọi thực thể trong
valid/test đều phải xuất hiện trong train (nếu không, link prediction vô nghĩa).
"""
import json, os, random, collections, argparse

D = os.path.expanduser("~/.cache/hanviet")

def load():
    t = json.load(open(f"{D}/triples_raw.json"))
    # bỏ self-loop: (h,r,h) không mang thông tin cho link prediction
    return [tuple(x) for x in t["inner"] if x[0] != x[2]]

def dedup_inverse(triples):
    """Gộp cặp nghịch đảo (h,r,t)+(t,r,h) về một hướng.

    Vì sao bắt buộc: quan hệ đối xứng như P190 (thành phố kết nghĩa, 83,5% đối
    xứng) và P47 (giáp ranh, 94,6%) tạo rò rỉ — nếu (h,r,t) ở train và (t,r,h) ở
    test, mô hình chỉ cần học "r đối xứng" là đoán đúng. Đo được 27,3% test bị
    rò rỉ theo kiểu này trước khi sửa. Đây đúng là lỗi từng buộc cộng đồng thay
    FB15k bằng FB15k-237 (Toutanova & Chen 2015).
    """
    seen, out = set(), []
    for h, p_, t in sorted(triples):
        if (t, p_, h) in seen: continue
        seen.add((h, p_, t)); out.append((h, p_, t))
    return out

def prune(triples, min_deg):
    """Lọc lặp cho tới khi mọi thực thể có bậc >= min_deg (k-core)."""
    tr = set(triples)
    while True:
        deg = collections.Counter()
        for h, _, t in tr:
            deg[h] += 1; deg[t] += 1
        drop = {e for e, d in deg.items() if d < min_deg}
        if not drop: return sorted(tr)
        tr = {(h, p, t) for h, p, t in tr if h not in drop and t not in drop}
        if not tr: return []

def split(triples, seed=42, ratio=(0.8, 0.1, 0.1)):
    rng = random.Random(seed)
    tr = list(triples); rng.shuffle(tr)
    n = len(tr); n_tr = int(n * ratio[0]); n_va = int(n * ratio[1])
    train, valid, test = tr[:n_tr], tr[n_tr:n_tr+n_va], tr[n_tr+n_va:]
    # đẩy về train mọi bộ ba chứa thực thể/quan hệ chưa từng thấy trong train
    seen_e = {x for h,_,t in train for x in (h,t)}
    seen_r = {p for _,p,_ in train}
    def fix(part):
        keep, push = [], []
        for h,p,t in part:
            if h in seen_e and t in seen_e and p in seen_r: keep.append((h,p,t))
            else:
                push.append((h,p,t)); seen_e.update((h,t)); seen_r.add(p)
        return keep, push
    v_keep, v_push = fix(valid); t_keep, t_push = fix(test)
    return train + v_push + t_push, v_keep, t_keep

def write(out, train, valid, test):
    os.makedirs(out, exist_ok=True)
    ents = sorted({x for s in (train,valid,test) for h,_,t in s for x in (h,t)})
    rels = sorted({p for s in (train,valid,test) for _,p,_ in s})
    e2i = {e:i for i,e in enumerate(ents)}; r2i = {r:i for i,r in enumerate(rels)}
    for name, m in (("entity2id.txt", e2i), ("relation2id.txt", r2i)):
        with open(f"{out}/{name}", "w") as f:
            f.write(f"{len(m)}\n")
            for k,v in m.items(): f.write(f"{k}\t{v}\n")
    # OpenKE: mỗi dòng "h t r"
    for name, s in (("train2id.txt",train), ("valid2id.txt",valid), ("test2id.txt",test)):
        with open(f"{out}/{name}", "w") as f:
            f.write(f"{len(s)}\n")
            for h,p,t in s: f.write(f"{e2i[h]} {e2i[t]} {r2i[p]}\n")
    return len(ents), len(rels)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-deg", type=int, default=5)
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/hanviet/ViMMKG"))
    a = ap.parse_args()
    raw = load(); print(f"bộ ba trong tập      : {len(raw):,}")
    dd = dedup_inverse(raw); print(f"sau gộp nghịch đảo   : {len(dd):,}  (bỏ {len(raw)-len(dd):,})")
    pr = prune(dd, a.min_deg); print(f"sau k-core (deg>={a.min_deg}) : {len(pr):,}")
    if not pr: raise SystemExit("k-core rỗng — giảm --min-deg")
    tr, va, te = split(pr)
    ne, nr = write(a.out, tr, va, te)
    print(f"ent={ne:,} rel={nr:,} train={len(tr):,} valid={len(va):,} test={len(te):,}")
    print(f"bậc trung bình = {2*len(tr)/ne:.2f}   (MKG-W 4.56, DB15K 12.34)")
    print(f"-> {a.out}")
