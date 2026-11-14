#!/usr/bin/env python3
"""Sinh type_constrain.txt theo đúng định dạng OpenKE.

Vì sao cần: TestDataLoader mặc định type_constrain=True và gọi importTypeFiles(),
hàm này fopen("type_constrain.txt") KHÔNG kiểm tra NULL (Reader.h:275). Thiếu tệp
=> segfault, không có thông báo lỗi. MKG-W có sẵn tệp này nên chạy được.

Định dạng: dòng đầu là relationTotal; sau đó mỗi quan hệ hai dòng
  <rel> <số head> <danh sách head id...>
  <rel> <số tail> <danh sách tail id...>
Tệp phải liệt kê ĐỦ relationTotal quan hệ, kể cả quan hệ không có bộ ba nào
(khi đó số lượng bằng 0), nếu không vòng lặp fscanf sẽ lệch.
"""
import sys, os, collections

def gen(ds_dir):
    nr = int(open(f"{ds_dir}/relation2id.txt").readline())
    heads = collections.defaultdict(set); tails = collections.defaultdict(set)
    # chỉ dùng train — dùng valid/test sẽ là rò rỉ
    L = open(f"{ds_dir}/train2id.txt").read().split("\n")[1:]
    for l in L:
        if not l.strip(): continue
        h, t, r = map(int, l.split())
        heads[r].add(h); tails[r].add(t)
    out = [str(nr)]
    for r in range(nr):
        hs = sorted(heads[r]); ts = sorted(tails[r])
        out.append("\t".join([str(r), str(len(hs))] + [str(x) for x in hs]))
        out.append("\t".join([str(r), str(len(ts))] + [str(x) for x in ts]))
    open(f"{ds_dir}/type_constrain.txt", "w").write("\n".join(out) + "\n")
    empty = sum(1 for r in range(nr) if not heads[r])
    print(f"{ds_dir}: {nr} quan hệ, {empty} quan hệ không có bộ ba train")
    return nr

if __name__ == "__main__":
    gen(sys.argv[1] if len(sys.argv) > 1 else
        "/Users/nhantt/Downloads/MS_HCMUS/Study/HP3/GraphML/upstream/APKGC/benchmarks/ViMMKG")
