#!/usr/bin/env python3
"""Sinh embeddings/missing.json cho ViMMKG theo ĐÚNG dữ liệu thật.

APKGC.py:81 rẽ nhánh theo `"MKG" in args.dataset`. Tên "ViMMKG" có chứa "MKG"
nên sẽ đi nhánh này và đọc missing.json để biết thực thể nào thiếu mô thái, rồi
thay bằng nhiễu Gauss lấy từ thống kê của các thực thể CÓ mô thái.

Khác biệt phải ghi rõ trong báo cáo: missing.json trong kho mã công bố liệt kê
2 và 1 chỉ số cho MKG-W, trong khi embedding MKG-W thật có 537 và 851 hàng toàn
0 — và các chỉ số được liệt kê thậm chí không phải hàng 0. Tệp cho ViMMKG này
liệt kê ĐÚNG các hàng toàn 0, nên cách xử lý mô thái thiếu của ViMMKG KHÔNG so
sánh trực tiếp được với số MKG-W đã công bố.
"""
import torch, json, os, sys

EMB = "/Users/nhantt/Downloads/MS_HCMUS/Study/HP3/GraphML/upstream/APKGC/embeddings"

def main(name="ViMMKG"):
    path = f"{EMB}/missing.json"
    data = json.load(open(path)) if os.path.exists(path) else {}
    for mod in ("visual", "textual"):
        f = f"{EMB}/{name}-{mod}.pth"
        if not os.path.exists(f):
            print(f"  bỏ qua {mod}: chưa có {f}"); continue
        e = torch.load(f, map_location="cpu")
        zero = (e.abs().sum(1) == 0).nonzero().flatten().tolist()
        data[f"{name}-{mod}.pth"] = zero
        print(f"  {name}-{mod}.pth: {len(zero):,}/{e.shape[0]:,} hàng toàn 0 "
              f"({len(zero)/e.shape[0]*100:.1f}%)")
    json.dump(data, open(path, "w"))
    print(f"-> {path}  (khoá: {list(data)})")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ViMMKG")
