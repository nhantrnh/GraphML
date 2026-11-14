"""Chạy phantom-parameter audit trên chính mô hình APKGC (AAAI 2025).

Khác với bản minh hoạ trong audit_gradient.py, script này nạp lớp APKGC thật
từ repo tác giả và chạy một lượt huấn luyện thu nhỏ, nên kết quả là bằng chứng
trực tiếp chứ không phải tái hiện.

Cách chạy:
    APKGC_DEVICE=cpu python audit_apkgc_real.py --repo /duong/dan/toi/APKGC
"""

import argparse
import os
import os.path as osp
import sys
import types

import torch


def build_args(dataset: str, dim: int = 32):
    """Dựng namespace tham số tối thiểu mà lớp APKGC cần."""
    a = types.SimpleNamespace()
    a.dataset = dataset
    a.dim = dim
    a.num_proj = 1
    a.use_pool = 1
    a.add_noise = 1
    a.noise_update = "epoch"
    a.noise_ratio = 0.2
    a.mask_ratio = 0.7
    a.joint_way = "Mformer_hd_mean"
    a.hidden_size = dim * 2
    a.intermediate_size = dim * 2
    a.num_attention_heads = 2
    a.num_hidden_layers = 1
    a.use_intermediate = 0
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Thư mục gốc của repo APKGC")
    ap.add_argument("--dataset", default="MKG-W")
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--steps", type=int, default=5)
    args_cli = ap.parse_args()

    repo = osp.abspath(args_cli.repo)
    sys.path.insert(0, repo)
    sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
    os.chdir(repo)

    from mmkgc.module.model import APKGC          # noqa: E402
    from audit_gradient import audit              # noqa: E402

    n_ent, n_rel = 15000, 169
    dim = args_cli.dim

    # Ưu tiên bộ nhúng thật nếu có; nếu không thì dùng vector ngẫu nhiên.
    # Đường đi của gradient do cấu trúc mã quyết định nên không phụ thuộc giá
    # trị đầu vào, nhưng dùng dữ liệu thật thì kết luận không còn chỗ để nghi
    # ngờ.
    emb_dir = osp.join(repo, "embeddings")
    vis_p = osp.join(emb_dir, f"{args_cli.dataset}-visual.pth")
    txt_p = osp.join(emb_dir, f"{args_cli.dataset}-textual.pth")

    if osp.exists(vis_p) and osp.exists(txt_p):
        img_emb = torch.load(vis_p, map_location="cpu", weights_only=False)
        text_emb = torch.load(txt_p, map_location="cpu", weights_only=False)
        n_ent = img_emb.shape[0]
        print(f"Dung bo nhung THAT: {args_cli.dataset}")
        print(f"  visual  {tuple(img_emb.shape)}")
        print(f"  textual {tuple(text_emb.shape)}")
    else:
        torch.manual_seed(0)
        img_emb = torch.randn(n_ent, 4096) * 0.1
        text_emb = torch.randn(n_ent, 768) * 0.1
        print("Khong tim thay bo nhung that, dung vector ngau nhien.")
    print()

    args = build_args(args_cli.dataset, dim)

    # Lớp APKGC đọc missing.json khi tên dataset chứa "MKG". Nếu không có file
    # đó thì dùng nhánh không-MKG để tránh phụ thuộc ngoài; nhánh này chỉ ảnh
    # hưởng cách lấp mô thái thiếu, không ảnh hưởng cấu trúc gradient.
    if not osp.exists(osp.join(emb_dir, "missing.json")):
        args.dataset = "DB15K"
        print("Khong co missing.json — dung nhanh khong-MKG.\n")

    model = APKGC(
        args=args, ent_tot=n_ent, rel_tot=n_rel,
        dim=dim, margin=12.0, epsilon=2.0,
        img_emb=img_emb, text_emb=text_emb,
    )

    batch = 128

    def forward_fn():
        model.update_noise()
        data = {
            "batch_h": torch.randint(0, n_ent, (batch,)),
            "batch_t": torch.randint(0, n_ent, (batch,)),
            "batch_r": torch.randint(0, n_rel, (batch,)),
            "mode": "normal",
        }
        score = model(data)
        return score.mean()

    report = audit(model, forward_fn, n_steps=args_cli.steps)
    report.summary()

    out = osp.join(osp.dirname(osp.abspath(__file__)), "audit_result.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Kết quả audit tham số — APKGC (AAAI 2025)\n\n")
        f.write(f"Mô hình thật từ `{repo}`, {report.n_steps} bước forward/backward.\n\n")
        if report.phantoms:
            f.write("## Tham số phantom\n\n")
            for p in report.phantoms:
                f.write(f"- `{p.name}` — khai báo `nn.Parameter`, "
                        f"`requires_grad=True`, nhưng không bao giờ nhận gradient.\n")
            f.write("\n")
        f.write("## Bảng đầy đủ\n\n")
        f.write(report.as_markdown() + "\n")
    print(f"Đã ghi: {out}")


if __name__ == "__main__":
    main()
