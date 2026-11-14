"""Vector hoá phép tính tích luỹ hậu tố trong Attention Penalty của APKGC.

Bản gốc (mmkgc/module/model/APKGC.py, dòng 361-363):

    penalty_scores = torch.zeros_like(attention_scores)
    for i in range(attention_scores.shape[-1]):
        penalty_scores[..., i] = attention_scores[..., i:].prod(dim=-1)

Vòng lặp này gọi kernel L lần, mỗi lần tính tích trên một lát cắt khác nhau.
Cùng phép toán đó viết được bằng một lời gọi `cumprod`:

    penalty_scores = attention_scores.flip(-1).cumprod(-1).flip(-1)

Script kiểm chứng hai điều: kết quả trùng khớp về số học, và mức chênh lệch
tốc độ. Với APKGC hiện tại chiều cuối chỉ bằng 3 (ba mô thái) nên lợi ích thực
tế nhỏ; phần đo với chuỗi dài hơn cho thấy điều gì xảy ra nếu mở rộng số mô
thái hoặc chuyển sang biểu diễn dạng token.
"""

import time
import torch


def suffix_prod_loop(a: torch.Tensor) -> torch.Tensor:
    """Bản gốc: vòng lặp Python trên chiều cuối."""
    out = torch.zeros_like(a)
    for i in range(a.shape[-1]):
        out[..., i] = a[..., i:].prod(dim=-1)
    return out


def suffix_prod_vec(a: torch.Tensor) -> torch.Tensor:
    """Bản vector hoá: lật, tích luỹ, lật lại."""
    return a.flip(-1).cumprod(-1).flip(-1)


def check_equivalence() -> bool:
    """Đối chiếu hai cách tính trên nhiều hình dạng và kiểu dữ liệu."""
    print("=" * 66)
    print("KIỂM CHỨNG TÍNH TƯƠNG ĐƯƠNG")
    print("=" * 66)

    shapes = [
        (2, 2, 3, 3),      # đúng hình dạng của APKGC: batch, head, 3 mô thái
        (8, 4, 3, 3),
        (4, 2, 16, 16),    # chuỗi dài hơn
        (1, 1, 64, 64),
    ]

    all_ok = True
    for shp in shapes:
        torch.manual_seed(0)
        # giá trị quanh 1 để tích không tràn số
        a = torch.randn(*shp) * 0.3 + 1.0

        r1 = suffix_prod_loop(a)
        r2 = suffix_prod_vec(a)
        diff = (r1 - r2).abs().max().item()
        ok = torch.allclose(r1, r2, rtol=1e-5, atol=1e-7)
        all_ok &= ok

        print(f"  {str(shp):<18} sai lệch lớn nhất = {diff:.3e}   "
              f"{'khớp' if ok else 'KHÁC NHAU'}")

    # Kiểm tra thêm với giá trị âm và giá trị lớn, vì tích dễ mất chính xác
    print()
    for name, gen in [("có giá trị âm", lambda: torch.randn(4, 2, 8, 8)),
                      ("biên độ lớn", lambda: torch.randn(4, 2, 8, 8) * 5)]:
        torch.manual_seed(1)
        a = gen()
        r1, r2 = suffix_prod_loop(a), suffix_prod_vec(a)
        rel = ((r1 - r2).abs() / (r1.abs() + 1e-12)).max().item()
        ok = torch.allclose(r1, r2, rtol=1e-4, atol=1e-6)
        all_ok &= ok
        print(f"  {name:<18} sai lệch tương đối = {rel:.3e}   "
              f"{'khớp' if ok else 'KHÁC NHAU'}")

    print()
    print("  Kết luận: hai cách tính " +
          ("cho cùng kết quả." if all_ok else "KHÔNG tương đương."))
    return all_ok


def benchmark(n_rep: int = 50) -> None:
    """Đo thời gian trên nhiều độ dài chuỗi."""
    print()
    print("=" * 66)
    print(f"ĐO TỐC ĐỘ  ({n_rep} lượt mỗi cấu hình)")
    print("=" * 66)
    print(f"  {'hình dạng':<22}{'vòng lặp':>12}{'vector hoá':>13}{'nhanh hơn':>12}")
    print("  " + "-" * 57)

    configs = [
        (1024, 2, 3, 3),      # cấu hình APKGC thật: batch 1024, 2 head, 3 mô thái
        (1024, 2, 8, 8),
        (1024, 2, 32, 32),
        (256, 4, 128, 128),
    ]

    for shp in configs:
        torch.manual_seed(0)
        a = torch.randn(*shp) * 0.3 + 1.0

        # chạy nóng máy trước khi đo
        suffix_prod_loop(a); suffix_prod_vec(a)

        t0 = time.perf_counter()
        for _ in range(n_rep):
            suffix_prod_loop(a)
        t_loop = (time.perf_counter() - t0) / n_rep

        t0 = time.perf_counter()
        for _ in range(n_rep):
            suffix_prod_vec(a)
        t_vec = (time.perf_counter() - t0) / n_rep

        speedup = t_loop / t_vec if t_vec > 0 else float("inf")
        print(f"  {str(shp):<22}{t_loop*1000:>10.3f} ms{t_vec*1000:>11.3f} ms"
              f"{speedup:>10.1f}x")

    print()
    print("  Chiều cuối của APKGC hiện tại bằng 3, nên lợi ích thực tế nhỏ.")
    print("  Mức chênh lệch tăng theo độ dài chuỗi, đáng kể nếu về sau mở rộng")
    print("  số mô thái hoặc chuyển sang biểu diễn dạng token.")


def patch_preview() -> None:
    """In ra thay đổi cần áp dụng vào mã nguồn."""
    print()
    print("=" * 66)
    print("THAY ĐỔI CẦN ÁP DỤNG")
    print("=" * 66)
    print("  Tệp: mmkgc/module/model/APKGC.py, dòng 361-363")
    print()
    print("  Trước:")
    print("      penalty_scores = torch.zeros_like(attention_scores)")
    print("      for i in range(attention_scores.shape[-1]):")
    print("          penalty_scores[..., i] = attention_scores[..., i:].prod(dim=-1)")
    print()
    print("  Sau:")
    print("      penalty_scores = attention_scores.flip(-1).cumprod(-1).flip(-1)")


if __name__ == "__main__":
    ok = check_equivalence()
    benchmark()
    patch_preview()
    print()
    if not ok:
        raise SystemExit("Hai cách tính không tương đương — không nên áp dụng.")
