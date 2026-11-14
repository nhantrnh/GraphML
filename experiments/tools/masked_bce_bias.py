#!/usr/bin/env python3
"""Độ chệch của ước lượng gradient trong masked BCE — mệnh đề chứng minh được.

Khác với các công cụ khác trong thư mục này (đo và đếm), đây là một dẫn xuất
giải tích: kết luận đúng cho mọi mô hình, mọi dữ liệu, không phụ thuộc thí
nghiệm nào.

BỐI CẢNH. Công thức (5) của ImparTail nói nhãn bị che phải đóng góp 0 vào hàm
mất mát. Mã công bố (stsg_base.py:305-318) thay vào đó dựng vector đích toàn 0
rồi set 1 cho nhãn sống sót, nên nhãn dương bị che trở thành nhãn ÂM và vẫn
sinh gradient.

PHÂN BIỆT HAI KIỂU CHE — chỗ dễ nhầm nhất.

  (A) Che = LOẠI số hạng khỏi hàm mất mát. Nếu mặt nạ độc lập với tham số và
      dữ liệu thì E[∇L̃] = (1-ρ)·∇L: ước lượng KHÔNG chệch hướng, chỉ khác một
      hệ số vô hướng đã biết, sửa được bằng cách chia cho (1-ρ). Toàn bộ vấn
      đề khi đó nằm ở PHƯƠNG SAI ρ(1-ρ)·Σ(∇ℓ_k)², không ở độ chệch. Đây là
      kết quả chuẩn trong học đa nhãn với nhãn thiếu MCAR (Schultheis &
      Babbar, TMLR 2025; Qaraei et al., WWW 2021).

  (B) Che = BIẾN nhãn dương THÀNH nhãn âm. Đây mới là điều mã công bố làm
      (stsg_base.py:315-316 dựng torch.zeros rồi chỉ set 1 cho nhãn sống sót).
      Kết quả khác hẳn: gradient ĐỔI DẤU, không phải co lại.

  Kiểm số (q=0.6, tỉ lệ giữ p=0.1, 200.000 mẫu):
      gradient đúng (y=1)        −1,667   (đẩy xác suất LÊN)
      (A) loại số hạng           −0,168 = 0,101 × đúng   -> cùng dấu
      (B) biến thành âm (mã)     +2,084 = −1,250 × đúng  -> NGƯỢC DẤU

  Vì vậy KHÔNG được viện dẫn lý thuyết dropout hay MCAR để nói mã này không
  chệch: cơ chế khác nhau. Dropout gây chệch qua bất đẳng thức Jensen BÊN
  TRONG hàm phi tuyến; ở đây mặt nạ nằm ngoài, nhưng nó đổi GIÁ TRỊ NHÃN chứ
  không bỏ số hạng — nên sinh chệch theo một đường hoàn toàn khác.

MỆNH ĐỀ 1 (độ chệch không phụ thuộc mô hình).
Xét một nhãn dương c, mô hình cho xác suất q_c ∈ (0,1), mặt nạ giữ nhãn đó với
xác suất p. Với BCE, ∂L/∂q = (q - y)/(q(1-q)).

  Mã công bố:  y_c ~ Bernoulli(p),  nên  E[∂L/∂q_c] ∝ (q_c - p)
  Công thức (5): nhãn bị che bị loại,      ∂L/∂q_c  ∝ (q_c - 1)

  Độ chệch tử số = (q_c - p) - (q_c - 1) = 1 - p.

Độ chệch bằng đúng TỈ LỆ CHE và KHÔNG phụ thuộc q_c. Nghĩa là không mô hình
nào, dù huấn luyện tốt đến đâu, làm nó nhỏ đi được — và vì nó cộng vào tử số
chứ không nhân, nó đổi được cả DẤU của gradient khi q_c < p, chứ không chỉ
làm yếu đi.

MỆNH ĐỀ 2 (độ chệch không đều theo lớp — chỗ gây hại thật).
Nếu độ chệch như nhau cho mọi lớp thì nó chỉ là một phép đổi thang, vô hại về
mặt thứ tự. Nhưng Algorithm 1 đặt Prob[rel] = 1/|P| nên mỗi lớp nhận cùng hạn
ngạch, kẹp theo số mẫu sẵn có:

    p_c = min(quota, n_c) / n_c,   quota = ρ·N/|P|

Lớp nhiều mẫu có p_c nhỏ (bị che gần hết), lớp hiếm có p_c = 1 (giữ trọn).
Do đó độ chệch 1 - p_c biến thiên theo lớp, và gradient giữa các lớp bị bóp
méo KHÁC NHAU. Đây là thiên lệch có hướng, không phải nhiễu.

Hệ quả kiểm chứng được: thiên lệch lớn nhất rơi đúng vào lớp head — tức
phương pháp vốn nhằm giảm ưu thế của lớp head lại làm sai lệch gradient của
chính chúng mạnh nhất.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "experiments/results/impartail_repro/ag_predicate_freq.json")
TI_LE_GIU = 0.10           # cấu hình S=10 của bài báo
NHOM = "contacting_relationship"


def main():
    if not os.path.isfile(DATA):
        sys.exit(f"không thấy {DATA}")
    c = json.load(open(DATA))["tan_suat"][NHOM]

    print(__doc__.split("\n\n")[0])
    print("\nMỆNH ĐỀ 1 — độ chệch = tỉ lệ che, độc lập với mô hình\n")
    print(f"  {'tỉ lệ giữ p':>14}{'độ chệch 1-p':>16}")
    for p in (0.9, 0.7, 0.4, 0.1):
        print(f"  {p:>14.1f}{1 - p:>16.1f}")
    print(f"\n  Cấu hình chính của bài báo giữ {TI_LE_GIU:.0%} → chệch ≈ {1-TI_LE_GIU:.1f},"
          f" gần tối đa.")

    n = sum(c.values())
    quota = n * TI_LE_GIU / len(c)
    print(f"\n\nMỆNH ĐỀ 2 — độ chệch KHÔNG đều theo lớp ({NHOM})\n")
    print(f"  {'lớp':<22}{'tần suất':>10}{'p_c':>8}{'chệch':>9}")
    print("  " + "-" * 49)
    ps = []
    for k, v in sorted(c.items(), key=lambda x: -x[1]):
        p = min(quota, v) / v
        ps.append(p)
        print(f"  {k:<22}{v:>10,}{p:>8.3f}{1 - p:>9.3f}")

    print("  " + "-" * 49)
    print(f"  chệch từ {1 - max(ps):.3f} (lớp hiếm) đến {1 - min(ps):.3f} (lớp phổ biến),"
          f" chênh {max(ps) - min(ps):.3f}")

    tron = sum(1 for p in ps if p >= 1.0)
    print(f"\n  {tron}/{len(c)} lớp giữ trọn vẹn (p_c = 1) → chệch bằng 0 ở đuôi,")
    print(f"  gần cực đại ở đầu. Gradient giữa các lớp bị bóp méo khác nhau,")
    print(f"  nên đây là thiên lệch CÓ HƯỚNG, không phải phép đổi thang vô hại.")

    print("\n  Nghịch lý đáng nêu trong bài: phương pháp nhằm giảm ưu thế của lớp")
    print("  head lại làm sai lệch gradient của chính lớp head mạnh nhất.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
