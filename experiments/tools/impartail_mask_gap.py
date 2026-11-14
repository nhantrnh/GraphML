#!/usr/bin/env python3
"""Đo khoảng cách giữa công thức (5) của ImparTail và mã đã công bố.

Bài báo, mục 4.2.1, công thức (5):

    L_ijk = (1 - m_ijk) * L_ijk = { L_ijk nếu m=0 ; 0 nếu m=1 }

kèm câu "A mask initialized i.e. setting m=1 voids the contribution of loss".
Tức nhãn bị che phải đóng góp ĐÚNG BẰNG 0 vào hàm mất mát.

Mã đã công bố, nhánh BCE (stsg_base.py:305-318):

    gt_masked = gt[mask == 1]
    label = torch.zeros([max_len], ...)
    label[gt_masked] = 1

dựng vector đích toàn 0 rồi chỉ set 1 cho các nhãn SỐNG SÓT. Nhãn dương bị che
trở thành 0 — tức nhãn ÂM — và vì nn.BCELoss() (train_sgg_base.py:50) dùng
reduction='mean' trên toàn bộ max_len phần tử, vị trí đó vẫn sinh gradient.

Hai sai lệch tách biệt:
  (1) dấu    — nhãn bị che bị đẩy xác suất XUỐNG thay vì được bỏ qua;
  (2) mẫu số — chia cho max_len thay vì số phần tử còn hiệu lực, làm co mọi
      gradient còn lại.

BCE có đạo hàm dạng đóng nên script không cần torch:
    L = -[y log p + (1-y) log(1-p)]   ->   dL/dp = (p - y) / (p(1-p))

Đây là mô tả MÃ ĐÃ CÔNG BỐ đối chiếu với CÔNG THỨC ĐÃ CÔNG BỐ, không phải kết
luận về việc tác giả đã chạy thí nghiệm thế nào.
"""
import random

# contacting_relationship có 17 lớp (train_sgg_base.py:354-360); spatial có 6
MAX_LEN = 17


def dbce(p, y):
    """Đạo hàm BCE theo xác suất dự đoán."""
    return (p - y) / (p * (1 - p))


def main():
    random.seed(0)
    p = [round(random.uniform(0.05, 0.95), 3) for _ in range(MAX_LEN)]
    duong = [3, 9, 14]      # nhãn dương thật của một cặp (người, vật)
    che = [9, 14]           # hai trong số đó bị mặt nạ che
    song = [i for i in duong if i not in che]

    print(__doc__.split("\n\n")[0])
    print(f"\nVí dụ: {MAX_LEN} lớp, nhãn dương {duong}, bị che {che}\n")
    print(f"{'lớp':>5}{'p':>8}{'nhãn (mã)':>12}{'grad (mã)':>12}{'grad đúng':>12}")
    print("-" * 49)

    n_dung = MAX_LEN - len(che)
    for i in che:
        g_ma = dbce(p[i], 0.0) / MAX_LEN     # mã coi là nhãn âm
        print(f"{i:>5}{p[i]:>8.3f}{'0 = âm':>12}{g_ma:>+12.4f}{0.0:>+12.4f}")

    print("\n(1) Dấu: mã đẩy xác suất các lớp bị che XUỐNG "
          "(gradient dương = giảm p);\n    công thức (5) yêu cầu gradient bằng 0. "
          "Mô hình bị dạy rằng quan hệ\n    đó KHÔNG tồn tại, thay vì không học gì từ nó.")

    heso = n_dung / MAX_LEN
    print(f"\n(2) Mẫu số: mã chia {MAX_LEN}, đúng ra chia {n_dung}. "
          f"Mọi gradient còn lại\n    bị co lại hệ số {heso:.3f} "
          f"(nhãn sống sót {song} cũng bị ảnh hưởng).")

    print("\nHệ quả: tỉ lệ che càng cao thì lệch càng lớn — đúng chiều mà phương")
    print("pháp hướng tới, vì lớp head là lớp bị che nhiều nhất.")
    print("\nSửa: nn.BCELoss(reduction='none'), nhân mặt nạ theo từng phần tử,")
    print("rồi chuẩn hoá theo số phần tử còn hiệu lực.")


if __name__ == "__main__":
    main()
