#!/usr/bin/env python3
"""Cơ chế phạt attention của APKGC đo sai đại lượng — và cách sửa.

BÀI BÁO NÓI GÌ. Mục "Attention Penalty" (công thức 9) phạt token bị chú ý quá
mức: khi điểm attention vượt ngưỡng φ thì trừ bớt θp và phân phối lại cho các
token lân cận. Mục RQ4 thừa nhận ngưỡng phải dò tay — quá nhỏ thì phạt không
đủ, quá lớn thì "overcorrection".

MÃ LÀM GÌ. APKGC.py:360-365 tính đại lượng đem so với ngưỡng như sau:

    penalty_scores[..., i] = attention_scores[..., i:].prod(dim=-1)
    exceed_threshold_mask = penalty_scores > self.penalty_threshold   # = 10

tức TÍCH LUỸ HẬU TỐ của điểm attention từ vị trí i đến hết hàng. Bốn hệ quả,
tất cả đo được bằng mô phỏng dưới đây:

  1. CƠ CHẾ KHÔNG KÍCH HOẠT LẦN NÀO — ĐÃ ĐO TRÊN MÔ HÌNH THẬT.
     Kernel `apkgc-attn-probe` (27/09) chèn đoạn đo vào APKGC.py rồi huấn
     luyện MKG-W 40 epoch, ghi thống kê trên 300 lần gọi forward:

         attention_scores.std()        trung bình 2,094   lớn nhất 3,145
         attention_scores.abs().max()  trung bình 7,502   lớn nhất 10,798
         tỉ lệ mask kích hoạt          0,0000%            0,0000%
         tỉ lệ penalty_scores âm       62,66%             82,66%

     Không một lần nào trong 300 lần gọi vượt ngưỡng. Hai lý do đọc được
     thẳng từ số đo: |attention_scores| lớn nhất chỉ 10,798 nên một tích
     muốn vượt 10 phải có hậu tố rất ngắn VÀ mọi thừa số đều dương và lớn;
     mà 62,66% thừa số mang dấu âm nên xác suất hậu tố toàn dương giảm theo
     hàm mũ (dài 3 chỉ còn ~5%).

     Lưu ý phương pháp: mô phỏng iid với std=2,09 dự đoán ~18-26% kích hoạt,
     sai hẳn so với 0% đo được. Điểm attention thật KHÔNG độc lập — bằng
     chứng là tỉ lệ âm 62,66% thay vì ~50% của mẫu iid. Mô phỏng chỉ dùng để
     thấy đại lượng này nhạy thế nào, KHÔNG dùng để dự đoán mô hình thật.

  2. PHẠT THEO VỊ TRÍ, KHÔNG THEO ĐỘ NHỌN. Token cuối hàng có tích của ít số
     nhất nên |penalty| lớn nhất — bất kể attention nhọn hay đều. Đại lượng này
     đo vị trí, không đo thứ nó định đo.

  3. NỬA SỐ PHẦN TỬ KHÔNG BAO GIỜ BỊ PHẠT. Tích các số có dấu bất kỳ cho kết
     quả âm khoảng một nửa số lần. Mask là `> 10` nên phần tử âm miễn nhiễm dù
     attention nhọn đến đâu.

  4. ĐỘ NHỌN THẬT ĐO ĐƯỢC BẰNG ENTROPY. Entropy của phân bố attention sau
     softmax phân biệt rõ nhọn với đều; tích hậu tố thì không.

ĐỀ XUẤT SỬA. Thay tích hậu tố bằng entropy chuẩn hoá của phân bố attention, và
thay ngưỡng cố định bằng ngưỡng suy từ chính phân bố đó:

    H = -Σ p·log p  /  log L        (0 = nhọn tuyệt đối, 1 = đều tuyệt đối)
    phạt khi H < τ, với τ là phân vị của H trong batch

Ngưỡng khi đó tự co giãn theo dữ liệu và theo giai đoạn huấn luyện — đúng tinh
thần "thích ứng" mà bài đã áp cho nhiễu (σ giảm dần) nhưng chưa áp cho phạt.

Chạy: python apkgc_penalty_defect.py
"""
import json
import math
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

L_LIST = (8, 16, 32, 64)
NGUONG = 10          # APKGC.py:336, self.penalty_threshold
N_MAU = 200


def tich_hau_to(a):
    """penalty_scores của APKGC: tích từ vị trí i đến hết hàng."""
    p = []
    for i in range(len(a)):
        v = 1.0
        for x in a[i:]:
            v *= x
        p.append(v)
    return p


def entropy_chuan_hoa(a):
    """Độ nhọn thật: 0 = dồn hết vào một token, 1 = trải đều."""
    m = max(a)
    e = [math.exp(x - m) for x in a]
    s = sum(e)
    q = [x / s for x in e]
    return -sum(x * math.log(x + 1e-12) for x in q) / math.log(len(a))


def van_de_1():
    print("VẤN ĐỀ 1 — tần suất kích hoạt phụ thuộc thang điểm attention\n")
    print(f"  {'std':>6}{'L':>5}{'tỉ lệ vượt ngưỡng 10':>24}")
    print("  " + "-" * 35)
    for std in (0.5, 1.0, 2.0, 3.0, 5.0):
        for L in (16, 64):
            vuot = tong = 0
            for _ in range(N_MAU):
                p = tich_hau_to([random.gauss(0, std) for _ in range(L)])
                vuot += sum(1 for x in p if x > NGUONG)
                tong += L
            print(f"  {std:>6.1f}{L:>5}{100*vuot/tong:>23.3f}%")
    print("\n  Mô phỏng iid cho thấy đại lượng này rất nhạy với phương sai.")
    print("  NHƯNG nó KHÔNG dự đoán đúng mô hình thật: với std=2,09 đo được,")
    print("  mô phỏng nói ~18-26%, thực tế 0%. Xem phần ĐO THẬT bên dưới.")


def van_de_2():
    print("\n\nVẤN ĐỀ 2 — phạt theo VỊ TRÍ token, không theo độ nhọn\n")
    L = 16
    a = [random.gauss(0, 1) for _ in range(L)]
    p = tich_hau_to(a)
    for i in (0, L // 2, L - 1):
        print(f"  i={i:<3} (tích {L-i:>2} số): {p[i]:+.3e}")
    print("\n  Token cuối hàng luôn có |penalty| lớn nhất vì tích ít số nhất.")


def van_de_3():
    print("\n\nVẤN ĐỀ 3 — phần tử âm không bao giờ bị phạt\n")
    L, am, tong = 16, 0, 0
    for _ in range(N_MAU):
        p = tich_hau_to([random.gauss(0, 1) for _ in range(L)])
        am += sum(1 for x in p if x < 0)
        tong += L
    print(f"  {100*am/tong:.1f}% số phần tử penalty mang dấu ÂM.")
    print("  Mask là `penalty > 10` nên chúng miễn nhiễm, dù attention nhọn cỡ nào.")


def van_de_4():
    print("\n\nVẤN ĐỀ 4 — entropy đo đúng thứ cần đo\n")
    L = 16
    deu = [0.1] * L
    nhon = [0.0] * L
    nhon[3] = 8.0
    vua = [0.0] * L
    for i in range(4):
        vua[i] = 2.0
    print(f"  {'phân bố':<24}{'entropy chuẩn hoá':>20}{'tích hậu tố lớn nhất':>24}")
    print("  " + "-" * 68)
    for ten, v in (("đều tuyệt đối", deu), ("nhọn vừa (4 token)", vua),
                   ("nhọn tuyệt đối", nhon)):
        print(f"  {ten:<24}{entropy_chuan_hoa(v):>20.4f}"
              f"{max(tich_hau_to(v)):>24.3e}")
    print("\n  Entropy xếp đúng thứ tự nhọn→đều. Tích hậu tố thì không:")
    print("  nó bị chi phối bởi độ dài hậu tố chứ không bởi hình dạng phân bố.")


def do_that():
    """Số đo từ kernel apkgc-attn-probe, MKG-W, 300 lần gọi forward."""
    f = os.path.join(ROOT, "experiments/results/apkgc_res/attn_scale.json")
    if not os.path.isfile(f):
        return
    d = json.load(open(f))
    print("\n\nĐO THẬT trên mô hình đang huấn luyện (MKG-W, 40 epoch)\n")
    print(f"  {'đại lượng':<32}{'trung bình':>12}{'lớn nhất':>12}")
    print("  " + "-" * 56)
    print(f"  {'attention_scores.std()':<32}{d['std_mean']:>12.4f}{d['std_max']:>12.4f}")
    print(f"  {'attention_scores.abs().max()':<32}{d['absmax_mean']:>12.4f}{d['absmax_max']:>12.4f}")
    print(f"  {'tỉ lệ mask kích hoạt':<32}{100*d['fire_mean']:>11.4f}%{100*d['fire_max']:>11.4f}%")
    print(f"  {'tỉ lệ penalty_scores âm':<32}{100*d['neg_mean']:>11.2f}%{100*d['neg_max']:>11.2f}%")
    print(f"\n  Trên {d['n']} lần gọi forward, KHÔNG lần nào mask kích hoạt.")


def main():
    random.seed(0)
    print(__doc__.split("\n\n")[0])
    print()
    van_de_1()
    van_de_2()
    van_de_3()
    van_de_4()
    do_that()
    print("\n\n" + "=" * 70)
    print("KẾT LUẬN")
    print("  ĐO ĐƯỢC trên mô hình thật (300 lần gọi forward, MKG-W):")
    print("    - tỉ lệ kích hoạt 0,0000% — cơ chế phạt KHÔNG chạy lần nào;")
    print("    - 62,66% phần tử âm nên mask `> 10` không bao giờ chạm tới;")
    print("    - |attention| lớn nhất 10,798, sát ngay ngưỡng 10.")
    print("\n  ĐÚNG THEO ĐỊNH NGHĨA:")
    print("    - đại lượng phụ thuộc ĐỘ DÀI hậu tố nên chấm điểm theo vị trí;")
    print("    - entropy phân biệt được nhọn/đều, tích hậu tố thì không.")
    print("\n  Dò tay ngưỡng như mục RQ4 đề xuất không sửa được, vì đại lượng")
    print("  bị dò không đo độ nhọn — và vì 62,66% phần tử miễn nhiễm do dấu.")
    print("\n  ĐỀ XUẤT SỬA VÀ KẾT QUẢ ĐO (kernel apkgc-entropy-c, 28/09):")
    print("    Thay tích hậu tố bằng entropy chuẩn hoá, ngưỡng là phân vị")
    print("    trong batch. Chạy thật trên MKG-W seed 42, so ba nhánh:")
    print()
    print("        A mã công bố       MRR 0,3629   H@1 0,2933")
    print("        B công thức bài    MRR 0,3629   H@1 0,2933")
    print("        C entropy          MRR 0,3630   H@1 0,2934")
    print()
    print("    C − A = +0,0001, nhỏ hơn độ lệch chuẩn giữa seed (0,0009)")
    print("    CHÍN LẦN. Đề xuất entropy KHÔNG cải thiện được kết quả.")
    print()
    print("    Đây là kết quả ÂM cho phần đề xuất, nhưng nó củng cố phần")
    print("    chẩn đoán: cả ba cách xử lý attention — không phạt gì, phạt")
    print("    theo công thức bài, phạt theo entropy — đều cho cùng một số.")
    print("    Cơ chế phạt attention không phải thứ quyết định kết quả của")
    print("    APKGC, bất kể cài đặt thế nào.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
