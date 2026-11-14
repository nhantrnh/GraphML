#!/usr/bin/env python3
"""mean Recall lấy trung bình các TỈ SỐ per-khung, không phải tỉ số của tổng.

ĐÍNH CHÍNH MỘT LẬP LUẬN SAI TRƯỚC ĐÓ. Bản đầu của tệp này khẳng định mã gom
điểm theo từng khung hình rồi chia cho 26 nên "17 số hạng bằng 0 vẫn nằm ở
mẫu". Đó là đọc sai cấu trúc dữ liệu:

    result_dict[mode + '_mean_recall_collect'][k] = [[] for _ in range(num_rel)]
                                              # evaluation_recall.py:33

là **26 danh sách, mỗi LỚP một danh sách** — không phải danh sách theo khung.
Dòng 348-349 append recall của lớp n vào danh sách riêng của lớp đó, chỉ khi
lớp đó có mặt trong khung. Khi tổng hợp, `np.mean(vi)` là recall trung bình
của LỚP n trên các khung chứa nó. Vì mọi lớp đều xuất hiện đâu đó trong tập
test, không lớp nào rơi vào nhánh `else 0.0`, nên **chia cho 26 là đúng
macro-average**. Lập luận cũ sai; giữ lại đính chính này để không tái phạm.

SAI LỆCH THẬT nằm ở chỗ khác, và nó tinh vi hơn.

  Mã (evaluation_recall.py:349 + 62):
      R_n = mean_{f ∈ F_n} ( TP_{f,n} / GT_{f,n} )      <- trung bình các TỈ SỐ
  Định nghĩa quy ước của SGG (Lorenz et al., CVPRW 2024,
  10.1109/cvprw63382.2024.00263; và Visual Genome):
      R_n = ( Σ_f TP_{f,n} ) / ( Σ_f GT_{f,n} )         <- TỈ SỐ của tổng

Hai đại lượng này khác nhau khi số triplet mỗi khung không đều. Trung bình
các tỉ số cho mỗi KHUNG trọng số bằng nhau, bất kể khung đó chứa 1 hay 10
triplet của lớp n. Lớp hiếm thường xuất hiện lẻ một lần mỗi khung, nên được
đội điểm so với định nghĩa chuẩn.

Đây là mô tả mã đã công bố đối chiếu định nghĩa đã công bố.
"""
import sys


def vi_du():
    """Lớp c xuất hiện ở hai khung: một khung 1 GT trúng, một khung 10 GT trúng 1."""
    khung = [(1, 1), (10, 1)]        # (GT, TP)
    macro = sum(tp / gt for gt, tp in khung) / len(khung)
    micro = sum(tp for _, tp in khung) / sum(gt for gt, _ in khung)
    return khung, macro, micro


def main():
    print(__doc__.split("\n\n")[0])
    khung, macro, micro = vi_du()

    print("\nVÍ DỤ TỐI THIỂU — một lớp, hai khung hình\n")
    for i, (gt, tp) in enumerate(khung):
        print(f"  khung {chr(65+i)}: {tp}/{gt} = {tp/gt:.3f}")
    print(f"\n  mã   (trung bình tỉ số) = {macro:.4f}")
    print(f"  chuẩn (tỉ số của tổng)  = {micro:.4f}")
    print(f"  chênh                   = {macro - micro:+.4f}"
          f"  ({100*(macro-micro)/micro:+.0f}%)")

    print("\nVÌ SAO ĐÁNG QUAN TÂM")
    print("  Khung chứa đúng 1 triplet của một lớp hiếm được tính trọng số")
    print("  ngang khung chứa 10 triplet của lớp phổ biến. Lớp càng hiếm càng")
    print("  hay xuất hiện lẻ một lần mỗi khung, nên hiệu ứng đội điểm dồn về")
    print("  đúng phía đuôi — tức phía mà mean Recall được tạo ra để đo.")

    print("\n  Hai đại lượng trùng nhau chỉ khi mọi khung có cùng số GT của lớp")
    print("  đó. Trên Action Genome điều đó không đúng: số vật thể mỗi khung")
    print("  chạy từ 1 đến 8.")

    print("\nGIỮ ĐÚNG MỨC")
    print("  Chưa đo được hiệu ứng này lớn bao nhiêu trên dữ liệu thật — cần")
    print("  tính lại R_n theo cả hai cách trên cùng checkpoint rồi so. Trước")
    print("  khi có số đó, chỉ nên phát biểu rằng hai định nghĩa KHÁC NHAU,")
    print("  không nên nói sai lệch là lớn.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
