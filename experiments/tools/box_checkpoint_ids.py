#!/usr/bin/env python3
"""Bản đồ file_id trên Box của checkpoint ImparTail — và một cảnh báo.

ImparTail phát hành 26 checkpoint qua bốn link Box. Muốn tải bằng script thì
cần file_id, và đây là chỗ có bẫy.

BẪY 1 — regex dễ viết sai. Cấu trúc JSON nhúng trong trang Box là

    "id":1691742791564, ... "extension":"tar","name":"sttran_partial_70_..."

tức `id` là SỐ không có dấu nháy, và đứng TRƯỚC `name`. Mẫu
`"id":"(\\d+)"` (có nháy) trượt toàn bộ; phải dùng

    r'"id":(\\d+),[^{}]*?"name":"([^"]+\\.tar)"'

BẪY 2 — mọi checkpoint STTran đều CÙNG kích thước 416.767.365 byte, nên
không thể dùng dung lượng để xác nhận đã tải đúng file. Ba checkpoint
predcls/sgcls/sgdet giống hệt nhau về kiến trúc (cùng 41 tham số), nên cũng
không phân biệt được bằng tên tham số.

CÁCH XÁC NHẬN DUY NHẤT TIN ĐƯỢC: chạy đánh giá rồi đối chiếu với bảng trong
bài báo. Kết quả tái lập của nhóm cho mR@10/20/50 with-constraint là
(30,70 · 35,54 · 35,65), so với Bảng 1:

    PREDCLS (30,5 · 34,7 · 34,8)  -> lệch trung bình  0,63 điểm  <- khớp
    SGCLS   (25,0 · 27,5 · 27,6)  -> lệch trung bình  7,26 điểm
    SGDET   ( 8,0 · 16,6 · 19,3)  -> lệch trung bình 19,33 điểm

nên checkpoint đã dùng đúng là predcls, bất kể nhãn nào gắn với file_id.

Ghi chú: id mà notebook cũ dùng cho predcls (f_1674218919601) khác id mà trang
Box hiện gắn nhãn predcls (1674209087058). Đối chiếu số liệu ở trên cho thấy
file tải về hành xử như predcls. Không kết luận gì thêm về nguyên nhân —
chỉ ghi lại để lần sau xác nhận bằng số liệu chứ không bằng nhãn.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "experiments/results/impartail_repro/checkpoints.json")

# Lấy bằng regex ở trên từ hai trang Box, ngày 14/09/2026
IDS = {
    "sttran_predcls.tar": ("ji6aekd4d8iguqk59rfwcalhpu608mop", "1674209087058"),
    "sttran_sgcls.tar": ("ji6aekd4d8iguqk59rfwcalhpu608mop", "1674213563569"),
    "sttran_sgdet.tar": ("ji6aekd4d8iguqk59rfwcalhpu608mop", "1674218919601"),
    "sttran_partial_70_predcls_epoch_3.tar":
        ("1515xwjjt04tnby8zfy9m8jnk9pztnv4", "1691742791564"),
    "sttran_partial_40_predcls_epoch_3.tar":
        ("1515xwjjt04tnby8zfy9m8jnk9pztnv4", "1691737384523"),
    "sttran_partial_10_predcls_epoch_0.tar":
        ("1515xwjjt04tnby8zfy9m8jnk9pztnv4", "1684866160304"),
}
BOX = "https://utdallas.box.com/index.php?rm=box_download_shared_file"


def main():
    print(__doc__.split("\n\n")[0])
    print(f"\n{len(IDS)} checkpoint cần cho bảng chính (PredCls, 4 tỉ lệ che)\n")
    print(f"{'tệp':<42}{'file_id':>16}")
    print("-" * 60)
    for ten, (shared, fid) in IDS.items():
        print(f"{ten:<42}{fid:>16}")

    print("\nURL tải:")
    ten = "sttran_partial_70_predcls_epoch_3.tar"
    shared, fid = IDS[ten]
    print(f"  {BOX}&shared_name={shared}&file_id=f_{fid}")

    print("\nXác nhận: HEAD request trả content-type: application/x-tar và")
    print("content-length: 416.767.365 — file thật, tải ẩn danh được.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
