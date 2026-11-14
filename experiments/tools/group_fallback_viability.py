#!/usr/bin/env python3
"""Fallback mức nhóm có khả thi không, hay là no-op về mặt toán học?

Ý tưởng: khi nhãn vị ngữ chi tiết bị mặt nạ che, thay vì vứt cả cặp
(stsg_base.py:297, :312 — lệnh `continue`), vẫn giám sát ở mức NHÓM.

Phản biện quan trọng: nếu mỗi nhóm đã là một softmax riêng thì tổng xác suất
các lớp trong nhóm LUÔN bằng 1, cross-entropy trên tổng đó bằng 0, gradient
bằng 0 — ý tưởng vô tác dụng. Phải kiểm mã trước khi xây bất cứ thứ gì.

Kết quả kiểm (lib/supervised/sgg/dsgdetr/dsgdetr.py:569-575):

    entry[ATTENTION_DISTRIBUTION]  = self.a_rel_compress(global_output)   # logits thô
    entry[SPATIAL_DISTRIBUTION]    = torch.sigmoid(...)                   # SIGMOID
    entry[CONTACTING_DISTRIBUTION] = torch.sigmoid(...)                   # SIGMOID

Nên:
  - attention (3 lớp): đi qua CrossEntropyLoss (train_sgg_base.py:333) tức có
    softmax nội bộ -> tổng = 1 -> marginalize LÀ no-op. Loại nhóm này.
  - spatial (6) và contacting (17): sigmoid, mỗi đầu ra ĐỘC LẬP, tổng KHÔNG bị
    ràng buộc bằng 1 -> marginalize có gradient thật. Dùng được.

Với sigmoid độc lập, công thức đúng không phải "cộng xác suất rồi lấy CE" (các
xác suất không lập thành phân phối) mà là noisy-OR: xác suất ít nhất một lớp
trong nhóm dương tính,

    P(nhóm) = 1 - prod_c (1 - p_c)

rồi lấy BCE trên đại lượng đó với đích = 1. Script kiểm nó có gradient không.
"""
import math

# Ba nhóm của Action Genome và cách mã xử lý chúng
NHOM = [
    ("attention", 3, "CrossEntropyLoss (softmax nội bộ)", False),
    ("spatial", 6, "sigmoid độc lập", True),
    ("contacting", 17, "sigmoid độc lập", True),
]


def noisy_or(ps):
    q = 1.0
    for p in ps:
        q *= (1.0 - p)
    return 1.0 - q


def main():
    print(__doc__.split("\n\n")[0])
    print(f"\n{'nhóm':<12}{'lớp':>5}  {'kích hoạt':<34}{'marginalize được?'}")
    print("-" * 76)
    for ten, k, kich_hoat, dung_duoc in NHOM:
        print(f"{ten:<12}{k:>5}  {kich_hoat:<34}{'CÓ' if dung_duoc else 'KHÔNG (no-op)'}")

    print("\n\nKiểm gradient của noisy-OR trên nhóm contacting (17 lớp sigmoid)\n")
    # d/dp_c [ -log(1 - prod(1-p_j)) ] = -(prod_{j!=c}(1-p_j)) / (1 - prod(1-p_j))
    for mo_ta, ps in [
        ("mô hình chưa tin lớp nào (p thấp đều)", [0.05] * 17),
        ("mô hình đã tin một lớp (p cao lẻ)", [0.9] + [0.05] * 16),
        ("trung bình", [0.3] * 17),
    ]:
        P = noisy_or(ps)
        # gradient theo lớp đầu tiên
        con_lai = 1.0
        for p in ps[1:]:
            con_lai *= (1.0 - p)
        grad = -con_lai / P if P > 0 else float("nan")
        print(f"  {mo_ta:<40} P(nhóm)={P:.4f}  dL/dp_0={grad:+.4f}")

    print("\n-> Gradient KHÁC 0 ở mọi trường hợp, và LỚN NHẤT khi mô hình chưa tin")
    print("   lớp nào — đúng lúc tín hiệu cần thiết nhất. Không phải no-op.")

    print("\n-> Nhưng chú ý: khi mô hình đã tin chắc một lớp (p=0.9), gradient rất")
    print("   nhỏ. Nghĩa là fallback tự tắt khi không cần. Đây là tính chất tốt,")
    print("   nhưng cũng nghĩa là lợi ích sẽ nhỏ ở giai đoạn cuối huấn luyện.")

    print("\nRỦI RO CÒN LẠI (chưa kiểm được bằng script):")
    print("  Nhãn bị che chủ yếu là lớp head, nên tập cặp được cứu lệch về head.")
    print("  Dạy 'đây là contacting' khi che 'holding' có thể đẩy mô hình về lại")
    print("  vùng head. Phải đo bằng thực nghiệm, không lập luận suông được.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
