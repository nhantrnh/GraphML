#!/usr/bin/env python3
"""Cài mặt nạ đúng theo công thức (5) của ImparTail, sau cờ --true_masked_bce.

Bài báo, mục 4.2.1, công thức (5), kèm câu "A mask ... voids the contribution
of loss":

    L = (1 - m) * L = { L nếu m=0 ; 0 nếu m=1 }

Nhãn bị che phải đóng góp ĐÚNG BẰNG 0.

Mã công bố (stsg_base.py, nhánh BCE) dựng vector đích toàn 0 rồi chỉ set 1 cho
các nhãn SỐNG SÓT. Vì nn.BCELoss() dùng reduction='mean' trên toàn bộ max_len
phần tử, nhãn dương bị che trở thành nhãn ÂM và vẫn sinh gradient. Hai sai lệch:

  (1) dấu    - nhãn bị che bị đẩy xác suất XUỐNG thay vì được bỏ qua;
  (2) mẫu số - chia max_len thay vì số phần tử còn hiệu lực, co mọi gradient
               còn lại (ví dụ 15/17 = 0.88x).

Bản vá thêm một đường thứ hai, bật bằng --true_masked_bce, dựng mặt nạ ở cấp
PHẦN TỬ rồi tính BCE có mặt nạ. Mặc định TẮT: không truyền cờ thì hành vi
giống hệt mã gốc, để hai nhánh so sánh được công bằng.

Ghi chú về phát biểu: đây là đối chiếu MÃ ĐÃ CÔNG BỐ với CÔNG THỨC ĐÃ CÔNG BỐ,
không phải kết luận về việc tác giả đã chạy thí nghiệm thế nào.

Dùng: python fix_masked_bce.py /đường/dẫn/ImparTail
"""
import os
import re
import sys

# ── 1. stsg_base.py: dựng thêm mặt nạ cấp phần tử ──────────────────
CU_STSG = '''                if gt_masked.shape[0] == 0:
                    continue
                else:
                    label = torch.zeros([max_len], dtype=torch.float32, device=self._device)
                    label[gt_masked] = 1
                    filtered_labels.append(label)
                    filtered_distribution.append(pred_distribution_i)'''

MOI_STSG = '''                if gt_masked.shape[0] == 0:
                    continue
                else:
                    label = torch.zeros([max_len], dtype=torch.float32, device=self._device)
                    label[gt_masked] = 1
                    filtered_labels.append(label)
                    filtered_distribution.append(pred_distribution_i)

                    # Công thức (5): nhãn bị che phải đóng góp 0. Ở nhánh gốc
                    # chúng thành 0 = nhãn âm và vẫn sinh gradient. Mặt nạ cấp
                    # phần tử dưới đây cho phép loại đúng chúng khỏi phép tính.
                    el = torch.ones([max_len], dtype=torch.float32,
                                    device=self._device)
                    if gt.numel() > 0:
                        el[gt[mask == 0]] = 0.0
                    element_masks.append(el)'''

# ── 2. khởi tạo và trả về element_masks ────────────────────────────
CU_INIT = '''        filtered_labels = []
        filtered_distribution = []
        if not self._conf.bce_loss:'''
MOI_INIT = '''        filtered_labels = []
        filtered_distribution = []
        element_masks = []
        if not self._conf.bce_loss:'''

CU_RET_NONE = '''        if len(filtered_labels) == 0 and len(filtered_distribution) == 0:
            return None, None, pred_distribution'''
MOI_RET_NONE = '''        if len(filtered_labels) == 0 and len(filtered_distribution) == 0:
            self._last_element_mask = None
            return None, None, pred_distribution'''

CU_RET = '''        filtered_labels = torch.stack(filtered_labels)
        filtered_distribution = torch.stack(filtered_distribution)

        return filtered_distribution, filtered_labels, pred_distribution'''
MOI_RET = '''        filtered_labels = torch.stack(filtered_labels)
        filtered_distribution = torch.stack(filtered_distribution)
        # Truyền qua thuộc tính thay vì đổi chữ ký hàm: hàm này được gọi từ
        # nhiều nơi (SGG và SGA), đổi chữ ký sẽ phải sửa hết.
        self._last_element_mask = torch.stack(element_masks) if element_masks else None

        return filtered_distribution, filtered_labels, pred_distribution'''

# ── 3. train_sgg_base.py: dùng mặt nạ khi cờ được bật ──────────────
CU_LOSS = '''            else:
                losses[const.{TEN}_RELATION_LOSS] = self._bce_loss(filtered_{v}_distribution, filtered_{v}_labels)'''
MOI_LOSS = '''            elif getattr(self._conf, "true_masked_bce", False):
                losses[const.{TEN}_RELATION_LOSS] = self._masked_bce(
                    filtered_{v}_distribution, filtered_{v}_labels, {v}_element_mask)
            else:
                losses[const.{TEN}_RELATION_LOSS] = self._bce_loss(filtered_{v}_distribution, filtered_{v}_labels)'''

# hàm _masked_bce, đặt cạnh _init_loss_functions
CU_LOSSFN = '''    def _init_loss_functions(self):
        self._bce_loss = nn.BCELoss()'''
MOI_LOSSFN = '''    def _masked_bce(self, pred_distribution, labels, element_mask):
        """BCE tôn trọng công thức (5): phần tử bị che đóng góp đúng 0.

        Khác hai điểm so với nn.BCELoss() mặc định:
          - reduction='none' rồi nhân mặt nạ -> phần tử bị che không sinh gradient;
          - chuẩn hoá theo số phần tử CÒN HIỆU LỰC, không theo max_len.
        """
        if element_mask is None:
            return self._bce_loss(pred_distribution, labels)
        per = nn.functional.binary_cross_entropy(
            pred_distribution, labels, reduction="none")
        tong = element_mask.sum()
        if tong.item() == 0:
            return pred_distribution.sum() * 0.0
        return (per * element_mask).sum() / tong

    def _init_loss_functions(self):
        self._bce_loss = nn.BCELoss()'''


def doc(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def ghi(p, s):
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)


def va_stsg(root):
    p = os.path.join(root, "stsg_base.py")
    s = doc(p)
    # Dấu hiệu nhận biết đã vá: phải là chuỗi CHỈ có trong bản vá. Dùng dòng
    # đầu của `moi` là sai — với khối mặt nạ, dòng đầu trùng mã gốc nên lần
    # chạy thứ hai chèn thêm một khối nữa.
    for cu, moi, ten, dau in [
            (CU_INIT, MOI_INIT, "khởi tạo element_masks", "element_masks = []"),
            (CU_STSG, MOI_STSG, "dựng mặt nạ cấp phần tử", "element_masks.append(el)"),
            (CU_RET_NONE, MOI_RET_NONE, "trả None", "self._last_element_mask = None"),
            (CU_RET, MOI_RET, "trả element_masks", "torch.stack(element_masks)")]:
        if dau in s:
            print(f"  - {ten}: đã vá trước đó")
            continue
        if cu not in s:
            sys.exit(f"  ✗ không khớp: {ten}")
        s = s.replace(cu, moi, 1)
        print(f"  ✓ {ten}")
    ghi(p, s)


def va_train(root):
    p = os.path.join(root, "train_sgg_base.py")
    s = doc(p)

    if "_masked_bce" not in s:
        if CU_LOSSFN not in s:
            sys.exit("  ✗ không khớp: _init_loss_functions")
        s = s.replace(CU_LOSSFN, MOI_LOSSFN, 1)
        print("  ✓ thêm hàm _masked_bce")
    else:
        print("  - _masked_bce: đã có")

    for ten, v in [("SPATIAL", "spatial"), ("CONTACTING", "contact")]:
        cu = CU_LOSS.format(TEN=ten, v=v)
        moi = MOI_LOSS.format(TEN=ten, v=v)
        if f"{v}_element_mask" in s:
            print(f"  - nhánh {v}: đã vá")
            continue
        if cu not in s:
            sys.exit(f"  ✗ không khớp nhánh {v}")
        s = s.replace(cu, moi, 1)
        print(f"  ✓ nhánh {v}")

    # lấy mặt nạ ngay sau mỗi lời gọi _prepare_labels_and_distribution
    for v, hang in [("spatial", "max_len=6"), ("contact", "max_len=17")]:
        neo = f"            {hang}\n        )\n"
        them = f"            {hang}\n        )\n        {v}_element_mask = getattr(self, \"_last_element_mask\", None)\n"
        if f"{v}_element_mask = getattr" in s:
            continue
        if neo not in s:
            sys.exit(f"  ✗ không tìm thấy neo {hang}")
        s = s.replace(neo, them, 1)
        print(f"  ✓ lấy mặt nạ cho {v}")

    ghi(p, s)


def va_config(root):
    p = os.path.join(root, "lib/supervised/config.py")
    s = doc(p)
    if "true_masked_bce" in s:
        print("  - cờ --true_masked_bce: đã có")
        return
    neo = "        parser.add_argument('--bce_loss', action='store_true')"
    if neo not in s:
        sys.exit("  ✗ không tìm thấy neo --bce_loss")
    them = (neo + "\n        # Mặc định TẮT: không truyền cờ thì hành vi giống hệt mã gốc.\n"
            "        parser.add_argument('--true_masked_bce', action='store_true')")
    ghi(p, s.replace(neo, them, 1))
    print("  ✓ thêm cờ --true_masked_bce")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    if not os.path.isfile(os.path.join(root, "stsg_base.py")):
        sys.exit(f"không thấy stsg_base.py trong {root}")
    print("stsg_base.py:")
    va_stsg(root)
    print("train_sgg_base.py:")
    va_train(root)
    print("lib/supervised/config.py:")
    va_config(root)
    print("\nXong. Bật bằng --true_masked_bce; không truyền thì chạy như mã gốc.")


if __name__ == "__main__":
    main()
