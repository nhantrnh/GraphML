#!/usr/bin/env python3
"""Xuất recall TỪNG LỚP ra JSON, để tính lại mean Recall khi loại lớp phủ định.

mean Recall trên Action Genome chia cho đủ 26 lớp:

    sum_recall = np.sum([np.mean(vi) if vi else 0.0 for vi in v])
    mean_recall_value = sum_recall / float(self.num_rel)   # num_rel = 26
                                          # evaluation_recall.py:62-63

Bốn trong 26 lớp mang nghĩa phủ định hoặc gom (chỉ số theo
dataloader/base_ag_dataset.py:69-83):

    index  1  not_looking_at      212.640 lần
    index  2  unsure               48.882
    index 17  not_contacting      117.088
    index 18  other_relationship   10.084

Chúng chiếm 15,4% trọng số mR (4/26) và 28,4% tổng số nhãn. Muốn biết việc
loại chúng có đảo thứ hạng giữa các phương pháp hay không thì phải có recall
của TỪNG lớp — CSV kết quả chỉ lưu mR đã tổng hợp.

Bản vá gồm hai phần, KHÔNG đổi bất kỳ con số nào đang có:
  1. evaluation_recall.py giữ lại danh sách recall từng lớp
  2. test_sgg_base.py ghi nó ra JSON sau khi đánh giá xong

Phần 2 bắt buộc ghi ra tệp chứ không đọc biến trong bộ nhớ: tiến trình
notebook Kaggle đã nạp torch 2.10 từ lúc khởi động, nên gọi evaluator trong
tiến trình vỡ ở "RuntimeError: operator torchvision::nms does not exist".
Chạy bằng subprocess thì không đọc được biến, nên evaluator tự ghi tệp.

Dùng: python dump_per_class_recall.py /đường/dẫn/ImparTail
Kết quả: $PER_CLASS_OUT (mặc định /kaggle/working/per_class.json)
"""
import os
import sys

# ── Phần 1: evaluation_recall.py ───────────────────────────────────
KHOI_TAO_NEO = "        self.num_rel = len(AG_all_predicates)"
KHOI_TAO_MOI = """        self.num_rel = len(AG_all_predicates)
        self.per_class_recall = {}"""

GHI_NEO = """        for k, v in self.result_dict[self.mode + '_mean_recall_collect'].items():
            sum_recall = np.sum([np.mean(vi) if vi else 0.0 for vi in v])
            mean_recall_value = sum_recall / float(self.num_rel)
            mean_recall_dict[k] = mean_recall_value"""
GHI_MOI = """        for k, v in self.result_dict[self.mode + '_mean_recall_collect'].items():
            sum_recall = np.sum([np.mean(vi) if vi else 0.0 for vi in v])
            mean_recall_value = sum_recall / float(self.num_rel)
            mean_recall_dict[k] = mean_recall_value
            # Giữ recall từng lớp để tính lại mR khi loại lớp âm/gom.
            # Không đổi mean_recall_value ở trên — mọi số cũ giữ nguyên.
            self.per_class_recall[k] = [
                float(np.mean(vi)) if vi else 0.0 for vi in v
            ]"""

# ── Phần 2: test_sgg_base.py ───────────────────────────────────────
BASE_NEO = "        self._collated_stats = self._collate_evaluation_stats()"
BASE_MOI = "\n".join([
    '        self._collated_stats = self._collate_evaluation_stats()',
    '        # Ghi recall từng lớp và (TP, GT) thô ra JSON.',
    '        import json as _json',
    '        _pc = {}',
    '        for _ten, _ev in (("with", self._with_constraint_evaluator),',
    '                          ("no", self._no_constraint_evaluator),',
    '                          ("semi", self._semi_constraint_evaluator)):',
    '            _r = {str(_k): _v for _k, _v',
    '                  in getattr(_ev, "per_class_recall", {}).items()}',
    '            _raw = {str(_k): _v for _k, _v in',
    '                    _ev.result_dict.get(_ev.mode + "_raw_tp_gt", {}).items()}',
    '            _pc[_ten] = {"recall": _r, "raw": _raw}',
    '        _out = os.environ.get("PER_CLASS_OUT",',
    '                              "/kaggle/working/per_class.json")',
    '        with open(_out, "w") as _f:',
    '            _json.dump(_pc, _f)',
    '        print("da ghi recall tung lop -> " + _out)',
])


RAW_NEO = "                result_dict[mode + '_mean_recall_collect'][k][n].append(float(recall_hit[n] / recall_count[n]))"
RAW_MOI = "                result_dict[mode + '_mean_recall_collect'][k][n].append(float(recall_hit[n] / recall_count[n]))\n                # Giữ (TP, GT) thô: tỉ số mất mẫu số nên không tính được\n                # R_global = ΣTP/ΣGT lẫn Cov(R_f, w_f) của Định lý 3.\n                result_dict.setdefault(mode + '_raw_tp_gt', {}).setdefault(\n                    k, [[] for _ in range(num_rel)])[n].append(\n                        (int(recall_hit[n]), int(recall_count[n])))"


def va(duong_dan, cap, ten_tep, dau_hieu):
    """Áp một loạt (cũ, mới) lên tệp; bỏ qua nếu đã vá."""
    if not os.path.isfile(duong_dan):
        sys.exit(f"  x khong thay {duong_dan}")
    s = open(duong_dan, encoding="utf-8").read()
    if dau_hieu in s:
        print(f"  - {ten_tep}: da va truoc do")
        return
    for cu, moi, nhan in cap:
        if cu not in s:
            sys.exit(f"  x khong khop: {nhan} trong {ten_tep}")
        s = s.replace(cu, moi, 1)
    open(duong_dan, "w", encoding="utf-8").write(s)
    print(f"  v {ten_tep}")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    va(os.path.join(root, "lib/supervised/evaluation_recall.py"),
       [(KHOI_TAO_NEO, KHOI_TAO_MOI, "khoi tao"),
        (GHI_NEO, GHI_MOI, "ghi recall tung lop")],
       "evaluation_recall.py", "per_class_recall")
    # Giữ (TP, GT) thô để tính R_global và Cov(R_f, w_f) của Định lý 3 —
    # tỉ số per-khung mất mẫu số nên không suy ngược được.
    va(os.path.join(root, "lib/supervised/evaluation_recall.py"),
       [(RAW_NEO, RAW_MOI, "giữ TP/GT thô")],
       "evaluation_recall.py (raw)", "_raw_tp_gt")
    va(os.path.join(root, "test_sgg_base.py"),
       [(BASE_NEO, BASE_MOI, "ghi JSON")],
       "test_sgg_base.py", "PER_CLASS_OUT")
    print("\nXong. Doc tep JSON tai $PER_CLASS_OUT sau khi danh gia.")


if __name__ == "__main__":
    main()
