#!/usr/bin/env python3
"""Thêm cờ --seed để ba lần chạy tái lập được.

Mã công bố KHÔNG đặt seed ở bất kỳ đâu (không có manual_seed trong toàn bộ
upstream, config.py không có cờ seed). Nên hai lần chạy cùng cấu hình vẫn cho
kết quả khác nhau, và không phân biệt được chênh lệch do phương pháp hay do
khởi tạo ngẫu nhiên.

Bản vá thêm --seed vào config.py và gieo ở đầu main() của train_sgg_methods.py,
phủ cả random, numpy và torch (gồm CUDA). KHÔNG truyền cờ thì seed mặc định
là None và hành vi giống hệt mã gốc.

Ghi chú về phát biểu: đây là đối chiếu MÃ ĐÃ CÔNG BỐ, không phải kết luận về
việc tác giả đã chạy thí nghiệm thế nào.

Dùng: python add_seed.py /đường/dẫn/ImparTail
"""
import os
import sys

CFG_CU = '        parser.add_argument("--use_wandb", action="store_true")'
CFG_MOI = '''        parser.add_argument("--use_wandb", action="store_true")
        # Mặc định None -> không gieo -> hành vi giống hệt mã gốc.
        parser.add_argument("--seed", default=None, type=int)'''

TR_CU = '''def main():
    conf = Config()'''
TR_MOI = '''def main():
    conf = Config()
    # Gieo seed trước khi dựng mô hình: khởi tạo trọng số, thứ tự dataloader và
    # mặt nạ của Algorithm 1 đều rút từ các nguồn ngẫu nhiên này.
    _seed = getattr(conf, "seed", None)
    if _seed is not None:
        import random as _rd
        import numpy as _np
        import torch as _th
        _rd.seed(_seed)
        _np.random.seed(_seed)
        _th.manual_seed(_seed)
        if _th.cuda.is_available():
            _th.cuda.manual_seed_all(_seed)
        print(f"[seed] da gieo seed = {_seed}", flush=True)'''


def va(duong_dan, cap, ten_tep, dau_hieu):
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
    va(os.path.join(root, "lib/supervised/config.py"),
       [(CFG_CU, CFG_MOI, "co --seed")], "config.py", '"--seed"')
    va(os.path.join(root, "train_sgg_methods.py"),
       [(TR_CU, TR_MOI, "gieo seed")], "train_sgg_methods.py", "[seed] da gieo")
    print("\\nXong. Dung: --seed 1234")


if __name__ == "__main__":
    main()
