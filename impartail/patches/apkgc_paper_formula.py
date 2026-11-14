#!/usr/bin/env python3
"""Cài đúng công thức (9)(10) của bài APKGC, thay cho mã công bố.

VÌ SAO CẦN BẢN NÀY. Đo trên mô hình thật (kernel apkgc-attn-probe) cho thấy
cơ chế phạt trong mã công bố KHÔNG kích hoạt lần nào trong 300 lần gọi
forward. Nhưng điều đó chỉ nói về REPO, không nói về PHƯƠNG PHÁP: tác giả có
thể đã đẩy nhầm bản, hoặc giữ lại bản thật.

Cách tránh hẳn tranh cãi đó là cài lại đúng như BÀI mô tả, rồi so ba nhánh.
Khi ấy kết luận không phụ thuộc vào việc repo có phải mã cuối hay không.

BÀI VIẾT GÌ (tr.15007, công thức 9 và 10):

    A_j = A_i − θp·A_i,                 nếu A_i > φ              (9)
    A_n = A_n + θn·(A_i − φ),   n ∈ (p−k, p+k)                   (10)

Đọc kỹ: A_i là ĐIỂM ATTENTION CỦA CHÍNH TOKEN ĐÓ. Phạt trực tiếp token vượt
ngưỡng, rồi chia phần bị trừ cho k token lân cận.

MÃ CÔNG BỐ LÀM GÌ (APKGC.py:360-365):

    penalty_scores[..., i] = attention_scores[..., i:].prod(dim=-1)
    exceed_threshold_mask = penalty_scores > self.penalty_threshold

Đem so ngưỡng với TÍCH LUỸ HẬU TỐ — tích các điểm từ vị trí i đến hết hàng.
Đây là đại lượng khác hẳn A_i, và đo được là nó không bao giờ vượt 10.

BA NHÁNH ĐỂ SO:

    A  mã gốc như công bố          (không đặt biến môi trường nào)
    B  đúng công thức (9)(10)      APKGC_PAPER_FORMULA=1
    B2 cap xác suất ở φ=0,5        APKGC_CAP_PROB=0.5   (theo HÌNH case study)
    C  entropy, đề xuất cải tiến   APKGC_ENTROPY_PENALTY=1

Hai câu hỏi tách bạch:
    B khác A?  -> mã công bố có lệch khỏi bài hay không
    C hơn B?   -> cải tiến có thắng bản đúng nhất của bài hay không

THAM SỐ. Bài không công bố giá trị θp, θn, φ, k. Lấy từ chính mã công bố để
không tự chọn số có lợi: φ = penalty_threshold = 10, k = window_size = 1,
θp = penalty_weights = 0.001, θn = θp / (2k).

Ghi chú phát biểu: đây là đối chiếu MÃ ĐÃ CÔNG BỐ với MÔ TẢ TRONG BÀI, không
kết luận về việc tác giả đã chạy thí nghiệm thế nào.

KẾT QUẢ CHẠY THẬT (kernel apkgc-three-branch, 27-28/09, MKG-W, seed 42):

    nhánh A (mã công bố)      MRR 0,3629   H@1 0,2933   252,4 phút
    nhánh B (công thức bài)   MRR 0,3629   H@1 0,2933   246,3 phút
    nhánh C (entropy)         bị cắt — kernel bị huỷ sau ~7h50

A và B TRÙNG đến bốn chữ số. Không phải lỗi cài đặt: thời lượng chênh 6 phút
nên nhánh B có chạy thật. Lý do nằm ở ĐỘ LỚN tham số lấy từ mã công bố:

    θp = penalty_weights = 0.001

Token vượt ngưỡng mạnh nhất đo được có A = 10,798, nên bị trừ 0,0108 — thay
đổi 0,10%. Lân cận nhận θn·(A−φ) = 0,0004. Sau softmax, chênh lệch xác suất
chỉ cỡ 1e-4 đến 1e-5. Quá nhỏ để dịch chuyển MRR ở chữ số thứ tư.

Nên phát biểu đúng là: cơ chế phạt attention KHÔNG ảnh hưởng đáng kể tới kết
quả, dù cài theo mã công bố (không kích hoạt lần nào) hay theo đúng công thức
trong bài (kích hoạt nhưng biên độ 0,1%). Cả hai đường đều dẫn tới cùng một
chỗ, và điều đó mạnh hơn phát hiện ban đầu vì nó không còn phụ thuộc vào việc
repo có phải mã cuối hay không.

CÒN THIẾU: nhánh C. Cần chạy lại để biết entropy có khác hai nhánh kia không.

Dùng: python apkgc_paper_formula.py /đường/dẫn/APKGC [--revert]
"""
import os
import sys

REL = "mmkgc/module/model/APKGC.py"

CU = """        exceed_threshold_mask = penalty_scores > self.penalty_threshold"""

MOI = '''        exceed_threshold_mask = penalty_scores > self.penalty_threshold

        # === NHANH B2: cap XAC SUAT attention o φ, chia du cho lan can ====
        # Hinh case study (slide 11, Fig.4 cua bai) ve nguong 0,5 tren attention
        # DA CHUAN HOA: cot 0,66 bi cat xuong 0,5, phan du chia cho token ke.
        # Day la mo ta bang HINH, khac voi hang so trong ma (10 tren tich logit)
        # va khac ca nhanh B (θp=0,001 tren logit). Bat: APKGC_CAP_PROB=0.5
        if os.environ.get("APKGC_CAP_PROB"):
            _phi = float(os.environ["APKGC_CAP_PROB"])
            _k = max(1, self.window_size)
            _p = nn.functional.softmax(attention_scores, dim=-1)
            _du = torch.clamp(_p - _phi, min=0.0)          # phan vuot nguong
            _p = _p - _du                                   # cat xuong φ
            for _d in range(1, _k + 1):                     # chia deu 2k lan can
                _p = _p + torch.roll(_du, shifts=_d, dims=-1) / (2 * _k)
                _p = _p + torch.roll(_du, shifts=-_d, dims=-1) / (2 * _k)
            _pr = self.dropout(_p)
            _cl = torch.matmul(_pr, value_layer).permute(0, 2, 1, 3).contiguous()
            _cl = _cl.view(_cl.size()[:-2] + (self.all_head_size,))
            return (_cl, _pr) if output_attentions else (_cl,)

        # === NHANH B: dung cong thuc (9)(10) cua bai ====================
        # Bat bang APKGC_PAPER_FORMULA=1. Khong bat -> chay nhanh goc ben duoi.
        if os.environ.get("APKGC_PAPER_FORMULA") == "1":
            _phi = self.penalty_threshold          # φ, cong thuc (9)
            _k = self.window_size                  # k, cong thuc (10)
            _tp = self.penalty_weights             # θp
            _tn = _tp / (2 * _k) if _k else 0.0    # θn, chia deu cho 2k lan can

            # (9): phat token co A_i > φ
            _vuot = attention_scores > _phi
            _tru = torch.where(_vuot, _tp * attention_scores,
                               torch.zeros_like(attention_scores))
            _kq = attention_scores - _tru

            # (10): chia phan du cho k token lan can moi ben
            _du = torch.where(_vuot, attention_scores - _phi,
                              torch.zeros_like(attention_scores))
            for _d in range(1, _k + 1):
                _kq = _kq + _tn * torch.roll(_du, shifts=_d, dims=-1)
                _kq = _kq + _tn * torch.roll(_du, shifts=-_d, dims=-1)

            attention_probs = nn.functional.softmax(_kq, dim=-1)
            attention_probs = self.dropout(attention_probs)
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            _new_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(_new_shape)
            # Ham goc tra ve TUPLE (APKGC.py:403); tra ve tensor tran se hong
            # o cho goi. Giu dung kieu.
            return ((context_layer, attention_probs) if output_attentions
                    else (context_layer,))
        # === HET NHANH B ================================================'''

DAU_HIEU = "APKGC_PAPER_FORMULA"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root, revert = sys.argv[1], "--revert" in sys.argv
    p = os.path.join(root, REL)
    if not os.path.isfile(p):
        sys.exit(f"  x khong thay {p}")
    s = open(p, encoding="utf-8").read()

    if revert:
        if DAU_HIEU not in s:
            print("  - chua va")
            return 0
        # Doan chen bat dau bang dong CU roi moi den khoi B, nen cat tu SAU
        # dong CU: chi bo khoi B, giu nguyen dong goc.
        i = s.index("        # === NHANH B")
        j = s.index("# === HET NHANH B ================================================")
        j = s.index("\n", j) + 1
        open(p, "w", encoding="utf-8").write(s[:i].rstrip("\n") + "\n" + s[j:])
        print("  v da hoan ve ban goc")
        return 0

    if DAU_HIEU in s:
        print("  - da va truoc do")
        return 0
    if CU not in s:
        sys.exit("  x khong khop doan can va")
    if s.count(CU) != 1:
        sys.exit(f"  x doan can va xuat hien {s.count(CU)} lan, can dung 1")
    if "\nimport os" not in s:
        s = s.replace("import math", "import math\nimport os", 1)
    open(p, "w", encoding="utf-8").write(s.replace(CU, MOI, 1))
    print("  v da va APKGC.py (nhanh B)")
    print("\nBat bang:  export APKGC_PAPER_FORMULA=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
