#!/usr/bin/env python3
"""Thay tiêu chí phạt attention của APKGC bằng entropy chuẩn hoá.

VÌ SAO. APKGC.py:360-365 chọn token để phạt bằng cách so ngưỡng với TÍCH LUỸ
HẬU TỐ của điểm attention:

    penalty_scores[..., i] = attention_scores[..., i:].prod(dim=-1)
    exceed_threshold_mask = penalty_scores > self.penalty_threshold   # = 10

Đại lượng này không đo độ nhọn của phân bố attention — thứ cơ chế định xử lý.
Ba hệ quả, đo được bằng mô phỏng tất định (apkgc_penalty_defect.py):

  1. ngưỡng 10 chỉ kích hoạt trên ~0,01-0,1% số phần tử, nên phạt gần như
     không chạy;
  2. tích của các số có dấu bất kỳ ra âm khoảng một nửa số lần, mà mask là
     `> 10` nên nửa đó miễn nhiễm bất kể attention nhọn cỡ nào;
  3. phần còn lại phụ thuộc ĐỘ DÀI hậu tố, nên chấm điểm token theo vị trí
     trong hàng chứ không theo độ nhọn.

Mục RQ4 của bài đề xuất dò tay ngưỡng. Dò tay không sửa được ba điều trên, vì
đại lượng bị dò không đo thứ cần đo.

BẢN VÁ LÀM GÌ. Chỉ thay tiêu chí CHỌN token bị phạt; toàn bộ phần phân phối
lại trọng số sang token lân cận giữ nguyên, nên so sánh là công bằng.

    p = softmax(attention_scores, dim=-1)
    H = -Σ p·log p / log L          ∈ [0, 1]; 0 = nhọn tuyệt đối, 1 = đều
    ngưỡng τ = phân vị `entropy_percentile` của H trong batch
    phạt hàng có H < τ, tại token có p lớn nhất của hàng đó

Ngưỡng là PHÂN VỊ nên tự co giãn theo dữ liệu và theo giai đoạn huấn luyện —
đúng tinh thần thích ứng mà bài đã áp cho nhiễu (σ giảm dần) nhưng chưa áp cho
phạt. Không thêm siêu tham số cần dò: `entropy_percentile` mặc định 0,10 nghĩa
là phạt 10% hàng nhọn nhất, đọc được trực tiếp.

KHÔNG bật cờ thì hành vi giống hệt mã gốc.

Ghi chú phát biểu: đây là đối chiếu MÃ ĐÃ CÔNG BỐ với mô tả trong bài, không
kết luận về việc tác giả đã chạy thí nghiệm thế nào.

Dùng: python apkgc_entropy_penalty.py /đường/dẫn/APKGC [--revert]
"""
import os
import sys

REL = "mmkgc/module/model/APKGC.py"

CU = """        penalty_scores = torch.zeros_like( attention_scores )
        for i in range( attention_scores.shape[-1] ):
            penalty_scores[..., i] = attention_scores[..., i:].prod( dim=-1 )

        exceed_threshold_mask = penalty_scores > self.penalty_threshold"""

MOI = '''        # === BẢN VÁ: tiêu chí phạt theo entropy ===========================
        # Bật bằng biến môi trường APKGC_ENTROPY_PENALTY=1. Không bật thì chạy
        # đúng nhánh gốc bên dưới, nên hai cấu hình so sánh được công bằng.
        if os.environ.get("APKGC_ENTROPY_PENALTY") == "1":
            _pct = float( os.environ.get( "APKGC_ENTROPY_PCT", "0.10" ) )
            _L = attention_scores.shape[-1]
            _p = F.softmax( attention_scores, dim=-1 )
            # entropy chuẩn hoá mỗi hàng: 0 = dồn hết vào một token, 1 = đều
            _H = -( _p * torch.log( _p + 1e-12 ) ).sum( dim=-1 ) / math.log( _L )
            # ngưỡng là phân vị trong batch -> tự co giãn theo dữ liệu
            _tau = torch.quantile( _H.flatten().float(), _pct )
            _hang_nhon = ( _H < _tau ).unsqueeze( -1 )          # (B,h,L,1)
            # trong hàng nhọn, phạt đúng token chiếm attention lớn nhất
            _dinh = _p == _p.max( dim=-1, keepdim=True ).values
            exceed_threshold_mask = _hang_nhon & _dinh
            penalty_scores = _p
        else:
            penalty_scores = torch.zeros_like( attention_scores )
            for i in range( attention_scores.shape[-1] ):
                penalty_scores[..., i] = attention_scores[..., i:].prod( dim=-1 )

            exceed_threshold_mask = penalty_scores > self.penalty_threshold
        # === HẾT BẢN VÁ ==================================================='''

DAU_HIEU = "APKGC_ENTROPY_PENALTY"


def main():
    if len( sys.argv ) < 2:
        sys.exit( __doc__ )
    root = sys.argv[1]
    revert = "--revert" in sys.argv
    p = os.path.join( root, REL )
    if not os.path.isfile( p ):
        sys.exit( f"  x khong thay {p}" )
    s = open( p, encoding="utf-8" ).read()

    if revert:
        if DAU_HIEU not in s:
            print( "  - chua va, khong can hoan" )
            return 0
        i = s.index( "        # === BẢN VÁ" )
        j = s.index( "# === HẾT BẢN VÁ ===================================================" )
        j = s.index( "\n", j ) + 1
        open( p, "w", encoding="utf-8" ).write( s[:i] + CU + "\n" + s[j:] )
        print( "  v da hoan ve ban goc" )
        return 0

    if DAU_HIEU in s:
        print( "  - da va truoc do" )
        return 0
    if CU not in s:
        sys.exit( "  x khong khop doan can va — mã upstream có thể đã đổi" )
    if "\nimport os" not in s and "^import os" not in s:
        s = s.replace( "import math", "import math\nimport os", 1 )
    open( p, "w", encoding="utf-8" ).write( s.replace( CU, MOI, 1 ) )
    print( "  v da va APKGC.py" )
    print( "\nBat bang:  export APKGC_ENTROPY_PENALTY=1" )
    print( "Phan vi :  export APKGC_ENTROPY_PCT=0.10   (mac dinh)" )
    return 0


if __name__ == "__main__":
    raise SystemExit( main() )
