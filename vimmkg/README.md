# vimmkg/src/ — mã và dữ liệu cho ViMMKG

Mọi con số trong `../PROPOSAL_ViMMKG.md` và `../TRACKING_ViMMKG.md` đều sinh ra
từ đây. Cài phụ thuộc: `pip install -r requirements.txt`.

## Thứ tự chạy

| # | Tệp | Việc | Đầu ra |
|---|---|---|---|
| 1 | `wdq.py` | Truy vấn Wikidata SPARQL lấy thực thể tiếng Việt có ảnh P18 | `wikidata_vi_entities.json` |
| 2 | — | Truy vấn cạnh giữa các thực thể đó | `triples_raw.json` (ở cache) |
| 3 | `build_vimmkg.py --min-deg 3` | Gộp cặp nghịch đảo, lọc k-core, chia 8/1/1 | `dataset/` |
| 4 | `gen_type_constrain.py` | Sinh `type_constrain.txt` — **bắt buộc**, thiếu sẽ segfault | trong `dataset/` |
| 5 | `imgtest.py`, `txttest.py` | Đo tỉ lệ tải ảnh và độ phủ văn bản | in ra màn hình |
| 6 | `extract_features.py` | Trích CLIP 512-d + e5-large 1024-d | `ViMMKG-{visual,textual}.pth` |
| 7 | `gen_missing.py` | Liệt kê chỉ số thiếu mô thái | `missing.json` |
| 8 | `sanity_transe.py` | Phép thử bộ dữ liệu có học được không | `transe_*.json` |
| — | `probe.py` | Thí nghiệm dò C4 — **đã bác bỏ**, giữ làm bằng chứng | không chạy được (thiếu nhãn) |

## Dữ liệu kèm theo

| Tệp | Nội dung |
|---|---|
| `wikidata_vi_entities.json` | 42.081 thực thể, 26 lớp |
| `hv_rad.json` | Ánh xạ Hán-Việt ↔ bộ thủ, ghép từ Unihan `kRSUnicode` + `ph0ngp/hanviet-pinyin-wordlist` |
| `dataset/` | ViMMKG-d3: 12.334 thực thể, 329 quan hệ, 45.284/5.612/5.646 |
| `transe_vimmkg_d3.json` | MRR 0,1651 |
| `transe_mkgw.json` | Đối chứng MKG-W cùng siêu tham số: MRR 0,1281 |

## Hai điều dễ vấp

1. **`type_constrain.txt` bắt buộc.** `TestDataLoader` mặc định `type_constrain=True`
   và `fopen` không kiểm tra NULL — thiếu tệp là segfault, exit code 0, không báo lỗi.
2. **Commons chặn tốc độ.** Trên 4 luồng sẽ bị HTTP 429. Cấu hình đã đo an toàn:
   4 luồng, nghỉ 0,5 s, lùi cấp số nhân 5/10/20/40 s.
