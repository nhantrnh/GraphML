# GraphML HP3 — mã tái lập cho báo cáo 25C15053

Kho mã đi kèm báo cáo cuối kỳ học phần GraphML (HP3), VNUHCM–University of Science.
Báo cáo gồm hai phần: **APKGC** (AAAI 2025, bài seminar) và **ImparTail** (CVPR 2025).
README này chỉ hướng dẫn cách chạy lại. Kết quả và phân tích nằm trong báo cáo
`report/final/25C15053.pdf`.

```
report/final/   báo cáo cuối kỳ (LaTeX, khổ IEEE) và trang bìa
seminar/        slide và dàn ý seminar APKGC (PDF)
slides/src/     nguồn LaTeX của slide và dàn ý
experiments/    notebook Kaggle, công cụ phân tích, tệp kết quả gốc
impartail/      bản vá dùng khi chạy mã của tác giả (APKGC và ImparTail)
qa/             hai bộ kiểm tra tự động cho số liệu và tài liệu
vimmkg/         bộ dữ liệu đồ thị tri thức đa phương thức tiếng Việt (phụ, không thuộc báo cáo)
papers/         PDF hai bài báo
upstream/       mã của tác giả, tải về khi chạy (không đưa lên git)
```

---

## 1. Môi trường và phần cứng

| Việc | Nơi chạy | Phần cứng | Phần mềm |
|---|---|---|---|
| Huấn luyện, đánh giá APKGC | Kaggle Notebook | GPU Tesla T4 16 GB, phiên tối đa 12 giờ | Python 3.10, torch có sẵn của Kaggle |
| Đánh giá, huấn luyện ImparTail | Kaggle Notebook | GPU Tesla P100 16 GB | `torch==2.4.1+cu121` (P100 là sm_60, torch mới không hỗ trợ) |
| Công cụ phân tích, QA, dựng báo cáo | máy cá nhân | CPU | Python 3.11, `pip install -r requirements.txt`, TeX Live có `xelatex` và `IEEEtran` |

Notebook Kaggle cần bật **GPU** và **Internet** (để clone mã tác giả và tải dữ liệu).

---

## 2. Phần I — APKGC

### 2.1. Chuẩn bị

```bash
git clone https://github.com/HubuKG/APKGC.git upstream/APKGC
```

Đặc trưng ảnh và văn bản của DB15K và MKG-W tải từ
[Google Drive của tác giả](https://drive.google.com/drive/folders/1wGvlzt0i6RYcMniwooFcLKVEs2NBMcpB)
vào `upstream/APKGC/embeddings/`. Các notebook dưới đây tự làm hai bước này.

Mã phát hành cần vài bản vá để chạy được; mọi notebook áp dụng chúng trong ô
"vá mã nguồn": bổ sung mã C++ `mmkgc/base/` từ OpenKE rồi biên dịch lại, chọn thiết
bị CUDA/MPS/CPU thay cho `.cuda()` cứng, bổ sung `scikit-learn` và `scipy`, ghi kết quả
vào `./output/`, kiểm tra `best_model_wts` khi chạy dưới 100 epoch. Bản vá
`impartail/patches/add_seed.py` thêm cờ `--seed`.

### 2.2. Notebook và thực nghiệm tương ứng

Mỗi notebook nằm trong `experiments/notebooks/kaggle/`, là đúng mã nguồn của kernel
Kaggle cùng tên (tải về bằng API). Cách chạy: tải notebook lên Kaggle, bật GPU T4 và
Internet, chọn *Run All*. Tệp kết quả ghi ra `/kaggle/working/`, sau đó chép vào
`experiments/results/`.

| Thực nghiệm (mục trong báo cáo) | Notebook | Kernel Kaggle | Tệp kết quả |
|---|---|---|---|
| Ba seed và cấu hình theo bài (IV-A) | `apkgc-remaining-exp.ipynb` | [nhantrnh/apkgc-remaining-exp](https://www.kaggle.com/code/nhantrnh/apkgc-remaining-exp) | `apkgc_res/seed*.json`, `apkgc_res2/paper_config.json` |
| DB15K (IV-A) | `apkgc-db15k-capped.ipynb` | [thnhnhntrng/apkgc-db15k-capped](https://www.kaggle.com/code/thnhnhntrng/apkgc-db15k-capped) | `apkgc_res/db15k_seed42.json` |
| Biến thể phạt A, B (IV-B) | `apkgc-three-branch.ipynb` | [thnhnhntrng/apkgc-three-branch](https://www.kaggle.com/code/thnhnhntrng/apkgc-three-branch) | `apkgc_res/three_branch.json` |
| Biến thể phạt C (IV-B) | `apkgc-entropy-c.ipynb` | [thnhnhntrng/apkgc-entropy-c](https://www.kaggle.com/code/thnhnhntrng/apkgc-entropy-c) | `apkgc_res/three_branch.json` |
| Biến thể phạt B2 (IV-B) | `apkgc-cap-prob.ipynb` | [nhantrnh/apkgc-cap-prob](https://www.kaggle.com/code/nhantrnh/apkgc-cap-prob) | `apkgc_res/three_branch.json` |
| Bộ đếm lệnh phạt (IV-B) | `apkgc-attn-probe.ipynb` | [thnhnhntrng/apkgc-attn-probe](https://www.kaggle.com/code/thnhnhntrng/apkgc-attn-probe) | `apkgc_res/attn_scale.json` |
| Bỏ lấy mẫu nhiễu (IV-C) | `apkgc-ablation-noise.ipynb` | [thnhnhntrng/apkgc-ablation-noise](https://www.kaggle.com/code/thnhnhntrng/apkgc-ablation-noise) | `apkgc_res/three_branch.json` |
| Huấn luyện lại NativE trên đặc trưng của APKGC (IV-A) | `../kaggle_native_mkgw.ipynb` (đặt `SEED = 42` hoặc `1234`) | [nhantrnh/native-mkgw-s42](https://www.kaggle.com/code/nhantrnh/native-mkgw-s42) (seed 42) | `native_res/native_s42_e1000.json`, `native_res/native_s1234_e1000.json` |
| APKGC biến thể A, seed 2024 (IV-D) | `apkgc-A-seed2024.ipynb` | [thnhnhntrng/apkgc-a-seed2024](https://www.kaggle.com/code/thnhnhntrng/apkgc-a-seed2024) | `apkgc_res5/A_goc_seed2024.json`, `robustness/robustness_apkgc_A_seed2024.json` |
| Độ bền khi làm hỏng đặc trưng lúc test (IV-D) | `mmkgc-robustness-eval.ipynb` | [nhantrnh/mmkgc-robustness-eval](https://www.kaggle.com/code/nhantrnh/mmkgc-robustness-eval) | `robustness/robustness_*.json` |

Notebook độ bền cần dataset Kaggle chứa ba checkpoint (APKGC biến thể A seed 42, NativE seed 42
và 1234) cùng `missing.json`; dựng lại bằng cách tải checkpoint tốt nhất từ output của
`apkgc-three-branch` và hai kernel NativE. Trước khi đo, công cụ kiểm MRR ở điều kiện sạch phải
khớp MRR đã ghi của checkpoint (tham số `--expect`), lệch thì dừng.

Tham số θ_r, θ_n sau huấn luyện (mục IV-C) được đọc từ checkpoint trong
output của `apkgc-three-branch` (phiên bản 1); cách đọc và nguồn ghi trong
`experiments/results/apkgc_res/learned_ratios.json`.

Biến thể B và C bật bằng biến môi trường, không bật thì mã giữ nguyên hành vi gốc:
`APKGC_PAPER_FORMULA=1` (bản vá `impartail/patches/apkgc_paper_formula.py`),
`APKGC_ENTROPY_PENALTY=1` và `APKGC_ENTROPY_PCT=0.10`
(`impartail/patches/apkgc_entropy_penalty.py`). B2 bật bằng `APKGC_CAP_PROB`. Hai bản
vá có `--revert` để hoàn nguyên mã gốc.

Notebook cũ hơn trong `experiments/notebooks/`: `kaggle_apkgc_seeds.ipynb`,
`kaggle_seed_study.ipynb`, `kaggle_remaining_exp.ipynb` (các đợt chạy seed đầu tiên),
`kaggle_native_mkgw.ipynb` (baseline NativE trên MKG-W), `APKGC_Colab.ipynb` và
`APKGC_Ablation_Colab.ipynb` (bản Colab), `MyGO_Trace_Colab.ipynb`,
`ViMMKG_Features_Colab.ipynb` và `kaggle_vimmkg_features.ipynb` (phục vụ ViMMKG).

---

## 3. Phần II — ImparTail

### 3.1. Chuẩn bị

```bash
git clone https://github.com/rohithpeddi/ImparTail.git upstream/ImparTail
```

Dữ liệu Action Genome cần đủ sáu phần khung hình (khoảng 58 GB) và tệp chú thích.
Checkpoint của tác giả (STTran PredCls và detector dùng chung) lấy theo bản đồ trong
`experiments/results/impartail_repro/checkpoints.json`;
`experiments/tools/box_checkpoint_ids.py` giữ mã tệp trên Box.

Bản vá để chạy trên Kaggle P100: `impartail/patches/fix_thc.py` (header `THC` bị gỡ
khỏi torch), `impartail/patches/fix_numpy_alias.py` (`np.float`, `np.bool`),
`impartail/patches/stub_analysis.py` (gói `analysis` thiếu khi clone). Checkpoint STTran
phải đổi tên thành `sttran_predcls_epoch_3.tar` cho khớp cách mã tách tên.

### 3.2. Notebook và thực nghiệm tương ứng

| Thực nghiệm (mục trong báo cáo) | Notebook | Kernel Kaggle | Tệp kết quả |
|---|---|---|---|
| Đánh giá STTran PredCls (VIII-A) | `impartail-eval.ipynb` | [nhantrnh/impartail-eval](https://www.kaggle.com/code/nhantrnh/impartail-eval) | `impartail_repro/sttran_predcls_v8.csv` |
| Huấn luyện mã gốc (VIII-B) | `impartail-train-original.ipynb` | [nhantrnh/impartail-train-original](https://www.kaggle.com/code/nhantrnh/impartail-train-original) | `impartail_repro/sttran_partial10_goc.csv` |
| Huấn luyện bản vá (VIII-B) | `impartail-train-patched.ipynb` | [nhantrnh/impartail-train-patched](https://www.kaggle.com/code/nhantrnh/impartail-train-patched) | `impartail_repro/sttran_partial10_va.csv` |

Bản vá mặt nạ theo công thức (5): `impartail/patches/fix_masked_bce.py`, bật bằng cờ
`--true_masked_bce`; không truyền cờ thì giữ hành vi gốc. Hai notebook huấn luyện chỉ
khác nhau ở cờ này. `impartail/patches/dump_per_class_recall.py` xuất recall từng lớp.
`impartail/notebooks/impartail_eval.ipynb` là bản cục bộ của notebook đánh giá.

---

## 4. Weights & Biases

Hai project công khai: [apkgc-repro](https://wandb.ai/nhantrnh-graphml/apkgc-repro) và
[impartail-repro](https://wandb.ai/nhantrnh-graphml/impartail-repro). Run được dựng từ
tệp kết quả đã lưu, không ghi trực tiếp lúc huấn luyện:

```bash
export WANDB_API_KEY=...            # khoá của bạn, không ghi vào tệp
python experiments/tools/wandb_upload.py --entity nhantrnh-graphml
python experiments/tools/wandb_upload_extra.py --entity nhantrnh-graphml
```

`wandb_upload.py` đẩy các lần chạy có vết đánh giá theo thời gian; `wandb_upload_extra.py`
đẩy các thực nghiệm chỉ còn số tổng kết, đặt project ở chế độ công khai và ghi link mọi
run vào `experiments/results/wandb_runs_all.json`.

---

## 5. Dựng báo cáo, slide và kiểm tra số liệu

```bash
# báo cáo cuối kỳ
cd report/final && xelatex cover.tex && xelatex 25C15053.tex && xelatex 25C15053.tex

# slide và dàn ý seminar APKGC
cd slides/src && xelatex apkgc_slides.tex && python md2tex_dany.py && xelatex dan_y_apkgc.tex

# kiểm tra số liệu và tài liệu
python qa/verify_numbers.py     # so từng con số trong tài liệu với tệp kết quả gốc
python qa/check_repo.py         # bắt tài liệu lệch với cây thư mục và kết quả
```

---

## 6. Công cụ phân tích (`experiments/tools/`)

Mỗi công cụ chạy bằng `python experiments/tools/<tên>.py`, đọc tệp trong
`experiments/results/` hoặc mã tác giả trong `upstream/`.

APKGC:
`audit_gradient.py`, `audit_apkgc_real.py` (dò tham số không nhận gradient),
`audit_static.py` (quét tĩnh bằng AST),
`apkgc_penalty_defect.py` (mô phỏng điều kiện kích hoạt của cơ chế phạt),
`vectorize_penalty.py` (đo tốc độ phép tích luỹ hậu tố),
`concrete_noise.py` (thử các biến thể mặt nạ nhiễu),
`pilot_seed_study.py` (đọc siêu tham số từ script tác giả),
`selection_bias.py`, `make_valid_selection.py`, `survey_selection.py`
(so cách chọn checkpoint theo tập test và tập phát triển),
`wandb_upload.py`, `wandb_upload_extra.py`, `wandb_upload_robustness.py` (mục 4).
`apkgc_robustness.py` (công cụ đo của mục IV-D; chạy bằng notebook `mmkgc-robustness-eval.ipynb`).

ImparTail:
`impartail_compare.py` (đối chiếu số tái lập với bảng của bài),
`impartail_tradeoff.py`, `vidsgg_hr_table.py` (tính Recall, mean Recall, hR từ PDF bài),
`impartail_mask_gap.py`, `masked_bce_bias.py` (phân tích mặt nạ trong hàm mất mát),
`compare_mask_branches.py` (so hai nhánh huấn luyện),
`impartail_dropped_pairs.py`, `group_fallback_viability.py`, `masking_equivalence.py`,
`recall_at_k_saturation.py`, `mrecall_denominator.py`, `macro_vs_global.py`,
`zipf_fit_check.py`, `rank_reversal_bound.py`, `four_theorems.py`,
`theorem4_prediction.py`, `global_equilibrium.py`, `decisive_test.py`,
`vidor_contrast.py` (các phân tích phụ về chỉ số và phân bố nhãn),
`box_checkpoint_ids.py` (mục 3.1).

Quy ước: số dòng trong tài liệu chỉ tới mã gốc chưa vá của tác giả.

---

## 7. ViMMKG (phụ)

Bộ dữ liệu đồ thị tri thức đa phương thức tiếng Việt, không thuộc báo cáo. Dựng lại:

```bash
python vimmkg/src/build_vimmkg.py --min-deg 3
python vimmkg/src/gen_type_constrain.py
python vimmkg/src/extract_features.py
```

Các script khác: `vimmkg/src/wdq.py` (truy vấn Wikidata), `vimmkg/src/gen_missing.py`,
`vimmkg/src/sanity_transe.py`, `vimmkg/src/imgtest.py`, `vimmkg/src/txttest.py`,
`vimmkg/src/probe.py`. Chi tiết: `vimmkg/README.md`.
