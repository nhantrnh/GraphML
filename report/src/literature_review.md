# LITERATURE REVIEW

## Các hạn chế của APKGC và tình trạng nghiên cứu liên quan

**Bài gốc:** APKGC — Noise-enhanced Multi-Modal Knowledge Graph Completion with Attention Penalty (AAAI 2025, Oral, tr. 15005–15013)

**Phạm vi rà soát:** arXiv, ACL Anthology, AAAI/IJCAI/SIGIR/WWW proceedings, ScienceDirect, MDPI, GitHub — cập nhật đến tháng 8/2026.

**Nguyên tắc:** mọi khẳng định đều kèm nguồn. Khi không tìm được bằng chứng, báo cáo ghi rõ "không tìm thấy" thay vì suy đoán.

---

## 0. Một lưu ý quan trọng về tên gọi

Trong quá trình rà soát xuất hiện **hai công trình khác nhau cùng viết tắt là APKGC**:

| | Bài của chúng ta | Bài trùng tên |
|---|---|---|
| Tên đầy đủ | **A**ttention **P**enalty **KGC** | **A**daptive and **P**rompt-tuning **KGC** |
| Nội dung | Tiêm nhiễu + phạt chú ý cho MMKGC | Prompt-tuning LLM cho KGC đơn mô thái |
| Công bố | AAAI 2025 | Knowledge-Based Systems 2025 |
| Nguồn | [ojs.aaai.org/.../33645](https://ojs.aaai.org/index.php/AAAI/article/view/33645) | [ScienceDirect S0950705125019884](https://www.sciencedirect.com/science/article/abs/pii/S0950705125019884) |

Khi đọc các bài trích dẫn "APKGC", cần kiểm tra kỹ đang nói tới bài nào. Trong báo cáo này, "APKGC" luôn chỉ bài AAAI 2025.

---

## 1. Tình trạng trích dẫn APKGC

**Kết quả rà soát:** APKGC xuất hiện trong danh mục tham khảo của một số công trình MMKGC 2025–2026, nhưng **không tìm thấy công trình nào phân tích sâu, phản biện, hay báo cáo thất bại khi tái lập** kết quả của nó.

Đáng chú ý: APKGC **vắng mặt trong bảng so sánh của nhiều bài MMKGC mới**, kể cả những bài công bố sau AAAI 2025 và dùng đúng DB15K, MKG-W. Ví dụ [RADD (arXiv 2604.25693)](https://arxiv.org/abs/2604.25693) không nhắc tới APKGC. Điều này có thể do độ trễ giữa thời điểm nộp bài và thời điểm APKGC xuất bản, nhưng cũng phản ánh mức độ ảnh hưởng còn hạn chế.

**Chưa có báo cáo tái lập độc lập nào** — không tìm thấy issue trên GitHub, bài blog, hay công trình nào xác nhận hoặc bác bỏ con số của tác giả. Đây chính là khoảng trống mà nhóm có thể lấp bằng phần thực nghiệm của mình.

---

## 2. Rà soát từng hạn chế

### H1 — Không báo cáo độ lệch chuẩn qua nhiều seed

**Trạng thái: chưa được khắc phục trong APKGC, và là vấn đề mang tính hệ thống của cả lĩnh vực.**

Đây là hạn chế nghiêm trọng nhất, vì khoảng cách giữa APKGC và NativE trên DB15K chỉ 0.002 MRR.

Công trình [**Link Prediction or Perdition: the Seeds of Instability in Knowledge Graph Embeddings**](https://arxiv.org/abs/2606.03365) (Méroué, Gandon, Monnin — arXiv 2606.03365, tháng 6/2026) cung cấp bằng chứng định lượng trực tiếp liên quan:

- **RotatE trên WN18RR: MRR = 0.326 ± 0.039.** Độ lệch chuẩn 0.039 — lớn gấp gần **20 lần** khoảng cách 0.002 mà APKGC dùng để tuyên bố vượt NativE.
- RotatE trên FB15k-237: 0.288 ± 0.001 — ổn định hơn nhiều. Nghĩa là mức bất ổn **phụ thuộc mạnh vào bộ dữ liệu**, không thể suy đoán cho DB15K/MKG-W nếu chưa đo.
- Phát hiện sâu hơn: ngay cả khi độ lệch chuẩn nhỏ, hai lần chạy khác seed **chỉ trùng trung bình 4.3 trên 10 thực thể dự đoán đầu bảng** ở FB15k-237, và **khoảng 40% bộ ba cho ra đáp án hạng nhất khác nhau**.

Điểm này đặc biệt đáng chú ý với APKGC vì mô hình **dùng chính RotatE làm hàm chấm điểm**.

Nhóm tác giả trên khuyến nghị rõ: báo cáo kết quả qua nhiều seed kèm thước đo độ ổn định, và xem ổn định là chiều đánh giá độc lập với hiệu năng.

Bài [**When Metrics Disagree: A Meta-Analysis of Knowledge-Graph-Completion Model Benchmarking**](https://arxiv.org/html/2606.10287v1) (arXiv 2606.10287, 2026) củng cố thêm rằng thực hành đánh giá trong lĩnh vực KGC còn thiếu nhất quán.

> **Kết luận H1:** Không tìm thấy bài MMKGC nào trên DB15K/MKG-W báo cáo độ lệch chuẩn qua nhiều seed. **Đây là khoảng trống thật, và là cơ hội đóng góp rõ ràng nhất cho nhóm.**

---

### H2 — Thua NativE ở Hits@10 trên DB15K

**Trạng thái: chưa ai giải thích.**

Không tìm thấy công trình nào phân tích vì sao attention penalty cải thiện Hits@1 nhưng không cải thiện Hits@10.

Giả thuyết có cơ sở lý thuyết: cơ chế phạt–tái phân bổ làm **sắc nét hoá** phân phối chú ý, giúp đẩy đáp án đúng lên hạng nhất, nhưng đồng thời **thu hẹp vùng ứng viên** mà mô hình cân nhắc, gây bất lợi ở các ngưỡng xếp hạng rộng. Đây là giả thuyết kiểm chứng được bằng thực nghiệm và là một hướng đóng góp tốt.

---

### H3 — Ngưỡng cứng không biện luận

**Trạng thái: chưa khắc phục trong MMKGC, nhưng lĩnh vực khác đã có lời giải sẵn sàng áp dụng.**

APKGC hardcode `penalty_threshold = 10`, `penalty_max_threshold = 15`, `penalty_weights = 0.001`, `window_size = 1` ngay trong mã, không phân tích độ nhạy.

Đối chiếu với nghiên cứu về attention sink — cùng bản chất hiện tượng:

- [**When Attention Sink Emerges in Language Models**](https://proceedings.iclr.cc/paper_files/paper/2025/file/f1b04face60081b689ba740d39ea8f37-Paper-Conference.pdf) (ICLR 2025) phân tích cơ chế hình thành attention sink, cung cấp nền lý thuyết mà APKGC còn thiếu.
- [**Attention Sink in Transformers: A Survey on Utilization, Interpretation, and Mitigation**](https://arxiv.org/pdf/2604.10098) (arXiv 2604.10098, 2026) tổng hợp các hướng giảm thiểu — tài liệu nền rất tốt cho báo cáo.
- [**Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free**](https://openreview.net/forum?id=1b7whO4SfY) đề xuất cổng sigmoid theo từng đầu chú ý, **loại bỏ attention sink từ gốc thay vì phạt sau khi nó xảy ra** — về mặt thiết kế là hướng sạch hơn APKGC.
- Nghiên cứu về entropy regularization đã dùng **ngưỡng động theo từng đầu chú ý**, tính như một tỉ lệ học được của entropy cực đại, kèm biên dung sai để tránh phạt quá tay. Đây chính xác là thứ APKGC thiếu.

> **Kết luận H3:** Giải pháp đã tồn tại ở lĩnh vực NLP/VLM nhưng **chưa ai mang sang MMKGC**. Ghép ngưỡng thích ứng vào APKGC là hướng khả thi, chi phí thấp, đóng góp rõ.

---

### H4 — Cấu hình khác nhau theo từng tập dữ liệu

**Trạng thái: chưa ai kiểm chứng.**

DB15K dùng nhiễu Gaussian + `Mformer_hd_graph`; MKG-W dùng nhiễu trung bình + `Mformer_hd_mean`. Bài báo trình bày đây là sự thích ứng theo đặc tính dữ liệu, nhưng không có thí nghiệm chéo chứng minh.

Thí nghiệm cần làm rất đơn giản: chạy cấu hình DB15K trên MKG-W và ngược lại. Nếu hiệu năng sụt mạnh, đó là dấu hiệu điều chỉnh riêng cho từng tập. **Không tìm thấy ai đã làm.**

---

### H5 — Chỉ hai bộ dữ liệu

**Trạng thái: hạn chế thật, nhưng là chuẩn mực chung của lĩnh vực.**

Phần lớn công trình MMKGC dùng DB15K, MKG-W, MKG-Y. APKGC bỏ MKG-Y — một thiếu sót vì đây là tập cùng họ, sẵn có, và các bài khác đều báo cáo. Ví dụ [RADD](https://arxiv.org/pdf/2604.25693) báo cáo đủ cả ba (MRR 41.17 / 36.48 / 38.96 trên DB15K / MKG-W / MKG-Y).

Bổ sung MKG-Y là thí nghiệm rẻ nhất mà nhóm có thể làm để mở rộng phạm vi kiểm chứng.

---

### H6 — Suffix product không vector hoá

**Trạng thái: lỗi kỹ thuật thuần tuý, chưa ai nêu, sửa được ngay.**

```python
# Mã gốc — O(L) lần gọi kernel
for i in range(attention_scores.shape[-1]):
    penalty_scores[..., i] = attention_scores[..., i:].prod(dim=-1)

# Tương đương, một lần gọi
penalty_scores = attention_scores.flip(-1).cumprod(-1).flip(-1)
```

Với APKGC, chiều cuối chỉ bằng 3 (ba mô thái) nên tác động thực tế nhỏ. Nhưng nếu mở rộng sang video/audio hoặc token mịn thì chi phí tăng tuyến tính. Đây là chi tiết cho thấy nhóm đã đọc mã ở mức triển khai.

---

### H7 — Chỉ ba mô thái

**Trạng thái: dữ liệu đã có sẵn từ 2023, nhưng chưa ai áp dụng cho APKGC.**

Chính tác giả nêu hướng mở rộng sang video và âm thanh trong phần kết luận. Hạ tầng dữ liệu đã tồn tại:

- [**TIVA-KG: A Multimodal Knowledge Graph with Text, Image, Video and Audio**](https://dl.acm.org/doi/abs/10.1145/3581783.3612266) (ACM MM 2023) — đủ bốn mô thái, kèm mô hình nền **QEB (Quadruple Embedding Baseline)**. Điểm đặc biệt: TIVA-KG gắn mô thái vào **bộ ba** chứ không chỉ vào thực thể. Dữ liệu công khai tại http://mn.cs.tsinghua.edu.cn/tivakg
- [**VAT-KG**](https://arxiv.org/pdf/2506.21556) (2025) — đồ thị tri thức đa mô thái hướng truy xuất.
- **SciMKG** (2026) — văn bản, ảnh, video, âm thanh cho giáo dục khoa học.

Cơ chế tương quan chéo `θ_ab` của APKGC về nguyên tắc mở rộng được cho n mô thái. **Không tìm thấy công trình nào đã thử.**

---

## 3. Hiện trạng SOTA trên DB15K và MKG-W

Cảnh báo về cách đọc: các bài dưới đây dùng bộ nền so sánh khác nhau, và phần lớn **không có APKGC trong bảng**. Con số vì vậy không hoàn toàn so trực tiếp được.

| Công trình | Nguồn | Kết quả công bố | Có so với APKGC? |
|---|---|---|---|
| APKGC (AAAI'25) | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/33645) | DB15K MRR 0.364 / MKG-W 0.374 | — |
| **RADD** | [arXiv 2604.25693](https://arxiv.org/abs/2604.25693) | MRR 41.17 (DB15K), 36.48 (MKG-W), 38.96 (MKG-Y) | ❌ Không nhắc |
| **RAPC** | [Applied Sciences 16(14):6944](https://doi.org/10.3390/app16146944) | +2.64 MRR, +3.57 H@1 (DB15K); +3.02 MRR, +3.72 H@1 (MKG-W) so với nền tốt nhất | Không xác minh được (bản đầy đủ bị chặn) |
| **MgHiSal** | [KBS S0950705125015916](https://www.sciencedirect.com/science/article/abs/pii/S0950705125015916) | +13.1% MRR (DB15K), +12.5% (MKG-W) so với á quân | Không xác minh được |
| **ELMM** | [arXiv 2510.16753](https://arxiv.org/abs/2510.16753) | Tuyên bố SOTA trên 4 tập | ❌ Không nhắc |
| **DiffusionCom** | [arXiv 2504.06543](https://arxiv.org/pdf/2504.06543) | Diffusion có nhận thức cấu trúc | Chưa rà |

**Nhận định:** nếu RADD đạt MRR 41.17 trên DB15K (tức 0.4117), con số này **vượt APKGC (0.364) khá xa**. Tuy nhiên RADD dùng khung retrieval-augmented discrete diffusion — khác hẳn về kiến trúc, và cần đọc kỹ bảng đầy đủ để xác nhận cùng thiết lập đánh giá.

Kết luận thận trọng: **APKGC nhiều khả năng không còn là SOTA tính đến giữa 2026**, nhưng chưa có bài nào so sánh trực tiếp và công bằng với nó.

---

## 4. Khoảng trống còn lại và đề xuất nghiên cứu

Tổng hợp: trong bảy hạn chế, **không hạn chế nào đã được giải quyết trực tiếp cho APKGC**. Dưới đây là năm hướng khả thi trong 1–2 tháng với một GPU.

### Đề xuất 1 — Kiểm định độ ổn định qua nhiều seed ⭐ ưu tiên cao nhất

**Vấn đề nhắm tới:** H1, H2

**Cách làm:** chạy APKGC với 5 seed trên cả DB15K và MKG-W, báo cáo trung bình ± độ lệch chuẩn. Bổ sung thước đo ổn định ở mức bộ ba theo phương pháp của [Méroué et al. 2026](https://arxiv.org/abs/2606.03365): tỉ lệ trùng lặp top-10 giữa các lần chạy.

**Vì sao đáng làm:** khoảng cách 0.002 MRR so với NativE quá nhỏ. Nếu độ lệch chuẩn vượt con số này, kết luận "vượt SOTA" của bài báo mất cơ sở — một phát hiện có giá trị thật. Ngược lại nếu ổn định, đó là bằng chứng củng cố bài báo.

**Chi phí:** 5 lần chạy × 2 tập. Cao nhất trong các đề xuất nhưng cũng giá trị nhất.

**Rủi ro:** thấp. Kết quả nào cũng đáng báo cáo.

---

### Đề xuất 2 — Thay ngưỡng cứng bằng ngưỡng thích ứng

**Vấn đề nhắm tới:** H3

**Cách làm:** thay `penalty_threshold = 10` bằng phân vị của phân phối `penalty_scores` trong từng lô, hoặc bằng một tỉ lệ học được như cách entropy regularization làm. So sánh với ngưỡng cố định trên cùng seed.

**Nền tham khảo:** [Attention Sink Survey](https://arxiv.org/pdf/2604.10098), [Gated Attention](https://openreview.net/forum?id=1b7whO4SfY)

**Vì sao đáng làm:** giải quyết trực tiếp điểm yếu rõ nhất về mặt thiết kế, và Hình 3 của bài báo đã chứng minh hiệu năng rất nhạy với ngưỡng — tức có nhiều dư địa cải thiện.

**Chi phí:** thấp. Sửa vài chục dòng, chạy 2–3 cấu hình.

---

### Đề xuất 3 — Thí nghiệm chéo cấu hình

**Vấn đề nhắm tới:** H4

**Cách làm:** chạy cấu hình DB15K (Gaussian + `Mformer_hd_graph`) trên MKG-W và ngược lại. Lập bảng 2×2.

**Vì sao đáng làm:** trả lời dứt điểm câu hỏi mô hình có bị điều chỉnh riêng cho từng tập hay không. Bốn lần chạy là đủ kết luận.

**Chi phí:** rất thấp.

---

### Đề xuất 4 — Mở rộng sang MKG-Y

**Vấn đề nhắm tới:** H5

**Cách làm:** chạy APKGC trên MKG-Y — tập cùng họ mà bài báo bỏ qua, và các công trình khác đều báo cáo.

**Vì sao đáng làm:** rẻ, và cho phép so sánh trực tiếp với RADD cùng nhiều bài khác trên ba tập thay vì hai.

**Lưu ý:** cần bộ nhúng MKG-Y. Nếu tác giả không chia sẻ, phải tự trích bằng BEiT và BERT — chi phí tăng đáng kể.

---

### Đề xuất 5 — Vector hoá và đo tốc độ

**Vấn đề nhắm tới:** H6

**Cách làm:** thay vòng lặp bằng `flip → cumprod → flip`, xác nhận kết quả trùng khớp, đo thời gian mỗi epoch trước và sau.

**Vì sao đáng làm:** nhỏ nhưng cụ thể, chứng minh nhóm đọc mã ở mức triển khai chứ không chỉ đọc bài báo.

**Chi phí:** rất thấp.

---

## 5. Tổng kết

Sau khi rà soát, ba nhận định chính:

**Thứ nhất,** APKGC chưa được cộng đồng kiểm chứng độc lập. Không có báo cáo tái lập, không có phân tích phản biện, và bài vắng mặt trong bảng so sánh của nhiều công trình mới cùng lĩnh vực.

**Thứ hai,** hạn chế nghiêm trọng nhất — thiếu độ lệch chuẩn — không phải lỗi riêng của APKGC mà là vấn đề hệ thống của lĩnh vực KGC. Nghiên cứu [Méroué et al. 2026](https://arxiv.org/abs/2606.03365) cho thấy RotatE, đúng hàm chấm điểm mà APKGC dùng, có thể dao động tới ±0.039 MRR — lớn gấp 20 lần khoảng cách mà bài báo dùng để tuyên bố vượt SOTA.

**Thứ ba,** lời giải cho vấn đề ngưỡng cứng đã tồn tại trong nghiên cứu về attention sink ở NLP và thị giác–ngôn ngữ, nhưng chưa ai mang sang MMKGC. Đây là khoảng trống có thể lấp bằng công sức vừa phải.

Với mục tiêu seminar và điểm cộng thực nghiệm, **Đề xuất 1 và 2** là kết hợp tốt nhất: một hướng kiểm định tính vững của kết quả đã công bố, một hướng cải tiến thiết kế, cả hai đều nằm trong khả năng của một GPU và vài tuần làm việc.

---

## Danh mục nguồn

**Bài gốc**
- [APKGC — AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33645) · [Mã nguồn](https://github.com/HubuKG/APKGC)

**Về độ ổn định và đánh giá**
- [Link Prediction or Perdition: the Seeds of Instability in Knowledge Graph Embeddings — arXiv 2606.03365](https://arxiv.org/abs/2606.03365)
- [When Metrics Disagree: A Meta-Analysis of KGC Model Benchmarking — arXiv 2606.10287](https://arxiv.org/html/2606.10287v1)
- [Bringing Light Into the Dark: A Large-scale Evaluation of KGE Models — arXiv 2006.13365](https://arxiv.org/html/2006.13365)

**Về attention sink và over-trust**
- [When Attention Sink Emerges in Language Models — ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/f1b04face60081b689ba740d39ea8f37-Paper-Conference.pdf)
- [Attention Sink in Transformers: A Survey — arXiv 2604.10098](https://arxiv.org/pdf/2604.10098)
- [Gated Attention for Large Language Models — OpenReview](https://openreview.net/forum?id=1b7whO4SfY)
- [Attention Sinks Are Provably Necessary in Softmax Transformers — arXiv 2603.11487](https://arxiv.org/pdf/2603.11487)

**MMKGC gần đây**
- [RADD: Retrieval-Augmented Discrete Diffusion — arXiv 2604.25693](https://arxiv.org/abs/2604.25693)
- [ELMM: Efficient Lightweight MLLM for MMKGC — arXiv 2510.16753](https://arxiv.org/abs/2510.16753)
- [DiffusionCom: Structure-Aware Multimodal Diffusion — arXiv 2504.06543](https://arxiv.org/pdf/2504.06543)
- [RAPC — Applied Sciences 16(14):6944](https://doi.org/10.3390/app16146944)
- [MgHiSal — Knowledge-Based Systems](https://www.sciencedirect.com/science/article/abs/pii/S0950705125015916)
- [When Modalities Remember: Continual Learning for MMKG — arXiv 2604.02778](https://arxiv.org/html/2604.02778)
- [NativE: MMKGC in the Wild — arXiv 2406.17605](https://arxiv.org/pdf/2406.17605)

**Đa mô thái mở rộng**
- [TIVA-KG: Text, Image, Video and Audio — ACM MM 2023](https://dl.acm.org/doi/abs/10.1145/3581783.3612266)
- [VAT-KG — arXiv 2506.21556](https://arxiv.org/pdf/2506.21556)
- [KG-MM-Survey — GitHub](https://github.com/zjukg/KG-MM-Survey)
