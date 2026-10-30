<p align="center">
  <a href="https://aicon-conf.eai-conferences.org/2026/"><img src="docs/images/eai-aicon-2026-readme-banner.svg" alt="EAI AICON 2026 — 6th EAI International Conference on Artificial Intelligence for Communications and Networks" width="900"></a>
</p>

<h1 align="center">ViSAQA</h1>

<p align="center">
  <a href="README.md"><img src="docs/images/badges/language-vi-active.svg" alt="Tiếng Việt" height="36"></a>
  <a href="README.en.md"><img src="docs/images/badges/language-en.svg" alt="English" height="36"></a>
</p>

<p align="center"><strong>Đánh giá tính đầy đủ của bằng chứng và khả năng nhận biết phạm vi<br>trong hỏi đáp công tác sinh viên bằng tiếng Việt</strong></p>

<p align="center"><em>ViSAQA: Benchmarking Evidence Sufficiency and Scope Awareness in Vietnamese Student Affairs Question Answering</em></p>

<p align="center">Huu Dung Nguyen · Phuoc Tran · Van Thinh Nguyen · <b>Bao Nhan Chau</b> · Hoang Kiet Duong · Van Dung Nguyen</p>

<p align="center">
  <a href="paper/ViSAQA.pdf"><img src="docs/images/badges/paper.svg" alt="Paper PDF" height="36"></a>
  <a href="https://huggingface.co/spaces/BaoNhan/tra-cuu-ctsv"><img src="docs/images/badges/demo.svg" alt="Live Demo — Hugging Face" height="36"></a>
  <a href="visaqa/"><img src="docs/images/badges/code.svg" alt="Source code" height="36"></a>
  <a href="#kết-quả-trong-bài-báo"><img src="docs/images/badges/results-vi.svg" alt="Kết quả" height="36"></a>
  <a href="docs/REPRODUCIBILITY.md"><img src="docs/images/badges/guide-vi.svg" alt="Hướng dẫn" height="36"></a>
  <a href="#trích-dẫn"><img src="docs/images/badges/cite-vi.svg" alt="Trích dẫn" height="36"></a>
</p>

<p align="center">
  <a href="https://aicon-conf.eai-conferences.org/2026/"><img src="docs/images/badges/status.svg" alt="EAI AICON 2026 — To Appear" height="36"></a>
</p>

Bài báo đã được chấp nhận tại Hội nghị quốc tế EAI lần thứ 6 về Trí tuệ nhân tạo cho Truyền thông và Mạng. **Trạng thái kỷ yếu: To Appear.**

## Giới thiệu

ViSAQA nghiên cứu bài toán hỏi đáp công tác sinh viên bằng tiếng Việt, tập trung vào tính đầy đủ của bằng chứng và khả năng nhận biết câu hỏi ngoài phạm vi. Bộ đánh giá gồm **15.879 mẫu thuộc 21 chủ đề**, với ba nhóm: có thể trả lời, không đủ thông tin và ngoài phạm vi. Nghiên cứu đánh giá riêng chất lượng truy xuất, chất lượng câu trả lời trên các câu có đáp án và hành vi từ chối, với hai cách chia dữ liệu: tách biệt theo chủ đề và chia ngẫu nhiên.

Bài báo đánh giá ViT5, BARTpho, mBART, mT5-base và mT5-small trong các điều kiện chỉ có câu hỏi, BM25, truy xuất dense và truy xuất hybrid. Hiệu quả của truy xuất phụ thuộc vào mô hình và cách chia dữ liệu; điểm tổng hợp có thể che khuất việc từ chối quá mức hoặc trả lời thiếu căn cứ.

## Kết quả trong bài báo

Các số dưới đây được chép từ [Table 2–4 trong bản PDF của bài báo](paper/ViSAQA.pdf), **không phải kết quả chạy lại bằng mã trong repo hoặc ứng dụng demo**. Toàn bộ điểm số được hiển thị theo phần trăm.

### Truy xuất — Table 2

| Protocol | Retriever | Rerank | Hit@1 | Recall@3 | Hit@3 | MRR@3 |
| --- | --- | --- | --- | --- | --- | --- |
| Theo chủ đề | BM25 | Trước | 26.68 | 42.17 | 47.13 | 35.29 |
| Theo chủ đề | BM25 | Sau | 41.77 | 58.67 | 63.84 | 51.06 |
| Theo chủ đề | Dense | Trước | 45.51 | 60.83 | 69.08 | 55.24 |
| Theo chủ đề | Dense | Sau | 48.38 | 65.12 | 74.19 | 59.23 |
| Theo chủ đề | Hybrid | Trước | 38.53 | 55.35 | 65.46 | 50.42 |
| Theo chủ đề | Hybrid | Sau | 47.01 | 63.95 | 72.44 | 57.73 |
| Ngẫu nhiên | BM25 | Trước | 44.08 | 54.46 | 60.31 | 51.26 |
| Ngẫu nhiên | BM25 | Sau | 56.91 | 65.38 | 72.15 | 63.51 |
| Ngẫu nhiên | Dense | Trước | 54.06 | 63.29 | 69.85 | 60.87 |
| Ngẫu nhiên | Dense | Sau | 58.77 | 66.93 | 74.12 | 65.42 |
| Ngẫu nhiên | Hybrid | Trước | 55.04 | 62.47 | 69.52 | 61.46 |
| Ngẫu nhiên | Hybrid | Sau | 58.99 | 66.91 | 74.23 | 65.53 |

Đánh giá trên **802** câu topic-test và **912** câu random-test có nhãn relevance. Hit@k đo tỷ lệ câu có ít nhất một mục liên quan trong top-k; Recall@3 đo tỷ lệ các mục liên quan được tìm thấy; MRR@3 xét thứ hạng mục liên quan đầu tiên. Chênh lệch Hybrid–Dense ở random-test Hit@3 là **0,11 điểm** (74,23 so với 74,12); bài báo không xem đây là bằng chứng xếp hạng hai phương pháp.

### Chất lượng câu trả lời — Table 3

Kết quả **chỉ trên 802 câu có đáp án của topic-test**. EM, Token-F1 và ROUGE-L là **trung bình ± độ lệch chuẩn qua 3 seed**; FR là tỷ lệ từ chối sai trung bình trên câu có đáp án.

| Mô hình | Đầu vào | EM | Token-F1 | ROUGE-L | FR |
| --- | --- | --- | --- | --- | --- |
| ViT5-base | Q-only | 0.00 ± 0.00 | 9.44 ± 0.14 | 8.92 ± 0.08 | 86.03 |
| ViT5-base | BM25 | 0.00 ± 0.00 | 7.55 ± 0.24 | 7.01 ± 0.17 | 98.84 |
| ViT5-base | Dense | 0.00 ± 0.00 | 7.78 ± 0.34 | 7.19 ± 0.24 | 97.22 |
| ViT5-base | Hybrid | 0.00 ± 0.00 | 7.76 ± 0.36 | 7.17 ± 0.25 | 97.46 |
| BARTpho-base | Q-only | 2.08 ± 0.52 | 15.38 ± 0.59 | 14.76 ± 0.49 | 52.62 |
| BARTpho-base | BM25 | 0.46 ± 0.19 | 15.60 ± 0.49 | 13.81 ± 0.44 | 41.06 |
| BARTpho-base | Dense | 0.54 ± 0.26 | 16.98 ± 0.63 | 14.84 ± 0.52 | 34.91 |
| BARTpho-base | Hybrid | 0.50 ± 0.25 | 16.72 ± 0.54 | 14.62 ± 0.40 | 37.61 |
| mBART-50 | Q-only | 1.37 ± 0.00 | 13.45 ± 0.43 | 12.62 ± 0.43 | 63.84 |
| mBART-50 | BM25 | 2.41 ± 0.19 | 18.24 ± 0.59 | 15.73 ± 0.44 | 29.05 |
| mBART-50 | Dense | 2.41 ± 0.31 | 18.88 ± 0.39 | 16.26 ± 0.34 | 30.09 |
| mBART-50 | Hybrid | 2.37 ± 0.25 | 18.82 ± 0.47 | 16.17 ± 0.41 | 29.38 |
| mT5-base | Q-only | 0.00 ± 0.00 | 3.18 ± 0.95 | 3.25 ± 0.47 | 0.00 |
| mT5-base | BM25 | 0.00 ± 0.00 | 7.96 ± 0.31 | 7.72 ± 0.41 | 0.00 |
| mT5-base | Dense | 0.00 ± 0.00 | 8.16 ± 0.51 | 7.96 ± 0.58 | 0.00 |
| mT5-base | Hybrid | 0.00 ± 0.00 | 8.07 ± 0.42 | 7.83 ± 0.48 | 0.00 |
| mT5-small | Q-only | 0.00 ± 0.00 | 3.14 ± 0.46 | 3.17 ± 0.48 | 0.00 |
| mT5-small | BM25 | 0.00 ± 0.00 | 8.98 ± 0.54 | 8.97 ± 0.40 | 0.00 |
| mT5-small | Dense | 0.00 ± 0.00 | 8.97 ± 0.46 | 8.92 ± 0.35 | 0.00 |
| mT5-small | Hybrid | 0.00 ± 0.00 | 9.14 ± 0.51 | 9.07 ± 0.41 | 0.00 |

### Nhận diện từ chối — Table 4

Kết quả trên **1.588 câu topic-test**: 802 có đáp án, 554 không đủ thông tin và 232 ngoài phạm vi. Binary P/R/F1 xem hai loại từ chối là một lớp; OOS F1 chỉ xét loại ngoài phạm vi. Các hàng mô hình là trung bình qua 3 seed; hai hàng luôn từ chối là baseline cố định.

| Mô hình | Đầu vào | Binary P | Binary R | Binary F1 | OOS F1 |
| --- | --- | --- | --- | --- | --- |
| ViT5-base | Q-only | 50.43 | 89.31 | 64.46 | 3.82 |
| ViT5-base | BM25 | 49.50 | 98.85 | 65.97 | 0.00 |
| ViT5-base | Dense | 49.67 | 97.88 | 65.89 | 0.00 |
| ViT5-base | Hybrid | 49.72 | 98.35 | 66.05 | 0.00 |
| BARTpho-base | Q-only | 58.54 | 75.74 | 66.00 | 1.93 |
| BARTpho-base | BM25 | 57.89 | 57.55 | 57.62 | 0.00 |
| BARTpho-base | Dense | 60.88 | 55.30 | 57.85 | 0.00 |
| BARTpho-base | Hybrid | 59.45 | 56.15 | 57.63 | 0.00 |
| mBART-50 | Q-only | 55.17 | 80.15 | 65.34 | 13.46 |
| mBART-50 | BM25 | 61.60 | 47.50 | 53.63 | 0.00 |
| mBART-50 | Dense | 61.72 | 49.45 | 54.90 | 0.00 |
| mBART-50 | Hybrid | 62.46 | 49.79 | 55.40 | 0.00 |
| mT5-base | Q-only | 0.00 | 0.00 | 0.00 | 0.00 |
| mT5-base | BM25 | 0.00 | 0.00 | 0.00 | 0.00 |
| mT5-base | Dense | 0.00 | 0.00 | 0.00 | 0.00 |
| mT5-base | Hybrid | 0.00 | 0.00 | 0.00 | 0.00 |
| mT5-small | Q-only | 0.00 | 0.00 | 0.00 | 0.00 |
| mT5-small | BM25 | 0.00 | 0.00 | 0.00 | 0.00 |
| mT5-small | Dense | 0.00 | 0.00 | 0.00 | 0.00 |
| mT5-small | Hybrid | 0.00 | 0.00 | 0.00 | 0.00 |
| Luôn không đủ thông tin | – | 49.50 | 100.00 | 66.22 | 0.00 |
| Luôn ngoài phạm vi | – | 49.50 | 100.00 | 66.22 | 25.49 |

Không cấu hình nào vượt baseline “Luôn không đủ thông tin” về Binary F1 (cao nhất 66,05 so với 66,22). Các giá trị 0 của mT5 chịu ảnh hưởng của sentinel ở đầu văn bản sinh, như giải thích trong bài. Bài báo không có bảng đầy đủ điểm sinh câu trả lời cho random-test; riêng ROUGE-L trung bình của mBART Q-only và Dense được nêu là **18,85** và **15,12**.

## Huấn luyện và suy luận

Mã nguồn có trong [visaqa/](visaqa/), cấu hình trong [configs/paper.json](configs/paper.json). Xem [hướng dẫn đầy đủ](docs/REPRODUCIBILITY.md) để chuẩn bị dữ liệu cục bộ, train, infer và đánh giá.

| Thông số trong bài | Giá trị |
| --- | --- |
| Mô hình | ViT5-base, BARTpho-syllable-base, mBART-50, mT5-base, mT5-small |
| Protocol / seed | topic, random / 22, 42, 202 |
| Epoch / learning rate / effective batch | 5 / 3e-5 / 256 |
| Input / target / generation tối đa | 1.024 / 256 / 256 token; greedy, beam 1 |
| Truy xuất | BM25, E5-base dense, Hybrid RRF (k=60) |
| Reranker / ứng viên / top-k | BGE-reranker-v2-m3 / 20 / 3 |

~~~bash
python -m pip install -e ".[ml]"
python -m visaqa --help
python -m visaqa matrix --plan runs/plan.json
~~~

Lệnh matrix chỉ tạo kế hoạch **60 lượt train và 120 cấu hình inference**; thêm --execute mới chạy. Các lệnh train/infer cụ thể nằm trong hướng dẫn. Mã mới bám các thông số đã công bố; [các mặc định chưa được bài xác định đầy đủ](docs/REPRODUCIBILITY.md#các-mặc-định-chưa-được-bài-xác-định-đầy-đủ) được ghi rõ. Chưa chạy lại toàn bộ thí nghiệm với dữ liệu gốc.

**Không phân phối dữ liệu hoặc checkpoint.** Pipeline đọc dữ liệu và split gốc từ máy của bạn; data/, artifacts/, runs/ và các tệp dữ liệu/trọng số được Git bỏ qua.

## Ứng dụng minh họa

Ứng dụng [Tra cứu Công tác Sinh viên](https://huggingface.co/spaces/BaoNhan/tra-cuu-ctsv) cung cấp giao diện tiếng Việt để đặt câu hỏi, xem câu trả lời và kiểm tra các đoạn ngữ cảnh cùng nguồn tài liệu được truy xuất.

**Cấu hình demo:** ứng dụng đang triển khai hiển thị mô hình sinh **Qwen3-4B + LoRA (seed 42)** và bộ truy xuất **Hybrid BM25 + BGE-M3, kết hợp reranking và chọn top-3**, cùng các bước kiểm tra phạm vi và bằng chứng. Cấu hình ứng dụng khác với các thí nghiệm sequence-to-sequence trong bài báo. Các ảnh dưới đây minh họa luồng sử dụng ứng dụng, không phải kết quả đánh giá benchmark.

### Thử một câu hỏi

1. Mở ứng dụng và nhập mã do chủ demo cung cấp vào ô **Mã truy cập demo**.
2. Chọn hoặc nhập câu hỏi bên dưới vào ô **Câu hỏi**.
3. Giữ lựa chọn **Context hệ thống (mặc định · dùng trong ứng dụng)** và nhấn **Hỏi**.
4. Đọc phần **Trả lời**, sau đó xem **Ngữ cảnh truy xuất + điểm rerank + nguồn** và **Các bước kiểm tra (cổng phạm vi / cổng bằng chứng)**.

> Bao nhiêu điểm rèn luyện thì được xếp loại xuất sắc?

**Câu trả lời thực tế trong lượt chạy minh họa:**

> Điểm rèn luyện từ 90 đến 100 điểm được xếp loại xuất sắc.

Câu trả lời dẫn hai nguồn `14.6.1.9.QD3002.2023_HBKT&Hotro.txt` và `1.9.1.9.QD3937.2023_DGRL.txt`. Ứng dụng truy xuất ba đoạn và sử dụng 897 token ngữ cảnh. Đây là một ví dụ quan sát được, không phải ước lượng độ chính xác của hệ thống.

### 1. Nhập câu hỏi

![Giao diện ứng dụng với câu hỏi ví dụ và chế độ ngữ cảnh hệ thống mặc định](docs/images/01-question.png)

### 2. Xem câu trả lời và nguồn tài liệu

![Câu trả lời đã hoàn tất cùng các đoạn ngữ cảnh, điểm reranking và nguồn tài liệu](docs/images/02-answer-and-sources.png)

### 3. Kiểm tra luồng xử lý và bằng chứng

![Các bước kiểm tra phạm vi, truy xuất, ghép ngữ cảnh, kiểm tra bằng chứng và sinh câu trả lời](docs/images/03-evidence-checks.png)

Ảnh được chụp trực tiếp từ ứng dụng ngày **30/10/2026**. Ô mã truy cập đã được che trong các ảnh sau khi nhập mã. Kết quả và thời gian xử lý có thể thay đổi theo mô hình và tập tài liệu đang triển khai.

## Nội dung repository

Repository có mã chuẩn bị dữ liệu, chunking, truy xuất, huấn luyện, suy luận và đánh giá; cấu hình theo bài; hướng dẫn song ngữ; ảnh demo; citation và [bản PDF do tác giả cung cấp](paper/ViSAQA.pdf). Không chứa dữ liệu benchmark hoặc trọng số huấn luyện.

## Trích dẫn

Trạng thái xuất bản: **To Appear**. DOI, số tập và số trang sẽ được bổ sung khi kỷ yếu xuất bản.

```bibtex
@inproceedings{nguyen2026visaqa,
  author    = {Nguyen, Huu Dung and Tran, Phuoc and Nguyen, Van Thinh and Chau, Bao Nhan and Duong, Hoang Kiet and Nguyen, Van Dung},
  title     = {{ViSAQA}: Benchmarking Evidence Sufficiency and Scope Awareness in Vietnamese Student Affairs Question Answering},
  booktitle = {Proceedings of the 6th EAI International Conference on Artificial Intelligence for Communications and Networks (EAI AICON)},
  year      = {2026},
  note      = {To Appear}
}
```

Thông tin này cũng có trong tệp [CITATION.bib](CITATION.bib).
