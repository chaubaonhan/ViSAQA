# Huấn luyện và suy luận ViSAQA

**Tiếng Việt** | [English](REPRODUCIBILITY.en.md) · [README](../README.md) · [Paper](../paper/ViSAQA.pdf)

Mã trong repository được triển khai từ mô tả bài báo, không phải bản lưu trữ mã đã tạo ra các bảng kết quả. Cấu hình có đầy đủ thông số được công bố; các lựa chọn bài không mô tả được liệt kê bên dưới. **Dữ liệu, split gốc và trọng số đã huấn luyện không được phân phối trong repo.** Pipeline đọc dữ liệu cục bộ và không có chức năng tải dữ liệu lên.

## Cài đặt

Khuyến nghị Python 3.10–3.12 và môi trường riêng. Cài PyTorch phù hợp GPU/CUDA theo [hướng dẫn chính thức](https://pytorch.org/get-started/locally/) trước, sau đó chạy từ thư mục gốc:

~~~bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[ml]"
python -m visaqa --help
~~~

Transformers cố định ở 4.57.1. Mã train ghi phiên bản Torch, Transformers và Accelerate trong run.json. API sử dụng theo [tài liệu Trainer 4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer). Sau khi tạo thư mục runs, có thể lưu toàn bộ môi trường bằng:

~~~bash
python -m pip freeze > runs/environment.txt
~~~

## Cấu hình theo bài

Tệp trung tâm: [configs/paper.json](../configs/paper.json).

| Thành phần | Cấu hình |
| --- | --- |
| ViT5 | VietAI/vit5-base |
| BARTpho | vinai/bartpho-syllable-base |
| mBART | facebook/mbart-large-50-many-to-many-mmt; ngôn ngữ vi_VN |
| mT5 | google/mt5-base, google/mt5-small |
| Protocol / seed huấn luyện | topic, random / 22, 42, 202 |
| Epoch / learning rate | 5 / 0.00003 |
| Effective batch size | 256; micro batch 8 × tích lũy 32 |
| Input / target tối đa | 1.024 / 256 token |
| Sinh câu trả lời | greedy, beam 1, max_length=256, không sampling |
| Dense encoder | intfloat/multilingual-e5-base |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Số ứng viên / top-k sau rerank | 20 / 3 |
| RRF | hằng số 60 |
| Chunk / overlap / stride | 512 / 80 / 432 token |

Bài dùng một GPU NVIDIA RTX PRO 6000 Blackwell Server Edition 96 GB trên GCP g4-standard-48 (48 vCPU, RAM 180 GB). Mã hỗ trợ CPU để kiểm tra; huấn luyện đầy đủ cần tài nguyên phù hợp. Nếu có nhiều GPU, chỉ hiển thị một GPU qua CUDA_VISIBLE_DEVICES. Giảm --batch-size vẫn giữ effective batch 256 bằng tăng tích lũy gradient; batch cuối epoch có thể không đầy.

Hai chế độ train: **q_only** chỉ có câu hỏi; **context** có câu hỏi và reference context trong train/dev. Mỗi checkpoint context dùng lại cho BM25, Dense và Hybrid ở test. Tổng cộng **60 lượt train và 120 cấu hình inference**. Pipeline của bài không thêm Qwen hoặc cổng từ chối theo ngưỡng; đó là các thành phần của demo riêng.

## Dữ liệu cục bộ

Đặt tệp dưới data/ (đã bị Git bỏ qua). Không commit dữ liệu.

data/records.jsonl: mỗi dòng là một đối tượng JSON với các trường:

| Trường | Ý nghĩa |
| --- | --- |
| id | ID câu hỏi dạng chuỗi, duy nhất |
| question, answer, context | Câu hỏi, đáp án tham chiếu, reference context; dạng chuỗi |
| topic_id | Số nguyên 1–21 theo Table 1 |
| subtopic | Chuỗi chủ đề con; có thể rỗng |
| source_document_ids | Danh sách ID tài liệu nguồn |
| answer_type | answerable, insufficient hoặc oos |
| is_oos | Boolean nhất quán với answer_type |
| splits | Đối tượng có khóa topic và random, giá trị train/dev/test từ split gốc |
| relevant_chunk_ids | Danh sách chunk liên quan theo nhãn đánh giá; rỗng/không có nếu chưa gán |
| relevant_document_ids | Tùy chọn cho đánh giá mức tài liệu; không tự suy diễn từ source IDs |

Hai câu từ chối chuẩn:

- Không có đủ thông tin trong tài liệu được cung cấp để trả lời câu hỏi này.
- Câu hỏi nằm ngoài phạm vi hệ thống RAG nghiệp vụ công tác sinh viên.

data/chunks.jsonl: mỗi dòng có chunk_id duy nhất, document_id và text. Ưu tiên 2.571 chunks gốc thuộc 83 tài liệu để giữ ánh xạ relevance.

~~~bash
python -m visaqa prepare --records data/records.jsonl --chunks data/chunks.jsonl --output data/splits --paper-counts
~~~

Lệnh kiểm tra tổng 15.879 mẫu, 2.571 chunks, 83 tài liệu và số mẫu từng lớp/từng split trong Table 1. Cả hai protocol có 12.703/1.588/1.588 mẫu train/dev/test. Split topic giữ nguyên chủ đề: train [1,2,6,7,9,10,12,15,16,18,19,20], dev [3,11,13,17], test [4,5,8,14,21].

**Không tạo lại random split chỉ bằng seed 42:** bài không cung cấp đủ chi tiết thuật toán chia và làm tròn để đảm bảo cùng ID. Mã yêu cầu split gốc, không lọc câu gần trùng giữa split.

Tiện ích tùy chọn tạo chunks mới từ data/documents.jsonl (document_id, text):

~~~bash
python -m visaqa chunk --documents data/documents.jsonl --output data/new-chunks.jsonl
~~~

Tokenizer E5 được chọn làm mặc định, tài liệu trùng văn bản hoàn toàn được loại bỏ. Chunk ID mới không tương đương bản gốc; cần ánh xạ lại relevance trước khi đánh giá.

## Chạy từng bước

~~~bash
python -m visaqa index --chunks data/chunks.jsonl --output artifacts/index
python -m visaqa retrieve --index artifacts/index --records data/splits/topic/test.jsonl --method dense --output runs/retrieval/topic-dense.jsonl
python -m visaqa train --data-dir data/splits --protocol topic --model mbart --seed 42 --mode context --output runs/checkpoints/topic/mbart/42/context
python -m visaqa infer --checkpoint runs/checkpoints/topic/mbart/42/context --records data/splits/topic/test.jsonl --condition dense --retrieval runs/retrieval/topic-dense.jsonl --output runs/predictions/topic-mbart-42-dense.jsonl
~~~

Model keys: vit5, bartpho, mbart, mt5-base, mt5-small. Đổi --mode q_only và dùng output khác để train baseline; infer dùng --condition q_only, không truyền index/cache. --resume nhận thư mục checkpoint-N trong cùng output; cấu hình và hash train/dev phải khớp.

[E5](https://huggingface.co/intfloat/multilingual-e5-base) dùng tiền tố query: / passage:, mean pooling có attention mask và chuẩn hóa L2. Hybrid lấy top-20 từ mỗi nhánh BM25/dense, hợp nhất RRF rồi giữ 20 ứng viên trước rerank. Cả ba phương pháp đều rerank. Cache giữ ứng viên trước rerank, top-3 sau rerank, văn bản, ID nguồn và điểm.

Hỏi một câu mới:

~~~bash
python -m visaqa infer --checkpoint runs/checkpoints/topic/mbart/42/context --question "Bao nhiêu điểm rèn luyện thì được xếp loại xuất sắc?" --condition dense --index artifacts/index --output runs/answer.jsonl
~~~

Kết quả giữ văn bản sinh, token ID, context thực tế, nguồn và thống kê truncation. Sentinel mT5 được giữ nếu mô hình sinh ra, không bị ép chèn hoặc xóa.

## Đánh giá

~~~bash
python -m visaqa evaluate --references data/splits/topic/test.jsonl --predictions runs/predictions/topic-mbart-42-dense.jsonl --extended --bertscore --output runs/metrics/topic-mbart-42-dense.json
python -m visaqa evaluate-retrieval --references data/splits/topic/test.jsonl --retrieval runs/retrieval/topic-dense.jsonl --unit chunk --output runs/metrics/retrieval-topic-dense.json
python -m visaqa evaluate --references data/splits/topic/test.jsonl --baseline insufficient --extended --output runs/metrics/always-insufficient.json
~~~

- Generation: EM, token-F1, ROUGE-1/2/L riêng cho toàn tập và answerable. --extended thêm SacreBLEU; --bertscore thêm BERTScore và tự bật BLEU.
- Rejection: precision/recall/F1 gộp insufficient + OOS, OOS F1 riêng, false-rejection rate trên answerable. Chỉ khớp **toàn bộ chuỗi đã chuẩn hóa** mới gán nhãn từ chối.
- Retrieval: Hit@1, Hit@3, Recall@3 và MRR@3 trước/sau rerank. Chỉ tính câu có relevance judgments; ghi số được tính và bị loại. Thiếu nhãn không bị tính là truy xuất sai.
- --unit chunk dùng chunk IDs; --unit document dùng document IDs và gộp tài liệu trùng trong xếp hạng. Không trộn hai loại nhãn. Bài có 802 câu topic-test và 912 câu random-test có judgments; cần nhãn gốc để khớp.
- Điểm chất lượng theo phần trăm; n/tp/fp/fn là số lượng. Nhóm answerable rỗng có điểm null.
- aggregate --reports <ba tệp metrics> --output runs/summary.json yêu cầu đủ seed 22/42/202, cùng model, protocol, condition, cấu hình và references; báo cáo mean/SD.

## Ma trận thí nghiệm

~~~bash
python -m visaqa matrix --data-dir data/splits --index artifacts/index --plan runs/plan.json
~~~

Mặc định chỉ tạo kế hoạch có thứ tự. Sau khi chuẩn bị dữ liệu và index, thêm --execute để chạy: 6 cache truy xuất, 60 lượt train, 120 lượt infer/evaluate, 4 baseline và 40 bảng tổng hợp. Lệnh dừng khi có lỗi, không tự ghi đè checkpoint cũ. Tiếp tục bằng các lệnh còn lại trong plan và train --resume khi cần.

## Các mặc định chưa được bài xác định đầy đủ

Đây là lựa chọn của **mã mới**, không phải khẳng định về mã thí nghiệm gốc:

- Prompt tiếng Việt; câu hỏi trước, context sau; ghép top-3 bằng dòng trống rồi truncate theo tokenizer riêng.
- AdamW, weight decay 0, linear scheduler, warmup 0, gradient clipping 1; FP32; micro batch 8; checkpoint epoch cuối; dropout kế thừa model.
- BM25 token Unicode lowercase, k1=1.5, b=0.75, IDF dương log(1+(N-df+0.5)/(df+0.5)); hợp nhất hybrid như trên; hòa điểm theo thứ tự corpus.
- E5 giới hạn 512 token gồm prefix/special tokens; reranker giới hạn 1.024. Điểm [BGE reranker](https://huggingface.co/BAAI/bge-reranker-v2-m3) là raw logit, không dùng làm ngưỡng từ chối.
- Chuẩn hóa NFC/lowercase, tách token Unicode, giữ sentinel; ROUGE macro token n-gram/LCS F1 không stemming; SacreBLEU tokenize=none trên token chuẩn hóa; BERTScore xlm-roberta-large, layer 17, không rescale. Hash BERTScore/signature BLEU được lưu.
- SD giữa seed dùng mẫu (ddof=1); bài không nêu quy ước SD.
- Package/revision model gốc chưa được công bố đầy đủ. Mã lưu cấu hình, phiên bản, hash dữ liệu và revision model khi có.

Không cam kết tái tạo đúng các bảng chỉ từ PDF. Đối chiếu số liệu cần dữ liệu/split/chunks gốc và các quy ước triển khai tương ứng.

## Kiểm thử

~~~bash
python -m unittest discover -s tests -v
~~~

Kiểm thử ML tùy chọn dùng mô hình rất nhỏ khởi tạo ngẫu nhiên và dữ liệu giả lập trong thư mục tạm:

~~~powershell
$env:VISAQA_ML_SMOKE="1"
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
python -m unittest discover -s tests -v
~~~

Linux/macOS đặt các biến tương ứng. Kiểm thử không gọi mạng và không chạy lại benchmark.
