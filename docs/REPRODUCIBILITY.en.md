# ViSAQA training and inference

[Tiếng Việt](REPRODUCIBILITY.md) | **English** · [README](../README.en.md) · [Paper](../paper/ViSAQA.pdf)

This is a new implementation of the paper's methods, not an archive of the code behind its tables. Published settings are configured explicitly; unspecified choices are documented below. **The dataset, original splits and trained weights are not distributed.** These utilities read local data and do not upload it.

## Install

Python 3.10–3.12 in an isolated environment is recommended. Install a suitable PyTorch GPU/CUDA build using the [official instructions](https://pytorch.org/get-started/locally/), then run from the repository root:

~~~bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[ml]"
python -m visaqa --help
~~~

Transformers is pinned to 4.57.1, using the [Trainer API](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer). Training records Torch/Transformers/Accelerate versions in run.json. After creating runs/, save the complete environment with:

~~~bash
python -m pip freeze > runs/environment.txt
~~~

## Published settings

See [configs/paper.json](../configs/paper.json).

| Component | Setting |
| --- | --- |
| Generators | VietAI/vit5-base; vinai/bartpho-syllable-base; facebook/mbart-large-50-many-to-many-mmt; google/mt5-base; google/mt5-small |
| mBART source/target language | vi_VN |
| Protocols / training seeds | topic-disjoint, random / 22, 42, 202 |
| Epochs / learning rate | 5 / 3e-5 |
| Effective batch size | 256; default micro batch 8 × accumulation 32 |
| Input / target limit | 1,024 / 256 tokens |
| Decoding | greedy, beam 1, max_length=256, no sampling |
| Encoder / reranker | intfloat/multilingual-e5-base / BAAI/bge-reranker-v2-m3 |
| Candidates / reranked top-k / RRF constant | 20 / 3 / 60 |
| Chunk size / overlap / stride | 512 / 80 / 432 tokens |

The reported hardware is one NVIDIA RTX PRO 6000 Blackwell Server Edition 96 GB on GCP g4-standard-48 (48 vCPU, 180 GB RAM). CPU is supported for smoke testing. Expose one GPU with CUDA_VISIBLE_DEVICES on multi-GPU hosts. Lower --batch-size increases accumulation to preserve the effective batch; the final epoch update may be partial.

Train **q_only** on questions and **context** on questions plus reference contexts. Reuse each context checkpoint across BM25, Dense and Hybrid test conditions: **60 training runs and 120 inference configurations**. The paper pipeline does not add Qwen or threshold gates from the separate demo.

## Local data contract

Keep all data under the ignored data/ directory. Do not commit it.

data/records.jsonl has one object per line:

| Field | Meaning |
| --- | --- |
| id | Unique string question ID |
| question, answer, context | Question, reference answer and reference context strings |
| topic_id | Integer 1–21 using Table 1 IDs |
| subtopic | String, possibly empty |
| source_document_ids | List of source document IDs |
| answer_type | answerable, insufficient or oos |
| is_oos | Boolean consistent with answer type |
| splits | Object with topic/random keys, each containing the original train/dev/test assignment |
| relevant_chunk_ids | Relevant chunk judgments; absent/empty if unjudged |
| relevant_document_ids | Optional document judgments, not inferred from source IDs |

Canonical rejection targets:

- Không có đủ thông tin trong tài liệu được cung cấp để trả lời câu hỏi này.
- Câu hỏi nằm ngoài phạm vi hệ thống RAG nghiệp vụ công tác sinh viên.

data/chunks.jsonl contains unique chunk_id, document_id and text. Prefer the original 2,571 chunks from 83 documents.

~~~bash
python -m visaqa prepare --records data/records.jsonl --chunks data/chunks.jsonl --output data/splits --paper-counts
~~~

This checks corpus totals and Table 1 per-class/per-split counts. Both protocols have 12,703/1,588/1,588 train/dev/test records. Topic assignments are train [1,2,6,7,9,10,12,15,16,18,19,20], dev [3,11,13,17], test [4,5,8,14,21].

Original random assignments are required: seed 42 alone cannot reconstruct IDs without the original split/rounding algorithm. Near-duplicate questions are not filtered across splits.

Optional chunking from data/documents.jsonl (document_id and text):

~~~bash
python -m visaqa chunk --documents data/documents.jsonl --output data/new-chunks.jsonl
~~~

The new chunker uses the E5 tokenizer and exact-text document deduplication. New IDs require new relevance mappings; this does not reconstruct the original chunks.

## Run the pipeline

~~~bash
python -m visaqa index --chunks data/chunks.jsonl --output artifacts/index
python -m visaqa retrieve --index artifacts/index --records data/splits/topic/test.jsonl --method dense --output runs/retrieval/topic-dense.jsonl
python -m visaqa train --data-dir data/splits --protocol topic --model mbart --seed 42 --mode context --output runs/checkpoints/topic/mbart/42/context
python -m visaqa infer --checkpoint runs/checkpoints/topic/mbart/42/context --records data/splits/topic/test.jsonl --condition dense --retrieval runs/retrieval/topic-dense.jsonl --output runs/predictions/topic-mbart-42-dense.jsonl
~~~

Model keys: vit5, bartpho, mbart, mt5-base, mt5-small. Train Q-only with --mode q_only in a separate directory; infer with --condition q_only and no retrieval cache/index. --resume accepts a checkpoint-N directory in the same output and verifies configuration and data hashes.

[E5](https://huggingface.co/intfloat/multilingual-e5-base) uses query: / passage: prefixes and attention-masked mean pooling with L2 normalization. Hybrid fuses each branch's top-20 with RRF and retains 20 candidates before reranking. All three methods use the reranker. Caches preserve candidates, selected passages, IDs and scores.

For a new question:

~~~bash
python -m visaqa infer --checkpoint runs/checkpoints/topic/mbart/42/context --question "Bao nhiêu điểm rèn luyện thì được xếp loại xuất sắc?" --condition dense --index artifacts/index --output runs/answer.jsonl
~~~

Predictions retain generated text/token IDs, actual context, sources and truncation diagnostics. Emitted mT5 sentinels are retained, never forced or removed.

## Evaluate

~~~bash
python -m visaqa evaluate --references data/splits/topic/test.jsonl --predictions runs/predictions/topic-mbart-42-dense.jsonl --extended --bertscore --output runs/metrics/topic-mbart-42-dense.json
python -m visaqa evaluate-retrieval --references data/splits/topic/test.jsonl --retrieval runs/retrieval/topic-dense.jsonl --unit chunk --output runs/metrics/retrieval-topic-dense.json
python -m visaqa evaluate --references data/splits/topic/test.jsonl --baseline insufficient --extended --output runs/metrics/always-insufficient.json
~~~

Generation reports EM, token-F1 and ROUGE-1/2/L overall and answerable-only. --extended adds SacreBLEU; --bertscore adds BERTScore and enables BLEU. Rejection reports binary precision/recall/F1, OOS F1 and answerable false-rejection rate. The parser requires a complete normalized rejection-sentence match.

Retrieval reports Hit@1, Hit@3, Recall@3 and MRR@3 before/after reranking with evaluated/excluded counts. Missing judgments are excluded, not counted as failures. --unit chunk uses chunk judgments; --unit document deduplicates ranked documents and uses document judgments. The paper has 802 judged topic-test and 912 judged random-test queries; matching these needs the original annotations.

Quality scores are percentages; n/tp/fp/fn are counts. Empty answerable subsets get null scores. Use aggregate --reports <three metric files> --output runs/summary.json for mean/SD, requiring exactly seeds 22/42/202 with matching model, protocol, condition, configuration and references.

## Full matrix

~~~bash
python -m visaqa matrix --data-dir data/splits --index artifacts/index --plan runs/plan.json
~~~

This only writes the ordered plan. After preparing data/index, add --execute to run 6 retrieval caches, 60 training runs, 120 inference/evaluation pairs, 4 baselines and 40 summaries. It stops on errors and does not overwrite checkpoints. Continue using remaining plan commands and train --resume where needed.

## Defaults not fully specified in the paper

These are choices of this new implementation, not claims about the original code:

- Vietnamese prompt templates, question before context, blank-line top-3 concatenation, tokenizer-specific truncation.
- AdamW, weight decay 0, linear schedule, warmup 0, clipping 1, FP32, micro batch 8, final-epoch checkpoint, inherited model dropout.
- Unicode lowercase BM25 tokens, k1=1.5, b=0.75, positive IDF log(1+(N-df+0.5)/(df+0.5)), the stated hybrid merge policy and corpus-order tie breaking.
- E5 limit 512 including prefixes/special tokens; reranker limit 1,024. [BGE](https://huggingface.co/BAAI/bge-reranker-v2-m3) scores are raw logits, not rejection thresholds.
- NFC/lowercase Unicode tokens preserving sentinels; macro token n-gram/LCS ROUGE F1 without stemming; SacreBLEU tokenize=none on normalized tokens; BERTScore xlm-roberta-large, layer 17, no baseline rescaling. Metric hashes/signatures are saved.
- Sample seed SD (ddof=1); the paper does not specify this convention.
- Original package/model revisions are not fully specified. Resolved configuration, versions, data hashes and available model revisions are recorded.

Matching the paper tables requires the original local artifacts and corresponding implementation conventions; the PDF alone cannot establish exact numerical reproducibility.

## Tests

~~~bash
python -m unittest discover -s tests -v
~~~

Optional offline CPU integration:

~~~powershell
$env:VISAQA_ML_SMOKE="1"
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
python -m unittest discover -s tests -v
~~~

Set equivalent variables on Linux/macOS. The test uses tiny randomly initialized models and temporary synthetic records, exercises train → retrieve → infer → evaluate, and does not run the research benchmark.
