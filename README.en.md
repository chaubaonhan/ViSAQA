<h1 align="center">ViSAQA</h1>

<p align="center">
  <a href="README.md"><img src="docs/images/badges/language-vi.svg" alt="Tiếng Việt" height="36"></a>
  <a href="README.en.md"><img src="docs/images/badges/language-en-active.svg" alt="English" height="36"></a>
</p>

<p align="center"><strong>Benchmarking Evidence Sufficiency and Scope Awareness<br>in Vietnamese Student Affairs Question Answering</strong></p>

<p align="center">Huu Dung Nguyen · Phuoc Tran · Van Thinh Nguyen · Bao Nhan Chau · Hoang Kiet Duong · Van Dung Nguyen</p>

<p align="center">
  <a href="paper/ViSAQA.pdf"><img src="docs/images/badges/paper.svg" alt="Paper PDF" height="36"></a>
  <a href="https://huggingface.co/spaces/BaoNhan/tra-cuu-ctsv"><img src="docs/images/badges/demo.svg" alt="Live Demo — Hugging Face" height="36"></a>
  <a href="visaqa/"><img src="docs/images/badges/code.svg" alt="Source code" height="36"></a>
  <a href="#paper-results"><img src="docs/images/badges/results-en.svg" alt="Results" height="36"></a>
  <a href="docs/REPRODUCIBILITY.en.md"><img src="docs/images/badges/guide-en.svg" alt="Guide" height="36"></a>
  <a href="#citation"><img src="docs/images/badges/cite-en.svg" alt="Citation" height="36"></a>
</p>

<p align="center">
  <a href="https://aicon-conf.eai-conferences.org/2026/"><img src="docs/images/badges/status.svg" alt="EAI AICON 2026 — To Appear" height="36"></a>
</p>

Accepted at the 6th EAI International Conference on Artificial Intelligence for Communications and Networks. **Proceedings: To Appear.**

## Overview

ViSAQA studies Vietnamese Student Affairs question answering with an emphasis on evidence sufficiency and scope awareness. The benchmark contains **15,879 examples across 21 topics**, including answerable, insufficient-information, and out-of-scope questions. The study separates retrieval quality, answer quality on answerable questions, and rejection behavior under topic-disjoint and random splits.

The paper evaluates ViT5, BARTpho, mBART, mT5-base, and mT5-small with question-only, BM25, dense, and hybrid conditions. Retrieval benefits depend on the model and split; aggregate scores alone can obscure over-rejection and unsupported answers.

## Paper results

The values below are transcribed from [Tables 2–4 of the paper PDF](paper/ViSAQA.pdf). They are **published results, not reruns of this repository code or the live demo**. All scores are percentages.

### Retrieval — Table 2

| Protocol | Retriever | Rerank | Hit@1 | Recall@3 | Hit@3 | MRR@3 |
| --- | --- | --- | --- | --- | --- | --- |
| Topic | BM25 | Before | 26.68 | 42.17 | 47.13 | 35.29 |
| Topic | BM25 | After | 41.77 | 58.67 | 63.84 | 51.06 |
| Topic | Dense | Before | 45.51 | 60.83 | 69.08 | 55.24 |
| Topic | Dense | After | 48.38 | 65.12 | 74.19 | 59.23 |
| Topic | Hybrid | Before | 38.53 | 55.35 | 65.46 | 50.42 |
| Topic | Hybrid | After | 47.01 | 63.95 | 72.44 | 57.73 |
| Random | BM25 | Before | 44.08 | 54.46 | 60.31 | 51.26 |
| Random | BM25 | After | 56.91 | 65.38 | 72.15 | 63.51 |
| Random | Dense | Before | 54.06 | 63.29 | 69.85 | 60.87 |
| Random | Dense | After | 58.77 | 66.93 | 74.12 | 65.42 |
| Random | Hybrid | Before | 55.04 | 62.47 | 69.52 | 61.46 |
| Random | Hybrid | After | 58.99 | 66.91 | 74.23 | 65.53 |

Evaluated on **802** topic-test and **912** random-test queries with relevance judgments. Hit@k is the share of queries with at least one relevant item in the top k; Recall@3 is the fraction of relevant items found; MRR@3 uses the first relevant rank. The random-test Hybrid–Dense Hit@3 difference is **0.11 points** (74.23 vs 74.12); the paper does not treat it as evidence of a reliable ranking.

### Answer generation — Table 3

Results on the **802 answerable topic-test questions only**. EM, Token-F1 and ROUGE-L are **mean ± SD across three seeds**; FR is the mean false-rejection rate on answerable questions.

| Model | Input | EM | Token-F1 | ROUGE-L | FR |
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

### Rejection detection — Table 4

Results on **1,588 topic-test questions**: 802 answerable, 554 insufficient-information and 232 out-of-scope. Binary P/R/F1 treat both rejection types as one class; OOS F1 assesses out-of-scope recognition alone. Model rows are three-seed means; the two always-reject rows are deterministic baselines.

| Model | Input | Binary P | Binary R | Binary F1 | OOS F1 |
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
| Always insufficient | – | 49.50 | 100.00 | 66.22 | 0.00 |
| Always OOS | – | 49.50 | 100.00 | 66.22 | 25.49 |

No configuration exceeds the always-insufficient baseline in Binary F1 (best model 66.05 vs 66.22). The mT5 zeros are affected by sentinel tokens at the start of generated text, as explained in the paper. The paper does not provide a full random-test generation table; it reports mBART Q-only and Dense mean ROUGE-L of **18.85** and **15.12**, respectively.

## Training and inference

Implementation: [visaqa/](visaqa/). Configuration: [configs/paper.json](configs/paper.json). See the [full guide](docs/REPRODUCIBILITY.en.md) for local data preparation, training, inference and evaluation.

| Paper setting | Value |
| --- | --- |
| Models | ViT5-base, BARTpho-syllable-base, mBART-50, mT5-base, mT5-small |
| Protocols / seeds | topic, random / 22, 42, 202 |
| Epochs / learning rate / effective batch | 5 / 3e-5 / 256 |
| Maximum input / target / generation | 1,024 / 256 / 256 tokens; greedy, beam 1 |
| Retrieval | BM25, E5-base dense, Hybrid RRF (k=60) |
| Reranker / candidates / top-k | BGE-reranker-v2-m3 / 20 / 3 |

~~~bash
python -m pip install -e ".[ml]"
python -m visaqa --help
python -m visaqa matrix --plan runs/plan.json
~~~

The matrix command only plans **60 training runs and 120 inference configurations**; execution requires --execute. The guide includes individual training/inference commands and [implementation defaults not fully specified in the paper](docs/REPRODUCIBILITY.en.md#defaults-not-fully-specified-in-the-paper). The full experiments have not been rerun with the original data.

**No dataset or checkpoint is distributed.** The pipeline reads original data/splits from your machine; data/, artifacts/, runs/ and data/weight files are excluded from Git.

## Application demo

The [Tra cứu Công tác Sinh viên application](https://huggingface.co/spaces/BaoNhan/tra-cuu-ctsv) provides a Vietnamese interface for asking a question, inspecting the generated answer, and checking retrieved passages and document sources.

**Demo configuration:** the deployed application identifies its generator as **Qwen3-4B + LoRA (seed 42)** and its retriever as **Hybrid BM25 + BGE-M3 with reranking and top-3 selection**, with scope and evidence checks. This application illustrates a deployment workflow; its configuration differs from the sequence-to-sequence experiments reported in the paper. The screenshots below are an application walkthrough, not benchmark results.

### Try the example

1. Open the live application and enter an access code supplied by the demo owner in **Mã truy cập demo**.
2. Select or enter the question below in **Câu hỏi**.
3. Keep **Context hệ thống (mặc định · dùng trong ứng dụng)** selected and click **Hỏi**.
4. Read **Trả lời**, then inspect **Ngữ cảnh truy xuất + điểm rerank + nguồn** and **Các bước kiểm tra (cổng phạm vi / cổng bằng chứng)**.

> Bao nhiêu điểm rèn luyện thì được xếp loại xuất sắc?
>
> *English: How many conduct-assessment points are needed for an excellent rating?*

**Observed answer from this run:**

> Điểm rèn luyện từ 90 đến 100 điểm được xếp loại xuất sắc.

The response cites `14.6.1.9.QD3002.2023_HBKT&Hotro.txt` and `1.9.1.9.QD3937.2023_DGRL.txt`. The application retrieved three passages and used 897 context tokens. This is one observed example, not an accuracy estimate.

### 1. Enter a question

![Vietnamese Student Affairs application with the example question selected and the default system context enabled](docs/images/01-question.png)

### 2. Inspect the answer and retrieved sources

![Completed answer alongside retrieved passages, reranking scores, and source document references](docs/images/02-answer-and-sources.png)

### 3. Inspect the evidence checks

![Expanded processing steps showing the scope check, retrieval, context assembly, evidence check, and answer generation](docs/images/03-evidence-checks.png)

Screenshots captured from the live application on **30 October 2026**. The access-code field is masked in authenticated screenshots. Outputs and timings may change with the deployed model and source collection.

## Release contents

This repository contains data preparation, chunking, retrieval, training, inference and evaluation code; paper settings; bilingual guides; demo screenshots; citation metadata; and the [author-supplied paper PDF](paper/ViSAQA.pdf). Benchmark data and trained weights are not included.

## Citation

Publication status: **To Appear**. DOI, volume, and page numbers are omitted until the proceedings are published.

```bibtex
@inproceedings{nguyen2026visaqa,
  author    = {Nguyen, Huu Dung and Tran, Phuoc and Nguyen, Van Thinh and Chau, Bao Nhan and Duong, Hoang Kiet and Nguyen, Van Dung},
  title     = {{ViSAQA}: Benchmarking Evidence Sufficiency and Scope Awareness in Vietnamese Student Affairs Question Answering},
  booktitle = {Proceedings of the 6th EAI International Conference on Artificial Intelligence for Communications and Networks (EAI AICON)},
  year      = {2026},
  note      = {To Appear}
}
```

The same entry is available in [CITATION.bib](CITATION.bib).
