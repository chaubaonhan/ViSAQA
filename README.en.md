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
