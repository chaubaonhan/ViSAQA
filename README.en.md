# ViSAQA

[Tiếng Việt](README.md) | **English**

**Benchmarking Evidence Sufficiency and Scope Awareness in Vietnamese Student Affairs Question Answering**

Huu Dung Nguyen · Phuoc Tran · Van Thinh Nguyen · Bao Nhan Chau · Hoang Kiet Duong · Van Dung Nguyen

Accepted at the [6th EAI International Conference on Artificial Intelligence for Communications and Networks (EAI AICON 2026)](https://aicon-conf.eai-conferences.org/2026/). **Proceedings: To Appear.**

[Live application](https://huggingface.co/spaces/BaoNhan/tra-cuu-ctsv) · [Citation](#citation)

## Overview

ViSAQA studies Vietnamese Student Affairs question answering with an emphasis on evidence sufficiency and scope awareness. The benchmark contains **15,879 examples across 21 topics**, including answerable, insufficient-information, and out-of-scope questions. The study separates retrieval quality, answer quality on answerable questions, and rejection behavior under topic-disjoint and random splits.

The paper evaluates ViT5, BARTpho, mBART, mT5-base, and mT5-small with question-only, BM25, dense, and hybrid conditions. Retrieval benefits depend on the model and split; aggregate scores alone can obscure over-rejection and unsupported answers.

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

This repository currently contains the bilingual documentation, application screenshots, and citation metadata. The implementation source has not yet been added. The paper PDF and proceedings files are not included.

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
