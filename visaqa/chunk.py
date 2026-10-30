"""Optional deterministic chunker. Original chunks are preferred for paper comparisons."""
import hashlib
from .core import chunk_windows, read_json, read_jsonl, sha256, unique_rows, write_json, write_jsonl


def run(args):
    from transformers import AutoTokenizer
    config = read_json(args.config)["chunking"]
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], use_fast=True)
    documents = read_jsonl(args.documents)
    unique_rows(documents, "document_id")
    seen, chunks, duplicates = set(), [], []
    for document in documents:
        text = document.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Each document needs nonempty text")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            duplicates.append(document["document_id"])
            continue
        seen.add(digest)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        for start, end in chunk_windows(len(encoded["input_ids"]), config["size"], config["overlap"]):
            chunks.append({
                "chunk_id": f"{document['document_id']}:{start}:{end}",
                "document_id": document["document_id"],
                "text": text[offsets[start][0]:offsets[end - 1][1]],
                "token_start": start, "token_end": end,
            })
    write_jsonl(args.output, chunks)
    write_json(str(args.output) + ".meta.json", {
        "config": config, "documents_sha256": sha256(args.documents),
        "dropped_exact_duplicate_document_ids": duplicates, "chunks": len(chunks),
        "note": "New chunk IDs; rebuild relevance judgments before retrieval evaluation.",
    })
