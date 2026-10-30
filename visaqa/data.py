"""Validate local data and materialize supplied original split assignments."""
from collections import Counter
from pathlib import Path
from .core import LABELS, REJECTIONS, normalize, read_jsonl, sha256, unique_rows, write_json, write_jsonl

TOPIC_SPLITS = {
    "train": {1, 2, 6, 7, 9, 10, 12, 15, 16, 18, 19, 20},
    "dev": {3, 11, 13, 17}, "test": {4, 5, 8, 14, 21},
}
PAPER_COUNTS = {
    "topic": {"train": (7072, 4564, 1067), "dev": (1018, 377, 193), "test": (802, 554, 232)},
    "random": {"train": (7105, 4425, 1173), "dev": (874, 552, 162), "test": (913, 518, 157)},
}


def validate_records(rows):
    unique_rows(rows, "id")
    for row in rows:
        for key in ("question", "answer", "context", "subtopic"):
            if not isinstance(row.get(key), str):
                raise ValueError(f"{row['id']}: {key} must be a string")
        if not row["question"].strip() or not row["answer"].strip():
            raise ValueError(f"{row['id']}: empty question/answer")
        label = row.get("answer_type")
        if label not in LABELS or type(row.get("is_oos")) is not bool:
            raise ValueError(f"{row['id']}: invalid answer_type/is_oos")
        if row["is_oos"] != (label == "oos"):
            raise ValueError(f"{row['id']}: inconsistent OOS flag")
        if label in REJECTIONS and normalize(row["answer"]) != normalize(REJECTIONS[label]):
            raise ValueError(f"{row['id']}: rejection target does not match the paper template")
        if type(row.get("topic_id")) is not int or row["topic_id"] not in range(1, 22):
            raise ValueError(f"{row['id']}: topic_id must be 1..21 (Table 1)")
        for key in ("source_document_ids", "relevant_chunk_ids", "relevant_document_ids"):
            if key == "source_document_ids" or key in row:
                value = row.get(key)
                if not isinstance(value, list) or not all(isinstance(x, str) and x for x in value):
                    raise ValueError(f"{row['id']}: {key} must be a list of string IDs")
                if len(value) != len(set(value)):
                    raise ValueError(f"{row['id']}: duplicate IDs in {key}")
    return rows


def validate_chunks(rows):
    unique_rows(rows, "chunk_id")
    for row in rows:
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("document_id", "text")):
            raise ValueError("Chunks require nonempty document_id and text")
    return rows


def prepare(args):
    rows = validate_records(read_jsonl(args.records))
    chunks = validate_chunks(read_jsonl(args.chunks))
    chunk_ids = {r["chunk_id"] for r in chunks}
    document_ids = {r["document_id"] for r in chunks}
    for row in rows:
        for key, available in (("relevant_chunk_ids", chunk_ids), ("relevant_document_ids", document_ids),
                               ("source_document_ids", document_ids)):
            if set(row.get(key, [])) - available:
                raise ValueError(f"{row['id']}: {key} references missing corpus items")
    partitions = {}
    summary = {}
    for protocol in PAPER_COUNTS:
        groups = {split: [] for split in ("train", "dev", "test")}
        for row in rows:
            split = row.get("splits", {}).get(protocol)
            if split not in groups:
                raise ValueError(f"{row['id']}: provide original splits.{protocol}; splits are not regenerated")
            if protocol == "topic" and row["topic_id"] not in TOPIC_SPLITS[split]:
                raise ValueError(f"{row['id']}: topic split disagrees with Table 1")
            groups[split].append(row)
        summary[protocol] = {}
        for split, subset in groups.items():
            if not subset:
                raise ValueError(f"Empty {protocol}/{split}")
            counts = Counter(r["answer_type"] for r in subset)
            summary[protocol][split] = {"n": len(subset), "labels": dict(counts)}
            if args.paper_counts and tuple(counts[k] for k in LABELS) != PAPER_COUNTS[protocol][split]:
                raise ValueError(f"{protocol}/{split}: class counts do not match Table 1")
            if args.paper_counts and protocol == "random" and {r['topic_id'] for r in subset} != set(range(1, 22)):
                raise ValueError(f"{protocol}/{split}: all 21 topics must be represented")
        partitions[protocol] = groups
    if args.paper_counts and (len(rows), len(chunks), len(document_ids)) != (15879, 2571, 83):
        raise ValueError("Expected 15,879 examples, 2,571 chunks and 83 documents")
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Use an empty output directory: {root}")
    for protocol, groups in partitions.items():
        for split, subset in groups.items():
            write_jsonl(root / protocol / f"{split}.jsonl", subset)
    write_json(root / "manifest.json", {"records_sha256": sha256(args.records),
               "chunks_sha256": sha256(args.chunks), "counts": summary,
               "original_assignments_preserved": True, "paper_counts_checked": args.paper_counts})
    print(f"Validated local data and wrote splits to {root}")
