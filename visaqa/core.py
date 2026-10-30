"""Shared I/O, labels and explicit implementation conventions."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

INSUFFICIENT = "Không có đủ thông tin trong tài liệu được cung cấp để trả lời câu hỏi này."
OOS = "Câu hỏi nằm ngoài phạm vi hệ thống RAG nghiệp vụ công tác sinh viên."
LABELS = ("answerable", "insufficient", "oos")
REJECTIONS = {"insufficient": INSUFFICIENT, "oos": OOS}


def read_json(path):
    with open(path, encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"Empty JSONL: {path}")
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokens(text):
    # Keep Vietnamese letters and mT5 sentinels; do not remove rejection prefixes.
    text = unicodedata.normalize("NFC", text).lower()
    return re.findall(r"<extra_id_\d+>|\w+", text, flags=re.UNICODE)


def normalize(text):
    return " ".join(tokens(text))


def category(text):
    normalized = normalize(text)
    for label, sentence in REJECTIONS.items():
        if normalized == normalize(sentence):
            return label
    return "answerable"


def unique_rows(rows, key):
    result = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value.strip() or value in result:
            raise ValueError(f"Missing/duplicate/non-string {key}: {value!r}")
        result[value] = row
    return result


def prompt(config, question, context, mode):
    if mode not in ("q_only", "context"):
        raise ValueError(f"Unknown training mode: {mode}")
    return config["prompt"][mode].format(question=question, context=context)


def decode_preserving_sentinels(tokenizer, ids):
    # Remove BOS/EOS/PAD/language tokens, but retain any emitted <extra_id_N>.
    special = set(tokenizer.all_special_ids)
    keep = {tokenizer.convert_tokens_to_ids(t) for t in tokenizer.all_special_tokens
            if re.fullmatch(r"<extra_id_\d+>", t)}
    filtered = [int(i) for i in ids if int(i) not in special - keep]
    return tokenizer.decode(filtered, skip_special_tokens=False, clean_up_tokenization_spaces=False).strip()


def chunk_windows(length, size, overlap):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < chunk size")
    for start in range(0, length, size - overlap):
        end = min(start + size, length)
        yield start, end
        if end == length:
            break
