"""BM25, normalized E5 dense retrieval, RRF fusion and BGE cross-encoder reranking."""
import math
from collections import Counter
from pathlib import Path
from .core import read_json, read_jsonl, sha256, tokens, write_json, write_jsonl
from .data import validate_chunks


class BM25:
    """Okapi with positive log(1 + (N-df+.5)/(df+.5)) IDF; see reproducibility notes."""
    def __init__(self, texts, k1=1.5, b=0.75):
        self.docs = [Counter(tokens(t)) for t in texts]
        self.lengths = [sum(doc.values()) for doc in self.docs]
        self.average = sum(self.lengths) / len(self.docs) if self.docs else 0
        self.df = Counter(term for doc in self.docs for term in doc)
        self.k1, self.b = k1, b

    def scores(self, query):
        scores = []
        for doc, length in zip(self.docs, self.lengths):
            score = 0.0
            for term in set(tokens(query)):
                frequency = doc.get(term, 0)
                if not frequency:
                    continue
                df = self.df[term]
                idf = math.log(1 + (len(self.docs) - df + 0.5) / (df + 0.5))
                norm = frequency + self.k1 * (1 - self.b + self.b * length / self.average)
                score += idf * frequency * (self.k1 + 1) / norm
            scores.append(score)
        return scores


def ranked(scores, k):
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:k]


def rrf(rankings, constant=60):
    scores = {}
    for ranking in rankings:
        for rank, item in enumerate(dict.fromkeys(ranking), start=1):
            scores[item] = scores.get(item, 0.0) + 1 / (constant + rank)
    return sorted(scores, key=lambda i: (-scores[i], i)), scores


class Encoder:
    def __init__(self, name, device, max_length, batch_size):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval()
        self.device, self.max_length, self.batch_size = device, max_length, batch_size

    def encode(self, texts, prefix):
        import numpy as np
        result = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer([prefix + t for t in texts[start:start + self.batch_size]],
                                   padding=True, truncation=True, max_length=self.max_length,
                                   return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                vectors = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            result.append(vectors.cpu().float().numpy())
        return np.concatenate(result)


def build_index(args):
    import numpy as np
    import torch
    config = read_json(args.config)
    settings = config["retrieval"]
    chunks = validate_chunks(read_jsonl(args.chunks))
    if len(chunks) < settings["top_k"]:
        raise ValueError("Corpus has fewer than top_k chunks")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty directory for a new index")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(settings["encoder"], device, settings["encoder_max_length"], settings["batch_size"])
    embeddings = encoder.encode([r["text"] for r in chunks], "passage: ")
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "embeddings.npy", embeddings)
    write_jsonl(output / "chunks.jsonl", chunks)
    write_json(output / "index.json", {
        "encoder": settings["encoder"], "encoder_max_length": settings["encoder_max_length"],
        "prefix": "passage: ", "pooling": "masked_mean_l2",
        "source_chunks_sha256": sha256(args.chunks),
        "chunks_sha256": sha256(output / "chunks.jsonl"),
        "embeddings_sha256": sha256(output / "embeddings.npy"),
        "encoder_revision": getattr(encoder.model.config, "_commit_hash", None),
        "count": len(chunks),
    })
    print(f"Indexed {len(chunks)} local chunks: {output}")


class Retriever:
    def __init__(self, index, config, device=None, method="hybrid"):
        import numpy as np
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if method not in ("bm25", "dense", "hybrid"):
            raise ValueError("Retrieval method must be bm25, dense or hybrid")
        self.method, self.config = method, config["retrieval"]
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        index = Path(index)
        manifest = read_json(index / "index.json")
        for key in ("encoder", "encoder_max_length"):
            if manifest[key] != self.config[key]:
                raise ValueError(f"Index config mismatch: {key}")
        for filename, key in (("chunks.jsonl", "chunks_sha256"), ("embeddings.npy", "embeddings_sha256")):
            if sha256(index / filename) != manifest[key]:
                raise ValueError(f"Corrupted/stale index: {filename}")
        self.chunks = validate_chunks(read_jsonl(index / "chunks.jsonl"))
        self.index_hash = sha256(index / "index.json")
        self.embeddings = np.load(index / "embeddings.npy", allow_pickle=False)
        if len(self.chunks) != self.embeddings.shape[0]:
            raise ValueError("Embedding/chunk row mismatch")
        self.bm25 = BM25([r["text"] for r in self.chunks], self.config["bm25_k1"], self.config["bm25_b"])
        self.encoder = None
        if method != "bm25":
            self.encoder = Encoder(self.config["encoder"], self.device,
                                   self.config["encoder_max_length"], self.config["batch_size"])
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(self.config["reranker"])
        self.reranker = AutoModelForSequenceClassification.from_pretrained(
            self.config["reranker"]).to(self.device).eval()

    def search(self, question_id, question):
        candidate_k = self.config["candidate_k"]
        lexical = self.bm25.scores(question)
        bm_order = ranked(lexical, candidate_k)
        dense = None
        if self.method != "bm25":
            query = self.encoder.encode([question], "query: ")[0]
            dense = self.embeddings @ query
            dense_order = ranked(dense, candidate_k)
        if self.method == "bm25":
            order, candidate_scores = bm_order, lexical
        elif self.method == "dense":
            order, candidate_scores = dense_order, dense
        else:
            order, candidate_scores = rrf([bm_order, dense_order], self.config["rrf_k"])
            order = order[:candidate_k]
        before = [{**self.chunks[i], "candidate_score": float(candidate_scores[i])} for i in order]
        scores = []
        for start in range(0, len(before), self.config["batch_size"]):
            batch = before[start:start + self.config["batch_size"]]
            encoded = self.rerank_tokenizer([question] * len(batch), [r["text"] for r in batch],
                                           padding=True, truncation=True,
                                           max_length=self.config["reranker_max_length"],
                                           return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                logits = self.reranker(**encoded).logits
            if logits.shape[-1] != 1:
                raise ValueError("Reranker must return one relevance logit per pair")
            scores.extend(logits.flatten().cpu().float().tolist())
        reranked = sorted(range(len(before)), key=lambda i: (-scores[i], i))
        after = [{**before[i], "rerank_score": scores[i]} for i in reranked[:self.config["top_k"]]]
        return {"id": question_id, "question": question, "method": self.method,
                "retrieval_config": self.config, "index_sha256": self.index_hash,
                "before": before, "after": after}


def retrieve(args):
    config = read_json(args.config)
    rows = read_jsonl(args.records)
    from .core import unique_rows
    unique_rows(rows, "id")
    retriever = Retriever(args.index, config, args.device, args.method)
    write_jsonl(args.output, (retriever.search(r["id"], r["question"]) for r in rows))
    print(f"Saved retrieval results: {args.output}")
