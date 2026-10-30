"""Generation, rejection, retrieval and across-seed metrics with explicit denominators."""
import math
import statistics
from collections import Counter
from pathlib import Path
from .core import LABELS, REJECTIONS, category, normalize, read_json, read_jsonl, sha256, tokens, unique_rows, write_json
from .data import validate_records


def f1_overlap(pred, ref):
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if not pred or not ref:
        return float(pred == ref)
    return 2 * overlap / (len(pred) + len(ref))


def ngrams(words, n):
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def rouge_l(pred, ref):
    if not pred or not ref:
        return float(pred == ref)
    previous = [0] * (len(ref) + 1)
    for p in pred:
        current = [0]
        for j, r in enumerate(ref, start=1):
            current.append(previous[j - 1] + 1 if p == r else max(previous[j], current[-1]))
        previous = current
    return 2 * previous[-1] / (len(pred) + len(ref))


def binary_scores(gold, predicted):
    tp = sum(g and p for g, p in zip(gold, predicted))
    fp = sum(not g and p for g, p in zip(gold, predicted))
    fn = sum(g and not p for g, p in zip(gold, predicted))
    return {
        "precision": 100 * tp / (tp + fp) if tp + fp else 0.0,
        "recall": 100 * tp / (tp + fn) if tp + fn else 0.0,
        "f1": 200 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
    }


def lexical_metrics(predictions, references):
    if not predictions:
        return {"n": 0, "em": None, "token_f1": None, "rouge1": None, "rouge2": None, "rougeL": None}
    p_tokens, r_tokens = [tokens(p) for p in predictions], [tokens(r) for r in references]
    pairs = list(zip(p_tokens, r_tokens))
    return {
        "n": len(predictions),
        "em": 100 * statistics.mean(float(p == r) for p, r in pairs),
        "token_f1": 100 * statistics.mean(f1_overlap(p, r) for p, r in pairs),
        "rouge1": 100 * statistics.mean(f1_overlap(p, r) for p, r in pairs),
        "rouge2": 100 * statistics.mean(
            f1_overlap(ngrams(p, 2), ngrams(r, 2)) if len(p) >= 2 and len(r) >= 2 else 0.0
            for p, r in pairs),
        "rougeL": 100 * statistics.mean(rouge_l(p, r) for p, r in pairs),
    }


def generation_metrics(rows, predictions):
    references = [r["answer"] for r in rows]
    labels = [r["answer_type"] for r in rows]
    predicted_labels = [category(p) for p in predictions]
    answerable = [i for i, label in enumerate(labels) if label == "answerable"]
    return {
        "scale": "percent", "overall": lexical_metrics(predictions, references),
        "answerable": lexical_metrics([predictions[i] for i in answerable], [references[i] for i in answerable]),
        "rejection": binary_scores([x != "answerable" for x in labels], [x != "answerable" for x in predicted_labels]),
        "oos": binary_scores([x == "oos" for x in labels], [x == "oos" for x in predicted_labels]),
        "false_rejection_rate": (100 * sum(predicted_labels[i] != "answerable" for i in answerable) /
                                 len(answerable)) if answerable else None,
        "predicted_category_counts": dict(Counter(predicted_labels)),
        "sentinel_prefix_rate": 100 * sum(p.lstrip().startswith("<extra_id_") for p in predictions) / len(rows),
    }


def run(args):
    rows = validate_records(read_jsonl(args.references))
    if args.baseline:
        predictions = [REJECTIONS[args.baseline]] * len(rows)
        metadata = {"baseline": args.baseline}
    else:
        if not args.predictions:
            raise ValueError("Provide --predictions or --baseline")
        prediction_rows = unique_rows(read_jsonl(args.predictions), "id")
        if set(prediction_rows) != {r["id"] for r in rows}:
            raise ValueError("Prediction/reference IDs differ; refusing partial evaluation")
        for row in rows:
            pred = prediction_rows[row["id"]]
            if pred.get("question") != row["question"] or not isinstance(pred.get("prediction"), str):
                raise ValueError(f"Prediction mismatch: {row['id']}")
        predictions = [prediction_rows[r["id"]]["prediction"] for r in rows]
        metadata_path = Path(str(args.predictions) + ".meta.json")
        metadata = read_json(metadata_path) if metadata_path.exists() else {}
        if metadata and metadata["predictions_sha256"] != sha256(args.predictions):
            raise ValueError("Prediction file changed after inference")
    result = generation_metrics(rows, predictions)
    metric_metadata = {"normalization": "NFC lowercase; Unicode word tokens; retain extra_id sentinels",
                       "rouge": "macro token n-gram/LCS F1; no stemming"}
    if args.extended or args.bertscore:
        from sacrebleu.metrics import BLEU
        bleu = BLEU(tokenize="none", effective_order=False)
        for subset, indices in (
            ("overall", list(range(len(rows)))),
            ("answerable", [i for i, r in enumerate(rows) if r["answer_type"] == "answerable"]),
        ):
            result[subset]["bleu"] = bleu.corpus_score(
                [normalize(predictions[i]) for i in indices],
                [[normalize(rows[i]["answer"]) for i in indices]]).score if indices else None
        metric_metadata["bleu"] = str(bleu.get_signature())
    if args.bertscore:
        from bert_score import score
        config = read_json(args.config)["evaluation"]
        p, r, f, model_hash = score(
            predictions, [row["answer"] for row in rows], model_type=config["bertscore_model"],
            num_layers=config["bertscore_layers"], lang="vi", rescale_with_baseline=False,
            batch_size=8, device=args.device, return_hash=True)
        metric_metadata["bertscore_hash"] = model_hash
        for subset, indices in (
            ("overall", list(range(len(rows)))),
            ("answerable", [i for i, row in enumerate(rows) if row["answer_type"] == "answerable"]),
        ):
            for key, values in (("bertscore_p", p), ("bertscore_r", r), ("bertscore_f1", f)):
                result[subset][key] = 100 * float(values[indices].mean()) if indices else None
    result.update(run=metadata, references_sha256=sha256(args.references), metric_settings=metric_metadata)
    write_json(args.output, result)
    print(f"Saved evaluation: {args.output}")


def retrieval_metrics(rows, results, unit="chunk"):
    key = "relevant_chunk_ids" if unit == "chunk" else "relevant_document_ids"
    item_key = "chunk_id" if unit == "chunk" else "document_id"
    by_id = unique_rows(results, "id")
    if set(by_id) != {r["id"] for r in rows}:
        raise ValueError("Retrieval/reference query IDs differ")
    eligible = [row for row in rows if row.get(key)]
    result = {"n_total": len(rows), "n_evaluated": len(eligible),
              "n_without_judgments": len(rows) - len(eligible), "unit": unit, "scale": "percent"}
    for phase in ("before", "after"):
        collected = {k: [] for k in ("hit@1", "hit@3", "recall@3", "mrr@3")}
        for row in eligible:
            retrieved = by_id[row["id"]]
            if retrieved["question"] != row["question"]:
                raise ValueError(f"Query text mismatch: {row['id']}")
            # Collapse repeated documents only for document-level judgments.
            ids = list(dict.fromkeys(hit[item_key] for hit in retrieved[phase]))
            gold = set(row[key])
            top = ids[:3]
            collected["hit@1"].append(float(bool(gold.intersection(ids[:1]))))
            collected["hit@3"].append(float(bool(gold.intersection(top))))
            collected["recall@3"].append(len(gold.intersection(top)) / len(gold))
            collected["mrr@3"].append(next((1 / rank for rank, item in enumerate(top, 1) if item in gold), 0.0))
        result[phase] = {k: 100 * statistics.mean(v) if v else None for k, v in collected.items()}
    return result


def evaluate_retrieval(args):
    rows = validate_records(read_jsonl(args.references))
    result = retrieval_metrics(rows, read_jsonl(args.retrieval), args.unit)
    write_json(args.output, result)
    print(f"Saved retrieval evaluation: {args.output}")


def flatten_metrics(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten_metrics(item, name))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            if not math.isfinite(item):
                raise ValueError(f"Non-finite metric: {name}")
            result[name] = item
    return result


def aggregate(args):
    reports = [read_json(path) for path in args.reports]
    seeds = [r.get("run", {}).get("seed") for r in reports]
    if sorted(s for s in seeds if s is not None) != [22, 42, 202]:
        raise ValueError("Aggregation requires exactly one report for each seed: 22, 42, 202")
    first = reports[0]
    for report in reports[1:]:
        for key in ("model", "protocol", "condition", "config"):
            if report["run"].get(key) != first["run"].get(key):
                raise ValueError(f"Cannot mix runs with different {key}")
        for key in ("references_sha256", "metric_settings"):
            if report[key] != first[key]:
                raise ValueError(f"Cannot aggregate different {key}")
    values = [flatten_metrics({key: r[key] for key in (
        "overall", "answerable", "rejection", "oos", "false_rejection_rate", "sentinel_prefix_rate")}) for r in reports]
    if any(set(v) != set(values[0]) for v in values):
        raise ValueError("Reports contain different metrics")
    ddof = first["run"]["config"]["evaluation"]["seed_sd_ddof"]
    if ddof not in (0, 1):
        raise ValueError("seed_sd_ddof must be 0 or 1")
    output = {key: {"mean": statistics.mean(v[key] for v in values),
                    "sd": (statistics.stdev if ddof else statistics.pstdev)(v[key] for v in values)}
              for key in values[0]}
    write_json(args.output, {"seeds": seeds, "sd_ddof": ddof, "metrics": output,
                           "model": first["run"]["model"], "protocol": first["run"]["protocol"],
                           "condition": first["run"]["condition"]})
