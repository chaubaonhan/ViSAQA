"""Inference never substitutes gold context for retrieved context."""
from pathlib import Path
from .core import prompt, read_json, read_jsonl, sha256, unique_rows, write_json, write_jsonl
from .modeling import generate, load_tokenizer, set_generation


def check_condition(mode, condition):
    expected = "q_only" if condition == "q_only" else "context"
    if mode != expected:
        raise ValueError(f"{condition} requires a {expected} checkpoint, got {mode}")


def run(args):
    import torch
    from transformers import AutoModelForSeq2SeqLM
    checkpoint = Path(args.checkpoint)
    manifest = read_json(checkpoint / "run.json")
    if manifest.get("status") != "complete":
        raise ValueError("Checkpoint is not marked as completed")
    config = manifest["config"]
    check_condition(manifest["mode"], args.condition)
    if bool(args.records) == bool(args.question):
        raise ValueError("Provide exactly one of --records or --question")
    if args.records:
        rows = read_jsonl(args.records)
        unique_rows(rows, "id")
        if any(not isinstance(r.get("question"), str) or not r["question"].strip() for r in rows):
            raise ValueError("Every query requires a nonempty question")
    else:
        rows = [{"id": "interactive", "question": args.question}]
    retrieval, retriever = {}, None
    if args.condition != "q_only":
        if bool(args.retrieval) == bool(args.index):
            raise ValueError("RAG inference needs exactly one of --retrieval or --index")
        if args.retrieval:
            retrieval = unique_rows(read_jsonl(args.retrieval), "id")
            for row in rows:
                hit = retrieval.get(row["id"])
                if not hit or hit["question"] != row["question"] or hit["method"] != args.condition:
                    raise ValueError(f"Retrieval cache mismatch for {row['id']}")
                if hit.get("retrieval_config") != config["retrieval"]:
                    raise ValueError("Retrieval cache config does not match training run config")
        else:
            from .retrieval import Retriever
            retriever = Retriever(args.index, config, args.device, args.condition)
    elif args.retrieval or args.index:
        raise ValueError("Q-only inference must not use a retrieval cache/index")
    tokenizer = load_tokenizer(checkpoint, manifest["model"])
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device).eval()
    set_generation(model, tokenizer, manifest["model"], config)
    predictions = []
    for row in rows:
        hits = []
        if args.condition != "q_only":
            result = retrieval[row["id"]] if args.retrieval else retriever.search(row["id"], row["question"])
            hits = result["after"][:config["retrieval"]["top_k"]]
            if len(hits) != config["retrieval"]["top_k"]:
                raise ValueError("Retrieved context must contain top_k chunks")
        context = "\n\n".join(h["text"] for h in hits)
        text = prompt(config, row["question"], context, manifest["mode"])
        prediction = generate(model, tokenizer, text, config, device)
        prediction.update(id=row["id"], question=row["question"], context=context,
                          retrieved_chunks=hits, condition=args.condition)
        predictions.append(prediction)
        if args.question:
            print(prediction["prediction"])
    write_jsonl(args.output, predictions)
    write_json(str(args.output) + ".meta.json", {
        "model": manifest["model"], "protocol": manifest["protocol"], "seed": manifest["seed"],
        "mode": manifest["mode"], "condition": args.condition, "config": config,
        "checkpoint_manifest_sha256": sha256(checkpoint / "run.json"),
        "queries_sha256": sha256(args.records) if args.records else None,
        "retrieval_sha256": sha256(args.retrieval) if args.retrieval else None,
        "predictions_sha256": sha256(args.output),
    })
