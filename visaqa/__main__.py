"""Run with python -m visaqa --help. Heavy dependencies are imported only by ML commands."""
import argparse
import importlib


def main():
    parser = argparse.ArgumentParser(prog="visaqa")
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name, module, function, help_text):
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(handler=(module, function))
        return p

    def config(p):
        p.add_argument("--config", default="configs/paper.json")

    def device(p):
        p.add_argument("--device", help="cpu, cuda, cuda:0; auto-select when omitted")

    p = command("prepare", "data", "prepare", "Validate local records/chunks and preserve original splits")
    p.add_argument("--records", required=True)
    p.add_argument("--chunks", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--paper-counts", action="store_true", help="Enforce paper corpus/class counts")

    p = command("chunk", "chunk", "run", "Optional 512-token chunker (original chunks are preferred)")
    config(p)
    p.add_argument("--documents", required=True)
    p.add_argument("--output", required=True)

    p = command("index", "retrieval", "build_index", "Encode local chunks with E5")
    config(p); device(p)
    p.add_argument("--chunks", required=True)
    p.add_argument("--output", required=True)

    p = command("retrieve", "retrieval", "retrieve", "Retrieve 20 candidates and rerank top-3")
    config(p); device(p)
    p.add_argument("--index", required=True)
    p.add_argument("--records", required=True)
    p.add_argument("--method", choices=["bm25", "dense", "hybrid"], required=True)
    p.add_argument("--output", required=True)

    p = command("train", "train", "run", "Fine-tune a Q-only or reference-context generator")
    config(p)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--protocol", choices=["topic", "random"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--mode", choices=["q_only", "context"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, help="Micro batch; accumulation maintains effective batch size")
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--resume", help="Existing trainer checkpoint directory; requires matching run.json")

    p = command("infer", "infer", "run", "Generate from a saved fine-tuned checkpoint")
    device(p)
    p.add_argument("--checkpoint", required=True)
    queries = p.add_mutually_exclusive_group(required=True)
    queries.add_argument("--records")
    queries.add_argument("--question")
    p.add_argument("--condition", choices=["q_only", "bm25", "dense", "hybrid"], required=True)
    p.add_argument("--retrieval", help="Cached retrieval JSONL")
    p.add_argument("--index", help="Retrieve live from this index")
    p.add_argument("--output", required=True)

    p = command("evaluate", "evaluate", "run", "Answerable/overall scores, rejection and constant baselines")
    config(p); device(p)
    p.add_argument("--references", required=True)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--predictions")
    inputs.add_argument("--baseline", choices=["insufficient", "oos"])
    p.add_argument("--extended", action="store_true", help="Add SacreBLEU")
    p.add_argument("--bertscore", action="store_true", help="Add BERTScore; downloads configured model")
    p.add_argument("--output", required=True)

    p = command("evaluate-retrieval", "evaluate", "evaluate_retrieval", "Hit/Recall/MRR before and after reranking")
    p.add_argument("--references", required=True)
    p.add_argument("--retrieval", required=True)
    p.add_argument("--unit", choices=["chunk", "document"], required=True)
    p.add_argument("--output", required=True)

    p = command("aggregate", "evaluate", "aggregate", "Mean and SD across seeds 22,42,202")
    p.add_argument("--reports", nargs="+", required=True)
    p.add_argument("--output", required=True)

    p = command("matrix", "matrix", "run", "Write the complete experiment plan; does not train by default")
    config(p)
    p.add_argument("--data-dir", default="data/splits")
    p.add_argument("--index", default="artifacts/index")
    p.add_argument("--output-dir", default="runs")
    p.add_argument("--plan", default="runs/plan.json")
    p.add_argument("--execute", action="store_true", help="Actually execute all runs (substantial GPU compute)")
    args = parser.parse_args()
    try:
        module, function = args.handler
        getattr(importlib.import_module(f"visaqa.{module}"), function)(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
