"""Plan the 60 training runs and 120 inference configurations; execution is opt-in."""
import subprocess
import sys
from pathlib import Path
from .core import read_json, write_json


def build_plan(config, config_path, data_dir, index, output_dir):
    jobs = []
    root = Path(output_dir)
    def add(kind, arguments):
        jobs.append({"kind": kind, "argv": [sys.executable, "-m", "visaqa", *map(str, arguments)]})
    for protocol in config["protocols"]:
        records = Path(data_dir) / protocol / "test.jsonl"
        for method in ("bm25", "dense", "hybrid"):
            cache = root / "retrieval" / f"{protocol}-{method}.jsonl"
            add("retrieve", ["retrieve", "--config", config_path, "--index", index,
                            "--records", records, "--method", method, "--output", cache])
            add("retrieval_eval", ["evaluate-retrieval", "--references", records, "--retrieval", cache,
                                  "--unit", "chunk", "--output", str(cache) + ".metrics.json"])
        for baseline in ("insufficient", "oos"):
            add("baseline", ["evaluate", "--references", records, "--baseline", baseline, "--extended",
                             "--output", root / "metrics" / f"{protocol}-always-{baseline}.json"])
        for model in config["models"]:
            for seed in config["seeds"]:
                for mode in ("q_only", "context"):
                    checkpoint = root / "checkpoints" / protocol / model / str(seed) / mode
                    add("train", ["train", "--config", config_path, "--data-dir", data_dir,
                                  "--protocol", protocol, "--model", model, "--seed", seed,
                                  "--mode", mode, "--output", checkpoint])
                    conditions = ["q_only"] if mode == "q_only" else ["bm25", "dense", "hybrid"]
                    for condition in conditions:
                        prediction = root / "predictions" / f"{protocol}-{model}-{seed}-{condition}.jsonl"
                        arguments = ["infer", "--checkpoint", checkpoint, "--records", records,
                                     "--condition", condition, "--output", prediction]
                        if condition != "q_only":
                            arguments += ["--retrieval", root / "retrieval" / f"{protocol}-{condition}.jsonl"]
                        add("infer", arguments)
                        add("evaluate", ["evaluate", "--config", config_path, "--references", records,
                                         "--predictions", prediction, "--extended", "--bertscore",
                                         "--output", root / "metrics" / f"{protocol}-{model}-{seed}-{condition}.json"])
            for condition in ("q_only", "bm25", "dense", "hybrid"):
                paths = [root / "metrics" / f"{protocol}-{model}-{seed}-{condition}.json" for seed in config["seeds"]]
                add("aggregate", ["aggregate", "--reports", *paths, "--output",
                                  root / "summary" / f"{protocol}-{model}-{condition}.json"])
    return jobs


def run(args):
    jobs = build_plan(read_json(args.config), args.config, args.data_dir, args.index, args.output_dir)
    write_json(args.plan, jobs)
    counts = {kind: sum(job["kind"] == kind for job in jobs) for kind in dict.fromkeys(j["kind"] for j in jobs)}
    print(f"Saved plan to {args.plan}: {counts}")
    if args.execute:
        for number, job in enumerate(jobs, start=1):
            print(f"[{number}/{len(jobs)}] {job['kind']}", flush=True)
            subprocess.run(job["argv"], check=True)
