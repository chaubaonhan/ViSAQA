"""Full fine-tuning, with separate Q-only and reference-context checkpoints."""
import importlib.metadata
import os
from pathlib import Path
from .core import read_json, read_jsonl, sha256, write_json
from .data import validate_records
from .modeling import encode_rows, load_tokenizer, set_generation


def accumulation_steps(effective, micro, world_size=1):
    divisor = micro * world_size
    if min(effective, micro, world_size) <= 0 or effective % divisor:
        raise ValueError("Effective batch size must be divisible by micro batch * world size")
    return effective // divisor


def run(args):
    import torch
    from transformers import (AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed)
    config = read_json(args.config)
    if args.model not in config["models"] or args.seed not in config["seeds"]:
        raise ValueError("Model/seed not listed in the config")
    if args.protocol not in config["protocols"]:
        raise ValueError("Protocol not listed in the config")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or torch.cuda.device_count() > 1:
        raise ValueError("This release matches the single-GPU setup; expose one GPU with CUDA_VISIBLE_DEVICES")
    settings = config["training"]
    if args.batch_size is not None:
        settings["per_device_batch_size"] = args.batch_size
    if args.precision:
        settings["precision"] = args.precision
    accumulation = accumulation_steps(settings["effective_batch_size"], settings["per_device_batch_size"])
    if settings["precision"] != "fp32" and not torch.cuda.is_available():
        raise ValueError("Use fp32 on CPU")
    if settings["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bf16")
    if settings["checkpoint_selection"] != "last_epoch":
        raise ValueError("Only the documented last_epoch policy is implemented")
    source = Path(args.data_dir) / args.protocol
    train_path, dev_path = source / "train.jsonl", source / "dev.jsonl"
    train_rows = validate_records(read_jsonl(train_path))
    dev_rows = validate_records(read_jsonl(dev_path))
    if {r["id"] for r in train_rows} & {r["id"] for r in dev_rows}:
        raise ValueError("Train/dev question ID overlap")
    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        if any(r.get("splits", {}).get(args.protocol) != split for r in rows):
            raise ValueError(f"{split}: inconsistent original split assignments")
    if args.mode == "context" and any(not r["context"].strip() for r in train_rows + dev_rows
                                     if r["answer_type"] == "answerable"):
        raise ValueError("Answerable context-training examples must have reference contexts")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError(f"Output is not empty; use a new directory or --resume: {output}")
    set_seed(args.seed)
    base_model = config["models"][args.model]
    tokenizer = load_tokenizer(base_model, args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    set_generation(model, tokenizer, args.model, config)
    train_data, train_audit = encode_rows(train_rows, tokenizer, config, args.mode)
    dev_data, dev_audit = encode_rows(dev_rows, tokenizer, config, args.mode)
    if settings["gradient_checkpointing"]:
        model.config.use_cache = False
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output), num_train_epochs=settings["epochs"],
        learning_rate=settings["learning_rate"],
        per_device_train_batch_size=settings["per_device_batch_size"],
        per_device_eval_batch_size=settings["eval_batch_size"],
        gradient_accumulation_steps=accumulation,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=False, predict_with_generate=False,
        optim=settings["optimizer"], weight_decay=settings["weight_decay"],
        warmup_ratio=settings["warmup_ratio"], lr_scheduler_type=settings["scheduler"],
        max_grad_norm=settings["max_grad_norm"],
        bf16=settings["precision"] == "bf16", fp16=settings["precision"] == "fp16",
        gradient_checkpointing=settings["gradient_checkpointing"],
        dataloader_num_workers=0, dataloader_drop_last=False,
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=args.seed, data_seed=args.seed, report_to=[],
        logging_steps=10, generation_max_length=config["generation"]["max_length"],
        generation_num_beams=config["generation"]["num_beams"],
    )
    manifest = {
        "model": args.model, "base_model": base_model, "protocol": args.protocol,
        "seed": args.seed, "mode": args.mode, "config": config,
        "gradient_accumulation_steps": accumulation,
        "train_sha256": sha256(train_path), "dev_sha256": sha256(dev_path),
        "train_audit": train_audit, "dev_audit": dev_audit,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "accelerate")},
        "status": "started",
    }
    if args.resume:
        previous = read_json(output / "run.json")
        for key in ("model", "protocol", "seed", "mode", "config", "train_sha256", "dev_sha256"):
            if previous[key] != manifest[key]:
                raise ValueError(f"Resume mismatch: {key}")
    write_json(output / "run.json", manifest)
    trainer = Seq2SeqTrainer(
        model=model, args=training_args, train_dataset=train_data, eval_dataset=dev_data,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100),
    )
    result = trainer.train(resume_from_checkpoint=args.resume)
    model.config.use_cache = True
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    manifest.update(status="complete", training_metrics=result.metrics)
    write_json(output / "run.json", manifest)
    print(f"Saved final-epoch checkpoint: {output}")
