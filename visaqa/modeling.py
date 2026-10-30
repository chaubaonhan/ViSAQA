"""Model/tokenizer setup shared by training and inference."""
from .core import decode_preserving_sentinels, prompt


def load_tokenizer(checkpoint, model_key):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    if model_key == "mbart":
        tokenizer.src_lang = "vi_VN"
        tokenizer.tgt_lang = "vi_VN"
    return tokenizer


def set_generation(model, tokenizer, model_key, config):
    model.generation_config.max_length = config["generation"]["max_length"]
    model.generation_config.max_new_tokens = None
    model.generation_config.num_beams = config["generation"]["num_beams"]
    model.generation_config.do_sample = config["generation"]["do_sample"]
    if model_key == "mbart":
        model.generation_config.forced_bos_token_id = tokenizer.convert_tokens_to_ids("vi_VN")
    return model


def encode_rows(rows, tokenizer, config, mode):
    encoded = []
    truncated = 0
    target_truncated = 0
    for row in rows:
        text = prompt(config, row["question"], row["context"], mode)
        raw = tokenizer(text, truncation=False)["input_ids"]
        raw_target = tokenizer(text_target=row["answer"], truncation=False)["input_ids"]
        truncated += len(raw) > config["training"]["max_input_length"]
        target_truncated += len(raw_target) > config["training"]["max_target_length"]
        inputs = tokenizer(text, max_length=config["training"]["max_input_length"], truncation=True,
                           return_token_type_ids=False)
        inputs["labels"] = tokenizer(text_target=row["answer"],
                                    max_length=config["training"]["max_target_length"],
                                    truncation=True)["input_ids"]
        encoded.append(inputs)
    return encoded, {"n": len(rows), "input_truncated": truncated, "target_truncated": target_truncated}


def generate(model, tokenizer, text, config, device):
    import torch
    raw = tokenizer(text, truncation=False)["input_ids"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=config["training"]["max_input_length"], return_token_type_ids=False).to(device)
    with torch.inference_mode():
        outputs = model.generate(**inputs, **config["generation"])
    ids = outputs[0].tolist()
    return {
        "prediction": decode_preserving_sentinels(tokenizer, ids),
        "output_token_ids": ids,
        "input_tokens_before_truncation": len(raw),
        "input_tokens_used": int(inputs["attention_mask"].sum()),
        "input_truncated": len(raw) > config["training"]["max_input_length"],
    }
