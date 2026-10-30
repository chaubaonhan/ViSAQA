"""Optional offline CPU integration test: randomly initialized tiny models, no benchmark data."""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ENABLED = os.environ.get("VISAQA_ML_SMOKE") == "1"


@unittest.skipUnless(ENABLED, "Set VISAQA_ML_SMOKE=1 with ML dependencies to run offline integration")
class PipelineSmokeTest(unittest.TestCase):
    def test_train_retrieve_generate_evaluate(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import (PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration,
                                  BertConfig, BertModel, BertForSequenceClassification)
        from visaqa.core import read_json, read_jsonl, write_json, write_jsonl
        from visaqa.data import prepare
        from visaqa.train import run as train
        from visaqa.infer import run as infer
        from visaqa.retrieval import build_index, retrieve
        from visaqa.evaluate import run as evaluate
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2, "<extra_id_0>": 3,
                     "Câu": 4, "hỏi": 5, "Ngữ": 6, "cảnh": 7, ":": 8, "mock": 9, "answer": 10,
                     "query": 11, "passage": 12}
            tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
            tok.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>",
                                               eos_token="</s>", unk_token="<unk>",
                                               additional_special_tokens=["<extra_id_0>"])
            tokenizer.model_input_names = ["input_ids", "attention_mask"]
            base = root / "tiny-t5"
            tokenizer.save_pretrained(base)
            T5ForConditionalGeneration(T5Config(vocab_size=len(vocab), d_model=16, d_ff=32,
                                               num_layers=1, num_decoder_layers=1, num_heads=2,
                                               d_kv=8, pad_token_id=0, eos_token_id=1,
                                               decoder_start_token_id=0)).save_pretrained(base)
            encoder, reranker = root / "tiny-encoder", root / "tiny-reranker"
            for dest in (encoder, reranker):
                tokenizer.save_pretrained(dest)
            bert_config = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                                     num_attention_heads=2, intermediate_size=32,
                                     max_position_embeddings=128, num_labels=1)
            BertModel(bert_config).save_pretrained(encoder)
            BertForSequenceClassification(bert_config).save_pretrained(reranker)
            config = read_json("configs/paper.json")
            config["models"] = {"vit5": str(base)}
            config["seeds"] = [42]
            config["training"].update(epochs=1, effective_batch_size=2, per_device_batch_size=1,
                                      eval_batch_size=1, max_input_length=32, max_target_length=8)
            config["generation"]["max_length"] = 8
            config["retrieval"].update(encoder=str(encoder), reranker=str(reranker),
                                      encoder_max_length=32, reranker_max_length=64)
            config_path = root / "config.json"
            write_json(config_path, config)
            records = []
            for identifier, topic, split in [("train1", 1, "train"), ("train2", 2, "train"),
                                              ("dev", 3, "dev"), ("test", 4, "test")]:
                records.append({"id": identifier, "question": "mock", "answer": "answer",
                                "context": "mock answer", "topic_id": topic, "subtopic": "",
                                "source_document_ids": ["doc"], "answer_type": "answerable",
                                "is_oos": False, "splits": {"topic": split, "random": split},
                                "relevant_chunk_ids": ["c0"]})
            chunks = [{"chunk_id": f"c{i}", "document_id": "doc", "text": "mock answer " * (i + 1)} for i in range(3)]
            write_jsonl(root / "records.jsonl", records)
            write_jsonl(root / "chunks.jsonl", chunks)
            prepare(SimpleNamespace(records=root / "records.jsonl", chunks=root / "chunks.jsonl",
                                    output=root / "splits", paper_counts=False))
            build_index(SimpleNamespace(config=config_path, chunks=root / "chunks.jsonl",
                                        output=root / "index", device="cpu"))
            references = root / "splits" / "topic" / "test.jsonl"
            for method in ("bm25", "dense", "hybrid"):
                retrieve(SimpleNamespace(config=config_path, index=root / "index", records=references,
                                         method=method, output=root / f"{method}.jsonl", device="cpu"))
                self.assertEqual(len(read_jsonl(root / f"{method}.jsonl")[0]["after"]), 3)
            for mode in ("q_only", "context"):
                checkpoint = root / mode
                train(SimpleNamespace(config=config_path, model="vit5", seed=42, protocol="topic",
                                      mode=mode, data_dir=root / "splits", output=checkpoint,
                                      batch_size=None, precision=None, resume=None))
                self.assertEqual(read_json(checkpoint / "run.json")["status"], "complete")
                for condition in (("q_only",) if mode == "q_only" else ("bm25", "dense", "hybrid")):
                    output = root / f"pred-{condition}.jsonl"
                    infer(SimpleNamespace(checkpoint=checkpoint, records=references, question=None,
                                          condition=condition, output=output, device="cpu", index=None,
                                          retrieval=None if mode == "q_only" else root / f"{condition}.jsonl"))
                    predictions = read_jsonl(output)
                    self.assertEqual(len(predictions), 1)
                    self.assertIsInstance(predictions[0]["prediction"], str)
                    self.assertEqual(bool(predictions[0]["context"]), mode == "context")
                    evaluate(SimpleNamespace(references=references, predictions=output, baseline=None,
                                             extended=True, bertscore=False, device="cpu",
                                             output=root / f"metrics-{condition}.json"))
                    metrics = read_json(root / f"metrics-{condition}.json")
                    self.assertEqual(metrics["overall"]["n"], 1)
                    self.assertIn("bleu", metrics["overall"])


if __name__ == "__main__":
    unittest.main()
