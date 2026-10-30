import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from visaqa.core import INSUFFICIENT, OOS, category, chunk_windows, decode_preserving_sentinels, prompt, read_json, write_jsonl
from visaqa.data import prepare, validate_records
from visaqa.evaluate import generation_metrics, retrieval_metrics
from visaqa.infer import check_condition
from visaqa.matrix import build_plan
from visaqa.retrieval import BM25, ranked, rrf
from visaqa.train import accumulation_steps


def record(identifier="q1", topic=1, split="train", label="answerable"):
    return {"id": identifier, "question": "Câu hỏi thử nghiệm", "answer": {
                "answerable": "Câu trả lời thử nghiệm", "insufficient": INSUFFICIENT, "oos": OOS}[label],
            "context": "Ngữ cảnh giả lập", "topic_id": topic, "subtopic": "",
            "source_document_ids": ["doc"], "answer_type": label, "is_oos": label == "oos",
            "splits": {"topic": split, "random": split}, "relevant_chunk_ids": ["c1"]}


class CoreTests(unittest.TestCase):
    def test_rejection_parser_is_exact_and_preserves_sentinel(self):
        self.assertEqual(category(INSUFFICIENT.upper()), "insufficient")
        self.assertEqual(category(OOS), "oos")
        self.assertEqual(category("<extra_id_0> " + INSUFFICIENT), "answerable")
        self.assertEqual(category(INSUFFICIENT + " Vui lòng liên hệ trường."), "answerable")
        self.assertEqual(category("Không có đủ thông tin"), "answerable")

    def test_decode_keeps_only_sentinel_special_tokens(self):
        class Tokenizer:
            all_special_ids = [0, 1, 2, 3]
            all_special_tokens = ["<pad>", "</s>", "vi_VN", "<extra_id_0>"]
            def convert_tokens_to_ids(self, token):
                return self.all_special_tokens.index(token)
            def decode(self, ids, **kwargs):
                return " ".join({3: "<extra_id_0>", 4: "Không"}[i] for i in ids)
        self.assertEqual(decode_preserving_sentinels(Tokenizer(), [0, 2, 3, 4, 1]), "<extra_id_0> Không")

    def test_chunk_overlap_and_no_redundant_tail(self):
        self.assertEqual(list(chunk_windows(1000, 512, 80)), [(0, 512), (432, 944), (864, 1000)])
        self.assertEqual(list(chunk_windows(944, 512, 80)), [(0, 512), (432, 944)])
        with self.assertRaises(ValueError):
            list(chunk_windows(10, 5, 5))

    def test_effective_batch_and_checkpoint_modes(self):
        self.assertEqual(accumulation_steps(256, 8), 32)
        self.assertEqual(accumulation_steps(256, 4, 2), 32)
        with self.assertRaises(ValueError):
            accumulation_steps(256, 3)
        with self.assertRaises(ValueError):
            check_condition("q_only", "dense")

    def test_q_only_does_not_leak_context(self):
        config = read_json("configs/paper.json")
        self.assertNotIn("SECRET", prompt(config, "q", "SECRET", "q_only"))
        self.assertIn("SECRET", prompt(config, "q", "SECRET", "context"))

    def test_constant_baseline_denominators(self):
        rows = [record("a"), record("b", label="insufficient"), record("c", label="oos")]
        report = generation_metrics(rows, [INSUFFICIENT] * 3)
        self.assertAlmostEqual(report["overall"]["em"], 100 / 3)
        self.assertEqual(report["answerable"]["em"], 0)
        self.assertEqual(report["rejection"]["recall"], 100)
        self.assertEqual(report["false_rejection_rate"], 100)
        self.assertEqual(report["oos"]["f1"], 0)

    def test_recall_is_not_hit_and_denominator_excludes_missing_judgments(self):
        rows = [dict(id="q", question="test", relevant_chunk_ids=["a", "b"]),
                dict(id="missing", question="unknown", relevant_chunk_ids=[])]
        hits = [dict(chunk_id="x"), dict(chunk_id="b"), dict(chunk_id="z")]
        output = [dict(id="q", question="test", before=hits, after=hits),
                  dict(id="missing", question="unknown", before=[], after=[])]
        report = retrieval_metrics(rows, output)
        self.assertEqual(report["n_evaluated"], 1)
        self.assertEqual(report["after"]["hit@3"], 100)
        self.assertEqual(report["after"]["recall@3"], 50)
        self.assertEqual(report["after"]["mrr@3"], 50)

    def test_bm25_vietnamese_and_rrf(self):
        scores = BM25(["học bổng sinh viên", "thời tiết"]).scores("học bổng")
        self.assertEqual(ranked(scores, 1), [0])
        order, _ = rrf([[0, 1], [1, 2]], 60)
        self.assertEqual(order[0], 1)

    def test_split_integrity_and_no_silent_regeneration(self):
        rows = [record(), record("dev", 3, "dev"), record("test", 4, "test")]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "records.jsonl", rows)
            write_jsonl(root / "chunks.jsonl", [{"chunk_id": "c1", "document_id": "doc", "text": "mock"}])
            args = SimpleNamespace(records=root / "records.jsonl", chunks=root / "chunks.jsonl",
                                   output=root / "splits", paper_counts=False)
            prepare(args)
            rows[0]["splits"]["topic"] = "test"
            write_jsonl(root / "records.jsonl", rows)
            args.output = root / "bad"
            with self.assertRaisesRegex(ValueError, "Table 1"):
                prepare(args)

    def test_reject_duplicate_ids_and_inconsistent_labels(self):
        with self.assertRaises(ValueError):
            validate_records([record(), record()])
        row = record(label="oos")
        row["is_oos"] = False
        with self.assertRaises(ValueError):
            validate_records([row])

    def test_grid_reuses_context_checkpoint(self):
        config = read_json("configs/paper.json")
        jobs = build_plan(config, "configs/paper.json", "data/splits", "artifacts/index", "runs")
        self.assertEqual(sum(j["kind"] == "train" for j in jobs), 60)
        self.assertEqual(sum(j["kind"] == "infer" for j in jobs), 120)
        self.assertEqual(sum(j["kind"] == "retrieve" for j in jobs), 6)
        self.assertEqual(sum(j["kind"] == "aggregate" for j in jobs), 40)
        inference = [j["argv"] for j in jobs if j["kind"] == "infer"]
        checkpoints = [a[a.index("--checkpoint") + 1] for a in inference]
        self.assertEqual(len(set(checkpoints)), 60)


if __name__ == "__main__":
    unittest.main()
