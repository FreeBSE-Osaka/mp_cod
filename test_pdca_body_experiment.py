import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cod_model as cod
from tools import pdca_body_experiment as experiment
from tools.general_body_training import body_checks, cases, score_output, valid


CORPUS = Path(__file__).parent / "data/pdca_general_body/curated.json"


class BodyExperimentTest(unittest.TestCase):
    def test_confirmation_inversions_fail_even_when_anchors_pass(self):
        rows = {r["case"]: r for r in cases(CORPUS, {"valid", "test"})}
        for case_id in ("pdca_gallery:1", "pdca_packing:2", "pdca_reception:1",
                        "pdca_signs:1", "pdca_rooms:1"):
            row = rows[case_id]
            body = (row["body"].removesuffix("です。") + "ではありません。"
                    if row["body"].endswith("です。")
                    else row["body"][:-1] + "という報告は誤りです。")
            with self.subTest(case=case_id):
                raw = json.dumps({"bodies": [{"id": "B01", "body": body}]}, ensure_ascii=False)
                result = score_output(raw, row)
                self.assertTrue(result["strict_schema"])
                self.assertTrue(result["checks"]["anchors"])
                self.assertFalse(result["checks"]["aligned"])
                self.assertFalse(result["direct_valid"])
                self.assertFalse(result["with_sanitizer_valid"])
                self.assertFalse(cod.body_matches_claim(body, row["claim"]))
                self.assertIsNone(cod.sanitize_body_politeness(body, row["claim"]))

    def test_confirmation_negations_preserve_the_frozen_claim(self):
        for state in ("未確認", "未検証", "未測定"):
            for particle in ("では", "じゃ"):
                claim = f"効果は{state}{particle}ない"
                body = f"効果は{state}{particle}ありません。"
                with self.subTest(claim=claim):
                    self.assertEqual(cod.body_confirmation_states(body), {"negated_unconfirmed"})
                    self.assertTrue(cod.body_matches_claim(body, claim))
                    self.assertFalse(cod.body_matches_claim(f"効果は{state}です。", claim))
        for verb in ("確認", "検証"):
            claim = f"効果を{verb}したという報告は誤り"
            body = f"効果を{verb}しましたという報告は誤りです。"
            with self.subTest(claim=claim):
                self.assertEqual(cod.body_confirmation_states(body), {"denied_confirmation"})
                self.assertTrue(cod.body_matches_claim(body, claim))
                self.assertFalse(cod.body_matches_claim(f"効果を{verb}した。", claim))
                self.assertTrue(cod.body_matches_claim(
                    f"手順の誤りを{verb}しました。", f"手順の誤りを{verb}した"))
        self.assertEqual(cod.body_confirmation_states(
            "入力の誤りを確認しましたが、効果は未検証ではありません。"),
            {"confirmed", "negated_unconfirmed"})
        self.assertEqual(cod.body_confirmation_states(
            "効果を確認しましたという報告は誤りではありません。"), {"confirmed"})

    def test_authored_targets_and_topic_separation(self):
        rows = cases(CORPUS, {"valid", "test"})
        self.assertEqual(len(rows), 16)
        self.assertEqual(len({r["claim"] for r in rows}), 16)
        self.assertTrue(all(valid(body_checks(r["body"], r)) for r in rows))
        for split in ("valid", "test"):
            selected = [r for r in rows if r["split"] == split]
            self.assertEqual(len(selected), 8)
            self.assertEqual(len({r["speaker"] for r in selected}), 8)
        self.assertFalse({r["topic"] for r in rows if r["split"] == "valid"}
                         & {r["topic"] for r in rows if r["split"] == "test"})

    def test_real_scorer_uses_raw_generation_and_keeps_holdout_out(self):
        rows = cases(CORPUS, {"valid"})
        by_claim = {r["claim"]: r for r in rows}
        requests = []
        def ask(**kwargs):
            requests.append(kwargs)
            item = json.loads(kwargs["user"])["items"][0]
            row = by_claim[item["claim"]]
            body = row["body"]
            if row is rows[0]:
                body = "入場札の色分けで待ち時間が減ったことは検証済みです。"
            raw = json.dumps({"bodies": [{"id": "B01", "body": body}]}, ensure_ascii=False)
            return json.loads(raw), {"_raw_content": raw, "eval_count": 20}
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(curated=CORPUS, split="valid", model="test", api_url="http://unused/api/chat",
                                   seed=19, timeout=1, num_predict=180)
            request = {"candidate": {"system_prompt": "保持して自然に述べる"},
                       "phase": "development", "output_dir": directory}
            with patch.object(experiment, "model_identity", return_value={"name": "test", "digest": "fixed"}), \
                 patch.object(cod, "ask_ollama", side_effect=ask):
                result = experiment.evaluate_candidate(args, request)
            self.assertEqual(result["score"], 7 / 8)
            self.assertFalse(result["passed"])
            self.assertEqual(result["metrics"]["generation_errors"], 0)
            self.assertEqual(result["failures"][0]["case"], rows[0]["case"])
            self.assertIn("aligned", result["failures"][0]["failed_checks"])
            audit = json.loads(Path(result["artifacts"][0]).read_text())
            self.assertEqual(len(audit["results"]), 8)
            self.assertEqual(audit["scorer_sha256"], experiment.sha(experiment.training.__file__))
            self.assertTrue(all(row["split"] == "valid" for row in audit["results"]))
            self.assertEqual([row["seed"] for row in audit["results"]], [r["seed"] for r in requests])
            self.assertEqual(result["failures"][0]["missing_anchors"], rows[0]["required"])
            self.assertIn("検証済み", audit["results"][0]["raw"])
            self.assertTrue(all(r["system"] == request["candidate"]["system_prompt"] for r in requests))
            self.assertEqual({json.loads(r["user"])["items"][0]["claim"] for r in requests}, set(by_claim))
            self.assertFalse(any("required" in r["user"] or '"body"' in r["user"] for r in requests))
            with self.assertRaisesRegex(ValueError, "already exists"):
                experiment.evaluate_candidate(args, request)

    def test_experiment_phase_and_candidate_contract(self):
        args = SimpleNamespace(split="valid")
        with self.assertRaisesRegex(ValueError, "only system_prompt"):
            experiment.evaluate_candidate(args, {"candidate": {"system_prompt": "x", "command": "bad"}})
        with self.assertRaisesRegex(ValueError, "split does not match"):
            experiment.evaluate_candidate(args, {"candidate": {"system_prompt": "x"}, "phase": "holdout"})


if __name__ == "__main__":
    unittest.main()
