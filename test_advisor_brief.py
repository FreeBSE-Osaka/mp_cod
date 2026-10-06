import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cod_model as cod
from tools.advisor_brief import build_brief, export_brief, strict_json


class AdvisorBriefTest(unittest.TestCase):
    def test_meeting_armtest_is_valid_and_keeps_multiple_options(self):
        ledger = cod.load_claim_ledger(Path(__file__).parent / "data/advisor_meeting_armtest/claim_ledger.json")
        self.assertTrue(ledger["evaluation_only"])
        self.assertEqual(len(ledger["role_preferences"]), 8)
        self.assertEqual({row["code"] for row in ledger["claim_catalog"] if row["contradicts"]},
                         {"PREPARE_A", "CONDITIONAL_B", "HOLD_SELECTION"})

    def setUp(self):
        self.ledger = {
            "schema_version": 1, "topic": "架空の秘書補佐試験。実際の予約依頼ではない。",
            "data": [{"id": "D01", "text": "案Aと案Bの費用見積はあるが、予約はしていない。"}],
            "claim_catalog": [{"code": code, "label": label, "supported_by": ["D01"],
                               "contradicts": [other]}
                              for code, label, other in (("A", "低費用案を検討する", "B"),
                                                         ("B", "短時間案を検討する", "A"))],
        }
        self.run = {
            "schema_version": 2, "portable_context_schema_version": 1,
            "ledger_snapshot": copy.deepcopy(self.ledger), "persona_order": ["p1", "p2"],
            "persona_names": {"p1": "費用観点", "p2": "時間観点"},
            "independent": {p: {"raw": "synthetic test", "rejected": []} for p in ("p1", "p2")},
            "events": [dict(self.record(code), persona_id=p, claim_id=f"C{i}",
                            target_claim_id=None, action="new", origin="model")
                       for i, (p, code) in enumerate((("p1", "A"), ("p2", "B")), 1)],
            "reconciliation": [{"round": 1, "votes": {"A|B": {
                "p1": self.record("A"), "p2": self.record("B")}}}],
            "runtime": {"elapsed_seconds": 1.0, "model_seconds": 0.5, "model_call_count": 4},
        }
        self.resummarize()

    def record(self, code):
        return {"code": code, "choice": code, "confidence": 60, "data_ids": ["D01"],
                "statement": f"見積を根拠に{code}を検討します。[D01] 予約はまだ行っていません。",
                "statement_origin": "model", "utterance": f"私は案{code}を検討したいです。",
                "utterance_origin": "model", "choice_origin": "model_json", "raw": "synthetic test"}

    def resummarize(self):
        self.run["summary"] = cod.synthesize_event_summary(
            self.run["events"], {c["code"]: c for c in self.ledger["claim_catalog"]},
            self.run["reconciliation"], len(self.run["persona_order"]))

    def test_unresolved_dissent_and_exact_quotes_survive_without_execution(self):
        before = copy.deepcopy(self.run)
        brief = build_brief(self.run, self.ledger)
        self.assertEqual(brief["status"], "HOLD")
        self.assertIn("unresolved_conflicts", brief["hold_reasons"])
        self.assertEqual(brief["untrusted_discussion"]["structural_summary"]["unresolved_conflicts"], [["A", "B"]])
        self.assertEqual(brief["untrusted_discussion"]["quotes"][0]["statement"], self.run["events"][0]["statement"])
        self.assertFalse(brief["execution_authorized"])
        self.assertTrue(brief["semantic_review_required"])
        self.assertEqual(self.run, before)

    def test_saved_gate_cannot_hide_fallback_and_consensus_is_not_truth(self):
        self.run["reconciliation"][0]["votes"]["A|B"]["p2"] = self.record("A")
        self.resummarize()
        good = build_brief(self.run, self.ledger)
        self.assertEqual(good["status"], "REVIEW_REQUIRED")
        self.assertTrue(good["semantic_review_required"])
        self.run["events"][0]["utterance_origin"] = "template_fallback"
        self.run["metrics"] = {"hard_gate_pass": True}
        brief = build_brief(self.run, self.ledger)
        self.assertEqual(brief["status"], "HOLD")
        self.assertFalse(brief["verification"]["recomputed_metrics"]["hard_gate_pass"])
        self.assertEqual(brief["untrusted_discussion"]["structural_summary"]["consensus"], ["A"])

    def test_invalid_ids_speakers_votes_summary_and_snapshot_are_rejected(self):
        mutations = [
            lambda r: r["events"][0].update(code="UNKNOWN"),
            lambda r: r["events"][0].update(data_ids=["D999"]),
            lambda r: r["events"][0].update(statement="知らない根拠[D999]を使います。"),
            lambda r: r["events"][0].update(persona_id="stranger"),
            lambda r: r["events"][1].update(claim_id="C1"),
            lambda r: r["events"][0].update(target_claim_id="C2"),
            lambda r: r["reconciliation"][0]["votes"]["A|B"].pop("p2"),
            lambda r: r["reconciliation"][0]["votes"]["A|B"]["p2"].update(data_ids=["D999"]),
            lambda r: r["summary"].update(consensus=["A"]),
            lambda r: r["ledger_snapshot"]["data"][0].update(text="改変"),
            lambda r: r.pop("runtime"),
            lambda r: r["runtime"].clear(),
            lambda r: r["reconciliation"][0]["votes"]["A|B"]["p2"].update(
                changed_from_previous=True, change_reason="変更の理由は[D999]です。"),
        ]
        for mutate in mutations:
            run = copy.deepcopy(self.run)
            mutate(run)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                build_brief(run, self.ledger)

    def test_public_quantity_disagreement_holds_even_when_saved_discussion_gate_passes(self):
        self.run['reconciliation'][0]['votes']['A|B']['p2']=self.record('A')
        self.resummarize()
        before=build_brief(self.run,self.ledger)
        self.assertEqual(before['status'],'REVIEW_REQUIRED')
        event=self.run['events'][0]
        event['statement']+='合計の発行枚数は1枚です。'
        event['utterance']+='合計2枚を想定します。'
        original=copy.deepcopy(self.run)
        brief=build_brief(self.run,self.ledger)
        self.assertEqual(self.run,original)
        self.assertTrue(brief['verification']['recomputed_metrics']['hard_gate_pass'])
        self.assertEqual(brief['status'],'HOLD')
        self.assertIn('public_arithmetic_or_quantity_inconsistency',brief['hold_reasons'])
        issue=brief['verification']['public_consistency_diagnostics'][0]
        self.assertEqual(issue['reference'],'C1')
        self.assertEqual(issue['issues'][0]['kind'],'supported_total_disagrees')
        self.assertFalse(brief['execution_authorized'])

    def test_both_and_abstention_are_preserved_not_converted_into_agreement(self):
        votes = self.run["reconciliation"][0]["votes"]["A|B"]
        votes["p1"] = self.record("BOTH")
        votes["p2"] = self.record("ABSTAIN")
        self.resummarize()
        brief = build_brief(self.run, self.ledger)
        self.assertEqual(brief["status"], "HOLD")
        self.assertEqual([q["code_or_choice"] for q in brief["untrusted_discussion"]["quotes"][-2:]],
                         ["BOTH", "ABSTAIN"])
        self.assertFalse(brief["untrusted_discussion"]["structural_summary"]["consensus"])

    def test_uncited_material_and_unspoken_options_remain_visible_without_scoring_them(self):
        previous = build_brief(self.run, self.ledger)
        self.ledger["data"].extend([
            {"id": "D02", "text": "資料上の追加の条件。"},
            {"id": "D03", "text": "予約の承認は未取得。"}])
        self.ledger["claim_catalog"][1]["supported_by"].append("D02")
        self.ledger["claim_catalog"].append({"code": "C", "label": "予約の承認を保留する",
                                             "supported_by": ["D03"], "contradicts": []})
        vote = self.run["reconciliation"][0]["votes"]["A|B"]["p2"]
        vote.update(data_ids=["D02"], statement="追加の条件を理由にBを検討します。[D02]")
        self.run["ledger_snapshot"] = copy.deepcopy(self.ledger)
        brief = build_brief(self.run, self.ledger)
        coverage = brief["untrusted_discussion"]["coverage"]
        self.assertTrue(coverage["diagnostic_only"])
        self.assertEqual(coverage["cited_data_ids"], ["D01", "D02"])
        self.assertEqual(coverage["uncited_source_data"], [self.ledger["data"][2]])
        self.assertEqual(coverage["citation_references"]["D02"], [{"reference": "R1:A|B:p2", "persona_id": "p2"}])
        self.assertEqual([c["code"] for c in coverage["unaddressed_catalog_claims"]], ["C"])
        self.assertEqual(brief["status"], previous["status"])
        self.assertEqual(brief["verification"], previous["verification"])

    def test_ambiguous_json_and_non_finite_numbers_are_rejected(self):
        for raw in (b'{"status":"HOLD","status":"OK"}', b'{"elapsed":NaN}',
                    b'{"elapsed":Infinity}', b'{"elapsed":1e999}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                strict_json(raw)

    def test_hash_binding_no_overwrite_and_input_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger, run, out = (root / name for name in ("ledger.json", "run.json", "brief.json"))
            cod.write_json(ledger, self.ledger)
            self.run["ledger_sha256"] = hashlib.sha256(ledger.read_bytes()).hexdigest()
            cod.write_json(run, self.run)
            original = (run.read_bytes(), ledger.read_bytes())
            exported = export_brief(run, ledger, out)
            self.assertEqual(exported["inputs"]["consistency_auditor_sha256"], hashlib.sha256(
                (Path(__file__).parent / "tools/verifier_consistency.py").read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                export_brief(run, ledger, out)
            with self.assertRaises(FileExistsError):
                export_brief(run, ledger, ledger)
            self.assertEqual((run.read_bytes(), ledger.read_bytes()), original)
            self.assertEqual(json.loads(out.read_text())["status"], "HOLD")
            self.run["ledger_sha256"] = "0" * 64
            cod.write_json(run, self.run)
            with self.assertRaises(ValueError):
                export_brief(run, ledger)
            partial = root / "run.partial.json"
            cod.write_json(partial, self.run)
            with self.assertRaises(ValueError):
                export_brief(partial, ledger)


if __name__ == "__main__":
    unittest.main()
