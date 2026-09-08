import copy
from pathlib import Path
import tempfile
import unittest
import json
import io
from contextlib import redirect_stdout
from unittest.mock import patch
from types import SimpleNamespace

import cod_model as cod
from tools.general_body_training import body_checks, cases, evaluate, rescore, valid


class GeneralDiscussionTest(unittest.TestCase):
    def test_percentage_units_and_signs_cannot_change(self):
        payload = {"claim": "電力費18%とCO2 11%の削減を優先する"}
        self.assertFalse(cod.dialogue_numbers_are_grounded("電力費18割とCO2 11割の削減を優先します。", payload))
        self.assertFalse(cod.dialogue_numbers_are_grounded("電力費-18%の削減を優先します。", payload))
        self.assertTrue(cod.dialogue_numbers_are_grounded("電力費１８％とCO2 １１パーセントの削減を優先します。", payload))
        self.assertTrue(cod.dialogue_numbers_are_grounded("費用を2割削減します。", {"claim": "費用を2割削減する"}))

    def test_adapter_isolation_requires_an_adapter_before_loading_mlx(self):
        with self.assertRaisesRegex(ValueError, "requires --adapter"):
            evaluate(SimpleNamespace(check_adapter_isolation=True, adapter=None))

    def test_qwen35_fresh_holdout_covers_all_eight_speakers(self):
        source = Path(__file__).parent / "data/general_body_qwen35_v1_holdout/curated.json"
        rows = cases(source, {"test"})
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row["speaker"] for row in rows}), 8)
        old = cases(Path(__file__).parent / "data/general_body_v5/curated.json", {"train", "valid", "test"})
        self.assertFalse({row["claim"] for row in rows} & {row["claim"] for row in old})
        for row in rows:
            with self.subTest(case=row["case"]):
                self.assertTrue(valid(body_checks(row["body"], row)))

    def test_saved_raw_rescore_keeps_source_immutable(self):
        curated = Path(__file__).parent / "data/general_body_v5/curated.json"
        case = cases(curated, {"valid"})[0]
        raw = json.dumps({"bodies": [{"id": "B01", "body": case["body"]}]})
        with tempfile.TemporaryDirectory() as directory:
            source, out = Path(directory) / "source.json", Path(directory) / "rescore.json"
            original = json.dumps({"results": [{**case, "raw": raw, "direct_valid": False}]})
            source.write_text(original)
            args = SimpleNamespace(curated=curated, input=source, out=out, legacy=None)
            with redirect_stdout(io.StringIO()):
                rescore(args)
            self.assertEqual(source.read_text(), original)
            self.assertEqual(json.loads(out.read_text())["summary"]["valid"]["direct_valid"], 1)
            with self.assertRaisesRegex(ValueError, "already exists"):
                rescore(args)

    def test_verb_plus_desu_is_not_a_polite_conjugation(self):
        for text in ("元の運用へ戻すです。", "仮の予定として伝えるです。", "状況を確認するです。",
                     "質問の時間も含めるです。", "仮の案内だけを出すです。"):
            self.assertFalse(cod.body_is_polite_sentence(text))
        for text in ("元の運用へ戻します。", "費用が高いです。", "無断では使わないです。"):
            self.assertTrue(cod.body_is_polite_sentence(text))

    def test_flexible_score_does_not_penalize_same_claim_agreement(self):
        event = {"code": "P", "origin": "model", "statement_origin": "model", "utterance_origin": "model_body_v2",
                 "statement": "試験導入を継続します。[D01]", "utterance": "試験導入を継続します。", "action": "propose"}
        run = {"discussion_style": "flexible", "events": [event], "independent": {"p1": {"raw": "model output"}}}
        first = cod.event_run_metrics(run)
        run["events"].append({**event, "action": "agree_extend"})
        same_opinion = cod.event_run_metrics(run)
        self.assertEqual(first["shadow_score"], same_opinion["shadow_score"])
        self.assertEqual(same_opinion["near_duplicate_pairs"], 0)
        self.assertEqual(same_opinion["dialogue_near_duplicate_pairs"], 0)

    def test_renderer_cannot_turn_a_plan_into_a_completed_action(self):
        self.assertFalse(cod.body_matches_claim("雨に備えて傘を持って出しました。", "雨に備えて傘を持って出る"))
        self.assertTrue(cod.body_matches_claim("雨に備えて傘を持って出ました。", "雨に備えて傘を持って出た"))
        body, reason = cod.normalize_renderer_body(
            "実証監査者にとって、折り畳み傘を持つことは重要です。",
            "荷物を減らすため折り畳み傘を持つ", speaker="実証監査者")
        self.assertIsNone(body)
        self.assertIn("speaker attribution", reason)

    def test_flexible_runtime_keeps_sparse_model_opinions_and_decodes_sides(self):
        roster = cod.load_domains()["general"]["personas"][:2]
        ledger = {"schema_version": 1, "topic": "架空の試験導入", "data": [{"id": "D01", "text": "試行条件を比較する。"}],
                  "claim_catalog": [{"code": code, "label": label, "kind": "proposal", "supported_by": ["D01"],
                                     "contradicts": ["Q" if code == "P" else "P"] if code in {"P", "Q"} else []}
                                    for code, label in (("P", "少人数で試験導入を続ける"), ("Q", "試験導入を止めて再検証する"), ("R", "確認項目を先に整理する"))],
                  "role_preferences": {p["id"]: ["P", "Q", "R"] for p in roster}}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "ledger.json"
            source.write_text(json.dumps(ledger))
            args = cod.parser().parse_args(["event-debate", "--domain", "general", "--backend", "ollama",
                                           "--no-renderer", "--ledger", str(source), "--out", str(Path(directory)/"runs"),
                                           "--reconcile-rounds", "1"])
            counter = 0
            def respond(**kwargs):
                nonlocal counter
                payload = json.loads(kwargs["user"])
                if "claim_catalog" in payload:
                    code = "P" if counter == 0 else "Q"
                    counter += 1
                    result = {"claims": [{"code": code, "data_ids": ["D01"], "confidence": 75,
                                          "statement": "試行条件を比較して進めます。根拠は[D01]です。"}]}
                else:
                    result = {"choice": "LEFT", "data_ids": ["D01"],
                              "statement": "試行条件を比較して選択します。根拠は[D01]です。", "change_reason": ""}
                return result, {"_raw_content": json.dumps(result)}
            with patch.object(cod, "ask_ollama", side_effect=respond), redirect_stdout(io.StringIO()):
                self.assertEqual(cod.run_event_debate(args), 0)
            run = json.loads(next((Path(directory)/"runs").glob("event_debate_*.json")).read_text())
            self.assertEqual(len(run["events"]), 2)
            self.assertTrue(all(e["origin"] == "model" for e in run["events"]))
            votes = run["reconciliation"][0]["votes"]["P|Q"]
            self.assertEqual([votes[p["id"]]["choice"] for p in roster], ["P", "Q"])
            self.assertTrue(all(v["choice_origin"] == "model_json" for v in votes.values()))

    def test_polite_conjugations_do_not_add_a_judgment_catchphrase(self):
        for claim, expected in (("雨に備えて傘を持って出る", "雨に備えて傘を持って出ます。"),
                                ("荷物を減らすため傘を置く", "荷物を減らすため傘を置きます。"),
                                ("その選択にも十分な理由がある", "その選択にも十分な理由があります。"),
                                ("確認前に機材を貸し出さない", "確認前に機材を貸し出しません。")):
            with self.subTest(claim=claim):
                self.assertEqual(cod.sanitize_body_politeness(claim + "。", claim), expected)

    def test_equivalent_wording_keeps_numeric_and_condition_checks(self):
        case = {"claim": "見積額が8000円を超える場合は作業前に再確認する", "required": ["8000円を超える場合", "作業前", "再確認"]}
        self.assertTrue(valid(body_checks("見積額が8000円を超える場合は作業前に再度確認します。", case)))
        self.assertFalse(valid(body_checks("見積額が8000円未満の場合は作業前に再度確認します。", case)))
        case = {"claim": "予約枠は午前4件と午後3件で当日枠は設けない", "required": ["午前4件", "午後3件", "当日枠は設けません"]}
        self.assertTrue(valid(body_checks("予約枠は午前4件と午後3件で当日の枠は設けません。", case)))
        self.assertFalse(valid(body_checks("予約枠は午前3件と午後4件で当日の枠は設けません。", case)))

    def test_body_rejects_split_fragments_and_lost_negative_condition(self):
        self.assertFalse(cod.body_matches_claim(
            "返却された機材は動作確認が済ましませんまで次の利用者へ貸し出します。",
            "返却された機材は動作確認が済むまで次の利用者へ貸し出さない"))
        self.assertFalse(cod.body_matches_claim(
            "貸出期間は9日間で、延長は次の予約がない場合だけ認めます。",
            "貸出期間は最長9日で延長は次の予約がない場合だけ認める"))
        self.assertFalse(cod.body_matches_claim(
            "試行期間は6週間で確認します。夜間利用者数と残業時間です。",
            "試行期間は6週間で確認するのは夜間利用者数と残業時間"))
        label = "延長開館は金曜日だけに限定し利用者が増えなければ元に戻す"
        self.assertFalse(cod.body_matches_claim(
            "延長開館は金曜日だけに限定し、利用者数に応じて元に戻します。", label))
        self.assertTrue(cod.body_matches_claim(
            "延長開館は金曜日だけに限定し、利用者が増えない場合は元に戻します。", label))

    def test_curated_targets_and_topic_split(self):
        rows = cases(Path(__file__).parent / "data/general_body_v5/curated.json", {"train", "valid", "test"})
        self.assertEqual(len(rows), len({r["claim"] for r in rows}))
        for row in rows:
            with self.subTest(case=row["case"]):
                self.assertTrue(valid(body_checks(row["body"], row)))
        self.assertGreaterEqual(len(cod.load_domains()["general"]["personas"]), 8)

    def test_critique_without_forced_alternative_and_plain_agreement(self):
        label = "利用者が少ないという理由だけでは必要性が低いとは言えない"
        body = "利用者が少ないという理由だけでは必要性が低いとは言えません。"
        critique = cod.compose_dialogue_body(body, label, "object", flexible=True)
        self.assertIsNotNone(critique)
        self.assertNotIn("代案", critique)
        self.assertIsNone(cod.validate_dialogue_move(critique, "object", label)[0])
        agreement = cod.compose_dialogue_body("試験導入を継続します。", "試験導入を継続する", "agree", flexible=True)
        self.assertIn("賛成", agreement)
        self.assertNotIn("加えて", agreement)
        self.assertIn("賛同と改善", cod.review_schema(["other"], flexible=True)["properties"]["rebuttal_type"]["enum"])

    def test_shared_support_critique_and_improvement_are_not_false_conflicts(self):
        ledger = {"schema_version": 1, "data": [{"id": "D01"}], "claim_catalog": [
            {"code": "P", "label": "試験導入を継続する", "supported_by": ["D01"], "kind": "proposal"},
            {"code": "C", "label": "平均だけでは一部の待ち時間を見落とす", "supported_by": ["D01"], "kind": "critique", "challenges": ["P"]},
            {"code": "I", "label": "対面の窓口も残して試験導入を続ける", "supported_by": ["D01"], "kind": "improvement", "extends": ["P"]},
        ]}
        catalog = {c["code"]: c for c in ledger["claim_catalog"]}
        first = {"claim_id": "C01", "code": "P", "persona_id": "p1"}
        action, target = cod.claim_reaction([first], {"code": "P"}, catalog)
        self.assertEqual((action, target), ("agree_extend", "C01"))
        self.assertEqual(cod.flexible_event_move({"code": "P", "action": action, "persona_id": "p1"}, first, catalog), "elaborate")
        self.assertEqual(cod.flexible_event_move({"code": "P", "action": action, "persona_id": "p2"}, first, catalog), "agree")
        action, target = cod.claim_reaction([first], {"code": "C"}, catalog)
        self.assertEqual((action, target), ("object", "C01"))
        self.assertEqual(cod.flexible_event_move({"code": "C", "action": action}, first, catalog), "object")
        action, target = cod.claim_reaction([first], {"code": "I"}, catalog)
        self.assertEqual(cod.flexible_event_move({"code": "I", "action": action}, first, catalog), "improve")
        summary = cod.synthesize_event_summary([first, {**first, "persona_id": "p2"},
                                               {"code": "C", "persona_id": "p3"}], catalog, [], 3)
        self.assertIn("P", summary["consensus"])
        self.assertEqual(summary["unresolved_conflicts"], [])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            path.write_text(json.dumps(ledger))
            cod.load_claim_ledger(path)
            bad = copy.deepcopy(ledger)
            bad["claim_catalog"][2]["extends"] = ["UNKNOWN"]
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "extends"):
                cod.load_claim_ledger(path)


if __name__ == "__main__":
    unittest.main()
