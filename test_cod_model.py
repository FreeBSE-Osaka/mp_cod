import tempfile
import unittest
import hashlib
from pathlib import Path
from types import SimpleNamespace
import json

import cod_model


class CodModelTest(unittest.TestCase):
    def test_profiles_are_distinct(self):
        domains = cod_model.load_domains()
        self.assertGreaterEqual(len(domains["software"]["personas"]), 4)
        for config in domains.values():
            ids = [persona["id"] for persona in config["personas"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(all(persona["utility"] and persona["loss"] for persona in config["personas"]))
            self.assertEqual(len({persona["utility"] for persona in config["personas"]}), len(ids))

    def test_software_holdout_ledger_matches_software_personas(self):
        ledger = cod_model.load_claim_ledger(
            Path("data/software_architecture_holdout/claim_ledger.json")
        )
        persona_ids = {persona["id"] for persona in cod_model.load_domains()["software"]["personas"]}
        self.assertEqual(set(ledger["role_preferences"]), persona_ids)
        self.assertTrue(ledger["fixture"])

    def test_general_holdout_ledger_matches_general_personas(self):
        ledger = cod_model.load_claim_ledger(
            Path("data/general_experiment_holdout/claim_ledger.json")
        )
        persona_ids = {persona["id"] for persona in cod_model.load_domains()["general"]["personas"]}
        self.assertEqual(set(ledger["role_preferences"]), persona_ids)
        self.assertTrue(ledger["fixture"])
        self.assertTrue(cod_model.is_mechanical_utterance("この点を今後の判断の軸にしたいです。"))
        self.assertFalse(cod_model.is_mechanical_utterance("8週間の期限を考えるとSwift先行が現実的です。"))

    def test_identical_blind_answers_trigger_collapse_proxy(self):
        answer = {
            "stance": "主案",
            "thesis": "同じ結論です",
            "recommendation": "同じ実装を採用する",
            "reasons": ["同じ根拠"],
        }
        metrics = cod_model.debate_metrics({"a": answer, "b": answer}, {})
        self.assertTrue(metrics["premature_consensus_proxy"])
        self.assertEqual(metrics["near_duplicate_pairs"], 1)

    def test_split_keeps_latest_examples_as_holdout(self):
        examples = [{"id": number} for number in range(10)]
        split = cod_model.split_examples(examples)
        self.assertEqual([row["id"] for row in split["train"]], list(range(8)))
        self.assertEqual(split["valid"], [{"id": 8}])
        self.assertEqual(split["test"], [{"id": 9}])

    def test_failed_verifier_replaces_moderator_answer(self):
        moderator = {"final_answer": "9/25 = 0.36"}
        verifier = {"verdict": "fail", "corrected_final_answer": "8/25 = 0.32"}
        self.assertEqual(cod_model.effective_answer(moderator, verifier), "8/25 = 0.32")

    def test_calculator_uses_exact_fractions(self):
        self.assertEqual(cod_model.calculate("16/50"), "8/25 = 0.32")
        with self.assertRaises(ValueError):
            cod_model.calculate("__import__('os').system('id')")

    def test_atomic_json_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            cod_model.write_json(path, {"ok": True})
            self.assertIn('"ok": true', path.read_text())

    def test_export_requires_explicit_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            runs.mkdir()
            run = {
                "schema_version": 1,
                "run_id": "approved-run",
                "domain": "software",
                "training_review": {"status": "approved", "approved_calls": [0]},
                "calls": [
                    {
                        "persona_id": "iterative_generalist",
                        "system": "system",
                        "user": "topic",
                        "assistant": json.dumps({"answer": "ok"}),
                    }
                ],
            }
            cod_model.write_json(runs / "run.json", run)
            output = root / "sft"
            result = cod_model.export_sft(SimpleNamespace(runs=str(runs), out=str(output), domain=None))
            self.assertEqual(result, 0)
            self.assertIn('"role": "assistant"', (output / "software/iterative_generalist/train.jsonl").read_text())

    def test_event_dialogue_export_requires_approved_untampered_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger_path = Path("data/software_architecture_holdout/claim_ledger.json").resolve()
            run_path = root / "event.json"
            run = {
                "schema_version": 2,
                "model": "test-model",
                "ledger": str(ledger_path),
                "ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
                "domain": "software",
                "independent": {
                    "iterative_generalist": {"raw": "raw", "rejected": []}
                },
                "events": [
                    {
                        "claim_id": "C01",
                        "persona_id": "iterative_generalist",
                        "action": "new",
                        "target_claim_id": None,
                        "code": "SWIFT_MLX_VERTICAL_SLICE_FIRST",
                        "data_ids": ["D01"],
                        "origin": "model",
                        "statement": "SwiftとMLXを先に試します。根拠は[D01]です。",
                        "statement_origin": "model",
                        "utterance": "8週間の期限があるため、まずSwiftとMLXで動く形を確かめます。",
                        "utterance_origin": "model",
                    }
                ],
                "reconciliation": [],
            }
            cod_model.write_json(run_path, run)
            self.assertEqual(
                cod_model.mark_event_run(
                    SimpleNamespace(
                        run=str(run_path),
                        status="approved",
                        reviewer="tester",
                        note="fixture verified",
                    )
                ),
                0,
            )
            out = root / "dialogue"
            self.assertEqual(
                cod_model.export_dialogue_sft(
                    SimpleNamespace(runs=[str(run_path)], out=str(out), min_per_persona=1)
                ),
                0,
            )
            manifest = json.loads(
                (out / "software/iterative_generalist/manifest.json").read_text()
            )
            self.assertEqual(manifest["total"], 1)
            self.assertFalse(manifest["ready_for_training"])
            tampered = json.loads(run_path.read_text())
            tampered["events"][0]["utterance"] = "承認後に変更しました。"
            cod_model.write_json(run_path, tampered)
            with self.assertRaisesRegex(ValueError, "review後"):
                cod_model.export_dialogue_sft(
                    SimpleNamespace(runs=[str(run_path)], out=str(out), min_per_persona=1)
                )

    def test_curriculum_is_exact_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            args = SimpleNamespace(out=str(first), count=30, seed=7)
            self.assertEqual(cod_model.make_auditor_curriculum(args), 0)
            self.assertEqual(
                cod_model.make_auditor_curriculum(SimpleNamespace(out=str(second), count=30, seed=7)),
                0,
            )
            self.assertEqual((first / "train.jsonl").read_bytes(), (second / "train.jsonl").read_bytes())
            manifest = json.loads((first / "manifest.json").read_text())
            self.assertEqual(manifest["counts"], {"train": 24, "valid": 3, "test": 3})
            row = json.loads((first / "benchmark.jsonl").read_text().splitlines()[0])
            self.assertIn("expected_fraction", row)
            evidence_row = json.loads((first / "evidence_benchmark.jsonl").read_text().splitlines()[0])
            self.assertIn("決定論的監査ツール結果", evidence_row["user"])
            self.assertNotIn("決定論的監査ツール結果", row["user"])

    def test_auditor_score_requires_json_fraction_and_verdict(self):
        output = json.dumps(
            {
                "stance": "対案",
                "thesis": "仮説は誤り。",
                "recommendation": "16/50=8/25（0.32）",
                "reasons": ["再計算"],
                "assumptions": ["一様分布"],
                "risks": [],
                "confidence": 100,
            }
        )
        score = cod_model.score_auditor_output(output, "8/25", False)
        self.assertTrue(score["all_correct"])
        synonym = json.dumps({"thesis": "計算値と一致しない。", "recommendation": "仮説を棄却し8/25を採用。"})
        self.assertTrue(cod_model.score_auditor_output(synonym, "8/25", False)["semantic_correct"])
        self.assertFalse(
            cod_model.score_auditor_output(
                json.dumps({"analysis": "仮説は誤り。答えは8/25。"}), "8/25", False
            )["contract_valid"]
        )
        self.assertFalse(cod_model.score_auditor_output("8/25", "8/25", False)["all_correct"])

    def test_weight_audit_input_always_contains_tool_evidence(self):
        case = cod_model.finite_audit_input(100, 2, 3, cod_model.Fraction(1, 3))
        self.assertIn("決定論的監査ツール結果", case["user"])
        self.assertEqual(case["effective_final_answer"], "8/25 = 0.32")
        self.assertFalse(case["hypothesis_correct"])

    def test_contract_normalizer_only_trims_valid_arrays(self):
        value = {
            "stance": "対案",
            "thesis": "仮説は誤り。",
            "recommendation": "8/25を採用。",
            "reasons": ["a", "b", "c"],
            "assumptions": ["a"],
            "risks": ["a"],
            "confidence": 101,
        }
        normalized = cod_model.normalize_auditor_contract(value)
        self.assertEqual(normalized["reasons"], ["a", "b"])
        self.assertEqual(normalized["confidence"], 100)
        self.assertIsNone(cod_model.normalize_auditor_contract({**value, "stance": ["対案"]}))

    def test_unsupported_claim_is_disqualified(self):
        ledger = {
            "claim_catalog": [
                {"code": "MAIN", "label": "main", "supported_by": ["D01"], "contradicts": []}
            ]
        }
        valid, reason = cod_model.validate_coded_claim(
            {"code": "MAIN", "data_ids": ["D99"], "confidence": 80}, ledger
        )
        self.assertIsNone(valid)
        self.assertIn("unsupported", reason)

    def test_named_confidence_is_normalized_after_evidence_validation(self):
        ledger = {
            "claim_catalog": [
                {"code": "MAIN", "label": "main", "supported_by": ["D01"], "contradicts": []}
            ]
        }
        valid, reason = cod_model.validate_coded_claim(
            {"code": "MAIN", "data_ids": ["D01"], "confidence": "High"}, ledger
        )
        self.assertIsNone(reason)
        self.assertEqual(valid["confidence"], 85)

    def test_public_statement_requires_a_selected_data_id(self):
        statement, reason = cod_model.validate_public_statement("主経路を採ります。根拠は[D01]です。", ["D01"])
        self.assertIsNone(reason)
        self.assertIn("D01", statement)
        statement, reason = cod_model.validate_public_statement("根拠は[D99]です。", ["D01"])
        self.assertIsNone(statement)
        self.assertIn("unselected", reason)
        sanitized = cod_model.sanitize_model_statement(
            "北東転向外れを独立シナリオとして残す。根拠は[D08]です。",
            ["D09"],
        )
        self.assertEqual(sanitized, "北東転向外れを独立シナリオとして残す。根拠は[D09]。")
        utterance, reason = cod_model.validate_dialogue_utterance(
            "いえ、その見方では大陸側の高気圧の影響を見落としてしまいます。"
        )
        self.assertIsNone(reason)
        self.assertTrue(utterance.startswith("いえ"))
        utterance, reason = cod_model.validate_dialogue_utterance("これはまだ文末がない発言です")
        self.assertIsNone(utterance)
        self.assertIn("complete sentence", reason)
        utterance, reason = cod_model.validate_dialogue_utterance(
            "私は北東転向の可能性を残します。根拠は[D09]です。"
        )
        self.assertIsNone(utterance)
        self.assertIn("metadata", reason)
        utterance, reason = cod_model.validate_dialogue_move(
            "私もその見方に賛成します。", "revise"
        )
        self.assertIsNone(utterance)
        self.assertIn("revise", reason)
        utterance, reason = cod_model.validate_dialogue_move(
            "確かにその影響を見落としていました。見方を改めます。", "revise"
        )
        self.assertIsNone(reason)
        self.assertIn("見方を改めます", utterance)

    def test_adapter_map_rejects_unknown_personas(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}")
            (adapter / "adapters.safetensors").write_bytes(b"weights")
            mapping = Path(directory) / "map.json"
            mapping.write_text(
                json.dumps({"schema_version": 1, "adapters": {"unknown": str(adapter)}})
            )
            with self.assertRaisesRegex(ValueError, "unknown personas"):
                cod_model.load_adapter_map(str(mapping), {"known"})

    def test_objection_speaks_before_new_topic(self):
        ledger = {
            "claim_catalog": [
                {"code": "MAIN", "label": "main", "supported_by": ["D01"], "contradicts": ["OUT"], "priority": 5},
                {"code": "OUT", "label": "out", "supported_by": ["D02"], "contradicts": ["MAIN"], "priority": 3},
                {"code": "NEW", "label": "new", "supported_by": ["D03"], "contradicts": [], "priority": 4},
            ]
        }
        claims = {
            "a": [{"code": "MAIN", "data_ids": ["D01"], "confidence": 90}],
            "b": [{"code": "OUT", "data_ids": ["D02"], "confidence": 60}],
            "c": [{"code": "NEW", "data_ids": ["D03"], "confidence": 99}],
        }
        events = cod_model.schedule_claim_events(claims, ledger, 3)
        self.assertEqual(events[0]["code"], "MAIN")
        self.assertEqual(events[1]["action"], "object")
        self.assertEqual(events[1]["code"], "OUT")
        self.assertEqual(events[1]["statement"], "out。根拠は[D02]。")
        self.assertEqual(events[1]["statement_origin"], "label_fallback")
        self.assertEqual(events[1]["utterance"], "out。")
        self.assertEqual(events[1]["utterance_origin"], "statement_fallback")

    def test_agreement_can_target_older_claim(self):
        ledger = {
            "claim_catalog": [
                {"code": "MAIN", "label": "main", "supported_by": ["D01"], "contradicts": [], "priority": 5},
                {"code": "SIDE", "label": "side", "supported_by": ["D02"], "contradicts": [], "priority": 4},
            ]
        }
        events = [
            {"claim_id": "C01", "code": "MAIN"},
            {"claim_id": "C02", "code": "SIDE"},
        ]
        action, target = cod_model.claim_reaction(events, {"code": "MAIN"}, {item["code"]: item for item in ledger["claim_catalog"]})
        self.assertEqual((action, target), ("agree_extend", "C01"))

    def test_first_reconciliation_compares_against_initial_position(self):
        claims = [{"code": "LOW"}, {"code": "OTHER"}]
        self.assertEqual(cod_model.prior_pair_choice(claims, ("LOW", "HIGH"), {}), "LOW")
        self.assertEqual(
            cod_model.prior_pair_choice(claims, ("LOW", "HIGH"), {"choice": "HIGH"}),
            "HIGH",
        )
        self.assertFalse(
            cod_model.reaction_is_aligned(
                "ただ、その見方では北東転向を低位に扱う点が抜けています。",
                "北東転向を独立シナリオとして残す",
                "北東転向を低位に扱う",
                "object",
            )
        )

    def test_three_of_four_resolves_conflict(self):
        catalog = {
            "A": {"contradicts": ["B"]},
            "B": {"contradicts": ["A"]},
            "C": {"contradicts": []},
        }
        events = [
            {"code": "A", "persona_id": "p1"},
            {"code": "B", "persona_id": "p2"},
            {"code": "C", "persona_id": "p3"},
            {"code": "C", "persona_id": "p4"},
        ]
        reconciliation = [
            {
                "votes": {
                    "A|B": {"p1": "A", "p2": "A", "p3": "A", "p4": "B"}
                }
            }
        ]
        summary = cod_model.synthesize_event_summary(events, catalog, reconciliation, 4)
        self.assertEqual(summary["resolved_conflicts"], {"A|B": "A"})
        self.assertIn("A", summary["consensus"])
        self.assertIn("C", summary["consensus"])
        self.assertTrue(
            cod_model.reconciliation_has_supermajority(
                {"A|B": {"A": 3, "B": 1}}, [("A", "B")], 4
            )
        )
        self.assertFalse(
            cod_model.reconciliation_has_supermajority(
                {"A|B": {"A": 2, "B": 2}}, [("A", "B")], 4
            )
        )

    def test_event_metrics_keep_model_and_fallback_separate(self):
        run = {
            "independent": {
                "p1": {"raw": "raw1", "rejected": []},
                "p2": {"raw": "raw2", "rejected": []},
            },
            "events": [
                {
                    "code": "A",
                    "action": "new",
                    "origin": "model",
                    "statement": "Aを採ります。根拠は[D01]です。",
                    "statement_origin": "model",
                    "utterance": "私はAの見方を採ります。",
                    "utterance_origin": "model_dialogue_v2",
                },
                {
                    "code": "B",
                    "action": "object",
                    "origin": "validated_fallback",
                    "statement": "Bです。根拠は[D02]です。",
                    "statement_origin": "label_fallback",
                    "utterance": "ただ、Bの可能性も残ります。",
                    "utterance_origin": "model_reaction",
                    "reaction_raw": '{"utterance":"ただ、Bの可能性も残ります。"}',
                },
            ],
            "reconciliation": [
                {
                    "votes": {
                        "A|B": {
                            "p1": {
                                "choice": "A",
                                "choice_origin": "model_json",
                                "statement_origin": "model_repair",
                                "utterance": "私はAの見方を採ります。",
                                "utterance_origin": "model_repair",
                                "raw": '{"choice":"A"}',
                            },
                            "p2": {
                                "choice": "B",
                                "choice_origin": "model_json",
                                "statement_origin": "model",
                                "utterance": "私はBを支持します。",
                                "utterance_origin": "model",
                                "raw": '{"choice":"B"}',
                            },
                        }
                    }
                }
            ],
        }
        metrics = cod_model.event_run_metrics(run)
        self.assertEqual(metrics["model_claims"], 1)
        self.assertEqual(metrics["validated_fallbacks"], 1)
        self.assertEqual(metrics["fallback_rate"], 0.5)
        self.assertEqual(metrics["model_statement_rate"], 0.75)
        self.assertEqual(metrics["model_utterance_rate"], 1.0)
        self.assertEqual(metrics["dialogue_v2_utterances"], 1)
        self.assertEqual(metrics["mechanical_utterance_rate"], 0.0)
        self.assertEqual(metrics["dialogue_near_duplicate_pairs"], 0)
        self.assertTrue(metrics["hard_gate_pass"])

    def test_changed_reconciliation_vote_requires_model_change_reason(self):
        run = {
            "independent": {"p1": {"raw": "raw", "rejected": []}},
            "events": [
                {
                    "code": "A",
                    "action": "object",
                    "origin": "model",
                    "statement": "Aを採ります。根拠は[D01]です。",
                    "statement_origin": "model",
                    "utterance": "私はAへ見方を改めます。",
                    "utterance_origin": "model",
                }
            ],
            "reconciliation": [
                {
                    "votes": {
                        "A|B": {
                            "p1": {
                                "choice": "A",
                                "choice_origin": "model_json",
                                "statement_origin": "model",
                                "utterance": "確かに、Aへ見方を改めます。",
                                "utterance_origin": "model",
                                "changed_from_previous": True,
                                "change_reason_origin": "label_fallback",
                                "raw": '{"choice":"A"}',
                            }
                        }
                    }
                }
            ],
        }
        metrics = cod_model.event_run_metrics(run)
        self.assertEqual(metrics["model_change_reason_rate"], 0.0)
        self.assertFalse(metrics["hard_gate_pass"])

    def test_bounded_rsi_requires_holdout_transfer_and_never_promotes(self):
        parent = {
            "shadow_score": 80.0,
            "fallback_rate": 0.2,
            "model_statement_rate": 0.7,
            "near_duplicate_rate": 0.1,
            "hard_gate_pass": False,
        }
        candidate = {
            "shadow_score": 82.0,
            "fallback_rate": 0.1,
            "model_statement_rate": 0.9,
            "near_duplicate_rate": 0.05,
            "hard_gate_pass": True,
        }
        decision = cod_model.decide_rsi_shadow(
            parent,
            candidate,
            {**parent, "shadow_score": 78.0},
            {**candidate, "shadow_score": 80.0},
            round_no=1,
            max_rounds=3,
            holdout_distinct=True,
        )
        self.assertEqual(decision["status"], "research_shadow_candidate")
        self.assertTrue(decision["continue_allowed"])
        self.assertFalse(decision["promotion_allowed"])
        stopped = cod_model.decide_rsi_shadow(
            parent,
            candidate,
            {**parent, "shadow_score": 78.0},
            {**candidate, "shadow_score": 78.5},
            round_no=1,
            max_rounds=3,
            holdout_distinct=True,
        )
        self.assertEqual(stopped["status"], "parent_retained")
        self.assertFalse(stopped["continue_allowed"])


if __name__ == "__main__":
    unittest.main()
