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
        self.assertIn(
            "pragmatic_operator",
            {persona["id"] for persona in domains["general"]["personas"]},
        )
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
        self.assertLessEqual(set(ledger["role_preferences"]), persona_ids)
        self.assertGreaterEqual(len(ledger["role_preferences"]), 3)
        self.assertTrue(ledger["fixture"])
        self.assertTrue(cod_model.is_mechanical_utterance("この点を今後の判断の軸にしたいです。"))
        self.assertTrue(cod_model.is_mechanical_utterance("現時点では、この案が有力です。"))
        self.assertTrue(cod_model.is_mechanical_utterance("全vendorへ展開することが判断です。"))
        self.assertFalse(cod_model.is_mechanical_utterance("8週間の期限を考えるとSwift先行が現実的です。"))

    def test_collection_ledgers_match_domain_personas(self):
        domains = cod_model.load_domains()
        for path, domain in (
            ("data/weather_update_holdout/claim_ledger.json", "weather"),
            ("data/software_offline_sync_holdout/claim_ledger.json", "software"),
            ("data/weather_steering_vs_impact_holdout/claim_ledger.json", "weather"),
            ("data/software_model_cache_holdout/claim_ledger.json", "software"),
            ("data/general_rollout_holdout/claim_ledger.json", "general"),
            ("data/general_alert_threshold_holdout/claim_ledger.json", "general"),
            ("data/general_inventory_policy_holdout/claim_ledger.json", "general"),
            ("data/general_energy_peak_holdout/claim_ledger.json", "general"),
            ("data/general_refund_automation_holdout/claim_ledger.json", "general"),
            ("data/general_quality_inspection_holdout/claim_ledger.json", "general"),
            ("data/general_subscription_pricing_holdout/claim_ledger.json", "general"),
            ("data/general_route_optimizer_holdout/claim_ledger.json", "general"),
            ("data/general_predictive_maintenance_holdout/claim_ledger.json", "general"),
            ("data/general_lightweight_packaging_holdout/claim_ledger.json", "general"),
            ("data/general_irrigation_policy_holdout/claim_ledger.json", "general"),
            ("data/general_weight_transfer_holdout/claim_ledger.json", "general"),
            ("data/general_conversation_v2_holdout/claim_ledger.json", "general"),
            ("data/general_v2_backup_restore_holdout/claim_ledger.json", "general"),
            ("data/general_v2_reply_assist_holdout/claim_ledger.json", "general"),
            ("data/general_v2_cold_storage_holdout/claim_ledger.json", "general"),
            ("data/general_v2_invoice_ocr_holdout/claim_ledger.json", "general"),
            ("data/general_v2_autoscaling_holdout/claim_ledger.json", "general"),
            ("data/general_v2_inventory_replenishment_holdout/claim_ledger.json", "general"),
            ("data/general_v2_email_triage_holdout/claim_ledger.json", "general"),
            ("data/general_v2_weight_transfer_holdout/claim_ledger.json", "general"),
        ):
            ledger = cod_model.load_claim_ledger(Path(path))
            persona_ids = {persona["id"] for persona in domains[domain]["personas"]}
            active_ids = set(ledger["role_preferences"])
            self.assertLessEqual(active_ids, persona_ids)
            self.assertGreaterEqual(len(active_ids), 3)
            if domain != "general":
                self.assertEqual(active_ids, persona_ids)
            self.assertTrue(ledger["fixture"])

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
        statement, reason = cod_model.validate_public_statement(
            "変えていない時は空文字です。根拠は[D01]です。", ["D01"]
        )
        self.assertIsNone(statement)
        self.assertIn("internal protocol", reason)
        sanitized = cod_model.sanitize_model_statement(
            "北東転向外れを独立シナリオとして残す。根拠は[D08]です。",
            ["D09"],
        )
        self.assertEqual(sanitized, "北東転向外れを独立シナリオとして残す。根拠は[D09]です。")
        sanitized = cod_model.sanitize_model_statement(
            "複数断層の評価は[D02]を統合した[ D13]で確認できます。",
            ["D13"],
        )
        self.assertNotIn("D02", sanitized)
        self.assertIn("D13", sanitized)
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
        proposal, reason = cod_model.validate_dialogue_move(
            "私の見立てでは、対象を絞ったpilotから始めるのが妥当です。", "propose"
        )
        self.assertIsNone(reason)
        proposal, reason = cod_model.validate_dialogue_move(
            "その案には賛成です。対象を絞ったpilotから始めます。", "propose"
        )
        self.assertIsNone(proposal)
        self.assertIn("must not pretend", reason)
        objection, reason = cod_model.validate_dialogue_move(
            "ただ、その案では処理上限を超えます。", "object"
        )
        self.assertIsNone(objection)
        self.assertIn("alternative", reason)
        objection, reason = cod_model.validate_dialogue_move(
            "その案には懸念があります。代わりに対象を絞ったpilotを先に試すべきです。",
            "object",
        )
        self.assertIsNone(reason)
        self.assertIn("代わりに", objection)
        objection, reason = cod_model.validate_dialogue_move(
            "全展開では予算超過を避けるには不十分です。batch job限定pilotを代案にします。",
            "object",
        )
        self.assertIsNone(reason)
        agreement, reason = cod_model.validate_dialogue_move(
            "予算超過を重大視する提案には賛同します。", "agree"
        )
        self.assertIsNone(reason)
        self.assertIn("賛同", agreement)
        objection = cod_model.sanitize_dialogue_move(
            "私の見立てでは、全networkへ展開する案が有力です。",
            "object",
        )
        self.assertTrue(objection.startswith("その案には懸念があります。代わりに、"))
        self.assertIsNone(
            cod_model.sanitize_dialogue_move(
                "その案には賛成です。代案として対象を絞ったpilotを試します。",
                "object",
            )
        )
        rewritten = cod_model.sanitize_dialogue_move(
            "その案には、『対象を絞ったpilotを行う』という観点も外せません。",
            "object",
        )
        self.assertNotIn("代わりに、その案には", rewritten)
        self.assertIsNone(cod_model.sanitize_dialogue_move("その案には賛成です。", "agree"))
        fallback_label = "利用不能率9%または受電600kW超を停止条件にする"
        fallback_statement = f"{fallback_label}と判断します。根拠は[D08]です。"
        composed, origin = cod_model.compose_dialogue_fallback(
            fallback_statement,
            fallback_label,
            "agree",
        )
        self.assertEqual(origin, "composed_statement_fallback")
        self.assertTrue(composed.startswith("その案には賛成です。加えて、"))
        self.assertIn(fallback_label, composed)
        self.assertNotIn("D08", composed)
        objection, origin = cod_model.compose_dialogue_fallback(
            fallback_statement,
            fallback_label,
            "object",
            1,
        )
        self.assertEqual(origin, "composed_statement_fallback")
        self.assertTrue(objection.startswith("その結論には異議があります。代案として、"))
        selected, origin = cod_model.compose_dialogue_fallback(
            f"{fallback_label}を採ります。根拠は[D08]です。",
            fallback_label,
            "maintain",
        )
        self.assertIn(f"『{fallback_label}』を採ります。", selected)
        self.assertNotIn(f"{fallback_label}を採ります。", selected)
        for move in ("object", "agree", "maintain", "revise"):
            examples = [
                cod_model.dialogue_move_example("対象を絞ったpilotを行う", move, index)
                for index in range(len(cod_model.MOVE_UTTERANCE_TEMPLATES[move]))
            ]
            self.assertEqual(len(set(examples)), len(examples))
            self.assertTrue(all("確かに" not in example for example in examples))
        self.assertEqual(
            cod_model.sanitize_dialogue_move(
                "私もその見方に賛同します。根拠は[D03]です。", "agree"
            ),
            "私もその見方に賛同します。",
        )
        maintained = cod_model.sanitize_dialogue_move(
            "送信前離脱率は旧12%から新18%へ上昇した。", "maintain"
        )
        self.assertTrue(maintained.startswith("結論は変わりません。"))
        orphaned, reason = cod_model.validate_dialogue_utterance(
            "私もその見方に賛同します。の結果を踏まえ、全配送へ展開します。"
        )
        self.assertIsNone(orphaned)
        self.assertIn("orphan particle", reason)
        repaired = cod_model.sanitize_dialogue_move(
            "私もその見方に賛同します。[D01]の結果を踏まえ、全配送へ展開します。",
            "agree",
        )
        self.assertIn("。その結果を踏まえ", repaired)
        repaired = cod_model.sanitize_dialogue_move(
            "結論を維持します。[D01]のデータに基づき、全networkへ展開します。",
            "maintain",
        )
        self.assertIn("。そのデータに基づき", repaired)
        repaired = cod_model.sanitize_dialogue_move(
            "D01のデータに基づき、全倉庫へ展開することを決定します。",
            "agree",
        )
        self.assertIn("。そのデータに基づき", repaired)
        self.assertEqual(
            cod_model.restore_claim_label(
                "現時点では、事前冷却を全zoneへ展開ると見ています。",
                "事前冷却を全zoneへ展開する",
            ),
            "現時点では、事前冷却を全zoneへ展開すると見ています。",
        )
        self.assertEqual(
            cod_model.restore_claim_label(
                "欠陥流出・再検柾費・生産遅延を統合するutilityを事前登録すると見ています。",
                "欠陥流出・再検査費・生産遅延を統合するutilityを事前登録する",
            ),
            "欠陥流出・再検査費・生産遅延を統合するutilityを事前登録すると見ています。",
        )
        self.assertEqual(
            cod_model.restore_claim_label(
                "確かに、光学条件を直した製品line限定pilotと人手再検柾を先に行います。",
                "光学条件を直した製品line限定pilotと人手再検査を先に行う",
            ),
            "確かに、光学条件を直した製品line限定pilotと人手再検査を先に行います。",
        )
        unrelated = "私は別の条件を先に確認すべきだと思います。"
        self.assertEqual(
            cod_model.restore_claim_label(unrelated, "欠陥検出率を優先して全製品lineへ展開する"),
            unrelated,
        )
        self.assertTrue(
            cod_model.dialogue_matches_claim(
                "都市部depotの道路工事map未更新が遅延の原因とされる可能性を検証します。",
                "都市部depotの道路工事map未更新を遅延の交絡候補として検証する",
            )
        )
        self.assertFalse(
            cod_model.dialogue_matches_claim(
                "都市部depotの道路工事map未更新を遅延の交絡候補として検証します。",
                "道路工事mapを更新した都市部depotでpilotを先に行う",
            )
        )
        self.assertTrue(
            cod_model.dialogue_matches_claim(
                "運用予算は20%までですが、全展開では41%増えています。この超過を重大視します。",
                "cloud費41%増が予算上限20%を超える事実を重大視する",
            )
        )
        self.assertTrue(
            cod_model.reaction_is_aligned(
                "全workload展開では予算を守るには不十分です。batch jobを分離したautoscaling pilotを代案にします。",
                "batch jobを分離してautoscaling pilotを先に行う",
                "予測autoscalingを全workloadへ展開する",
                "object",
            )
        )
        self.assertTrue(
            cod_model.dialogue_is_aligned(
                "その案には賛成です。満足度が上がり、処理時間も短縮できることがポイントです。",
                "処理時間と満足度の改善を優先してAI返信支援を全agentへ展開する",
                ["誤案内が集中する請求例外queueでAI返信支援pilotを先に行う"],
            )
        )
        self.assertFalse(
            cod_model.dialogue_is_aligned(
                "その案には賛成です。",
                "処理時間と満足度の改善を優先してAI返信支援を全agentへ展開する",
                ["誤案内が集中する請求例外queueでAI返信支援pilotを先に行う"],
            )
        )
        claims = [
            {"code": "PILOT", "label": "道路工事mapを更新した都市部depotでpilotを先に行う"},
            {"code": "MAP", "label": "都市部depotの道路工事map未更新を遅延の交絡候補として検証する"},
        ]
        self.assertFalse(
            cod_model.independent_utterance_is_aligned(
                "都市部depotの道路工事map未更新を遅延の交絡候補として検証します。",
                "PILOT",
                claims,
            )
        )
        natural_claims = [
            {"code": "REGION", "label": "近畿全域50〜60%を長期的な地震懸念として重く伝える"},
            {"code": "TRIANGLE", "label": "近畿三角帯40〜60%を主要な懸念として扱う"},
        ]
        self.assertTrue(
            cod_model.independent_utterance_is_aligned(
                "まずはこの広範な確率帯を無視せず、長期的な懸念として真剣に考えたいです。",
                "REGION",
                natural_claims,
            )
        )

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

    def test_event_parser_supports_ollama_backend(self):
        args = cod_model.parser().parse_args(
            [
                "event-debate",
                "--ledger",
                "ledger.json",
                "--domain",
                "general",
                "--backend",
                "ollama",
                "--ollama-model",
                "qwen3.5:4b",
                "--prompt-profile",
                "orthogonal_bare",
                "--no-renderer",
            ]
        )
        self.assertEqual(args.backend, "ollama")
        self.assertEqual(args.ollama_model, "qwen3.5:4b")
        self.assertFalse(args.portable_context)
        portable = cod_model.parser().parse_args(
            [
                "event-debate", "--ledger", "ledger.json", "--domain", "general",
                "--portable-context",
            ]
        )
        self.assertTrue(portable.portable_context)
        self.assertEqual(args.prompt_profile, "orthogonal_bare")
        self.assertTrue(args.no_renderer)
        self.assertIsNone(args.model_path)

    def test_renderer_v3_requires_exact_ids_and_fast_mode_is_bounded(self):
        values, warning = cod_model.parse_renderer_utterances(
            {
                "utterances": [
                    {"id": "C02", "utterance": "二つ目の発言です。"},
                    {"id": "C01", "utterance": "一つ目の発言です。"},
                ]
            },
            ["C01", "C02"],
        )
        self.assertIsNone(warning)
        self.assertEqual(set(values), {"C01", "C02"})
        values, warning = cod_model.parse_renderer_utterances(
            {"utterances": [{"id": "C01", "utterance": "一つだけです。"}]},
            ["C01", "C02"],
        )
        self.assertEqual(values, {})
        self.assertIn("every requested id", warning)
        settings = cod_model.event_execution_settings(
            SimpleNamespace(fast=True, max_turns=10, reconcile_rounds=2, max_tokens=600),
            4,
        )
        self.assertEqual(
            settings,
            {
                "claims_per_persona": 2,
                "max_turns": 8,
                "reconcile_rounds": 0,
                "decision_max_tokens": 320,
            },
        )

    def test_claim_body_renderer_accepts_only_safe_single_id_compatibility(self):
        self.assertIn("idをJSONキーにしてはならない", cod_model.BODY_RENDERER_SYSTEM)
        self.assertIn("claimの時制と確実性を保つ", cod_model.BODY_RENDERER_SYSTEM)
        values, warning, repaired = cod_model.parse_renderer_bodies(
            {"bodies": [{"id": "C01", "body": "段階導入を先に試します。"}]},
            ["C01"],
        )
        self.assertIsNone(warning)
        self.assertFalse(repaired)
        self.assertEqual(values["C01"], "段階導入を先に試します。")
        values, warning, repaired = cod_model.parse_renderer_bodies(
            {"C01": "段階導入を先に試すのが現実的です"},
            ["C01"],
        )
        self.assertTrue(repaired)
        self.assertIn("single-id", warning)
        body, reason = cod_model.normalize_renderer_body(values["C01"])
        self.assertIsNone(reason)
        self.assertEqual(body, "段階導入を先に試すのが現実的です。")
        values, warning, repaired = cod_model.parse_renderer_bodies(
            {"C02": "別の主張です。"},
            ["C01"],
        )
        self.assertEqual(values, {})
        self.assertFalse(repaired)
        self.assertIn("only bodies", warning)
        values, warning, repaired = cod_model.parse_renderer_bodies(
            {"B01": "短いtransport idなら長い討論idを復唱しなくて済みます。"},
            ["B01"],
        )
        self.assertTrue(repaired)
        self.assertEqual(set(values), {"B01"})
        self.assertTrue(cod_model.body_is_neutral("段階導入を先に試します。"))
        self.assertTrue(cod_model.body_is_neutral("長期的な懸念として扱います。"))
        self.assertTrue(cod_model.body_is_neutral("契約案に同意します。"))
        self.assertFalse(cod_model.body_is_neutral("その案には賛成です。"))
        self.assertFalse(cod_model.body_is_neutral("その案には懸念があります。"))
        self.assertFalse(cod_model.body_is_neutral("『段階導入する』を採ります。"))
        self.assertTrue(cod_model.body_is_polite_sentence("段階導入を先に試します。"))
        self.assertTrue(cod_model.body_is_polite_sentence("直ちには展開しません。"))
        self.assertFalse(cod_model.body_is_polite_sentence("段階導入を先に試す。"))
        self.assertFalse(cod_model.body_is_polite_sentence("削減を優先。"))
        self.assertEqual(
            cod_model.sanitize_body_politeness(
                "利用不能率9%到達を停止条件候補にする。",
                "利用不能率9%到達を停止条件候補にする",
            ),
            "利用不能率9%到達を停止条件候補にします。",
        )
        self.assertIsNone(
            cod_model.sanitize_body_politeness(
                "観測値を確認する。", "停止条件候補にする"
            )
        )
        self.assertEqual(
            cod_model.sanitize_body_politeness(
                "72時間後には位置と弱化速度が中程度の信頼に留まる。",
                "72時間以後の位置と弱化速度は中程度の信頼に留める",
            ),
            "72時間後には位置と弱化速度が中程度の信頼に留まります。",
        )
        self.assertEqual(
            cod_model.sanitize_body_politeness(
                "主経路は琉球から東シナ海を西〜南西進し中国東岸方向。",
                "主経路は琉球から東シナ海を西〜南西進し中国東岸方向",
            ),
            "主経路は琉球から東シナ海を西〜南西進し中国東岸方向と判断します。",
        )
        self.assertTrue(
            cod_model.body_matches_claim(
                "需要が集中する通勤station群でpilotを先に行います。",
                "需要が集中する通勤station群でpilotを先に行う",
            )
        )
        self.assertFalse(
            cod_model.body_matches_claim(
                "貸出不能の71%が通勤stationへ集中していました。",
                "需要が集中する通勤station群でpilotを先に行う",
            )
        )
        self.assertFalse(
            cod_model.body_matches_claim(
                "応答時間の削減を優先して自動triageを全mailへ展開した。",
                "応答時間の削減を優先して自動triageを全mailへ展開する",
            )
        )
        self.assertFalse(
            cod_model.body_matches_claim(
                "自動triageを全mailへ展開することを検証済み。",
                "自動triageを全mailへ展開する",
            )
        )
        self.assertFalse(
            cod_model.body_matches_claim(
                "自動triageの全mail展開により応答時間削減が実現されています。",
                "応答時間の削減を優先して自動triageを全mailへ展開する",
            )
        )
        self.assertFalse(
            cod_model.body_matches_claim(
                "自動triageを全mailへ展開することを提案。",
                "自動triageを全mailへ展開する",
            )
        )
        self.assertTrue(
            cod_model.body_matches_claim(
                "応答時間の削減を優先して自動triageを全mailへ展開します。",
                "応答時間の削減を優先して自動triageを全mailへ展開する",
            )
        )
        composed = cod_model.compose_dialogue_body(
            "段階導入を先に試します。",
            "段階導入を先に試す",
            "agree",
        )
        self.assertTrue(composed.startswith("その案には賛成です。加えて、"))
        metrics = cod_model.event_run_metrics(
            {
                "events": [
                    {
                        "code": "A",
                        "action": "new",
                        "origin": "model",
                        "statement": "段階導入を先に試します。根拠は[D01]です。",
                        "statement_origin": "model",
                        "utterance": composed,
                        "utterance_origin": "model_body_v2_schema_repair",
                        "renderer_cached": True,
                    }
                ],
                "independent": {},
                "renderer_batches": [{"renderer_kind": "claim_body_v2", "raw": "raw"}],
            }
        )
        self.assertEqual(metrics["body_model_utterance_rate"], 1.0)
        self.assertEqual(metrics["body_renderer_model_calls"], 1)
        self.assertEqual(metrics["body_renderer_cache_hits"], 1)
        self.assertEqual(metrics["body_schema_repairs"], 1)
        self.assertEqual(metrics["model_utterance_rate"], 0.0)

    def test_event_portable_context_embeds_ledger_and_active_personas(self):
        ledger = {"schema_version": 1, "topic": "portable"}
        context = cod_model.event_portable_context(
            ledger,
            [
                {"id": "p1", "name": "提案役"},
                {"id": "p2", "name": "反証役"},
            ],
        )
        self.assertEqual(context["portable_context_schema_version"], 1)
        self.assertIs(context["ledger_snapshot"], ledger)
        self.assertEqual(context["persona_order"], ["p1", "p2"])
        self.assertEqual(context["persona_names"], {"p1": "提案役", "p2": "反証役"})


    def test_renderer_v3_training_batch_uses_runtime_contract(self):
        persona = cod_model.load_domains()["general"]["personas"][0]
        rows = [
            {
                "messages": [
                    {"role": "system", "content": "legacy"},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "phase": "event",
                                "move": "agree_extend",
                                "own_claim": "段階導入する",
                                "target_claim": "一括導入する",
                                "evidence": ["監視期間は2週間"],
                            },
                            ensure_ascii=False,
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {"utterance": "その案に賛成です。まず段階導入で確かめましょう。"},
                            ensure_ascii=False,
                        ),
                    },
                ]
            }
        ]
        batch = cod_model.batch_renderer_examples(rows, persona, 3)[0]["messages"]
        self.assertEqual(batch[0]["content"], cod_model.renderer_system(persona))
        self.assertEqual(json.loads(batch[1]["content"])["items"][0]["move"], "agree")
        self.assertEqual(
            set(json.loads(batch[2]["content"])),
            {"utterances"},
        )
        shared = cod_model.batch_shared_renderer_examples([(persona, rows[0])], 3)[0]["messages"]
        shared_item = json.loads(shared[1]["content"])["items"][0]
        self.assertEqual(shared[0]["content"], cod_model.renderer_system(None))
        self.assertEqual(shared_item["speaker"], persona["name"])
        self.assertIn("賛同", shared_item["speech_act"])
        self.assertTrue(
            cod_model.dialogue_selects_competing_claim(
                "前案の『全面導入』を選びます。",
                ["全面導入"],
            )
        )
        restricted = "pairwiseを本線2頭専用のshadow headとして残し、公開3頭BOXの全面置換には使わない"
        reversed_utterance = (
            "その案には異議があります。代わりに、pairwiseを本線2頭専用のshadow headとして残し、"
            "公開3頭BOXを直ちに置換する方が安全です。"
        )
        self.assertTrue(cod_model.dialogue_reverses_restriction(reversed_utterance, restricted))
        self.assertFalse(cod_model.dialogue_is_aligned(reversed_utterance, restricted, []))
        self.assertFalse(
            cod_model.dialogue_reverses_restriction(
                "pairwiseは本線専用に留め、公開3頭BOXへは置換しない方が安全です。",
                restricted,
            )
        )
        self.assertFalse(
            cod_model.dialogue_reverses_restriction(
                "data4はWeight保管に使いますが、horse本体の恒久依存にはしません。",
                "data4はWeight保管に使うがhorse本体の恒久依存にはしない",
            )
        )
        incomplete, reason = cod_model.validate_dialogue_utterance(
            "pairwiseを本線2頭専用のshadow headとして残します。採用条件として、8/30のBOX改善を。"
        )
        self.assertIsNone(incomplete)
        self.assertIn("complete sentence", reason)
        self.assertFalse(
            cod_model.dialogue_is_aligned(
                "pairwiseを本線2頭専用のshadow headとして残します。",
                restricted,
                [],
            )
        )
        self.assertTrue(
            cod_model.dialogue_selects_competing_claim(
                "その案には懸念があります。代わりに、8月106競走の3頭BOXを直ちに置換する条件で試します。",
                ["8/30のBOX改善を根拠にpairwise上位3頭で公開BOXを直ちに置換する"],
            )
        )
        rejected_replacement = (
            "ただ、本線で残すpairwiseは2頭で決めるのが現実的です。"
            "代わりに、公開3頭BOXを直ちに置換する案は危険です。"
        )
        self.assertTrue(cod_model.dialogue_preserves_restriction(rejected_replacement, restricted))
        self.assertFalse(cod_model.dialogue_reverses_restriction(rejected_replacement, restricted))
        self.assertFalse(
            cod_model.dialogue_selects_competing_claim(
                rejected_replacement,
                ["8/30のBOX改善を根拠にpairwise上位3頭で公開BOXを直ちに置換する"],
            )
        )
        numeric_payload = {
            "own_claim": "on-call上限30件/日をhard constraintとして扱う",
            "evidence": ["対象は240件、処理上限は180件である。"],
        }
        self.assertTrue(
            cod_model.dialogue_numbers_are_grounded(
                "賛成です。上限30件/日と対象240件を確認しましょう。",
                numeric_payload,
            )
        )
        self.assertFalse(
            cod_model.dialogue_numbers_are_grounded(
                "賛成です。12か月ではなく18か月で再確認しましょう。",
                numeric_payload,
            )
        )

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
                    "utterance_origin": "model_sanitized",
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
        self.assertEqual(metrics["reaction_failures"], 0)
        self.assertEqual(metrics["reconciliation_model_repairs"], 0)
        self.assertFalse(metrics["hard_gate_pass"])

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
