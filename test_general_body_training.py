import copy
from pathlib import Path
import tempfile
import unittest
import json
import hashlib
import io
import sys
from contextlib import redirect_stdout
from unittest.mock import patch
from types import ModuleType, SimpleNamespace

import cod_model as cod
from tools.general_body_training import body_checks, build, cases, evaluate, example, guarded_job, rescore, train, valid


class GeneralDiscussionTest(unittest.TestCase):
    def test_v9_reuses_train_only_and_compares_layer_keys_with_fixed_system(self):
        root = Path(__file__).parent
        path = root / "data/general_body_qwen35_v9/curated.json"
        rows = cases(path, {"train", "valid", "test"})
        prior = cases(root / "data/general_body_qwen35_v8/curated.json", {"train", "valid", "test"})
        self.assertEqual({r["claim"] for r in rows if r["split"] == "train"},
                         {r["claim"] for r in prior if r["split"] == "train"})
        old = []
        for corpus in root.glob("data/general_body*/curated.json"):
            if corpus != path:
                old.extend(cases(corpus, {"train", "valid", "test"}))
        old.extend(cases(root / "data/general_body_input_diagnostic_20261005/fresh.json", {"test"}))
        self.assertFalse({r["claim"] for r in rows if r["split"] in {"valid", "test"}}
                         & {r["claim"] for r in old})
        for split, count in (("train", 96), ("valid", 8), ("test", 16)):
            self.assertEqual(sum(r["split"] == split for r in rows), count)
            if split != "train":
                self.assertEqual(len({r["speaker"] for r in rows if r["split"] == split}), 8)
        for row in rows:
            self.assertTrue(valid(body_checks(row["body"], row)), row["case"])
            if row.get("counterexample"):
                self.assertFalse(valid(body_checks(row["counterexample"], row)), row["case"])
        mlp = (root / "configs/claim-body-qwen35-4b-v9-mlp.yaml").read_text()
        attention = (root / "configs/claim-body-qwen35-4b-v9-attention.yaml").read_text()
        additions = "    - linear_attn.in_proj_qkv\n    - self_attn.q_proj\n    - self_attn.v_proj\n"
        self.assertEqual(attention.replace(additions, ""), mlp)
        record = json.loads((root / "promotions/qwen3.5-4b-claim-body-v9-mlp-step48.json").read_text())
        self.assertEqual(record["base_model"], "Qwen3.5-4B-4bit")
        self.assertEqual(record["corpus_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertTrue(record["same_original_system_both_modes"])
        for flag in ("promotion_allowed", "parent_replacement_allowed", "automatic_publish_allowed"):
            self.assertIs(record[flag], False)

    def test_trial_tense_survives_internal_unconfirmed_clauses_without_anchors(self):
        for stem, progressive, past in (("変更を試み", "変更を試みています", "変更を試みました"),
                                        ("短縮を試し", "短縮を試しています", "短縮を試しました")):
            current = stem + "ているが負担への効果は未確認"
            completed = stem + "たが負担への効果は未確認"
            current_body = progressive + "が、負担への効果は未確認です。"
            past_body = past + "が、負担への効果は未確認です。"
            self.assertTrue(cod.body_matches_claim(current_body, current))
            self.assertTrue(cod.body_matches_claim(past_body, completed))
            self.assertFalse(cod.body_matches_claim(past_body, current))
            self.assertFalse(cod.body_matches_claim(current_body, completed))
            self.assertIsNone(cod.sanitize_body_politeness(past_body, current))
        self.assertEqual(cod.body_trial_states("短縮を試みています。"),
                         cod.body_trial_states("短縮を試している"))
        self.assertTrue(cod.body_matches_claim("貸出の表示を統一して、混乱の減少を試みています。",
                                              "貸出の表示を統一して混乱の減少を試している"))
        self.assertEqual(cod.body_trial_states("変更を試みていないが効果は未確認"), {"negated"})
        self.assertEqual(cod.body_trial_states("短縮は試さない"), {"negated"})
        self.assertEqual(cod.body_trial_states("短縮は試しません"), {"negated"})
        self.assertFalse(cod.body_matches_claim("短縮を試していますが、効果は未確認です。",
                                               "短縮を試していないが効果は未確認"))

    def test_v8_contrast_pairs_are_authored_and_previous_final_answers_are_not_trained(self):
        root = Path(__file__).parent
        path = root / "data/general_body_qwen35_v8/curated.json"
        current = cases(path, {"train", "valid", "test"})
        self.assertEqual({s: sum(r["split"] == s for r in current) for s in ("train", "valid", "test")},
                         {"train": 96, "valid": 8, "test": 16})
        self.assertEqual(len(current), len({r["claim"] for r in current}))
        prior = cases(root / "data/general_body_qwen35_v7/curated.json", {"train", "valid", "test"})
        train = {r["claim"] for r in current if r["split"] == "train"}
        self.assertEqual(train & {r["claim"] for r in prior},
                         {r["claim"] for r in prior if r["split"] == "train"})
        previous = []
        for old in root.glob("data/general_body*/curated.json"):
            if old != path:
                previous.extend(cases(old, {"train", "valid", "test"}))
        self.assertFalse(train & {r["claim"] for r in previous if r["split"] in {"valid", "test"}})
        fresh = {r["claim"] for r in current if r["split"] in {"valid", "test"}}
        self.assertFalse(fresh & {r["claim"] for r in previous})
        for split in ("valid", "test"):
            self.assertEqual(len({r["speaker"] for r in current if r["split"] == split}), 8)
        for row in current:
            self.assertTrue(valid(body_checks(row["body"], row)), row["case"])
            if row.get("counterexample"):
                self.assertFalse(valid(body_checks(row["counterexample"], row)), row["case"])

    def test_v7_rehearses_train_only_and_keeps_fresh_dual_mode_cases_separate(self):
        root = Path(__file__).parent
        current = cases(root / "data/general_body_qwen35_v7/curated.json", {"train", "valid", "test"})
        self.assertEqual({s: sum(r["split"] == s for r in current) for s in ("train", "valid", "test")},
                         {"train": 48, "valid": 8, "test": 16})
        self.assertEqual(len(current), len({r["claim"] for r in current}))
        prior = cases(root / "data/general_body_qwen35_v6/curated.json", {"train", "valid", "test"})
        self.assertEqual({r["claim"] for r in current if r["split"] == "train"}
                         & {r["claim"] for r in prior}, {r["claim"] for r in prior if r["split"] == "train"})
        previous = []
        for path in root.glob("data/general_body*/curated.json"):
            if path.parent.name != "general_body_qwen35_v7":
                previous.extend(cases(path, {"train", "valid", "test"}))
        fresh = {r["claim"] for r in current if r["split"] in {"valid", "test"}}
        self.assertFalse(fresh & {r["claim"] for r in previous})
        for split in ("valid", "test"):
            self.assertEqual(len({r["speaker"] for r in current if r["split"] == split}), 8)
        for row in current:
            self.assertTrue(valid(body_checks(row["body"], row)), row["case"])
            if row.get("counterexample"):
                self.assertFalse(valid(body_checks(row["counterexample"], row)), row["case"])
        system = (root / "configs/claim-body-qwen35-4b-v7-system.txt").read_text().strip()
        self.assertTrue(system.startswith(cod.BODY_RENDERER_SYSTEM))

    def test_dual_input_rehearsal_keeps_same_sources_and_targets_without_routing(self):
        from collections import defaultdict
        curated = Path(__file__).parent / "data/general_body_qwen35_v6/curated.json"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old"
            old.mkdir()
            for split in ("train", "valid", "test"):
                rows = [example(f"旧手順{split}{i}を維持する", f"旧手順{split}{i}を維持します。", "仮説構築者")
                        for i in range(3)]
                (old / f"{split}.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows))
            plain = Path(__file__).parent / "configs/claim-body-qwen35-4b-v5-system.txt"
            hinted = Path(__file__).parent / "configs/claim-body-qwen35-4b-v7-system.txt"
            args = SimpleNamespace(curated=curated, rehearsal=old, out=root / "first",
                                   renderer_system_file=hinted, constraint_hints=True,
                                   plain_system_file=plain, rehearsal_train_limit=2)
            with redirect_stdout(io.StringIO()):
                build(args)
                args.out = root / "second"
                build(args)
            self.assertEqual((root / "first/train.jsonl").read_bytes(), (root / "second/train.jsonl").read_bytes())
            manifest = json.loads((args.out / "manifest.json").read_text())
            self.assertEqual(manifest["input_modes"], ["source_hints", "plain"])
            self.assertEqual(manifest["counts"]["train"]["rehearsal"], 4)
            self.assertEqual(manifest["counts"]["train"]["curated"], 512)
            grouped = defaultdict(list)
            for line in (args.out / "train.jsonl").read_text().splitlines():
                row = json.loads(line)
                item = json.loads(row["messages"][1]["content"])["items"][0]
                grouped[item["claim"], item["speaker"]].append(row)
            for pair in grouped.values():
                self.assertEqual(len(pair), 2)
                self.assertEqual(pair[0]["messages"][2], pair[1]["messages"][2])
                self.assertEqual({row["messages"][0]["content"] for row in pair},
                                 {plain.read_text().strip(), hinted.read_text().strip()})
                self.assertEqual(sum("rendering_hints" in row["messages"][1]["content"] for row in pair), 1)
            args.out = root / "invalid"
            args.constraint_hints = False
            with self.assertRaisesRegex(ValueError, "requires constraint hints"):
                build(args)

    def test_renderer_hints_are_source_only_and_default_contract_is_unchanged(self):
        claim = "予算は1900円までで人数は最低7名"
        old = example(claim, "TARGET_SENTINEL", "仮説構築者")
        self.assertEqual(json.loads(old["messages"][1]["content"]),
                         {"items": [{"id": "B01", "speaker": "仮説構築者", "claim": claim}]})
        new = example(claim, "TARGET_SENTINEL", "仮説構築者", constraint_hints=True, kind="condition")
        payload = json.loads(new["messages"][1]["content"])
        item = payload["items"][0]
        self.assertEqual(item, cod.body_input_item("B01", "仮説構築者", claim,
                                                  constraint_hints=True, kind="condition"))
        self.assertEqual(item["rendering_hints"], {
            "claim_kind": "condition", "quantity_bounds": [
                {"value_fraction": "7", "unit": "人", "relation": ">="},
                {"value_fraction": "1900", "unit": "円", "relation": "<="},
            ]})
        self.assertNotIn("TARGET_SENTINEL", new["messages"][1]["content"])
        self.assertNotIn("required", new["messages"][1]["content"])
        self.assertNotIn("body", item)
        unknown = cod.body_input_item("B01", "仮説構築者", "方法を変えて混雑に備える", constraint_hints=True)
        self.assertEqual(unknown["rendering_hints"]["claim_kind"], "unspecified")
        for kind in ("fact", [], 5):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "unknown source claim kind"):
                cod.body_input_item("B01", "A", claim, constraint_hints=True, kind=kind)

    def test_renderer_hints_require_adapter_before_any_model_load(self):
        ledger = Path(__file__).parent / "data/general_language_choice/claim_ledger.json"
        args = cod.parser().parse_args(["event-debate", "--domain", "general", "--backend", "mlx",
                                       "--model-path", "unused", "--ledger", str(ledger)])
        self.assertFalse(args.body_constraint_hints)
        args.body_constraint_hints = True
        with self.assertRaisesRegex(ValueError, "hints require --body-adapter"):
            cod.run_event_debate(args)

    def test_v6_hints_corpus_has_new_topics_and_eight_speakers(self):
        root = Path(__file__).parent
        new = cases(root / "data/general_body_qwen35_v6/curated.json", {"train", "valid", "test"})
        self.assertEqual({s: sum(r["split"] == s for r in new) for s in ("train", "valid", "test")},
                         {"train": 32, "valid": 8, "test": 16})
        self.assertEqual(len(new), len({r["claim"] for r in new}))
        for split in ("valid", "test"):
            self.assertEqual(len({r["speaker"] for r in new if r["split"] == split}), 8)
        old = []
        for path in root.glob("data/general_body*/curated.json"):
            if path.parent.name not in {"general_body_qwen35_v6", "general_body_qwen35_v7", "general_body_qwen35_v8", "general_body_qwen35_v9"}:
                old.extend(cases(path, {"train", "valid", "test"}))
        old.extend(cases(root / "data/pdca_general_body/curated.json", {"valid", "test"}))
        self.assertFalse({r["claim"] for r in new} & {r["claim"] for r in old})
        for row in new:
            with self.subTest(case=row["case"]):
                self.assertIn(row["kind"], cod.DISCUSSION_KINDS)
                self.assertTrue(valid(body_checks(row["body"], row)))
                if row.get("counterexample"):
                    self.assertFalse(valid(body_checks(row["counterexample"], row)))
                request = example(row["claim"], "unused", row["speaker"],
                                  constraint_hints=True, kind=row["kind"])["messages"][1]["content"]
                self.assertNotIn('"required"', request)
                self.assertNotIn('"counterexample"', request)
                self.assertNotIn('"body"', request)

    def test_v5_corpus_is_frozen_separated_and_preserves_each_target(self):
        root = Path(__file__).parent
        rows = cases(root / "data/general_body_qwen35_v5/curated.json", {"train", "valid", "test"})
        self.assertEqual({s: sum(r["split"] == s for r in rows) for s in ("train", "valid", "test")},
                         {"train": 32, "valid": 8, "test": 16})
        self.assertEqual(len(rows), len({r["claim"] for r in rows}))
        for split in ("valid", "test"):
            self.assertEqual(len({r["speaker"] for r in rows if r["split"] == split}), 8)
        previous = []
        for path in ("data/general_body_qwen35_v2/curated.json", "data/general_body_qwen35_v3/curated.json",
                     "data/general_body_qwen35_v4/curated.json", "data/general_body_qwen35_v4/confirmed_holdout.json",
                     "data/pdca_general_body/curated.json"):
            previous.extend(cases(root / path, {"train", "valid", "test"}))
        self.assertFalse({r["claim"] for r in rows} & {r["claim"] for r in previous})
        for row in rows:
            with self.subTest(case=row["case"]):
                self.assertTrue(valid(body_checks(row["body"], row)))
                if row.get("counterexample"):
                    self.assertFalse(valid(body_checks(row["counterexample"], row)))
        self.assertEqual((root / "configs/claim-body-qwen35-4b-v5-system.txt").read_text().strip(),
                         cod.BODY_RENDERER_SYSTEM)

    def test_native_training_masks_nonthinking_prefix_and_restores_state(self):
        for fails in (False, True):
            with self.subTest(fails=fails), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                parent = root / "parent.safetensors"
                parent.write_text("unit-test placeholder")
                args = SimpleNamespace(out=root / "new", parent_adapter=parent, model=root / "model",
                                       data=root / "data", config=root / "config.yaml")
                settings = []
                class Tokenizer:
                    def apply_chat_template(self, messages, **kwargs):
                        settings.append(kwargs)
                        return "template"
                class Wrapper:
                    def __init__(self):
                        object.__setattr__(self, "_tokenizer", Tokenizer())
                    def apply_chat_template(self, *a, **k):
                        k.setdefault("enable_thinking", True)
                        return self._tokenizer.apply_chat_template(*a, **k)
                    def __setattr__(self, name, value):
                        setattr(self._tokenizer, name, value)
                package = ModuleType("mlx_lm")
                native = SimpleNamespace(load=lambda *a, **k: (object(), Wrapper()))
                original_load, original_argv = native.load, sys.argv
                def main():
                    self.assertIn("--mask-prompt", sys.argv)
                    self.assertIn(str(parent), sys.argv)
                    _, tokenizer = native.load(str(args.model))
                    tokenizer.apply_chat_template([], add_generation_prompt=True)
                    tokenizer.apply_chat_template([], return_dict=False)
                    tokenizer.apply_chat_template([], enable_thinking=True)
                    if fails:
                        raise RuntimeError("training failure")
                native.main = main
                package.lora = native
                with patch.dict(sys.modules, {"mlx_lm": package}):
                    if fails:
                        with self.assertRaisesRegex(RuntimeError, "training failure"):
                            train(args)
                    else:
                        train(args)
                self.assertTrue(all(s["enable_thinking"] is False for s in settings))
                self.assertIs(native.load, original_load)
                self.assertIs(sys.argv, original_argv)
                args.out.mkdir()
                (args.out / "protected").write_text("existing")
                with self.assertRaisesRegex(ValueError, "already contains"):
                    train(args)

    def test_body_job_guard_prevents_heavy_work_and_records_terminal_errors(self):
        from cod_pdca import _ResourceGuard
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(out=root / "low.json", command="evaluate",
                                   min_free_gib=20, resource_check_seconds=1)
            calls = []
            with patch.object(_ResourceGuard, "_observe", return_value={"ok": False, "reason": "low_free_space"}):
                self.assertEqual(guarded_job(args, lambda a: calls.append(a)), 2)
            self.assertFalse(calls)
            record = json.loads((root / "low.json.job.json").read_text())
            self.assertEqual(record["status"], "resource_stopped")
            args.min_free_gib = 0
            args.out = root / "failure.json"
            def fail(_):
                raise RuntimeError("measurement failed")
            with self.assertRaisesRegex(RuntimeError, "measurement failed"):
                guarded_job(args, fail)
            self.assertEqual(json.loads((root / "failure.json.job.json").read_text())["status"], "failed")
            args.out = root / "exit.json"
            def exit_two(_):
                raise SystemExit(2)
            self.assertEqual(guarded_job(args, exit_two), 2)
            self.assertEqual(json.loads((root / "exit.json.job.json").read_text())["returncode"], 2)
            with self.assertRaisesRegex(ValueError, "journal already exists"):
                guarded_job(args, lambda _: None)

    def test_v4_action_does_not_imply_verified_benefit(self):
        rows = cases(Path(__file__).parent / "data/general_body_qwen35_v4/curated.json", {"train", "valid", "test"})
        self.assertEqual({s: sum(r["split"] == s for r in rows) for s in ("train", "valid", "test")},
                         {"train": 32, "valid": 8, "test": 16})
        self.assertEqual(len(rows), len({r["claim"] for r in rows}))
        self.assertEqual(len({r["speaker"] for r in rows if r["split"] == "test"}), 8)
        previous = cases(Path(__file__).parent / "data/general_body_qwen35_v3/curated.json", {"valid", "test"})
        self.assertFalse({r["claim"] for r in rows if r["split"] == "train"} & {r["claim"] for r in previous})
        for row in rows:
            with self.subTest(case=row["case"]):
                self.assertTrue(valid(body_checks(row["body"], row)))
                if row.get("counterexample"):
                    self.assertFalse(valid(body_checks(row["counterexample"], row)))

    def test_runtime_preserves_confirmation_without_evaluation_anchors(self):
        rows = cases(Path(__file__).parent / "data/general_body_qwen35_v4/confirmed_holdout.json", {"test"})
        for row in rows:
            with self.subTest(case=row["case"]):
                self.assertTrue(cod.body_matches_claim(row["body"], row["claim"]))
                self.assertFalse(cod.body_matches_claim(row["counterexample"], row["claim"]))
                self.assertFalse(body_checks(row["counterexample"], row)["aligned"])
        claim = "音量設定の変更による聴取の負担軽減は検証済み"
        for body in (
            "音量設定の変更による聴取の負担軽減は未確認です。",
            "音量設定の変更による聴取の負担軽減です。",
            "音量設定の変更による聴取の負担軽減は検証済みではありません。",
        ):
            with self.subTest(body=body):
                self.assertFalse(cod.body_matches_claim(body, claim))
                self.assertIsNone(cod.sanitize_body_politeness(body, claim))
        self.assertTrue(cod.body_matches_claim("音量設定の変更による聴取の負担軽減は、検証済みです。", claim))
        self.assertTrue(cod.body_matches_claim("音量設定の変更による聴取の負担軽減を確認しました。", claim))
        self.assertTrue(cod.body_matches_claim("負担の軽減を検証済みです。", "負担の軽減を確認した"))
        self.assertFalse(cod.body_matches_claim("負担の軽減を確認しました。", "負担の軽減は未確認"))
        self.assertTrue(cod.body_matches_claim("負担の軽減はまだ確認していません。", "負担の軽減は未確認"))
        self.assertTrue(cod.body_matches_claim("負担の軽減を試しています。", "負担の軽減を試している"))
        fallback, origin = cod.compose_dialogue_fallback(
            "音量設定の変更による聴取の負担軽減は未確認です。", claim,
            "agree", flexible=True, frozen_only=True)
        self.assertEqual(origin, "frozen_claim_fallback")
        self.assertIn("検証済み", fallback)
        self.assertNotIn("未確認", fallback)

    def test_runtime_citation_failure_keeps_frozen_fallback_after_repair(self):
        roster = cod.load_domains()["general"]["personas"][:2]
        ledger = {"schema_version": 1, "topic": "架空の引用検査",
                  "data": [{"id": "D05", "text": "移植工数は未算定。"},
                           {"id": "D06", "text": "安全性の優劣は未実測。"}],
                  "claim_catalog": [{"code": code, "label": label, "kind": "proposal",
                                     "supported_by": ["D06"], "contradicts": [other]}
                                    for code, other, label in (("P", "Q", "安全性の実測を先に行う"),
                                                               ("Q", "P", "安全性の検証計画を先に整理する"))],
                  "role_preferences": {p["id"]: ["P", "Q"] for p in roster}}
        malicious = "Swiftコアなら移植不要で安全性も確保されています。根拠は[D05]です。"
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
                    result = {"claims": [{"code": code, "data_ids": ["D06"], "confidence": 75,
                                          "statement": malicious}]}
                else:
                    result = {"choice": "LEFT", "data_ids": ["D06"],
                              "statement": malicious, "change_reason": malicious}
                return result, {"_raw_content": json.dumps(result)}
            with patch.object(cod, "ask_ollama", side_effect=respond), redirect_stdout(io.StringIO()):
                self.assertEqual(cod.run_event_debate(args), 0)
            run = json.loads(next((Path(directory)/"runs").glob("event_debate_*.json")).read_text())
            claims = [claim for entry in run["independent"].values() for claim in entry["valid"]]
            votes = list(run["reconciliation"][0]["votes"]["P|Q"].values())
            self.assertEqual(len(claims), 2)
            self.assertEqual(len(votes), 2)
            for row in claims + votes:
                with self.subTest(row=row):
                    self.assertEqual(row["statement_origin"], "label_fallback")
                    self.assertNotIn("移植不要", row["statement"])
                    self.assertNotIn("D05", row["statement"])
                    self.assertEqual(cod.validate_public_statement(row["statement"], ["D06"])[1], None)
            self.assertTrue(all(v["repair_raw"] for v in votes))

    def test_empty_custom_renderer_system_fails_before_loading_mlx(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "empty.txt"
            p.write_text(" ")
            with self.assertRaisesRegex(ValueError, "empty evaluation renderer system"):
                evaluate(SimpleNamespace(renderer_system_file=p))

    def test_confirmed_effect_holdout_preserves_explicit_evidence(self):
        rows = cases(Path(__file__).parent / "data/general_body_qwen35_v4/confirmed_holdout.json", {"test"})
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({r["speaker"] for r in rows}), 8)
        for row in rows:
            with self.subTest(case=row["case"]):
                self.assertTrue(valid(body_checks(row["body"], row)))
                self.assertFalse(valid(body_checks(row["counterexample"], row)))

    def test_v3_temporal_corpus_keeps_plan_ongoing_and_unverified_effect_distinct(self):
        rows = cases(Path(__file__).parent / "data/general_body_qwen35_v3/curated.json", {"train", "valid", "test"})
        self.assertEqual({s: sum(r["split"] == s for r in rows) for s in ("train", "valid", "test")},
                         {"train": 32, "valid": 8, "test": 16})
        self.assertEqual(len(rows), len({r["claim"] for r in rows}))
        self.assertEqual(len({r["speaker"] for r in rows if r["split"] == "test"}), 8)
        old = cases(Path(__file__).parent / "data/general_body_qwen35_v2/curated.json", {"train", "valid", "test"})
        self.assertFalse({r["claim"] for r in rows} & {r["claim"] for r in old})
        for row in rows:
            with self.subTest(case=row["case"]):
                self.assertTrue(valid(body_checks(row["body"], row)))
                if row.get("counterexample"):
                    self.assertFalse(valid(body_checks(row["counterexample"], row)))

    def test_body_failure_fallback_does_not_publish_unverified_base_reason(self):
        label = "共通コアを試験導入して将来の移植に備える"
        text, origin = cod.compose_dialogue_fallback(
            "安全性は実測で確認済みなので全面導入します。[D01]", label, "counterproposal",
            flexible=True, frozen_only=True)
        self.assertEqual(origin, "frozen_claim_fallback")
        self.assertNotIn("確認済み", text)
        self.assertNotIn("全面導入", text)
        self.assertIn("将来の移植に備える", text)

    def test_v2_scope_corpus_preserves_constraints_and_rejects_counterexamples(self):
        rows = cases(Path(__file__).parent / "data/general_body_qwen35_v2/curated.json", {"train", "valid", "test"})
        self.assertEqual({s: sum(r["split"] == s for r in rows) for s in ("train", "valid", "test")},
                         {"train": 32, "valid": 8, "test": 16})
        self.assertEqual(len(rows), len({r["claim"] for r in rows}))
        self.assertEqual(len({r["speaker"] for r in rows if r["split"] == "test"}), 8)
        old = cases(Path(__file__).parent / "data/general_body_qwen35_v1_holdout/curated.json", {"test"})
        self.assertFalse({r["claim"] for r in rows} & {r["claim"] for r in old})
        for row in rows:
            with self.subTest(case=row["case"]):
                self.assertTrue(valid(body_checks(row["body"], row)))
                if row.get("counterexample"):
                    self.assertFalse(valid(body_checks(row["counterexample"], row)))

    def test_every_restricted_action_must_survive_rendering(self):
        claim = "安全な共同作業のため個人メモは保存しないし集計結果は公開しない"
        self.assertFalse(cod.dialogue_preserves_restriction("安全な共同作業のため集計結果は公開しません。", claim))
        self.assertFalse(cod.body_matches_claim("安全な共同作業のため集計結果は公開しません。", claim))
        self.assertTrue(cod.body_matches_claim("安全な共同作業のため個人メモは保存せず、集計結果も公開しません。", claim))
        self.assertTrue(cod.body_matches_claim("未承認の資料は使いません。", "未承認の資料は使用しない"))
        self.assertFalse(cod.dialogue_preserves_restriction("集計結果は公開しません。", "集計結果は公開しないし未承認の資料は使用しない"))
        self.assertFalse(cod.body_matches_claim("安全な共同作業のため個人メモは保存し、集計結果は公開しません。", claim))

    def test_named_exclusions_cannot_flip_or_move_to_another_target(self):
        claim = "計測対象は傷果率17%と運搬時間23分で売上は含めない"
        wrong = "計測対象は傷果率17%と運搬時間23分で売上は含めます。"
        self.assertFalse(cod.body_matches_claim(wrong, claim))
        self.assertIsNone(cod.sanitize_body_politeness(wrong, claim))
        for body in (
            "計測対象は傷果率17%と運搬時間23分で、売上は含めません。",
            "計測対象は傷果率17%と運搬時間23分で、売上を含めずに計測します。",
            "計測対象は傷果率17%と運搬時間23分で、売上は対象に含みません。",
            "計測対象は傷果率17%と運搬時間23分で、売上は除外します。",
        ):
            self.assertTrue(cod.body_matches_claim(body, claim), body)
        mixed = "見学者は人数に含めないが受付係は人数に含める"
        self.assertTrue(cod.body_matches_claim("見学者は人数に含めませんが、受付係は人数に含めます。", mixed))
        for body in (
            "見学者は人数に含めますが、受付係は人数に含めません。",
            "受付係は人数に含めます。",
            "見学者は人数に含めませんが、見学者も人数に含めます。",
            "臨時見学者は人数に含めませんが、見学者は人数に含めます。",
        ):
            self.assertFalse(cod.body_matches_claim(body, mixed), body)
        self.assertTrue(cod.body_matches_claim("受付係も人数に含めます。", "受付係も人数に含める"))

    def test_rehearsal_limit_is_deterministic_and_rejects_zero(self):
        curated = Path(__file__).parent / "data/general_body_v5/curated.json"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old"
            old.mkdir()
            for split in ("train", "valid", "test"):
                rows = [{"messages": [{"role": "system", "content": "old"},
                          {"role": "user", "content": json.dumps({"items": [{"claim": f"過去の教材{split}{i}は確認する", "speaker": "仮説構築者"}]})},
                          {"role": "assistant", "content": json.dumps({"bodies": [{"body": f"過去の教材{split}{i}は確認します。"}]})}]} for i in range(5)]
                (old / f"{split}.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            args = SimpleNamespace(curated=curated, rehearsal=old, rehearsal_train_limit=2,
                renderer_system_file=Path(__file__).parent / "configs/claim-body-qwen35-4b-v1-system.txt", out=root / "first")
            with redirect_stdout(io.StringIO()):
                build(args)
                args.out = root / "second"
                build(args)
            self.assertEqual((root / "first/train.jsonl").read_bytes(), (root / "second/train.jsonl").read_bytes())
            manifest = json.loads((args.out / "manifest.json").read_text())
            self.assertEqual(manifest["counts"]["train"]["rehearsal"], 2)
            self.assertEqual(manifest["counts"]["valid"]["rehearsal"], 5)
            self.assertEqual(manifest["counts"]["test"]["rehearsal"], 5)
            args.rehearsal_train_limit = 0
            args.out = root / "invalid"
            with self.assertRaisesRegex(ValueError, "must be positive"):
                build(args)

    def test_percentage_units_and_signs_cannot_change(self):
        payload = {"claim": "電力費18%とCO2 11%の削減を優先する"}
        self.assertFalse(cod.dialogue_numbers_are_grounded("電力費18割とCO2 11割の削減を優先します。", payload))
        self.assertFalse(cod.dialogue_numbers_are_grounded("電力費-18%の削減を優先します。", payload))
        self.assertTrue(cod.dialogue_numbers_are_grounded("電力費１８％とCO2 １１パーセントの削減を優先します。", payload))
        self.assertTrue(cod.dialogue_numbers_are_grounded("費用を2割削減します。", {"claim": "費用を2割削減する"}))

    def test_quantity_bounds_cannot_be_dropped_or_changed(self):
        claim = "共有用具の予算は2600円までで飲食費は含めない"
        dropped = "共有用具の予算は2600円で、飲食費は含めません。"
        self.assertFalse(cod.body_matches_claim(dropped, claim))
        self.assertIsNone(cod.sanitize_body_politeness(dropped, claim))
        text, origin = cod.compose_dialogue_fallback(dropped, claim, "agree", flexible=True, frozen_only=True)
        self.assertEqual(origin, "frozen_claim_fallback")
        self.assertIn("2600円まで", text)
        for body in (
            "共有用具の予算は2600円以下で、飲食費は含めません。",
            "共有用具の予算は最大2600円で、飲食費は含めません。",
            "共有用具の予算は2600円を上限とし、飲食費は含めません。",
            "共有用具の予算は2600円を超えませんが、飲食費は含めません。",
        ):
            with self.subTest(body=body):
                self.assertTrue(cod.body_matches_claim(body, claim))
        for body in ("予算は2600円未満です。", "予算は2600円以上です。"):
            self.assertNotEqual(cod.body_numeric_relations(body), cod.body_numeric_relations(claim))
        for source, body in (
            ("見積額が5400円を超える場合だけ再確認する", "見積額が5400円以上の場合だけ再確認します。"),
            ("参加者は最低8人で見学者は数えない", "参加者は8人で、見学者は数えません。"),
            ("見学は最長3日で人数は10人まで", "見学は3日で、人数は10人までです。"),
            ("受入れは20人未満で荷物は預からない", "受入れは20人以下で、荷物は預かりません。"),
        ):
            with self.subTest(source=source):
                self.assertFalse(cod.body_matches_claim(body, source))

    def test_numeric_relation_equivalences_are_quantity_specific(self):
        for source, body in (
            ("料金は1200円まで", "料金の上限は１，２００円です"),
            ("参加者は最低8人", "参加者は8名以上"),
            ("負担率は12%以下", "負担率は最大12パーセント"),
            ("点検は2週間以内", "点検は最長2週"),
            ("定員は20人以上ではない", "定員は20人未満"),
            ("代金は500円以下ではありません", "代金は500円を超える"),
            ("料金は0.5万円まで", "料金は.5万円以下"),
            ("団体見学の上限は1回18人", "団体見学は1回18人まで"),
        ):
            with self.subTest(source=source):
                self.assertEqual(cod.body_numeric_relations(source), cod.body_numeric_relations(body))
        self.assertNotEqual(cod.body_numeric_relations("予算は700円までで人数は7人以上"),
                            cod.body_numeric_relations("予算は700円以上で人数は7人まで"))
        self.assertNotEqual(cod.body_numeric_relations("代金は600円まで"),
                            cod.body_numeric_relations("代金は600円以下かつ600円以上"))
        self.assertFalse(cod.body_numeric_relations("Qwen3.5はローカルで使う"))

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
                     "質問の時間も含めるです。", "仮の案内だけを出すです。", "仮の案内だけを出す、です。",
                     "状況を確認する, です。", "元の運用へ戻す です。"):
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
        self.assertFalse(cod.body_matches_claim("初期工数を増やして将来の移植に備えています。", "初期工数を増やして将来の移植に備える"))
        self.assertTrue(cod.body_matches_claim("初期工数を増やして将来の移植に備えています。", "初期工数を増やして将来の移植に備えている"))
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
            system = Path(directory) / "empty_system.txt"
            system.write_text(" ")
            args.body_system_file = str(system)
            with self.assertRaisesRegex(ValueError, "custom body system requires --body-adapter"):
                cod.run_event_debate(args)
            args.body_adapter = "unused"
            with self.assertRaisesRegex(ValueError, "empty body renderer system"):
                cod.run_event_debate(args)
            args.body_system_file = None
            args.body_adapter = None
            args.body_cache_scope = "speaker"
            with self.assertRaisesRegex(ValueError, "requires --body-adapter"):
                cod.run_event_debate(args)
            args.body_cache_scope = "claim"
            counter = 0
            def respond(**kwargs):
                nonlocal counter
                payload = json.loads(kwargs["user"])
                if "claim_catalog" in payload:
                    self.assertEqual(payload["response_contract"]["type"], "object")
                    self.assertEqual(payload["response_contract"]["required"], ["claims"])
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
            self.assertEqual(run["execution"]["decision_temperature"], args.temperature)
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
