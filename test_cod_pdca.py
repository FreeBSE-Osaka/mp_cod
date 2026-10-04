import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cod_pdca


EVALUATOR = '''import json, os, pathlib, subprocess, sys, time
request = json.load(sys.stdin)
x = request["candidate"]["x"]
if x == -1:
    print("not JSON")
    sys.exit(0)
if x == -2:
    print("broken process", file=sys.stderr)
    sys.exit(7)
if x == -3:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    (pathlib.Path(request["output_dir"]) / "worker_pids.json").write_text(json.dumps([os.getpid(),child.pid]))
    time.sleep(5)
if x == -4:
    print('{"schema_version":1,"score":NaN,"passed":true,"metrics":{},"failures":[],"artifacts":[]}')
    sys.exit(0)
if x == -5:
    print('{"schema_version":1,"score":0,"passed":false,"metrics":{"nested":1e999},"failures":[],"artifacts":[]}')
    sys.exit(0)
artifact = pathlib.Path(request["output_dir"]) / "evidence.json"
artifact.write_text(json.dumps(request))
score = x if request["phase"] == "development" else 2-x
passed = x >= 2 and x != 3 if request["phase"] == "development" else True
print(json.dumps({"schema_version":1,"score":score,"passed":passed,
    "metrics":{"x":x,"phase":request["phase"]},
    "failures":[] if x >= 2 else [{"reason":"x must reach 2", "raw":"observed failure"}],
    "artifacts":[str(artifact)]}))
'''


class PdcaTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        evaluator = self.root / "evaluator.py"
        evaluator.write_text(EVALUATOR, encoding="utf-8")
        self.task = {"schema_version": 1, "objective": "Improve the observed score", "domain": "general",
                     "initial_candidate": {"x": 0}, "candidate_schema": {
                         "type": "object", "properties": {"x": {"type": "integer", "minimum": -5, "maximum": 9}},
                         "required": ["x"], "additionalProperties": False},
                     "experiment": {"argv": [sys.executable, str(evaluator)], "timeout_seconds": 2},
                     "target_score": 2}
        self.args = SimpleNamespace(task=str(self.root / "task.json"), out=str(self.root / "run"),
                                    model="test", api_url="http://unused", timeout=2, num_predict=500,
                                    seed=33, max_rounds=2, min_rounds=2,
                                    personas=["hypothesis_builder", "falsifier"], min_free_gib=0)
        self.calls = []

    def execute(self, responder):
        Path(self.args.task).write_text(json.dumps(self.task), encoding="utf-8")
        def ask(**kwargs):
            prompt = json.loads(kwargs["user"])
            self.calls.append((prompt, kwargs))
            response = responder(prompt)
            return response, {"_raw_content": json.dumps(response), "eval_count": 10}
        with patch.object(cod_pdca.cod, "ask_ollama", side_effect=ask):
            self.assertEqual(cod_pdca.run_pdca(self.args), 0)
        return json.loads((Path(self.args.out) / "result.json").read_text())

    @staticmethod
    def responder(prompt):
        ids = [item["id"] for item in prompt["measurements"]]
        if prompt["stage"] == "proposal":
            value = 1 if not prompt["previous_round"] else 2
            return {"candidate": {"x": value}, "rationale": "Try a measured change", "measurement_ids": [ids[-1]]}
        if prompt["stage"] == "review":
            proposal = next(p for p in prompt["proposals"] if p["accepted"] and not p.get("duplicate_of"))
            return {"selected_candidate_id": proposal["candidate_id"], "critique": "Run this hypothesis", "measurement_ids": [ids[-1]]}
        return {"observations": "Observed actual experiment", "next_change": "Use x=2 next", "measurement_ids": [ids[-1]]}

    def test_real_experiment_feedback_and_frozen_holdout(self):
        self.task["holdout"] = dict(self.task["experiment"])
        result = self.execute(self.responder)
        self.assertEqual(result["status"], "improved")
        self.assertEqual(result["best_measurement_id"], "M002")
        self.assertEqual([m["score"] for m in result["measurements"]], [0, 1, 2, 2, 0])
        self.assertEqual(result["holdout"]["status"], "regression")
        self.assertFalse(result["holdout"]["supported_improvement"])
        self.assertFalse(result["promotion_allowed"])
        self.assertEqual(result["api_url"], self.args.api_url)
        self.assertEqual(result["timeout"], self.args.timeout)
        self.assertEqual(result["num_predict"], self.args.num_predict)
        self.assertEqual(result["temperature"], 0.35)
        self.assertEqual(len(result["engine_sha256"]), 64)
        self.assertEqual(len(result["controller_sha256"]), 64)
        second = next(p for p, _ in self.calls if p["stage"] == "proposal" and p["previous_round"])
        self.assertEqual(second["previous_round"]["reflection"]["next_change"], "Use x=2 next")
        self.assertEqual(second["last_experiment"]["candidate"], {"x": 1})
        self.assertEqual(second["last_experiment"]["result"]["failures"][0]["raw"], "observed failure")
        self.assertEqual(second["previous_round"]["review"]["selected_candidate_id"], "R01P01")
        self.assertTrue(all("H000" not in kw["user"] and "H001" not in kw["user"] for _, kw in self.calls))
        self.assertEqual(len({kw["seed"] for _, kw in self.calls}), len(self.calls))
        for measurement in result["measurements"]:
            directory = Path(measurement["directory"])
            self.assertEqual(json.loads((directory / "measurement.json").read_text()), measurement)
            request = json.loads((directory / "evidence.json").read_text())
            self.assertEqual(request["candidate"], measurement["candidate"])
            self.assertEqual(measurement["candidate_sha256"], cod_pdca.candidate_hash(request["candidate"]))
        self.assertEqual(json.loads((Path(self.args.out) / "best_candidate.json").read_text()), {"x": 2})
        self.assertEqual(json.loads((Path(self.args.out) / "journal.json").read_text()), result)

    def test_bad_initial_measurement_can_recover_and_failed_round_is_retained(self):
        self.task["initial_candidate"] = {"x": -1}
        result = self.execute(self.responder)
        self.assertEqual(result["status"], "recovered")
        self.assertIsNone(result["measurements"][0]["score"])
        self.assertEqual(result["best_measurement_id"], "M002")
        initial_prompt = self.calls[0][0]
        self.assertEqual(initial_prompt["last_experiment"]["status"], "error")
        self.assertEqual(initial_prompt["last_experiment"]["candidate"], {"x": -1})

    def test_project_domains_use_their_existing_roster(self):
        self.args.personas = None
        for domain in ("software", "weather"):
            with self.subTest(domain=domain):
                self.task["domain"] = domain
                self.args.out = str(self.root / domain)
                result = self.execute(self.responder)
                self.assertEqual(result["domain"], domain)
                self.assertEqual(result["personas"], [p["id"] for p in cod_pdca.cod.load_domains()[domain]["personas"]])

    def test_unknown_evidence_is_rejected_once_then_repaired(self):
        bad_once = True
        def responder(prompt):
            nonlocal bad_once
            response = self.responder(prompt)
            if bad_once:
                bad_once = False
                response["measurement_ids"] = ["M999"]
            return response
        result = self.execute(responder)
        self.assertFalse(result["calls"][0]["accepted"])
        self.assertIn("M999", result["calls"][0]["response"]["measurement_ids"])
        self.assertIn("Unknown/missing", self.calls[1][0]["repair"]["error"])
        self.assertEqual(len(result["measurements"]), 3)

    def test_latest_measurement_is_required_even_when_old_citation_exists(self):
        def responder(prompt):
            response = self.responder(prompt)
            if prompt["required_measurement_id"] != "M000" and "repair" not in prompt:
                response["measurement_ids"] = ["M000"]
            return response
        result = self.execute(responder)
        failures = [call for call in result["calls"] if not call["accepted"]]
        self.assertTrue(failures)
        self.assertTrue(all("Latest measurement ID" in call["error"] for call in failures))
        self.assertEqual(result["best_measurement_id"], "M002")

    def test_text_limits_are_enforced_in_schema_and_local_repair(self):
        def responder(prompt):
            response = self.responder(prompt)
            if "repair" not in prompt:
                key = {"proposal": "rationale", "review": "critique", "reflection": "observations"}[prompt["stage"]]
                response[key] = "長" * (cod_pdca.TEXT_LIMITS[key] + 1)
            return response
        result = self.execute(responder)
        self.assertEqual(result["best_measurement_id"], "M002")
        self.assertTrue(all("exceeds" in call["error"] for call in result["calls"] if not call["accepted"]))
        examples = {
            "proposal": {"candidate": {"x": 1}, "rationale": "理由", "measurement_ids": ["M000"]},
            "review": {"selected_candidate_id": "P1", "critique": "批評", "measurement_ids": ["M000"]},
            "reflection": {"observations": "観測", "next_change": "変更", "measurement_ids": ["M000"]},
        }
        for stage, response in examples.items():
            schema = cod_pdca._response_schema(stage, self.task["candidate_schema"], ["M000"])
            for key, limit in cod_pdca.TEXT_LIMITS.items():
                if key not in response:
                    continue
                with self.subTest(stage=stage, key=key):
                    self.assertEqual(schema["properties"][key]["maxLength"], limit)
                    boundary = {**response, key: "文" * limit}
                    cod_pdca._validate_response(boundary, stage, self.task["candidate_schema"], {"M000"}, {"P1"}, "M000")
                    with self.assertRaisesRegex(ValueError, "exceeds"):
                        cod_pdca._validate_response({**boundary, key: boundary[key] + "文"}, stage,
                                                    self.task["candidate_schema"], {"M000"}, {"P1"}, "M000")

    def test_reflection_uses_selected_experiment_without_repeating_all_proposals(self):
        result = self.execute(self.responder)
        for prompt, request in self.calls:
            self.assertIn("実測データを優先", request["system"])
            self.assertIn("誤りがあり得る", request["system"])
            self.assertFalse({"argv", "cwd", "directory", "timeout_seconds"} & prompt["last_experiment"].keys())
            self.assertEqual(set(prompt["last_experiment"]["result"]), {"metrics", "failures"})
            if prompt["stage"] == "review":
                self.assertEqual(len(prompt["proposals"]), len(self.args.personas))
            if prompt["stage"] == "reflection":
                current = prompt["current_round"]
                self.assertNotIn("proposals", current)
                self.assertNotIn("previous_round", prompt)
                self.assertEqual(current["selected_proposal"]["candidate_id"], current["review"]["selected_candidate_id"])
                self.assertEqual(prompt["last_experiment"]["id"], current["measurement_id"])
                self.assertEqual(prompt["required_measurement_id"], current["measurement_id"])
                self.assertEqual(prompt["last_experiment"]["candidate"], current["selected_proposal"]["response"]["candidate"])
        self.assertTrue(all(len(row["proposals"]) == len(self.args.personas) for row in result["rounds"]))
        self.assertTrue(all("argv" in row and "directory" in row for row in result["measurements"]))

    def test_unknown_reviewer_and_reflection_ids_cannot_become_evidence(self):
        def responder(prompt):
            response = self.responder(prompt)
            if prompt["stage"] in ("review", "reflection"):
                response["measurement_ids"] = ["invented"]
            return response
        result = self.execute(responder)
        self.assertEqual(len(result["measurements"]), 1)
        self.assertTrue(all(r["decision"] == "review_rejected" for r in result["rounds"]))
        self.assertTrue(all(r["reflection"] is None for r in result["rounds"]))

    def test_malformed_candidates_are_rejected_and_measured_duplicates_not_executed(self):
        def responder(prompt):
            response = self.responder(prompt)
            if prompt["stage"] == "proposal":
                response["candidate"] = {"x": 0} if "repair" in prompt else {"x": 1, "argv": "untrusted"}
            return response
        result = self.execute(responder)
        self.assertEqual(len(result["measurements"]), 1)
        self.assertTrue(all(r["decision"] == "no_new_valid_candidate" for r in result["rounds"]))
        self.assertTrue(all(not p["accepted"] for r in result["rounds"] for p in r["proposals"]))
        self.assertEqual(sum(not call["accepted"] for call in result["calls"]), 8)
        self.assertFalse(any(prompt["stage"] == "review" for prompt, _ in self.calls))

    def test_measured_duplicate_is_repaired_by_model_and_independent_agreement_remains_allowed(self):
        def responder(prompt):
            response = self.responder(prompt)
            if prompt["stage"] == "proposal" and "repair" not in prompt:
                response["candidate"] = prompt["last_experiment"]["candidate"]
                response["rationale"] = "Claiming an edit without actually changing the candidate"
            return response
        result = self.execute(responder)
        self.assertEqual([m["candidate"] for m in result["measurements"]], [{"x": 0}, {"x": 1}, {"x": 2}])
        for row in result["rounds"]:
            proposals = row["proposals"]
            self.assertTrue(all(p["accepted"] for p in proposals))
            self.assertEqual(proposals[1]["duplicate_of"], proposals[0]["candidate_id"])
            proposal_calls = [c for c in result["calls"] if c["stage"] == "proposal" and c["round"] == row["round"]]
            self.assertEqual([c["attempt"] for c in proposal_calls], [1, 2, 1, 2])
            self.assertEqual([c["accepted"] for c in proposal_calls], [False, True, False, True])
            self.assertTrue(all("identical to measured" in c["error"] for c in proposal_calls if not c["accepted"]))
            blind_prompts = [c["user"] for c in proposal_calls if c["attempt"] == 1]
            self.assertEqual(blind_prompts[0], blind_prompts[1])
        for prompt, request in self.calls:
            self.assertIn("system_promptを含む", request["system"])
            self.assertIn("編集対象データ", request["system"])
            if prompt["stage"] == "proposal":
                self.assertNotIn("proposals", prompt)
                self.assertNotIn("current_round", prompt)
                if "repair" in prompt:
                    self.assertIn("revise candidate itself", prompt["repair"]["error"])

    def test_repeated_measured_duplicate_stops_after_one_repair_without_execution(self):
        def responder(prompt):
            response = self.responder(prompt)
            if prompt["stage"] == "proposal":
                response["candidate"] = {"x": 0}
            return response
        result = self.execute(responder)
        self.assertEqual(len(result["measurements"]), 1)
        self.assertTrue(all(r["decision"] == "no_new_valid_candidate" for r in result["rounds"]))
        calls = [c for c in result["calls"] if c["stage"] == "proposal"]
        self.assertEqual(len(calls), self.args.max_rounds * len(self.args.personas) * 2)
        self.assertTrue(all(not c["accepted"] and "identical to measured M000" in c["error"] for c in calls))
        self.assertFalse(any(c["stage"] == "review" for c in result["calls"]))

    def test_real_nonimprovement_and_process_failures_keep_parent(self):
        self.task["initial_candidate"] = {"x": 2}
        self.task["target_score"] = 9
        self.args.max_rounds = 6
        self.task["experiment"]["timeout_seconds"] = 0.5
        values = [1, -1, -2, -3, -4, -5]
        def responder(prompt):
            response = self.responder(prompt)
            if prompt["stage"] == "proposal":
                round_number = prompt["previous_round"]["round"] if prompt["previous_round"] else 0
                response["candidate"] = {"x": values[round_number]}
            return response
        result = self.execute(responder)
        self.assertEqual(result["status"], "no_improvement")
        self.assertEqual(result["best_measurement_id"], "M000")
        self.assertEqual([m["status"] for m in result["measurements"]], ["measured", "measured"] + ["error"] * 5)
        self.assertIn("timed out", result["measurements"][4]["error"])
        self.assertEqual(result["measurements"][3]["returncode"], 7)
        self.assertTrue(all(r["decision"] == "best_retained" for r in result["rounds"]))

    def test_passed_then_score_retains_feasible_and_recovers_at_lower_score(self):
        for initial, proposal, expected_best, improved in ((2, 3, "M000", False), (3, 2, "M001", True)):
            with self.subTest(initial=initial):
                self.args.out = str(self.root / f"policy-{initial}")
                self.task["initial_candidate"] = {"x": initial}
                self.task["target_score"] = 9
                def responder(prompt):
                    response = self.responder(prompt)
                    if prompt["stage"] == "proposal":
                        response["candidate"] = {"x": proposal if not prompt["previous_round"] else 1}
                    return response
                result = self.execute(responder)
                self.assertEqual(result["comparison_policy"], "passed_then_score")
                self.assertEqual(result["best_measurement_id"], expected_best)
                self.assertEqual(result["development_improved"], improved)

    def test_interrupt_stops_owned_evaluator_group_and_preserves_journal(self):
        self.task["initial_candidate"] = {"x": -3}
        self.task["experiment"]["timeout_seconds"] = 10
        Path(self.args.task).write_text(json.dumps(self.task), encoding="utf-8")
        script = "import json,sys; from types import SimpleNamespace; import cod_pdca; cod_pdca.run_pdca(SimpleNamespace(**json.loads(sys.argv[1])))"
        process = subprocess.Popen([sys.executable, "-c", script, json.dumps(vars(self.args))],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        pid_file = Path(self.args.out) / "measurements/M000/worker_pids.json"
        try:
            deadline = time.monotonic() + 4
            while not pid_file.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(pid_file.exists())
            worker_pids = json.loads(pid_file.read_text())
            process.send_signal(signal.SIGINT)
            process.communicate(timeout=4)
            journal = json.loads((Path(self.args.out) / "journal.json").read_text())
            self.assertEqual(journal["status"], "interrupted")
            self.assertEqual(journal["measurements"][0]["status"], "interrupted")
            for pid in worker_pids:
                status = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
                self.assertTrue(not status or status.startswith("Z"), f"Worker still running: {pid} {status}")
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
            if pid_file.exists():
                for pid in json.loads(pid_file.read_text()):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_candidate_and_schema_boundaries_and_output_reuse(self):
        schema = self.task["candidate_schema"]
        for candidate in ({"x": True}, {"x": 10}, {"x": 1, "argv": "execute"}, {}):
            with self.assertRaises(ValueError):
                cod_pdca.validate_candidate(candidate, schema)
        unsupported = json.loads(json.dumps(schema))
        unsupported["properties"]["x"]["pattern"] = ".*"
        with self.assertRaises(ValueError):
            cod_pdca.validate_candidate({"x": 1}, unsupported)
        self.execute(self.responder)
        with self.assertRaises(ValueError):
            cod_pdca.run_pdca(self.args)
        self.args.min_rounds = 3
        with self.assertRaises(ValueError):
            cod_pdca.run_pdca(self.args)

    def test_model_generation_and_request_time_are_bounded(self):
        for name in ("timeout", "num_predict"):
            original = getattr(self.args, name)
            for value in (0, -1, True, 1.5):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    setattr(self.args, name, value)
                    cod_pdca.run_pdca(self.args)
            setattr(self.args, name, original)

    def test_initial_low_or_unreadable_space_prevents_all_heavy_work(self):
        Path(self.args.task).write_text(json.dumps(self.task), encoding="utf-8")
        del self.args.min_free_gib  # The production default must stay at 20 GiB.
        for mode in ("low", "unreadable"):
            with self.subTest(mode=mode):
                self.args.out = str(self.root / f"resource-{mode}")
                def disk_usage(path):
                    if mode == "unreadable":
                        raise OSError("required filesystem unavailable")
                    return SimpleNamespace(free=1024 ** 3)
                with patch.object(cod_pdca.shutil, "disk_usage", side_effect=disk_usage), patch.object(cod_pdca.cod, "ask_ollama") as model:
                    self.assertEqual(cod_pdca.run_pdca(self.args), 2)
                    model.assert_not_called()
                journal = json.loads((Path(self.args.out) / "journal.json").read_text())
                self.assertEqual(journal["status"], "resource_stopped")
                self.assertEqual(journal["min_free_gib"], 20)
                self.assertEqual(journal["resource_check_seconds"], 2)
                self.assertFalse(journal["measurements"])
                self.assertFalse(journal["calls"])
                self.assertFalse((Path(self.args.out) / "measurements").exists())
                self.assertFalse(journal["resource_stop"]["ok"])
                self.assertTrue(journal["resource_stop"]["observed_at"])
                self.assertEqual(json.loads((Path(self.args.out) / "result.json").read_text()), journal)
        self.assertFalse(any(t.name == "cod-pdca-resource-guard" for t in threading.enumerate()))

    def test_resource_guard_interrupts_blocked_model_and_owned_evaluator(self):
        script = '''import json, pathlib, sys, threading, time
from types import SimpleNamespace
import cod_pdca
args = SimpleNamespace(**json.loads(sys.argv[1]))
mode = sys.argv[2]
out = pathlib.Path(args.out)
marker = out / "model_entered.txt" if mode == "model" else out / "measurements/M000/worker_pids.json"
cod_pdca.shutil.disk_usage = lambda path: SimpleNamespace(free=(1 if marker.exists() else 30) * 1024**3)
def blocked_model(**kwargs):
    marker.write_text("entered")
    time.sleep(30)
    raise AssertionError("Guard did not interrupt the model request")
cod_pdca.cod.ask_ollama = blocked_model
result = cod_pdca.run_pdca(args)
assert not any(t.name == "cod-pdca-resource-guard" for t in threading.enumerate())
time.sleep(.15)
(out / "guard_exited.txt").write_text("no late signal")
sys.exit(result)
'''
        self.args.min_free_gib = 20
        self.args.resource_check_seconds = .05
        for mode in ("model", "evaluator"):
            with self.subTest(mode=mode):
                self.task["initial_candidate"] = {"x": 0 if mode == "model" else -3}
                self.task["experiment"]["timeout_seconds"] = 10
                self.args.out = str(self.root / f"guard-{mode}")
                Path(self.args.task).write_text(json.dumps(self.task), encoding="utf-8")
                started = time.monotonic()
                completed = subprocess.run([sys.executable, "-c", script, json.dumps(vars(self.args)), mode],
                                           capture_output=True, text=True, timeout=5)
                self.assertEqual(completed.returncode, 2, completed.stderr)
                self.assertLess(time.monotonic() - started, 3)
                journal = json.loads((Path(self.args.out) / "journal.json").read_text())
                self.assertEqual(journal["status"], "resource_stopped")
                self.assertEqual(journal["resource_stop"]["minimum_free_gib"], 20)
                self.assertTrue((Path(self.args.out) / "guard_exited.txt").exists())
                self.assertFalse(journal["promotion_allowed"])
                if mode == "model":
                    self.assertEqual(len(journal["calls"]), 1)
                    self.assertFalse(journal["calls"][0]["accepted"])
                    self.assertEqual(journal["calls"][0]["error"], "model request interrupted")
                    self.assertEqual(journal["measurements"][0]["status"], "measured")
                else:
                    self.assertEqual(journal["measurements"][0]["status"], "interrupted")
                    pids = json.loads((Path(self.args.out) / "measurements/M000/worker_pids.json").read_text())
                    for pid in pids:
                        status = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
                        self.assertTrue(not status or status.startswith("Z"), f"Owned worker still running: {pid}")

    def test_resource_guard_rejects_background_thread_before_start(self):
        from concurrent.futures import ThreadPoolExecutor
        self.args.min_free_gib = 20
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(cod_pdca.run_pdca, self.args)
            with self.assertRaisesRegex(ValueError, "main thread of its own process"):
                future.result()
        self.assertFalse(Path(self.args.out).exists())

    def test_resource_guard_stops_after_success_and_validates_limits(self):
        self.args.min_free_gib = 20
        self.args.resource_check_seconds = .01
        with patch.object(cod_pdca.shutil, "disk_usage", return_value=SimpleNamespace(free=30 * 1024 ** 3)):
            result = self.execute(self.responder)
        self.assertEqual(result["status"], "improved")
        self.assertTrue(result["resource_observation"]["ok"])
        self.assertFalse(any(t.name == "cod-pdca-resource-guard" for t in threading.enumerate()))
        for name, values in (("min_free_gib", (-1, True, float("inf"), float("nan"))),
                             ("resource_check_seconds", (0, -1, True, float("inf")))):
            original = getattr(self.args, name)
            for value in values:
                setattr(self.args, name, value)
                with self.assertRaises(ValueError):
                    cod_pdca.run_pdca(self.args)
            setattr(self.args, name, original)


if __name__ == "__main__":
    unittest.main()
