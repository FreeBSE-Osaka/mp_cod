import copy
import json
from pathlib import Path
import re
import unittest

from tools.distributed_phone_trial import ADAPTER_SHA, JOBS, SYSTEM, validate_body, validate_phone


class DistributedPhoneTrialTest(unittest.TestCase):
    def test_fixture_and_prompt_match_the_existing_device_renderer(self):
        source = (Path(__file__).parent / "ios/ClaimBodyDeviceHarness/Sources/ClaimBodyDeviceHarnessApp.swift").read_text()
        system = re.search(r'let system = ("(?:[^"\\]|\\.)*")', source)
        self.assertEqual(json.loads(system[1]), SYSTEM)
        for persona, claim, terms in JOBS:
            self.assertIn(persona, source)
            self.assertIn(claim, source)
            self.assertTrue(all(term in claim for term in terms))

    def test_phone_requires_fresh_request_matching_jobs_actual_raw_and_safe_resources(self):
        indices = [3]
        raw = json.dumps({"bodies": [{"id": "B01", "body": "暴風が強まる前に安全確保を優先します"}]}, ensure_ascii=False)
        row = {"persona": JOBS[3][0], "claim": JOBS[3][1], "body": validate_body(raw, 3),
               "raw_output": raw, "generation_tokens": 30}
        payload = {"schema_version": 3, "mode": "distributed_body_worker", "distributed_request_id": "request",
                   "distributed_job_indices": indices, "model_weights_sha256": "base", "adapter_weights_sha256": ADAPTER_SHA,
                   "adapter_unloaded": True, "outputs_distinct": True, "thermal_state": "nominal",
                   "memory_samples": [{"stage": "after_adapter_unload", "mlx_cache_bytes": 0}],
                   "minimum_limit_bytes_remaining": 1024 ** 3, "utterances": [row]}
        self.assertEqual(validate_phone(payload, "request", indices, "base"), [row])
        for mutate in (
            lambda p: p.update(distributed_request_id="old"),
            lambda p: p.update(model_weights_sha256="wrong"),
            lambda p: p.update(distributed_job_indices=[0]),
            lambda p: p.update(thermal_state="serious"),
            lambda p: p.update(minimum_limit_bytes_remaining=0),
            lambda p: p.update(adapter_unloaded=False),
            lambda p: p.update(outputs_distinct=False),
            lambda p: p.update(memory_samples=[]),
            lambda p: p["utterances"][0].update(body="偽の出力"),
            lambda p: p["utterances"][0].update(generation_tokens=0),
        ):
            changed = copy.deepcopy(payload); mutate(changed)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                validate_phone(changed, "request", indices, "base")


if __name__ == "__main__":
    unittest.main()
