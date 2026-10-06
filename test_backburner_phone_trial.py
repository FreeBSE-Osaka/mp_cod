import copy
import unittest

from tools.backburner_phone_trial import complete_response, offload_proof, phone_status, strict_json, validate_phone_status


class BackburnerPhoneTrialTest(unittest.TestCase):
    def test_requires_phone_compute_and_state_merge_without_fallback(self):
        before, after = {"tail_chunks": 4}, {"tail_chunks": 7}
        line = "split_finish: split 192 tokens done: wait 10 ms, merge 2 ms, phone ms/chunk [4 4 4]"
        proof = offload_proof(before, after, line)
        self.assertTrue(proof["verified"])
        self.assertEqual(proof["merged_remote_tokens"], 192)
        self.assertFalse(offload_proof(before, before, line)["verified"])
        self.assertFalse(offload_proof(before, after, "connected")["verified"])
        self.assertFalse(offload_proof(before, after, line + "\nsplit prefill off for 60 s")["verified"])
        self.assertFalse(offload_proof(before, after, line + "\nrerunning the batch locally")["verified"])

    def test_refuses_wifi_public_loopback_and_ipv6_before_connecting(self):
        for address in ("192.168.0.102", "8.8.8.8", "127.0.0.1", "::1", "fe80::1", "hostname"):
            with self.subTest(address=address), self.assertRaises(ValueError):
                phone_status(address)

    def test_rejects_duplicate_json_keys_and_nonfinite_numbers(self):
        for data in ('{"a":1,"a":2}', '{"a":{"x":1,"x":2}}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}'):
            with self.subTest(data=data), self.assertRaises(ValueError):
                strict_json(data)
        self.assertEqual(strict_json('{"a":1}'), {"a": 1})

    def test_machine_status_requires_real_numeric_resources_and_safe_state(self):
        good = {"thermal": 0, "avail_mb": 1024, "tail_chunks": 4, "cable_if_type": 2,
                "wifi_paired": False, "tail_state": "ready"}
        self.assertEqual(validate_phone_status(good), good)
        for key, value in (("thermal", False), ("thermal", 2), ("avail_mb", float("nan")),
                           ("avail_mb", 511), ("tail_chunks", True), ("tail_chunks", -1),
                           ("wifi_paired", True), ("cable_if_type", 1), ("tail_state", "error")):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_phone_status(good | {key: value})
        with self.assertRaises(ValueError):
            offload_proof({"tail_chunks": False}, {"tail_chunks": 1}, "split_finish: split 64 tokens done:")

    def test_response_requires_actual_tokens_complete_eos_and_fresh_prefill(self):
        good = {"stop": True, "stop_type": "eos", "truncated": False, "content": "未確認です",
                "tokens": [12, 13], "tokens_predicted": 2, "timings": {"cache_n": 0, "prompt_n": 485}}
        self.assertTrue(complete_response(good, 485))
        for change in ({"tokens": [True, 13]}, {"tokens_predicted": 0}, {"stop_type": "limit"},
                       {"truncated": True}, {"content": ""}, {"tokens": []},
                       {"timings": {"cache_n": 1, "prompt_n": 484}}):
            mutated = copy.deepcopy(good); mutated.update(change)
            self.assertFalse(complete_response(mutated, 485))


if __name__ == "__main__":
    unittest.main()
