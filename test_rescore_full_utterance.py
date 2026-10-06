import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.general_utterance_training import cases, example, profile_case
from tools.rescore_full_utterance import rescore


class SavedFullUtteranceRescoreTest(unittest.TestCase):
    def test_legacy_metadata_needs_exact_request_and_inputs_are_never_rewritten(self):
        curated = Path(__file__).parent / "data/general_utterance_qwen35_v2/curated.json"
        case = profile_case(next(c for c in cases(curated) if c["case"] == "sample_tags:3"), "flexible_plain")
        row = {"case": case["case"], "profile": "flexible_plain", "request": example(case)["messages"][:2],
               "raw": json.dumps({"utterances": [{"id": case["id"], "utterance": case["utterance"]}]}, ensure_ascii=False),
               "strict_schema": True, "direct_valid": True, "study_valid": True}
        payload = {"results": [row]}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "saved.json"
            source.write_text(json.dumps(payload, ensure_ascii=False))
            original = source.read_bytes()
            result = rescore(source, curated)
            self.assertEqual(source.read_bytes(), original)
            self.assertTrue(result["selection_unchanged"])
            self.assertFalse(result["promotion_allowed"])
            self.assertTrue(result["rows"][0]["current"]["study_valid"])
            for mutate in (
                lambda p: p.update(curated_sha256="wrong"),
                lambda p: p["results"][0].update(claim="different"),
                lambda p: p["results"][0]["request"][1].update(content="different"),
                lambda p: p["results"].append(copy.deepcopy(p["results"][0])),
            ):
                changed = copy.deepcopy(payload); mutate(changed)
                source.write_text(json.dumps(changed, ensure_ascii=False))
                before = source.read_bytes()
                with self.assertRaises(ValueError):
                    rescore(source, curated)
                self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
