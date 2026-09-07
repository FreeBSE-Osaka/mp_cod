import copy
import unittest

from tools.prepare_iphone_3d_shadow import shader_packet
from tools.validate_iphone_3d_shadow import audit, compare_reference, validate_body_workload
from tools.validate_iphone_native_cod import validate_fresh_body_contract


class IPhone3DShadowTest(unittest.TestCase):
    def test_fresh_body_requires_actual_calls_and_preserves_cache(self):
        native = {"body_cache_bypassed": True, "body_cache_persisted": False,
                  "body_cache_prime_mode": True, "start_thermal_state": "nominal",
                  "body_model_loaded": True, "body_adapter_loaded": True,
                  "ledger": {"claims": [{"code": "A"}, {"code": "B"}]},
                  "body_calls": [{"claim": "A"}, {"claim": "B"}],
                  "events": [{"body_origin": "model_body_v2_cache", "utterance": "新しい本文です。"}]}
        result = {"mode": "concurrent", "fresh_body": True, "cod_result": native}
        self.assertTrue(validate_body_workload(result))
        validate_fresh_body_contract(native)
        for key, value in (("body_calls", []), ("body_cache_persisted", True), ("body_model_loaded", False)):
            with self.subTest(key=key):
                bad = copy.deepcopy(result)
                bad["cod_result"][key] = value
                with self.assertRaises(ValueError):
                    validate_body_workload(bad)
                with self.assertRaises(SystemExit):
                    validate_fresh_body_contract(bad["cod_result"])
        bad = {**native, "events": [{"body_origin": "model_body_v2_persistent_cache"}]}
        with self.assertRaises(SystemExit):
            validate_fresh_body_contract(bad)
        with self.assertRaises(ValueError):
            validate_body_workload({"mode": "handoff", "fresh_body": True,
                                    "stop_inference_progress": "盲検選択: 役A"})
        reference = {**native, "hard_gate_pass": True, "events": [{"utterance": "以前の本文です。"}]}
        self.assertFalse(compare_reference(native, reference, True)["utterances_identical"])
        with self.assertRaisesRegex(ValueError, "reference hard gate"):
            compare_reference(native, {**reference, "hard_gate_pass": False}, True)
        with self.assertRaisesRegex(ValueError, "Base decisions"):
            compare_reference({**native, "initial_tally": {"B": 4}}, reference, True)

    def test_shader_extraction_rejects_interpolation_and_is_auditable(self):
        source = b'''let source = """
        vertex void regional_volume_vertex() {}
        fragment float4 regional_volume_fragment() {}
        fragment void regional_precipitation_fragment() {}
        """'''
        packet = shader_packet(source)
        self.assertTrue(packet["shader"].startswith("vertex void"))
        self.assertEqual(len(packet["source_sha256"]), 64)
        with self.assertRaises(ValueError):
            shader_packet(source.replace(b'vertex void', b'\\(unsafe) vertex void'))

    def test_handoff_gate_rejects_early_3d_and_retained_mlx(self):
        result = {
            "schema_version": 1, "physical_device": True, "production_integration_allowed": False,
            "input_kind": "synthetic_fixed_cloud_fields_v1", "renderer": "ExtremeWeather Regional3DMetalVolumeView",
            "source_sha256": "a" * 64, "shader_sha256": "b" * 64,
            "render_width": 384, "render_height": 256, "target_fps": 20, "ray_steps": 24,
            "mode": "handoff", "cancelled": True, "stop_reason": "3d_requested",
            "stop_requested_at": 4.0, "inference_finished_at": 4.5,
            "phases": [{"name": name, "started_at": start, "ended_at": end} for name, start, end in
                       [("baseline_3d", 0, 3), ("inference", 3, 5), ("recovery_3d", 5, 8)]],
            "frames": [{"phase": name, "submitted_at": start + i * .05,
                        "completed_at": start + i * .05 + .01, "gpu_seconds": .01, "succeeded": True}
                       for name, start in [("baseline_3d", 0), ("recovery_3d", 5)] for i in range(60)],
            "samples": [{"thermal": "nominal", "memory": {"stage": "after_inference_release",
                         "limit_bytes_remaining": 1024 * 1_048_576, "footprint_peak_bytes": 123,
                         "mlx_active_bytes": 0, "mlx_cache_bytes": 0}}],
        }
        self.assertEqual(audit(result)["status"], "valid")
        invalid = copy.deepcopy(result)
        invalid["frames"].append({"phase": "inference", "submitted_at": 3.5,
                                  "completed_at": 3.51, "gpu_seconds": .01, "succeeded": True})
        with self.assertRaisesRegex(ValueError, "before MLX release"):
            audit(invalid)
        invalid = copy.deepcopy(result)
        invalid["samples"][0]["memory"]["mlx_active_bytes"] = 2 * 1_048_576
        with self.assertRaisesRegex(ValueError, "buffers remained"):
            audit(invalid)


if __name__ == "__main__":
    unittest.main()
