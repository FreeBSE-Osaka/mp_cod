import importlib.util
import unittest
import json
import tempfile
from pathlib import Path


class ArchitectureFailureRecordTest(unittest.TestCase):
    def test_numerical_failure_retains_prior_evidence_without_promotion(self):
        from tools.benchmark_memory_mosaics_hybrid import record_failure
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'failed.json'
            prior = [{'variant': 'mamba_mosaics', 'status': 'cpu_smoke_pass'}]
            record_failure(out, 'rwkv7_mosaics', 'gradients', ['recurrent.attn.k_k'], prior)
            result = json.loads(out.read_text())
            self.assertEqual(result['results'], prior)
            self.assertEqual(result['status'], 'failed')
            self.assertFalse(result['all_variants_completed'])
            self.assertFalse(result['promotion_allowed'])
            self.assertFalse(result['default_replacement_allowed'])
            self.assertFalse(result['whole_advisor_goal_complete'])

HAS_MLX = importlib.util.find_spec("mlx") is not None
if HAS_MLX:
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx.utils import tree_flatten
    mx.set_default_device(mx.cpu)
    mx.disable_compile()
    from tools.memory_mosaics_hybrid import ContextMemory, HybridBlock, VARIANTS


@unittest.skipUnless(HAS_MLX, "requires the existing optional MLX research environment")
class MemoryMosaicsHybridTest(unittest.TestCase):
    def setUp(self):
        mx.random.seed(240506394)
        self.x = mx.random.normal((2, 10, 16)) * 0.2

    def test_prefix_is_causal_for_every_real_variant(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                layer = HybridBlock(variant)
                original = layer(self.x)
                changed = self.x.at[:, 5:].add(3)
                altered = layer(changed)
                prefix = layer(self.x[:, :5])
                mx.eval(original, altered, prefix)
                self.assertEqual(original.shape, self.x.shape)
                self.assertTrue(bool(mx.all(mx.isfinite(original)).item()))
                np.testing.assert_allclose(np.array(original[:, :5]), np.array(altered[:, :5]), rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(np.array(original[:, :5]), np.array(prefix), rtol=1e-5, atol=1e-5)

    def test_no_cross_persona_batch_memory(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                layer = HybridBlock(variant)
                before = layer(self.x)
                after = layer(self.x.at[1].add(3))
                mx.eval(before, after)
                np.testing.assert_allclose(np.array(before[0]), np.array(after[0]), rtol=1e-5, atol=1e-5)

    def test_empty_memory_and_real_parameter_gradients(self):
        layer = ContextMemory(16, 2, delay=4)
        output = layer(self.x)
        mx.eval(output)
        np.testing.assert_array_equal(np.array(output[:, :4]), np.zeros((2, 4, 16)))
        loss, grads = nn.value_and_grad(layer, lambda model: mx.mean((model(self.x)-0.1)**2))(layer)
        mx.eval(loss, grads)
        flattened = dict(tree_flatten(grads))
        self.assertTrue(all(bool(mx.all(mx.isfinite(v)).item()) for v in flattened.values()))
        self.assertGreater(float(mx.linalg.norm(flattened['key.weight']).item()), 0)
        self.assertGreater(float(mx.linalg.norm(flattened['value.weight']).item()), 0)

    def test_invalid_dimensions_inputs_and_resource_bounds(self):
        for values in ((3, 1), (16, 3), (True, 1), (128, 2)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ContextMemory(*values)
        for variant in ("mamba2", "production", ""):
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                HybridBlock(variant)
        layer = HybridBlock()
        for x in (mx.ones((1, 65, 16)), mx.ones((5, 2, 16)), mx.ones((1, 2, 8)),
                  mx.array([[[float('nan')]*16]]), mx.ones((1, 2, 16), dtype=mx.int32),
                  mx.ones((1, 2, 16), dtype=mx.float16), mx.full((1, 2, 16), 100)):
            with self.subTest(shape=x.shape), self.assertRaises(ValueError):
                layer(x)
        with self.assertRaises(ValueError):
            layer([1, 2, 3])

    def test_zero_vector_memory_gradients_are_finite(self):
        layer = ContextMemory(16, 2)
        zeros = mx.zeros((1, 8, 16))
        loss, grads = nn.value_and_grad(layer, lambda model: mx.mean((model(zeros)-.1)**2))(layer)
        mx.eval(loss, grads)
        self.assertTrue(all(bool(mx.all(mx.isfinite(v)).item())for _, v in tree_flatten(grads)))

    def test_fresh_rwkv_key_initialization_has_finite_fixture_gradients(self):
        for variant in ('rwkv7_attention', 'rwkv7_mosaics'):
            with self.subTest(variant=variant):
                layer = HybridBlock(variant)
                self.assertTrue(bool(mx.all(layer.recurrent.attn.k_k == 1).item()))
                loss, grads = nn.value_and_grad(layer, lambda model: mx.mean((model(self.x)-self.x)**2))(layer)
                mx.eval(loss, grads)
                self.assertTrue(all(bool(mx.all(mx.isfinite(v)).item())for _, v in tree_flatten(grads)))


if __name__ == "__main__":
    unittest.main()
